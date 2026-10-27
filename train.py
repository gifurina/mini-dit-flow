"""训练 Fashion-MNIST 条件 DiT，保存日志、模型和固定噪声生成图。

用法：python train.py --config configs/default.yaml
续训：在配置中设置 train.resume，再运行同一命令。
"""

import csv
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from config import parse_config, select_device
from data import create_loaders, create_test_loader, save_preview
from flow import euler_sample, flow_matching_loss
from model import DiT


# 辅助评估：只计算损失，不反向传播、不更新参数。
@torch.no_grad()
def evaluate(model, loader, device, seed, amp_enabled=False, amp_dtype=torch.bfloat16):
    """用固定噪声和时间评估 MSE，让各轮验证结果具有可比性。"""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    total = torch.zeros((), device=device)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            loss = flow_matching_loss(model, images, labels, generator)
        total += loss * len(images)
    return total.item() / len(loader.dataset)


def save_checkpoint(path, state):
    """先写临时文件再替换，避免保存中断破坏上一份 checkpoint。"""
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def main():
    # ===== 一、准备工作：读取配置，设置随机种子和运行设备 =====
    config, checkpoints = parse_config("train")
    resume = config["train"]["resume"]

    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["num_threads"])
    device = select_device(config["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # 准备训练/验证数据、模型和优化器；测试集仅在训练结束后评估。
    train_loader, val_loader = create_loaders(
        **config["data"], seed=seed, pin_memory=device.type == "cuda",
    )
    model = DiT(**config["model"]).to(device)
    settings = config["train"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    amp_enabled = device.type == "cuda" and settings["amp"]
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    # BF16 不需要梯度缩放；不支持 BF16 的 CUDA 设备使用 FP16 + GradScaler。
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled and amp_dtype == torch.float16)
    start_epoch, best_val = 1, float("inf")
    # 续训时恢复参数、优化器和随机状态，使训练能从断点继续。
    if resume:
        checkpoint = checkpoints["resume"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_val = checkpoint["epoch"] + 1, checkpoint["best_val"]
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        train_loader.generator.set_state(checkpoint["loader_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])

    # 准备输出目录、日志和配置快照。
    output = Path(config["output_dir"])
    for name in ("checkpoints", "logs", "samples"):
        (output / name).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            output / "logs/train.log", mode="a" if resume else "w", encoding="utf-8",
        )],
    )
    with (output / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    # 准备固定类别和噪声，仅用于定期生成预览、观察训练效果。
    per_class = config["sample"]["samples_per_class"]
    sample_labels = torch.arange(config["model"]["num_classes"], device=device).repeat_interleave(per_class)
    fixed_generator = torch.Generator(device=device).manual_seed(seed + 1)
    fixed_noise = torch.randn(
        len(sample_labels), model.in_channels, model.img_size, model.img_size,
        device=device, generator=fixed_generator,
    )
    logging.info("Device: %s; parameters: %s; train/val: %s/%s; AMP: %s",
                 device, f"{sum(p.numel() for p in model.parameters()):,}",
                 len(train_loader.dataset), len(val_loader.dataset), amp_dtype if amp_enabled else "off")
    if device.type == "cuda":
        logging.info("GPU: %s", torch.cuda.get_device_name(device))

    log_path = output / "logs/loss.csv"
    has_rows = resume and log_path.exists() and log_path.stat().st_size > 0
    with log_path.open("a" if has_rows else "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not has_rows:
            writer.writerow(["epoch", "train_loss", "val_loss", "seconds"])
        # ===== 二、训练阶段：每轮更新参数，再验证和保存结果 =====
        for epoch in range(start_epoch, settings["epochs"] + 1):
            started = time.perf_counter()
            model.train()
            total = torch.zeros((), device=device)
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not sys.stderr.isatty())
            # >>> 训练核心开始：遍历批次，学习从噪声到真实图像的速度场。
            for images, labels in iterator:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                # 1. 清空上一批次的梯度。
                optimizer.zero_grad(set_to_none=True)
                # 2. 采样噪声和时间，构造中间图像，让 DiT 预测速度并计算 MSE。
                #    具体的路径构造和损失公式在 flow.py 的 flow_matching_loss 中。
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    loss = flow_matching_loss(model, images, labels)
                # 遇到数值异常立即保留上一轮 checkpoint，避免继续写入无效模型。
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Epoch {epoch}: loss is not finite")
                # 3. 反向传播并更新模型参数；scaler 负责 FP16 的梯度缩放。
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total += loss.detach() * len(images)
            # <<< 训练核心结束。

            # 汇总本轮训练损失，并在验证集上评估；验证不更新参数。
            train_loss = total.item() / len(train_loader.dataset)
            val_loss = evaluate(model, val_loader, device, seed + 2, amp_enabled, amp_dtype)
            seconds = time.perf_counter() - started
            writer.writerow([epoch, train_loss, val_loss, seconds])
            handle.flush()
            logging.info("Epoch %02d/%02d train=%.6f val=%.6f time=%.1fs",
                         epoch, settings["epochs"], train_loss, val_loss, seconds)

            # 保存训练状态，按验证损失挑选最佳模型。
            improved = val_loss < best_val
            best_val = min(best_val, val_loss)
            state = {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "config": config, "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val,
                "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                "loader_rng": train_loader.generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
            }
            save_checkpoint(output / "checkpoints/last.pt", state)
            if improved:
                save_checkpoint(output / "checkpoints/best.pt", state)
            # 定期生成固定噪声预览；这一步只观察效果，不参与参数更新。
            if epoch % settings["sample_every"] == 0 or epoch == settings["epochs"]:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    samples = euler_sample(model, fixed_noise, sample_labels, config["sample"]["steps"])
                save_preview(samples.float().clamp(-1, 1), sample_labels,
                             output / "samples" / f"epoch_{epoch:03d}.png",
                             max_images=len(sample_labels), nrow=per_class)
                save_checkpoint(output / "checkpoints" / f"epoch_{epoch:03d}.pt", state)

    # ===== 三、训练结束：加载最佳模型，在官方测试集上做最终评估 =====
    # 测试集只报告结果，不用于挑选 epoch。
    best = torch.load(output / "checkpoints/best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    test_loader = create_test_loader(
        root=config["data"]["root"], batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"], download=False,
        pin_memory=device.type == "cuda",
    )
    test_loss = evaluate(model, test_loader, device, seed + 3, amp_enabled, amp_dtype)
    metrics = {"best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
               "test_loss": test_loss, "test_samples": len(test_loader.dataset)}
    (output / "logs/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    logging.info("Finished: best epoch=%d, val=%.6f, held-out test=%.6f",
                 best["epoch"], best["val_loss"], test_loss)


if __name__ == "__main__":
    main()
