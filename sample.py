"""按配置加载 checkpoint 并生成图片，可比较多个 Euler 采样步数。"""

import csv
import time
from pathlib import Path

import torch

from config import parse_config, select_device
from data import save_preview
from flow import euler_sample
from model import DiT


def main():
    # ===== 一、准备工作：读取配置，创建模型并加载已训练的权重 =====
    config, checkpoints = parse_config("sample")
    settings = config["sample"]
    device = select_device(config["device"])
    torch.set_num_threads(config["num_threads"])
    model = DiT(**config["model"]).to(device)
    model.load_state_dict(checkpoints["checkpoint"]["model"])
    model.eval()

    # ===== 二、生成条件与起点：指定类别 y，采样初始高斯噪声 x0 =====
    # 每张图片对应一个类别编号；可生成全部十类或单个指定类别。
    if settings["class_id"] is None:
        nrow = settings["samples_per_class"]
        labels = torch.arange(config["model"]["num_classes"], device=device).repeat_interleave(nrow)
    else:
        labels = torch.full(
            (settings["num_samples"],), settings["class_id"], device=device, dtype=torch.int64,
        )
        nrow = min(8, settings["num_samples"])
    steps_list = settings["compare_steps"] or [settings["steps"]]
    # 固定种子使初始噪声可复现；步数对比也复用同一份噪声。
    generator = torch.Generator(device=device).manual_seed(config["seed"])
    noise = torch.randn(
        len(labels), model.in_channels, model.img_size, model.img_size,
        device=device, generator=generator,
    )
    output = Path(settings["output"])
    timings = []
    # 以下预热和 CUDA 同步服务于计时，预热的模型输出不用于生成结果。
    with torch.no_grad():
        model(noise, torch.zeros(len(labels), device=device), labels)
    # ===== 三、生成阶段：按配置的步数积分速度场并保存图片 =====
    # 通常只生成一次；compare_steps 有多个值时，分别生成以便对比。
    for steps in steps_list:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        # >>> 生成核心开始：从 x0 出发，沿模型预测的速度场从 t=0 积分到 t=1。
        # euler_sample 内部重复执行 x = x + dt * model(x, t, labels)。
        # 这里只做前向计算，不计算梯度，也不更新模型参数。
        samples = euler_sample(model, noise, labels, steps)
        # <<< 生成核心结束：samples 是生成的图像张量，尚未转换为显示范围。
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        # 输出处理：裁剪显示范围并保存网格；save_preview 内部将 [-1, 1] 转为 [0, 1]。
        path = output.with_name(f"{output.stem}_steps_{steps}{output.suffix}") if settings["compare_steps"] else output
        save_preview(samples.float().clamp(-1, 1), labels, path, max_images=len(labels), nrow=nrow)
        timings.append([steps, len(labels), elapsed, elapsed / len(labels)])
        print(f"steps={steps}, samples={len(labels)}, seconds={elapsed:.3f}, saved={path}", flush=True)
    # 附加记录：步数对比时保存采样耗时，不影响生成过程。
    if settings["compare_steps"]:
        with output.with_name(f"{output.stem}_timings.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["steps", "samples", "seconds", "seconds_per_image"])
            writer.writerows(timings)


if __name__ == "__main__":
    main()
