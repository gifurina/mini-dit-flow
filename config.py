"""训练和生成共用的配置读取与启动校验。"""

import argparse
import math
from pathlib import Path

import torch
import yaml


def positive_int(value):
    return type(value) is int and value > 0


def nonnegative_int(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def path(value):
    return isinstance(value, str) and bool(value.strip())


def optional_path(value):
    return value is None or path(value)


# 所有字段都显式写在 YAML 中；这里集中定义类型和取值范围。
COMMON = {
    "seed": lambda v: nonnegative_int(v) and v < 2**63,
    "device": path,
    "num_threads": positive_int,
}
SECTIONS = {
    "data": {
        "root": path, "batch_size": positive_int,
        "val_size": lambda v: positive_int(v) and v < 60000,
        "num_workers": nonnegative_int, "download": lambda v: type(v) is bool,
    },
    "model": {
        "img_size": lambda v: type(v) is int and v == 28,
        "patch_size": positive_int,
        "in_channels": lambda v: type(v) is int and v == 1,
        "emb_size": positive_int, "depth": positive_int,
        "num_heads": positive_int, "mlp_ratio": positive_int,
        "num_classes": lambda v: type(v) is int and v == 10,
    },
    "train": {
        "epochs": positive_int,
        "learning_rate": lambda v: number(v) and v > 0,
        "weight_decay": lambda v: number(v) and v >= 0,
        "amp": lambda v: type(v) is bool,
        "sample_every": positive_int, "resume": optional_path,
    },
    "sample": {
        "checkpoint": path, "output": path,
        "steps": positive_int, "samples_per_class": positive_int,
        "class_id": lambda v: v is None or (type(v) is int and 0 <= v < 10),
        "num_samples": positive_int,
        "compare_steps": lambda v: v is None or (
            isinstance(v, list) and len(v) > 0 and all(positive_int(n) for n in v)
        ),
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(f"配置错误：{message}")


def validate_fields(values, rules, prefix):
    require(isinstance(values, dict), f"{prefix} 必须是映射")
    require(values.keys() == rules.keys(),
            f"{prefix} 字段不匹配；缺少 {sorted(rules.keys() - values.keys())}，"
            f"多余 {sorted(values.keys() - rules.keys())}")
    for key, rule in rules.items():
        require(rule(values[key]), f"{prefix}.{key} 的值不合法：{values[key]!r}")


def validate_model(model):
    validate_fields(model, SECTIONS["model"], "model")
    require(model["img_size"] % model["patch_size"] == 0, "model.patch_size 必须整除 img_size")
    require(model["emb_size"] % 4 == 0, "model.emb_size 必须是 4 的倍数")
    require(model["emb_size"] % model["num_heads"] == 0,
            "model.emb_size 必须能被 num_heads 整除")


def select_device(name):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def load_config(filename, mode):
    """校验配置并读取所需 checkpoint；成功后才允许进入训练或生成。"""
    with open(filename, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    sections = ("data", "model", "train", "sample")
    rules = dict(COMMON, **{name: lambda v: isinstance(v, dict) for name in sections})
    rules["output_dir"] = path
    validate_fields(config, rules, "config")
    for name in sections:
        if name == "model":
            validate_model(config[name])
        else:
            validate_fields(config[name], SECTIONS[name], name)

    device = select_device(config["device"])
    require(device.type in ("cpu", "cuda"), "device 只支持 auto、cpu、cuda 或 cuda:N")
    if device.type == "cuda":
        require(torch.cuda.is_available(), "指定了 CUDA，但当前 CUDA 不可用")
        require(device.index is None or device.index < torch.cuda.device_count(),
                "device 的 CUDA 编号超出范围")

    if mode == "train":
        paths = {"resume": config["train"]["resume"]}
    else:
        paths = {"checkpoint": config["sample"]["checkpoint"]}
        require(Path(config["sample"]["output"]).suffix.lower() in (".png", ".jpg", ".jpeg"),
                "sample.output 必须是 .png、.jpg 或 .jpeg 图片路径")
    checkpoints = {}
    for key, filename in paths.items():
        if filename is None:
            continue
        require(Path(filename).is_file(), f"{mode}.{key} 文件不存在：{filename}")
        checkpoint = torch.load(filename, map_location="cpu", weights_only=True)
        require(checkpoint["config"]["model"] == config["model"],
                f"model 与 {key} 中保存的模型配置不一致")
        if mode == "train":
            require(checkpoint["epoch"] < config["train"]["epochs"],
                    "train.epochs 必须大于续训 checkpoint 的已完成轮数")
        checkpoints[key] = checkpoint
    return config, checkpoints


def parse_config(mode):
    parser = argparse.ArgumentParser(description=f"{mode}：从 YAML 读取全部运行参数")
    parser.add_argument("--config", default="configs/default.yaml", help="配置文件路径")
    return load_config(parser.parse_args().config, mode)
