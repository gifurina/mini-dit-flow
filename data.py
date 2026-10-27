"""Fashion-MNIST 数据加载、归一化和图片预览。"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def load_dataset(root="data", train=True, download=False):
    """加载官方训练集或测试集，图片为 [1, 28, 28]，像素范围 [-1, 1]。"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    return datasets.FashionMNIST(
        root=str(root), train=train, download=download, transform=transform,
    )


def create_loaders(
    root="data", batch_size=128, val_size=5000, num_workers=0,
    download=False, seed=42, pin_memory=False,
):
    """固定划分训练/验证集；训练集打乱，验证集保持顺序，均保留尾批。"""
    dataset = load_dataset(root, train=True, download=download)
    train_set, val_set = random_split(
        dataset, [len(dataset) - val_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    options = dict(
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
    )
    # 划分、训练打乱、验证各用独立生成器，方便复现和断点续训。
    train_loader = DataLoader(
        train_set, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **options,
    )
    val_loader = DataLoader(
        val_set, shuffle=False,
        generator=torch.Generator().manual_seed(seed + 1), **options,
    )
    return train_loader, val_loader


def create_test_loader(
    root="data", batch_size=128, num_workers=0, download=False, pin_memory=False,
):
    """加载官方测试集，保持顺序并保留尾批。"""
    return DataLoader(
        load_dataset(root, train=False, download=download),
        batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
        shuffle=False, generator=torch.Generator().manual_seed(42),
    )


def save_preview(
    images, labels, output_path="outputs/data-preview.png", max_images=32, nrow=8,
):
    """将 [-1, 1] 灰度图片保存成带类别标签的网格，返回保存路径。"""
    count = min(max_images, len(images))
    pixels = ((images[:count].detach().cpu() + 1) / 2).clamp(0, 1)
    pixels = (pixels * 255).round().to(torch.uint8)
    labels = labels[:count].detach().cpu().tolist()
    columns = min(nrow, count)
    rows = (count + columns - 1) // columns
    canvas = Image.new("RGB", (columns * 144, rows * 148), "white")
    draw = ImageDraw.Draw(canvas)

    for index, label in enumerate(labels):
        left, top = (index % columns) * 144, (index // columns) * 148
        image = Image.fromarray(pixels[index, 0].numpy())
        image = image.resize((112, 112), Image.Resampling.NEAREST)
        canvas.paste(image, (left + 16, top + 8))
        draw.text(
            (left + 8, top + 126),
            f"{label}: {datasets.FashionMNIST.classes[label]}", fill="black",
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="预览 Fashion-MNIST 图片")
    parser.add_argument("--root", default="data")
    parser.add_argument("--download", action="store_true", help="允许下载缺失数据")
    parser.add_argument("--preview", default="outputs/data-preview.png")
    parser.add_argument("--num-images", type=int, default=32)
    parser.add_argument("--nrow", type=int, default=8)
    args = parser.parse_args()
    if args.num_images <= 0 or args.nrow <= 0:
        parser.error("num-images 和 nrow 必须大于 0")

    dataset = load_dataset(args.root, download=args.download)
    images, labels = next(iter(DataLoader(dataset, batch_size=args.num_images)))
    path = save_preview(
        images, labels, args.preview, max_images=args.num_images, nrow=args.nrow,
    )
    print(f"已保存 {len(images)} 张图片的预览: {path}")
