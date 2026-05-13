import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

from PIL import Image
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

class ECSSDDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # 遍历读取图片路径
        self.image_paths = self._collect_images(self.image_dir)
        # 匹配 image 和 mask
        self.mask_paths = self._match_masks(self.image_paths, self.mask_dir)

        # 定义转换器
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def _collect_images(self, image_dir):
        image_paths = sorted([p for p in image_dir.iterdir()])      # 保证数据读取顺序稳定、可复现

        if len(image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        return image_paths

    def _match_masks(self, image_paths: List[Path], mask_dir: Path) -> List[Path]:
        mask_paths = []

        for img_path in image_paths:
            matched = None

            stem = img_path.stem
            candidate = mask_dir / f"{stem}.png"
            if candidate.exists():
                matched = candidate

            if matched is None:
                raise FileNotFoundError(
                    f"Cannot find mask for image: {img_path.name} in {mask_dir}"
                )

            mask_paths.append(matched)

        return mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.image_transform(image)
        mask = self.mask_transform(mask)

        return {
            "image": image,
            "mask": mask,
        }

def build_ecssd_dataloader(
    root_dir,
    image_folder="images",
    mask_folder="masks",
    val_ratio=0.3,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    seed=42
):
    image_dir = os.path.join(root_dir, image_folder)
    mask_dir = os.path.join(root_dir, mask_folder)

    dataset = ECSSDDataset(
        image_dir=image_dir,
        mask_dir=mask_dir
    )

    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return dataset, dataloader, train_dataset, train_loader, val_dataset, val_loader

def show_samples_by_size(dataset, top_k_sizes=4, samples_per_size=6, cell_width=220):
    """ 按尺寸分组展示样本 """

    size_to_indices = defaultdict(list)

    # 统计每张图像的原始尺寸
    for idx, image_path in enumerate(dataset.image_paths):
        with Image.open(image_path) as img:
            size = img.size  # (width, height)

        size_to_indices[size].append(idx)

    size_counter = Counter({
        size: len(indices)
        for size, indices in size_to_indices.items()
    })

    top_sizes = [size for size, _ in size_counter.most_common(top_k_sizes)]

    rows = []

    for size in top_sizes:
        indices = size_to_indices[size][:samples_per_size]

        image_blocks = []
        mask_blocks = []

        for idx in indices:
            image_path = dataset.image_paths[idx]
            mask_path = dataset.mask_paths[idx]

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L").convert("RGB")

            w, h = image.size
            new_h = int(h * cell_width / w)

            image = image.resize((cell_width, new_h), Image.Resampling.BILINEAR)
            mask = mask.resize((cell_width, new_h), Image.Resampling.NEAREST)

            image_blocks.append(image)
            mask_blocks.append(mask)

        row_w = cell_width * len(image_blocks)
        row_h = max(img.height for img in image_blocks)

        image_row = Image.new("RGB", (row_w, row_h), "white")
        mask_row = Image.new("RGB", (row_w, row_h), "white")

        for j, (img, mask) in enumerate(zip(image_blocks, mask_blocks)):
            x = j * cell_width
            image_row.paste(img, (x, 0))
            mask_row.paste(mask, (x, 0))

        rows.append((image_row, f"Images - size {size}"))
        rows.append((mask_row, f"Masks  - size {size}"))

    total_w = max(row.width for row, _ in rows)
    total_h = sum(row.height for row, _ in rows)

    canvas = Image.new("RGB", (total_w, total_h), "white")

    y = 0
    for row, _ in rows:
        canvas.paste(row, (0, y))
        y += row.height

    plt.figure(figsize=(total_w / 100, total_h / 100))
    plt.imshow(np.array(canvas))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

if __name__ == "__main__":
    root_dir = "./data/ECSSD"

    dataset, dataloader, _, _, _, _ = build_ecssd_dataloader(
        root_dir=root_dir,
        image_folder="images",
        mask_folder="masks",
        batch_size=32,
        num_workers=0,
        shuffle=True,
    )

    print(f"Dataset size: {len(dataset)}")

    batch = next(iter(dataloader))
    print("Image batch shape:", batch["image"].shape)
    print("Mask batch shape :", batch["mask"].shape)

    show_samples_by_size(
        dataset,
        top_k_sizes=3,
        samples_per_size=6,
        cell_width=220,
    )