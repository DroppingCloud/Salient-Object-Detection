import os
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from PIL import Image, UnidentifiedImageError
from collections import defaultdict, Counter
import matplotlib.pyplot as plt

from .config import (
    RESIZE_SIZE, CROP_SIZE, FLIP_PROB, MASK_THRESH,
    IMAGENET_MEAN, IMAGENET_STD,
    VAL_RATIO, BATCH_SIZE, NUM_WORKERS, SEED,
    DATA_ROOT, TEST_DIR
)

class SaliencyDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # 遍历读取图片路径
        self.image_paths = self._collect_images(self.image_dir)
        # 匹配 image 和 mask
        self.mask_paths = self._match_masks(self.image_paths, self.mask_dir)

    def _collect_images(self, image_dir):
        image_paths = sorted([p for p in image_dir.iterdir()])      # 保证数据读取顺序稳定、可复现
        return image_paths

    def _match_masks(self, image_paths, mask_dir):
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

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        sample = {
            "image": image,
            "mask": mask,
        }

        return sample
    
class JointTransform:
    def __init__(self, train=True, resize_size=RESIZE_SIZE, crop_size=CROP_SIZE):
        self.train = train
        self.resize_size = resize_size
        self.crop_size = crop_size

    def __call__(self, image, mask):
        if self.train:
            # 缩放到 2256×256
            image = TF.resize(
                image,
                (self.resize_size, self.resize_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask,
                (self.resize_size, self.resize_size),
                interpolation=InterpolationMode.NEAREST
            )

            # 随机裁剪到 224×224
            i, j, h, w = transforms.RandomCrop.get_params(
                image,
                output_size=(self.crop_size, self.crop_size)
            )

            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # 随机水平翻转
            if torch.rand(1).item() < FLIP_PROB:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        else:
            # 验证/测试阶段不随机裁剪
            image = TF.resize(
                image,
                (self.crop_size, self.crop_size),
                interpolation=InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask,
                (self.crop_size, self.crop_size),
                interpolation=InterpolationMode.NEAREST
            )

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        mask = (mask > MASK_THRESH).float()

        image = TF.normalize(
            image,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        return image, mask

def build_saliency_dataloader(
    root_dir,
    image_folder=None,
    mask_folder=None,
    val_ratio=None,
    batch_size=None,
    num_workers=None,
    seed=None,
):
    if image_folder is None:
        from .config import IMAGE_FOLDER as image_folder
    if mask_folder is None:
        from .config import MASK_FOLDER as mask_folder
    if val_ratio is None:
        val_ratio = VAL_RATIO
    if batch_size is None:
        batch_size = BATCH_SIZE
    if num_workers is None:
        num_workers = NUM_WORKERS
    if seed is None:
        seed = SEED

    image_dir = os.path.join(root_dir, image_folder)
    mask_dir = os.path.join(root_dir, mask_folder)

    base_dataset = SaliencyDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=None
    )
    base_loader = DataLoader(
        base_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 划分数据集
    val_size = int(len(base_dataset) * val_ratio)
    train_size = len(base_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        base_dataset,
        [train_size, val_size],
        generator=generator
    )

    # 分别创建 Dataset & DataLoader
    train_dataset = SaliencyDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=JointTransform(train=True, resize_size=256, crop_size=224)
    )

    val_dataset = SaliencyDataset(
        image_dir=image_dir,
        mask_dir=mask_dir,
        transform=JointTransform(train=False, resize_size=256, crop_size=224)
    )

    train_dataset = torch.utils.data.Subset(train_dataset, train_subset.indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_subset.indices)

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
    return base_dataset, train_dataset, train_loader, val_dataset, val_loader

def check_saliency_dataset(image_dir, mask_dir, mask_suffix=".png", max_print=20):
    """ 数据完整性检查 """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])

    bad_images = []
    missing_masks = []
    bad_masks = []
    size_mismatch = []

    image_sizes = []
    mask_sizes = []
    image_exts = Counter()

    for img_path in image_paths:
        image_exts[img_path.suffix.lower()] += 1
        mask_path = mask_dir / f"{img_path.stem}{mask_suffix}"

        # 检查 image
        try:
            with Image.open(img_path) as img:
                img.verify()

            with Image.open(img_path) as img:
                img_size = img.size

        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            bad_images.append((img_path, str(e)))
            continue

        # 检查 mask 是否存在
        if not mask_path.exists():
            missing_masks.append(mask_path)
            continue

        # 检查 mask
        try:
            with Image.open(mask_path) as m:
                m.verify()

            with Image.open(mask_path) as m:
                mask_size = m.size

        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            bad_masks.append((mask_path, str(e)))
            continue

        image_sizes.append(img_size)
        mask_sizes.append(mask_size)

        # 检查尺寸是否一致
        if img_size != mask_size:
            size_mismatch.append((img_path, mask_path, img_size, mask_size))

    print()
    print("=" * 60)
    print("Dataset Check Result")
    print("=" * 60)

    print(f"Image dir        : {image_dir}")
    print(f"Mask dir         : {mask_dir}")
    print(f"Total images     : {len(image_paths)}")
    print(f"Valid pairs      : {len(image_sizes)}")
    print(f"Bad images       : {len(bad_images)}")
    print(f"Missing masks    : {len(missing_masks)}")
    print(f"Bad masks        : {len(bad_masks)}")
    print(f"Size mismatches  : {len(size_mismatch)}")

    print("\nImage extensions:")
    for ext, count in image_exts.items():
        print(f"  {ext}: {count}")

    print("\nTop image sizes:")
    for size, count in Counter(image_sizes).most_common(10):
        print(f"  {size}: {count}")

    def print_examples(title, items):
        if len(items) == 0:
            return

        print(f"\n[{title}]")
        for item in items[:max_print]:
            print(" ", item)

        if len(items) > max_print:
            print(f"  ... and {len(items) - max_print} more")

    print_examples("Bad images", bad_images)
    print_examples("Missing masks", missing_masks)
    print_examples("Bad masks", bad_masks)
    print_examples("Size mismatches", size_mismatch)

    print("=" * 60)
    print()

    return {
        "total_images": len(image_paths),
        "valid_pairs": len(image_sizes),
        "bad_images": bad_images,
        "missing_masks": missing_masks,
        "bad_masks": bad_masks,
        "size_mismatch": size_mismatch,
        "image_size_counter": Counter(image_sizes),
        "image_ext_counter": image_exts,
    }

def show_samples_by_size(
    dataset,
    top_k_sizes=4,
    samples_per_size=6,
    cell_width=220,
    col_gap=8,
    row_gap=8,
):
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

        rows.append(image_blocks)
        rows.append(mask_blocks)

    # 每一行的高度
    row_heights = [
        max(img.height for img in row)
        for row in rows
    ]

    # 总宽度
    max_cols = max(len(row) for row in rows)
    total_w = max_cols * cell_width + (max_cols - 1) * col_gap

    # 总高度
    total_h = sum(row_heights) + (len(rows) - 1) * row_gap

    canvas = Image.new("RGB", (total_w, total_h), "white")

    y = 0
    for row, row_h in zip(rows, row_heights):
        x = 0

        for img in row:
            canvas.paste(img, (x, y))
            x += cell_width + col_gap

        y += row_h + row_gap

    plt.figure(figsize=(total_w / 100, total_h / 100))
    plt.imshow(np.array(canvas))
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def save_val_subset(val_dataset, save_root, max_samples=None, prefix="val_"):

    save_root = Path(save_root)
    image_save_dir = save_root / "images"
    mask_save_dir = save_root / "masks"

    image_save_dir.mkdir(parents=True, exist_ok=True)
    mask_save_dir.mkdir(parents=True, exist_ok=True)

    # 获取真实 Dataset 和索引
    if isinstance(val_dataset, torch.utils.data.Subset):
        dataset = val_dataset.dataset
        indices = list(val_dataset.indices)
    else:
        dataset = val_dataset
        indices = list(range(len(dataset)))

    if max_samples is not None:
        indices = indices[:max_samples]

    for count, idx in enumerate(indices, 1):
        sample = dataset[idx]

        image = sample["image"]
        mask = sample["mask"]

        # -------------------------
        # image: tensor -> PIL RGB
        # -------------------------
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()

            # 如果 image 是经过 ImageNet normalize 的，需要反归一化
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            image = image * std + mean

            image = image.clamp(0, 1)
            image = TF.to_pil_image(image)

        image = image.convert("RGB")

        # -------------------------
        # mask: tensor -> PIL L, 0/255
        # -------------------------
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().squeeze()
            mask = (mask > 0.5).numpy().astype(np.uint8) * 255
            mask = Image.fromarray(mask, mode="L")
        else:
            mask = mask.convert("L")
            mask = np.array(mask)
            mask = (mask > 127).astype(np.uint8) * 255
            mask = Image.fromarray(mask, mode="L")

        # -------------------------
        # 文件名：image 保存 jpg，mask 保存 png
        # -------------------------
        if hasattr(dataset, "image_paths"):
            stem = Path(dataset.image_paths[idx]).stem
        else:
            stem = f"{prefix}{idx}"

        image_name = f"{stem}.jpg"
        mask_name = f"{stem}.png"

        image.save(image_save_dir / image_name, quality=95)
        mask.save(mask_save_dir / mask_name)

    print(f"Saved {len(indices)} samples to:")
    print(f"  Images: {image_save_dir}")
    print(f"  Masks : {mask_save_dir}")

if __name__ == "__main__":
    root_dir = DATA_ROOT

    dataset, _, train_loader, val_dataset, _ = build_saliency_dataloader(
        root_dir=root_dir,
        image_folder="images",
        mask_folder="masks",
        batch_size=32,
        num_workers=0
    )

    print(f"Dataset size: {len(dataset)}")

    batch = next(iter(train_loader))

    check_result = check_saliency_dataset(
        image_dir=f"{root_dir}/images",
        mask_dir=f"{root_dir}/masks",
        mask_suffix=".png"
    )

    show_samples_by_size(
        dataset,
        top_k_sizes=3,
        samples_per_size=6,
        cell_width=220,
    )

    save_val_subset(val_dataset, save_root=TEST_DIR, max_samples=None)
