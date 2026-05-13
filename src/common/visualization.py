import os
import random
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import torch
import torch.nn.functional as F

# 配色
TOP_PAPER_COLORS = [
    "#2B9CD7",  # blue
    "#DB726B",  # orange
    "#2AA371",  # green
    "#9B9c2D",  # yellow
    "#A977A6",  # purple
]

plt.rcParams.update({
    # =====================================================
    # Color cycle
    # =====================================================
    "axes.prop_cycle": cycler(color=TOP_PAPER_COLORS),

    # =====================================================
    # Font
    # =====================================================
    "font.family": "Carlito",
    "font.size": 14,

    # Axis title / label
    "axes.titlesize": 18,
    "axes.labelsize": 14,

    # Tick label
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Legend
    "legend.fontsize": 12,

    # Math font (LaTeX-like)
    "mathtext.fontset": "stix",

    # Minus sign fix
    "axes.unicode_minus": False,

    # =====================================================
    # Lines
    # =====================================================
    "lines.linewidth": 2.2,
    "lines.markersize": 5,

    # Rounded line caps
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",

    # Dashed line style
    "lines.dash_capstyle": "round",
    "lines.dash_joinstyle": "round",

    # =====================================================
    # Axes
    # =====================================================
    "axes.linewidth": 1.2,

    # Title padding
    "axes.titlepad": 12,

    # Label padding
    "axes.labelpad": 8,

    # Grid
    "axes.grid": False,

    # =====================================================
    # Grid
    # =====================================================
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "grid.alpha": 0.35,

    # =====================================================
    # Ticks
    # =====================================================
    "xtick.direction": "in",
    "ytick.direction": "in",

    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,

    "xtick.major.size": 5,
    "ytick.major.size": 5,

    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    # =====================================================
    # Legend
    # =====================================================
    "legend.frameon": False,
    "legend.handlelength": 2.0,

    # =====================================================
    # Histogram
    # =====================================================
    "patch.linewidth": 0.8,

    # =====================================================
    # Errorbar
    # =====================================================
    "errorbar.capsize": 3,
})

def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = image.detach().cpu().clone()

    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    image = image * std + mean

    image = image.clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()

    return image

def tensor_to_mask(mask):
    mask = mask.detach().cpu()

    if mask.dim() == 3:
        mask = mask.squeeze(0)

    return mask.numpy()

def plot_training_curves(
    train_losses,
    val_losses,
    val_maes=None,
    val_fmeasures=None,
    title_prefix="",
    save_path=None
):
    """ 绘制训练 Loss 曲线、MAE 曲线、F-measure 曲线"""
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss 曲线
    axes[0].plot(epochs, train_losses, "o-", label="Train Loss", color=TOP_PAPER_COLORS[0])
    axes[0].plot(epochs, val_losses, "o-", label="Val Loss", color=TOP_PAPER_COLORS[1])

    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # MAE 曲线
    axes[1].plot(epochs, val_maes, "o-", label="Val MAE", color=TOP_PAPER_COLORS[2])

    axes[1].set_title("MAE Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    # F-measure 曲线
    axes[2].plot(epochs, val_fmeasures, "o-", label="Val F-measure", color=TOP_PAPER_COLORS[3])

    axes[2].set_title("F-measure Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F-measure")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    if title_prefix:
        fig.suptitle(f"{title_prefix} Training Curves", fontsize=18)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def visualize_predictions(
    model,
    dataloader,
    device,
    num_samples=4,
    mean=None,
    std=None,
    threshold=0.5,
    save_path=None,
    seed=42
):
    """ 预测效果展示 """
    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device).float()

    batch_size = images.size(0)
    rng = random.Random(seed)
    indices = rng.sample(range(batch_size), min(num_samples, batch_size))

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds >= threshold).float()

    fig, axes = plt.subplots(
        len(indices),
        3,
        figsize=(10, 3.2 * len(indices))
    )

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        image = denormalize_image(images[idx], mean=mean, std=std)
        gt_mask = tensor_to_mask(masks[idx])
        pred_mask = tensor_to_mask(preds[idx])

        axes[row, 0].imshow(image)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].axis("off")

        if row == 0:
            axes[row, 0].set_title("Image", fontsize=16, fontweight="bold")
            axes[row, 1].set_title("GT Mask", fontsize=16, fontweight="bold")
            axes[row, 2].set_title("Pred Mask", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()

def visualize_predictions_with_error(
    model,
    dataloader,
    device,
    num_samples=4,
    mean=None,
    std=None,
    threshold=0.5,
    save_path=None,
    seed=42
):
    """ 展示原图、GT、预测结果、误差图 """
    model.eval()

    batch = next(iter(dataloader))
    images = batch["image"].to(device)
    masks = batch["mask"].to(device).float()

    batch_size = images.size(0)
    rng = random.Random(seed)
    indices = rng.sample(range(batch_size), min(num_samples, batch_size))

    with torch.no_grad():
        outputs = model(images)
        preds = torch.sigmoid(outputs)
        preds = (preds >= threshold).float()

    fig, axes = plt.subplots(
        len(indices),
        4,
        figsize=(13, 3.2 * len(indices))
    )

    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, idx in enumerate(indices):
        image = denormalize_image(images[idx], mean=mean, std=std)
        gt_mask = tensor_to_mask(masks[idx])
        pred_mask = tensor_to_mask(preds[idx])

        error_map = np.abs(pred_mask - gt_mask)

        axes[row, 0].imshow(image)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].axis("off")

        axes[row, 3].imshow(error_map, cmap="hot", vmin=0, vmax=1)
        axes[row, 3].axis("off")

        if row == 0:
            axes[row, 0].set_title("Image", fontsize=16, fontweight="bold")
            axes[row, 1].set_title("GT Mask", fontsize=16, fontweight="bold")
            axes[row, 2].set_title("Pred Mask", fontsize=16, fontweight="bold")
            axes[row, 3].set_title("Error Map", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()