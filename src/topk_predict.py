"""
加载指定模型权重，对测试集进行推理，返回预测效果最好的 Top-K 张掩码并标注原图

用法:
    python src/topk_predict.py --model PoolNetCFMRRM --ckpt outputs/PoolNetCFMRRM/best_model.pth --topk 10
    python src/topk_predict.py --model PoolNetCFMRRM --topk 5 --metric adpF
    python src/topk_predict.py --model PoolNetCFMRRM --topk 5 --metric mae
    python src/topk_predict.py --model PoolNetCFMRRM --topk 5 --metric sm
"""

import os
import sys
import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

from common.config import (
    MODEL_REGISTRY, DEVICE, OUTPUT_DIR, TEST_DIR,
    CROP_SIZE, IMAGENET_MEAN, IMAGENET_STD, THRESHOLD,
)
from common.data import JointTransform, SaliencyDataset


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    img = tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def to_uint8_mask(x):
    x = np.asarray(x)

    if x.ndim == 3:
        x = np.squeeze(x)

    x = x.astype(np.float32)

    if x.max() > 1.0:
        x = x / 255.0

    x = np.clip(x, 0.0, 1.0)
    return (x * 255).round().astype(np.uint8)


def compute_sod_metrics(pred_np, mask_np):
    """ 单张图像的 SOD 指标 """

    pred_u8 = to_uint8_mask(pred_np)
    gt_u8 = to_uint8_mask(mask_np)

    fm_metric = Fmeasure()
    mae_metric = MAE()
    sm_metric = Smeasure()
    em_metric = Emeasure()
    wfm_metric = WeightedFmeasure()

    fm_metric.step(pred=pred_u8, gt=gt_u8)
    mae_metric.step(pred=pred_u8, gt=gt_u8)
    sm_metric.step(pred=pred_u8, gt=gt_u8)
    em_metric.step(pred=pred_u8, gt=gt_u8)
    wfm_metric.step(pred=pred_u8, gt=gt_u8)

    fm = fm_metric.get_results()["fm"]
    mae = mae_metric.get_results()["mae"]
    sm = sm_metric.get_results()["sm"]
    em = em_metric.get_results()["em"]
    wfm = wfm_metric.get_results()["wfm"]

    return {
        "mae": float(mae),
        "sm": float(sm),
        "adpF": float(fm["adp"]),
        "maxF": float(fm["curve"].max()),
        "meanF": float(fm["curve"].mean()),
        "adpE": float(em["adp"]),
        "maxE": float(em["curve"].max()),
        "meanE": float(em["curve"].mean()),
        "wfm": float(wfm),
    }


def get_sort_score(metrics, metric_name):
    """ 返回用于排序的 score """

    if metric_name not in metrics:
        raise ValueError(
            f"Unsupported metric: {metric_name}. "
            f"Available metrics: {list(metrics.keys())}"
        )

    value = metrics[metric_name]

    if metric_name == "mae":
        return -value

    return value


def run_topk(model, test_dir, device, topk=10, metric="adpF", crop_size=CROP_SIZE):
    """ 返回效果最好的 Top-K 样本 """

    transform = JointTransform(train=False, crop_size=crop_size)
    dataset = SaliencyDataset(
        image_dir=os.path.join(test_dir, "images"),
        mask_dir=os.path.join(test_dir, "masks"),
        transform=transform,
    )

    model.eval()
    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            sample = dataset[idx]
            image = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"]

            outputs = model(image)
            pred = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            pred = torch.sigmoid(pred)
            pred_np = pred.squeeze().detach().cpu().numpy()
            pred_np = np.clip(pred_np, 0.0, 1.0).astype(np.float32)

            mask_np = mask.squeeze().detach().cpu().numpy()
            mask_np = np.clip(mask_np.astype(np.float32), 0.0, 1.0)

            metrics = compute_sod_metrics(pred_np, mask_np)
            score = get_sort_score(metrics, metric)
            metric_value = metrics[metric]

            image_np = denormalize(sample["image"])

            results.append({
                "index": idx,
                "image_path": str(dataset.image_paths[idx]),
                "score": score,
                "metric_value": metric_value,
                "metrics": metrics,
                "pred_np": pred_np,
                "mask_np": mask_np,
                "image_np": image_np,
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:topk]


def _resize_to_cell(arr, cell_size, is_mask=False):
    """
    将 numpy 图像 resize 到统一 cell_size。
    
    Args:
        arr: image [H,W,3] or mask [H,W]
        cell_size: int or tuple
        is_mask: 是否为 mask
    """
    if isinstance(cell_size, int):
        cell_size = (cell_size, cell_size)

    if arr.ndim == 2:
        arr = np.clip(arr, 0, 1)
        arr = (arr * 255).astype(np.uint8)
        pil = Image.fromarray(arr, mode="L").convert("RGB")
    else:
        pil = Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    pil = pil.resize(cell_size, resample=resample)
    return pil


def _make_overlay(image_np, pred_np, threshold=THRESHOLD, alpha=0.45):
    """
    生成红色预测区域 overlay。
    """
    image = image_np.astype(np.float32).copy()
    pred_binary = pred_np > threshold

    red = np.zeros_like(image)
    red[..., 0] = 255

    image[pred_binary] = image[pred_binary] * (1 - alpha) + red[pred_binary] * alpha
    return np.clip(image, 0, 255).astype(np.uint8)


def visualize_topk(
    topk_results,
    save_path,
    metric="adpF",
    threshold=THRESHOLD,
    cell_size=224,
    gap=5,
    label_height=42,
    include_overlay=False,
    show_metric=False,
):
    """ Top-K 可视化：Image | GT | Ours | Overlay(optional) """
    
    if len(topk_results) == 0:
        raise ValueError("topk_results is empty.")

    if isinstance(cell_size, int):
        cell_w, cell_h = cell_size, cell_size
    else:
        cell_w, cell_h = cell_size

    col_names = ["Image", "GT", "Ours"]
    if include_overlay:
        col_names.append("Overlay")

    num_rows = len(topk_results)
    num_cols = len(col_names)

    canvas_w = num_cols * cell_w + (num_cols - 1) * gap
    canvas_h = num_rows * cell_h + (num_rows - 1) * gap + label_height

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    # 使用 Times New Roman
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf"
    small_font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, size=28)
    else:
        print(f"Warning: font not found: {font_path}, using default font.")
        font = ImageFont.load_default()

    if os.path.exists(small_font_path):
        small_font = ImageFont.truetype(small_font_path, size=16)
    else:
        print(f"Warning: font not found: {small_font_path}, using default font.")
        small_font = ImageFont.load_default()

    for row_idx, item in enumerate(topk_results):
        y = row_idx * (cell_h + gap)

        image_np = item["image_np"]
        mask_np = item["mask_np"]
        pred_np = item["pred_np"]

        image_pil = _resize_to_cell(image_np, (cell_w, cell_h), is_mask=False)
        gt_pil = _resize_to_cell(mask_np, (cell_w, cell_h), is_mask=True)
        pred_pil = _resize_to_cell(pred_np, (cell_w, cell_h), is_mask=True)

        row_images = [image_pil, gt_pil, pred_pil]

        if include_overlay:
            overlay_np = _make_overlay(image_np, pred_np, threshold=threshold)
            overlay_pil = _resize_to_cell(overlay_np, (cell_w, cell_h), is_mask=False)
            row_images.append(overlay_pil)

        for col_idx, pil_img in enumerate(row_images):
            x = col_idx * (cell_w + gap)
            canvas.paste(pil_img, (x, y))

        if show_metric:
            value = item.get("metric_value", None)
            if value is not None:
                text = f"#{row_idx + 1} {metric}={value:.4f}"
                x = 6
                y_text = y + 6

                text_bbox = draw.textbbox((x, y_text), text, font=small_font)
                draw.rectangle(
                    [text_bbox[0] - 3, text_bbox[1] - 2, text_bbox[2] + 3, text_bbox[3] + 2],
                    fill="white"
                )
                draw.text((x, y_text), text, fill="black", font=small_font)

    # 底部列名
    label_y = num_rows * cell_h + (num_rows - 1) * gap

    for col_idx, name in enumerate(col_names):
        x0 = col_idx * (cell_w + gap)
        text_bbox = draw.textbbox((0, 0), name, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_x = x0 + (cell_w - text_w) // 2
        text_y = label_y + (label_height - text_h) // 2 - 2

        draw.text((text_x, text_y), name, fill="black", font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)

    print(f"Top-K visualization saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Top-K Best Prediction Visualization based on py_sod_metrics")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="模型名称",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="模型权重路径，默认: outputs/<ModelName>/best_model.pth",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="返回效果最好的前 K 张，默认 10",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="adpF",
        choices=["adpF", "maxF", "meanF", "mae", "sm", "adpE", "maxE", "meanE", "wfm"],
        help="排序指标，默认 adpF",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="测试集目录，默认使用 config 中的 TEST_DIR",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出图片路径",
    )
    args = parser.parse_args()

    device = DEVICE
    print(f"Device: {device}")

    model = MODEL_REGISTRY[args.model]().to(device)
    ckpt_path = args.ckpt or os.path.join(
        OUTPUT_DIR,
        model.__class__.__name__,
        "best_model.pth",
    )

    if not os.path.exists(ckpt_path):
        print(f"Error: checkpoint not found at {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    print(f"Loaded checkpoint: {ckpt_path}")

    test_dir = args.test_dir or TEST_DIR

    print(f"Running inference on {test_dir} ...")
    topk_results = run_topk(
        model=model,
        test_dir=test_dir,
        device=device,
        topk=args.topk,
        metric=args.metric,
    )

    print(f"\nTop-{args.topk} results sorted by {args.metric}:")
    print("-" * 100)
    for i, item in enumerate(topk_results):
        fname = os.path.basename(item["image_path"])
        m = item["metrics"]

        print(
            f"#{i + 1:2d}  "
            f"{args.metric}={item['metric_value']:.4f}  "
            f"MAE={m['mae']:.4f}  "
            f"adpF={m['adpF']:.4f}  "
            f"maxF={m['maxF']:.4f}  "
            f"SM={m['sm']:.4f}  "
            f"wFm={m['wfm']:.4f}  "
            f"{fname}"
        )
    print("-" * 100)

    save_path = args.output or os.path.join(
        OUTPUT_DIR,
        model.__class__.__name__,
        f"topk_{args.metric}_{args.topk}.png",
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    visualize_topk(
        topk_results=topk_results,
        save_path=save_path,
        metric=args.metric,
        cell_size=224,
        gap=5,
        label_height=30,
        include_overlay=False,
        show_metric=False,
    )


if __name__ == "__main__":
    main()