import os
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
import numpy as np

import sys
from .data import SaliencyDataset, JointTransform
from .visualization import visualize_predictions, visualize_predictions_with_error
from model.cpd import CPDResNet

class Evaluator:
    def __init__(
        self,
        model,
        test_dir,
        device,
        output_dir="./outputs",
        threshold=0.5,
        batch_size=8,
        num_workers=0,
        crop_size=224,
    ):
        self.model = model
        self.device = device
        self.threshold = threshold

        self.mae = 0.0
        self.Fmeasure = 0.0
        self.maxF = 0.0
        self.w_Fmeasure = 0.0
        self.Emeasure = 0.0
        self.Smeasure = 0.0

        model_name = model.__class__.__name__
        self.save_dir = os.path.join(output_dir, model_name)
        os.makedirs(self.save_dir, exist_ok=True)

        transform = JointTransform(train=False, crop_size=crop_size)
        dataset = SaliencyDataset(
            image_dir=os.path.join(test_dir, "images"),
            mask_dir=os.path.join(test_dir, "masks"),
            transform=transform,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    # ──────────────────────────────────────────
    # 指标计算
    # ──────────────────────────────────────────

    def _compute_mae(self, pred, mask):
        """ MAE """
        self.mae = torch.mean(torch.abs(pred - mask), dim=(1, 2, 3)).mean().item()
        return self.mae

    def _compute_fmeasure(self, pred, mask, beta2=0.3):
        """ F-measure """
        pred_bin = (pred >= self.threshold).float()
        mask_bin = (mask >= 0.5).float()
        scores = []
        for p, g in zip(pred_bin, mask_bin):
            tp = (p * g).sum()
            precision = tp / (p.sum() + 1e-8)
            recall    = tp / (g.sum() + 1e-8)
            fm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
            scores.append(fm)
        self.Fmeasure = torch.stack(scores).mean().item()
        return self.Fmeasure

    def _compute_emeasure(self, pred, mask, eps=1e-8):
        """ E-measure """
        mask_bin = (mask >= 0.5).float()
        scores = []
        for p, g in zip(pred, mask_bin):
            p, g = p.squeeze(), g.squeeze()
            if g.sum() == 0:
                scores.append(1.0 - p.mean()); continue
            if g.sum() == g.numel():
                scores.append(p.mean()); continue
            p_align = p - p.mean()
            g_align = g - g.mean()
            align_matrix = 2 * p_align * g_align / (p_align ** 2 + g_align ** 2 + eps)
            scores.append(((align_matrix + 1) ** 2 / 4).mean())
        self.Emeasure = torch.stack(scores).mean().item()
        return self.Emeasure

    def _compute_smeasure(self, pred, mask, eps=1e-8):
        """ S-measure """
        mask_bin = (mask >= 0.5).float()
        scores = []
        for p, g in zip(pred, mask_bin):
            p, g = p.squeeze(), g.squeeze()
            fg_mean = g.mean()
            if fg_mean == 0 or fg_mean == 1:
                scores.append(torch.tensor(0.0, device=p.device)); continue

            fg = g
            p_fg = p * fg
            mu_fg = p_fg.sum() / (fg.sum() + eps)
            var_fg = ((p_fg - mu_fg) ** 2 * fg).sum() / (fg.sum() + eps)
            s_fg = 2 * mu_fg / (mu_fg ** 2 + 1 + var_fg + eps)

            bg = 1 - g
            p_bg = p * bg
            mu_bg = p_bg.sum() / (bg.sum() + eps)
            var_bg = ((p_bg - mu_bg) ** 2 * bg).sum() / (bg.sum() + eps)
            s_bg = 2 * (1 - mu_bg) / ((1 - mu_bg) ** 2 + 1 + var_bg + eps)

            scores.append(fg_mean * s_fg + (1 - fg_mean) * s_bg)
        self.Smeasure = torch.stack(scores).mean().item()
        return self.Smeasure

    def _compute_maxf(self, all_preds, all_masks, beta2=0.3, num_thresh=255):
        """ Max-F """
        thresholds = torch.linspace(1 / 255, 254 / 255, num_thresh)
        preds_flat = torch.stack([p.squeeze().flatten() for p in all_preds])  # (N, HW)
        masks_flat = torch.stack([m.squeeze().flatten() for m in all_masks])  # (N, HW)
        masks_bin  = (masks_flat >= 0.5).float()
        gt_sum = masks_bin.sum(dim=1)  # (N,)

        best_fm = 0.0
        for t in thresholds:
            pred_bin  = (preds_flat >= t).float()
            tp        = (pred_bin * masks_bin).sum(dim=1)
            precision = tp / (pred_bin.sum(dim=1) + 1e-8)
            recall    = tp / (gt_sum + 1e-8)
            fm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
            best_fm = max(best_fm, fm.mean().item())

        self.maxF = best_fm
        return self.maxF

    def _compute_weighted_fmeasure(self, all_preds, all_masks, beta2=1.0, eps=1e-8):
        """ Weighted F-measure """
        scores = []
        for pred, mask in zip(all_preds, all_masks):
            p = pred.squeeze().numpy()             # (H, W) 连续值
            g = (mask.squeeze().numpy() >= 0.5).astype(np.float32)  # (H, W) 0/1

            if g.sum() == 0 or g.sum() == g.size:
                scores.append(0.0)
                continue

            # 用 distance transform 构造边缘权重
            dist_fg = distance_transform_edt(g)          # 到背景边界的距离（在前景内）
            dist_bg = distance_transform_edt(1 - g)      # 到前景边界的距离（在背景内）

            mu_fg = dist_fg[g == 1].mean() + eps
            mu_bg = dist_bg[g == 0].mean() + eps
            w = np.exp(-dist_fg / mu_fg) * g + np.exp(-dist_bg / mu_bg) * (1 - g)
            w = w / (w.sum() + eps)

            tp_w = (w * p * g).sum()
            fp_w = (w * p * (1 - g)).sum()
            fn_w = (w * (1 - p) * g).sum()

            prec = tp_w / (tp_w + fp_w + eps)
            rec  = tp_w / (tp_w + fn_w + eps)
            wfm  = (1 + beta2) * prec * rec / (beta2 * prec + rec + eps)
            scores.append(float(wfm))

        self.w_Fmeasure = float(np.mean(scores))
        return self.w_Fmeasure
    
    def _save_test_result(self):
        """ 保存测试结果 """
        log = {
            "MAE": self.mae,
            "F-measure": self.Fmeasure,
            "Max F-measure": self.maxF,
            "Weighted F": self.w_Fmeasure,
            "E-measure": self.Emeasure,
            "S-measure": self.Smeasure,
        }

        log_path = os.path.join(self.save_dir, "test_log.json")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)

        print("✅ Evaluation results saved")

    def evaluate(self):
        self.model.eval()

        totals = {"mae": 0.0, "fmeasure": 0.0, "emeasure": 0.0, "smeasure": 0.0}
        n = 0
        all_preds, all_masks = [], []   # 用于 maxF / WFm

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)
                bs = images.size(0)

                outputs = self.model(images)
                preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                preds = torch.sigmoid(preds)

                totals["mae"]      += self._compute_mae(preds, masks)      * bs
                totals["fmeasure"] += self._compute_fmeasure(preds, masks) * bs
                totals["emeasure"] += self._compute_emeasure(preds, masks) * bs
                totals["smeasure"] += self._compute_smeasure(preds, masks) * bs
                n += bs

                # 收集到 CPU，逐图存储
                for p, m in zip(preds.cpu(), masks.cpu()):
                    all_preds.append(p.unsqueeze(0))
                    all_masks.append(m.unsqueeze(0))

        results = {k: v / n for k, v in totals.items()}
        results["maxf"]  = self._compute_maxf(all_preds, all_masks)
        results["wfm"]   = self._compute_weighted_fmeasure(all_preds, all_masks)

        print("=" * 50)
        print(f"  Test Results")
        print("-" * 50)
        for name, key in [
            ("MAE",              "mae"),
            ("F-measure",        "fmeasure"),
            ("Max F-measure",    "maxf"),
            ("Weighted F",       "wfm"),
            ("E-measure",        "emeasure"),
            ("S-measure",        "smeasure"),
        ]:
            print(f"  {name:<20} {results[key]:.4f}")
        print("=" * 50)

        self._save_test_result()

        return results

    # ──────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────

    def visualize(self, num_samples=4, seed=42):
        """ 原图 / GT / 预测 """
        visualize_predictions(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            threshold=self.threshold,
            save_path=os.path.join(self.save_dir, "test_predictions.png"),
            seed=seed,
        )

    def visualize_with_error(self, num_samples=4, seed=42):
        """ 原图 / GT / 预测 / 误差图 """
        visualize_predictions_with_error(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            threshold=self.threshold,
            save_path=os.path.join(self.save_dir, "test_predictions_with_error.png"),
            seed=seed,
        )


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载模型
    model = CPDResNet(pretrained=False).to(device)
    ckpt_path = os.path.join(os.path.dirname(__file__), f"../../outputs/{model.__class__.__name__}/best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 构建 Evaluator
    test_dir = os.path.join(os.path.dirname(__file__), "../../data/test")
    evaluator = Evaluator(
        model=model,
        test_dir=test_dir,
        device=device,
        output_dir=os.path.join(os.path.dirname(__file__), f"../../outputs"),
        threshold=0.5,
        batch_size=8,
        num_workers=0,
    )

    # 指标评估
    results = evaluator.evaluate()

    # 可视化
    evaluator.visualize(num_samples=4)
    evaluator.visualize_with_error(num_samples=4)
