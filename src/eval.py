import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.ndimage import distance_transform_edt
import numpy as np
from py_sod_metrics import MAE, Fmeasure, Emeasure, Smeasure, WeightedFmeasure

from common import (
    build_saliency_dataloader,
    Trainer,
    plot_training_curves,
    visualize_predictions,
    visualize_predictions_with_error,
    config,
)
from common.data import JointTransform, SaliencyDataset

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
        self.maxE = 0.0
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
    
    def _save_test_result(self, results):
        """ 保存测试结果 """
        log = {
            "MAE":                results["mae"],

            "Max F-measure":      results["maxf"],
            "Mean F-measure":     results["meanf"],
            "Adaptive F-measure": results["adpf"],

            "Weighted F":         results["wfm"],

            "Max E-measure":      results["maxe"],
            "Mean E-measure":     results["meane"],
            "Adaptive E-measure": results["adpe"],

            "S-measure":          results["smeasure"],
        }

        log_path = os.path.join(self.save_dir, "test_log.json")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)

        print("✅ Evaluation results saved")

    # ──────────────────────────────────────────
    # 可视化
    # ──────────────────────────────────────────

    def visualize(self, num_samples=None, seed=None):
        """ 原图 / GT / 预测 """
        if num_samples is None:
            num_samples = config.VIZ_NUM_SAMPLES
        if seed is None:
            seed = config.VIZ_SEED

        visualize_predictions(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            threshold=self.threshold,
            save_path=os.path.join(self.save_dir, "test_predictions.png"),
            seed=seed,
        )


    def visualize_with_error(self, num_samples=None, seed=None):
        """ 原图 / GT / 预测 / 误差图 """
        if num_samples is None:
            num_samples = config.VIZ_NUM_SAMPLES
        if seed is None:
            seed = config.VIZ_SEED

        visualize_predictions_with_error(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            threshold=self.threshold,
            save_path=os.path.join(self.save_dir, "test_predictions_with_error.png"),
            seed=seed,
        )
        
    def evaluate(self):
        self.model.eval()

        # ─────────────────────────────────────
        # 初始化标准 SOD 指标
        # ─────────────────────────────────────
        mae_metric = MAE()
        fm_metric = Fmeasure()
        em_metric = Emeasure()
        sm_metric = Smeasure()
        wfm_metric = WeightedFmeasure()

        with torch.no_grad():
            for batch in self.dataloader:
                images = batch["image"].to(self.device)
                masks  = batch["mask"].to(self.device)

                outputs = self.model(images)
                preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                preds = torch.sigmoid(preds)

                # 逐图送入 pysodmetrics
                for pred, mask in zip(preds, masks):
                    pred_np = pred.squeeze().detach().cpu().numpy()
                    mask_np = mask.squeeze().detach().cpu().numpy()

                    # 确保范围在 [0, 1]
                    pred_np = np.clip(pred_np, 0, 1)
                    mask_np = np.clip(mask_np, 0, 1)

                    # pysodmetrics 通常按 uint8 灰度图协议计算
                    pred_np = (pred_np * 255).astype(np.uint8)
                    mask_np = (mask_np * 255).astype(np.uint8)

                    mae_metric.step(pred=pred_np, gt=mask_np)
                    fm_metric.step(pred=pred_np, gt=mask_np)
                    em_metric.step(pred=pred_np, gt=mask_np)
                    sm_metric.step(pred=pred_np, gt=mask_np)
                    wfm_metric.step(pred=pred_np, gt=mask_np)

        # ─────────────────────────────────────
        # 获取结果
        # ─────────────────────────────────────
        mae = mae_metric.get_results()["mae"]

        fm = fm_metric.get_results()["fm"]
        em = em_metric.get_results()["em"]
        sm = sm_metric.get_results()["sm"]
        wfm = wfm_metric.get_results()["wfm"]

        results = {
            "mae": float(mae),

            # F-measure
            "maxf": float(fm["curve"].max()),
            "meanf": float(fm["curve"].mean()),
            "adpf": float(fm["adp"]),

            # E-measure
            "maxe": float(em["curve"].max()),
            "meane": float(em["curve"].mean()),
            "adpe": float(em["adp"]),

            # S-measure / Weighted F
            "smeasure": float(sm),
            "wfm": float(wfm),
        }

        print("=" * 50)
        print(f"  Test Results")
        print("-" * 50)
        for name, key in [
            ("MAE",              "mae"),
            ("Max F-measure",    "maxf"),
            ("Mean F-measure",   "meanf"),
            ("Adaptive F",       "adpf"),
            ("Weighted F",       "wfm"),
            ("Max E-measure",    "maxe"),
            ("Mean E-measure",   "meane"),
            ("Adaptive E",       "adpe"),
            ("S-measure",        "smeasure"),
        ]:
            print(f"  {name:<20} {results[key]:.4f}")
        print("=" * 50)

        self._save_test_result(results)

        return results

if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PoolNetCFM",
                        choices=list(config.MODEL_REGISTRY.keys()),
                        help="选择模型")
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")

    # 加载模型
    model = config.MODEL_REGISTRY[args.model]().to(config.DEVICE)
    ckpt_path = os.path.join(config.OUTPUT_DIR, model.__class__.__name__, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    # 构建 Evaluator
    evaluator = Evaluator(
        model=model,
        test_dir=config.TEST_DIR,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR,
        threshold=config.THRESHOLD,
        batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # 指标评估
    results = evaluator.evaluate()

    # 可视化
    evaluator.visualize()
    evaluator.visualize_with_error()
