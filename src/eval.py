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
from common import distributed as dist_utils
from common.data import JointTransform, SaliencyDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
    ):
        self.model = model
        self.device = device
        self.threshold = threshold
        self.use_amp = config.USE_AMP and torch.device(device).type == "cuda"

        self.mae = 0.0
        self.Fmeasure = 0.0
        self.maxF = 0.0
        self.w_Fmeasure = 0.0
        self.Emeasure = 0.0
        self.maxE = 0.0
        self.Smeasure = 0.0

        model_name = dist_utils.unwrap_model(model).__class__.__name__
        self.save_dir = os.path.join(output_dir, model_name)
        if dist_utils.is_main_process():
            os.makedirs(self.save_dir, exist_ok=True)
        dist_utils.barrier()

        transform = JointTransform(train=False)
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
        if not dist_utils.is_main_process():
            return

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
        if not dist_utils.is_main_process():
            return

        if num_samples is None:
            num_samples = config.VIZ_NUM_SAMPLES
        if seed is None:
            seed = config.VIZ_SEED

        visualize_predictions(
            model=dist_utils.unwrap_model(self.model),
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
        if not dist_utils.is_main_process():
            return

        if num_samples is None:
            num_samples = config.VIZ_NUM_SAMPLES
        if seed is None:
            seed = config.VIZ_SEED

        visualize_predictions_with_error(
            model=dist_utils.unwrap_model(self.model),
            dataloader=self.dataloader,
            device=self.device,
            num_samples=num_samples,
            mean=config.IMAGENET_MEAN,
            std=config.IMAGENET_STD,
            threshold=self.threshold,
            save_path=os.path.join(self.save_dir, "test_predictions_with_error.png"),
            seed=seed,
        )

    def save_result(self, res):
        if not dist_utils.is_main_process():
            return

        result_path = os.path.join(config.OUTPUT_DIR, "..", "result.json")

        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        else:
            results = {}

        model_name = dist_utils.unwrap_model(self.model).__class__.__name__

        results[model_name] = {
            "mae": res["mae"],
            "maxf": res["maxf"],
            "smeasure": res["smeasure"],
            "wfm": res["wfm"],
        }

        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✅ Scaling result saved to: {result_path}")
        
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

                with torch.cuda.amp.autocast(enabled=self.use_amp):
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

        if dist_utils.is_main_process():
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

def main():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PoolNetCFM",
                        choices=list(config.MODEL_REGISTRY.keys()),
                        help="选择模型")
    args = parser.parse_args()

    device = dist_utils.init_distributed(config) if config.EVAL_USE_MULTI_GPU else torch.device(config.DEVICE)
    use_distributed = config.EVAL_USE_MULTI_GPU and dist_utils.can_use_distributed(config)

    if dist_utils.is_main_process():
        print(f"Using device: {device}")
        print(f"Multi-GPU evaluation: {use_distributed}")

    # 加载模型
    model_cls = config.MODEL_REGISTRY[args.model]
    try:
        model = model_cls(backbone_name=config.BACKBONE).to(device)
    except TypeError:
        model = model_cls().to(device)
    ckpt_path = os.path.join(config.OUTPUT_DIR, model.__class__.__name__, "best_model.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    dist_utils.load_model_state(model, ckpt)
    model = dist_utils.wrap_model(model, device, config) if use_distributed else model

    # 构建 Evaluator
    evaluator = Evaluator(
        model=model,
        test_dir=config.TEST_DIR,
        device=device,
        output_dir=config.OUTPUT_DIR,
        threshold=config.THRESHOLD,
        batch_size=config.EVAL_BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    # 指标评估
    results = evaluator.evaluate()

    # 保存指标
    if config.SCALING:
        evaluator.save_result(results)

    # 可视化
    evaluator.visualize()
    evaluator.visualize_with_error()


if __name__ == "__main__":
    try:
        main()
    finally:
        dist_utils.destroy_distributed()
