import os
import json
from tqdm import tqdm

import torch
import torch.nn.functional as F

import numpy as np
from py_sod_metrics import MAE, Fmeasure, Emeasure, Smeasure, WeightedFmeasure

from .config import (
    THRESHOLD, EPS, SCALING,
    MAIN_LOSS_WEIGHT, AUX_LOSS_WEIGHT, EDGE_LOSS_WEIGHT, LR_ETA_MIN, USE_AMP,
)
from . import distributed as dist_utils

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, device, output_dir="./outputs", threshold=THRESHOLD):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

        model_name = dist_utils.unwrap_model(self.model).__class__.__name__
        self.output_dir = os.path.join(output_dir, model_name)
        if dist_utils.is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
        dist_utils.barrier()

        self.best_mae = float("inf")
        self.best_maxf = 0.0
        self.best_smeasure = 0.0
        self.best_wfm = 0.0

        self.val_maes = []
        self.val_maxfs = []
        self.val_meanfs = []
        self.val_adpfs = []
        self.val_wfms = []
        self.val_maxes = []
        self.val_meanes = []
        self.val_adpes = []
        self.val_smeasures = []

        # 二值化阈值
        self.threshold = threshold

        # 深监督 warm-up 比例
        self.ds_warmup_ratio = 0.4
        self.use_amp = USE_AMP and torch.device(device).type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _to_uint8_numpy(self, x):
        x = x.squeeze().detach().cpu().numpy()
        x = np.clip(x, 0, 1)
        x = (x * 255).astype(np.uint8)
        return x

    def _compute_bce_iou_loss(self, pred_logits, masks):
        """ BCE + Soft IoU Loss """
        bce = self.criterion(pred_logits, masks)

        pred = torch.sigmoid(pred_logits)
        inter = (pred * masks).sum(dim=(1, 2, 3))
        union = (pred + masks - pred * masks).sum(dim=(1, 2, 3))
        iou = (inter / (union + EPS)).mean()

        return bce + (1.0 - iou)

    def _ssim_loss(self, pred, mask, window_size=11, sigma=1.5):
        """
        SSIM Loss

        pred / mask 均为 sigmoid 后的概率图，形状 [B, 1, H, W]，值域 [0, 1]
        返回标量 1 - mean_SSIM，值越小表示预测与 GT 结构越相似
        """
        # ── 构造高斯核 ────────────────────────────────────────
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
        coords -= window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        kernel = g.unsqueeze(0) * g.unsqueeze(1)          # [ws, ws]
        kernel = kernel.unsqueeze(0).unsqueeze(0)          # [1, 1, ws, ws]

        pad = window_size // 2

        def _conv(x):
            return torch.nn.functional.conv2d(x, kernel, padding=pad)

        # ── 局部统计量 ────────────────────────────────────────
        mu_p  = _conv(pred)
        mu_g  = _conv(mask)
        mu_p2 = mu_p * mu_p
        mu_g2 = mu_g * mu_g
        mu_pg = mu_p * mu_g

        sigma_p2  = _conv(pred * pred) - mu_p2
        sigma_g2  = _conv(mask * mask) - mu_g2
        sigma_pg  = _conv(pred * mask) - mu_pg

        # ── SSIM 公式 ────────────────
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim_map = (2 * mu_pg + C1) * (2 * sigma_pg + C2) / (
            (mu_p2 + mu_g2 + C1) * (sigma_p2 + sigma_g2 + C2) + EPS
        )

        ssim_loss = torch.clamp((1.0 - ssim_map) / 2.0, 0, 1)
        return ssim_loss.mean()

    def _compute_bce_ssim_iou_loss(self, pred_logits, masks):
        """ BCE + SSIM + Soft IoU Loss  """
        bce = self.criterion(pred_logits, masks)

        pred = torch.sigmoid(pred_logits)

        ssim = self._ssim_loss(pred, masks)

        inter = (pred * masks).sum(dim=(1, 2, 3))
        union = (pred + masks - pred * masks).sum(dim=(1, 2, 3))
        iou = (inter / (union + EPS)).mean()

        return bce + ssim + (1.0 - iou)

    def _edge_loss(self, pred_edge, masks):
        """ Edge Loss """
        # Sobel 边缘检测
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=masks.dtype, device=masks.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=masks.dtype, device=masks.device).view(1, 1, 3, 3)

        edge_x = F.conv2d(masks, sobel_x, padding=1)
        edge_y = F.conv2d(masks, sobel_y, padding=1)
        edge_gt = torch.sqrt(edge_x ** 2 + edge_y ** 2 + EPS)
        # 按样本归一化，保持 batch 维度
        B = edge_gt.shape[0]
        e_min = edge_gt.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        e_max = edge_gt.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        edge_gt = (edge_gt - e_min) / (e_max - e_min + EPS)

        # BCE loss
        return self.criterion(pred_edge, edge_gt)

    def _compute_loss(self, outputs, masks):
        """ 
        单输出模型/多输出模型损失计算。

        outputs 可以是：
          - Tensor：单输出模型
          - dict：键名驱动的多输出，支持以下键：
              'main'    (必须) 主显著性图
              'aux_sal' (list) 多尺度辅助显著性图，权重从 AUX_LOSS_WEIGHT 读取
              'edge'    (list) 边缘预测图，固定权重 0.3
        """

        # ── 单 Tensor ────────────────────────────────────────────────
        if not isinstance(outputs, dict):
            return self._compute_bce_ssim_iou_loss(outputs, masks)

        # ── dict：键名路由 ───────────────────────────────────────────
        loss = MAIN_LOSS_WEIGHT * self._compute_bce_ssim_iou_loss(outputs['main'], masks)

        for i, out in enumerate(outputs.get('aux_sal', [])):
            w = AUX_LOSS_WEIGHT[i] if i < len(AUX_LOSS_WEIGHT) else AUX_LOSS_WEIGHT[-1]
            loss = loss + w * self._compute_bce_iou_loss(out, masks)

        for out in outputs.get('edge', []):
            loss = loss + EDGE_LOSS_WEIGHT * self._edge_loss(out, masks)

        return loss

    def _run_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1:02d} [Training]",
            disable=not dist_utils.is_main_process(),
        ):
            # 清空梯度
            self.optimizer.zero_grad()

            # 数据上传
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # 模型输出
                outputs = self.model(images)

                # 损失计算
                loss = self._compute_loss(outputs, masks)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # 梯度反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        # 该 Epoch 平均样本损失
        total_loss = dist_utils.reduce_sum(total_loss, self.device)
        total_samples = dist_utils.reduce_sum(total_samples, self.device)
        avg_loss = total_loss / total_samples

        return avg_loss

    def _evaluate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_samples = 0

        # 标准 SOD 指标
        mae_metric = MAE()
        fm_metric = Fmeasure()
        em_metric = Emeasure()
        sm_metric = Smeasure()
        wfm_metric = WeightedFmeasure()

        with torch.no_grad():
            for batch in tqdm(
                self.valid_loader,
                desc=f"Epoch {epoch+1:02d} [Eval]",
                disable=not dist_utils.is_main_process(),
            ):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, masks)

                preds = outputs['main'] if isinstance(outputs, dict) else outputs

                # 如果输出尺寸和 mask 不一致，先对齐
                if preds.shape[-2:] != masks.shape[-2:]:
                    preds = torch.nn.functional.interpolate(
                        preds,
                        size=masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                preds = torch.sigmoid(preds)

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                # 逐图更新 pysodmetrics
                for pred, mask in zip(preds, masks):
                    pred_np = self._to_uint8_numpy(pred)
                    mask_np = self._to_uint8_numpy(mask)

                    mae_metric.step(pred=pred_np, gt=mask_np)
                    fm_metric.step(pred=pred_np, gt=mask_np)
                    em_metric.step(pred=pred_np, gt=mask_np)
                    sm_metric.step(pred=pred_np, gt=mask_np)
                    wfm_metric.step(pred=pred_np, gt=mask_np)

        total_loss = dist_utils.reduce_sum(total_loss, self.device)
        total_samples = dist_utils.reduce_sum(total_samples, self.device)
        avg_loss = total_loss / total_samples

        mae = mae_metric.get_results()["mae"]

        fm = fm_metric.get_results()["fm"]
        em = em_metric.get_results()["em"]
        sm = sm_metric.get_results()["sm"]
        wfm = wfm_metric.get_results()["wfm"]

        val_metrics = {
            "mae": float(mae),

            "maxf": float(fm["curve"].max()),
            "meanf": float(fm["curve"].mean()),
            "adpf": float(fm["adp"]),

            "maxe": float(em["curve"].max()),
            "meane": float(em["curve"].mean()),
            "adpe": float(em["adp"]),

            "smeasure": float(sm),
            "wfm": float(wfm),
        }

        return avg_loss, val_metrics
    
    def _save_result(self):                                                                                                                                                                                                                       
        result_path = os.path.join(self.output_dir, "..", "result.json")                                                                                       
                                                                                                                                                       
        if os.path.exists(result_path):                                                                                                                          
            with open(result_path, "r", encoding="utf-8") as f:                                                                                                  
                results = json.load(f)                                                                                                                           
        else:                                                                                                                                                    
            results = {}                                                                                                                                         
                                                                                                                                                 
        model_name = dist_utils.unwrap_model(self.model).__class__.__name__
        results[model_name] = {
            "best_mae": self.best_mae,
            "best_maxf": self.best_maxf,
            "best_smeasure": self.best_smeasure,
            "best_wfm": self.best_wfm,
        }                                                                                                                                             

        with open(result_path, "w", encoding="utf-8") as f:                                                                                               
            json.dump(results, f, indent=4, ensure_ascii=False)  

        print(f"✅ Model result saved to: {result_path}!")
        

    def _save_checkpoints(self, val_metrics):
        if not dist_utils.is_main_process():
            return

        val_mae = val_metrics["mae"]

        if val_mae < self.best_mae:
            self.best_mae = val_mae
            self.best_maxf = val_metrics["maxf"]
            self.best_smeasure = val_metrics["smeasure"]
            self.best_wfm = val_metrics["wfm"]

            save_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": dist_utils.state_dict(self.model),
                    "optimizer_state_dict": self.optimizer.state_dict(),

                    "best_mae": self.best_mae,
                    "best_maxf": self.best_maxf,
                    "best_smeasure": self.best_smeasure,
                    "best_wfm": self.best_wfm,
                },
                save_path,
            )

            print(
                f"✅ Best Model Saved! "
                f"MAE: {val_metrics['mae']:.4f} | "
                f"maxF: {val_metrics['maxf']:.4f} | "
                f"wF: {val_metrics['wfm']:.4f} | "
                f"S-measure: {val_metrics['smeasure']:.4f} | "
                f"maxE: {val_metrics['maxe']:.4f}"
            )

    def _save_training_log(self):
        """ 保存训练日志 """
        if not dist_utils.is_main_process():
            return

        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,

            "val_maes": self.val_maes,

            "val_maxfs": self.val_maxfs,
            "val_meanfs": self.val_meanfs,
            "val_adpfs": self.val_adpfs,

            "val_wfms": self.val_wfms,

            "val_maxes": self.val_maxes,
            "val_meanes": self.val_meanes,
            "val_adpes": self.val_adpes,

            "val_smeasures": self.val_smeasures,

            "best_mae": self.best_mae,
            "best_maxf": self.best_maxf,
            "best_smeasure": self.best_smeasure,
            "best_wfm": self.best_wfm,
        }

        log_path = os.path.join(self.output_dir, "training_log.json")

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=4, ensure_ascii=False)

    def train(self, epochs):
        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_fmeasures = []
        self.val_emeasures = []

        self._ds_warmup_epochs = max(1, int(epochs * self.ds_warmup_ratio))

        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=epochs,
        #     eta_min=LR_ETA_MIN
        # )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[15, 22],
            gamma=0.1
        )

        for epoch in range(epochs):
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self._run_epoch(epoch)
            val_loss, val_metrics = self._evaluate(epoch)

            # 学习率调度
            scheduler.step()                    

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.val_maes.append(val_metrics["mae"])

            self.val_maxfs.append(val_metrics["maxf"])
            self.val_meanfs.append(val_metrics["meanf"])
            self.val_adpfs.append(val_metrics["adpf"])

            self.val_wfms.append(val_metrics["wfm"])

            self.val_maxes.append(val_metrics["maxe"])
            self.val_meanes.append(val_metrics["meane"])
            self.val_adpes.append(val_metrics["adpe"])

            self.val_smeasures.append(val_metrics["smeasure"])

            current_lr = self.optimizer.param_groups[0]['lr']

            if dist_utils.is_main_process():
                print(
                    f"Epoch {epoch + 1:02d}/{epochs} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"val_loss: {val_loss:.4f} | "
                    f"MAE: {val_metrics['mae']:.4f} | "
                    f"maxF: {val_metrics['maxf']:.4f} | "
                    f"wF: {val_metrics['wfm']:.4f} | "
                    f"S: {val_metrics['smeasure']:.4f} | "
                    f"maxE: {val_metrics['maxe']:.4f} | "
                    f"lr: {current_lr:.6f}"
                )

            self._save_checkpoints(val_metrics)
            self._save_training_log()
            dist_utils.barrier()

        # ECSSD 训练/测试时在训练阶段保存结果
        if SCALING is not True and dist_utils.is_main_process():
            self._save_result()
        if dist_utils.is_main_process():
            print(f"✅ Model Training Finished! Best MAE: {self.best_mae}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,

            "val_maes": self.val_maes,

            "val_maxfs": self.val_maxfs,
            "val_meanfs": self.val_meanfs,
            "val_adpfs": self.val_adpfs,

            "val_wfms": self.val_wfms,

            "val_maxes": self.val_maxes,
            "val_meanes": self.val_meanes,
            "val_adpes": self.val_adpes,

            "val_smeasures": self.val_smeasures,

            "best_mae": self.best_mae,
            "best_maxf": self.best_maxf,
            "best_smeasure": self.best_smeasure,
            "best_wfm": self.best_wfm,
        }
