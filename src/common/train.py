import os
import json
from tqdm import tqdm

import torch

class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, valid_loader, device, output_dir="./outputs", threshold=0.5):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.device = device

        model_name = self.model.__class__.__name__
        self.output_dir = os.path.join(output_dir, model_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_mae = float("inf")
        self.best_fmeasure = 0.0

        self.train_losses = []
        self.val_losses = []
        self.val_maes = []
        self.val_fmeasures = []
        self.val_emeasures = []

        # 二值化阈值
        self.threshold = threshold

    def _compute_mae(self, pred, mask):
        """ 计算 MAE """
        return torch.mean(torch.abs(pred - mask)).item()

    def _compute_fmeasure(self, pred, mask, beta2=0.3):
        """ 计算 F-measure（per-image 平均）"""
        pred_binary = (pred >= self.threshold).float()
        mask_binary = (mask >= 0.5).float()

        scores = []
        for p, g in zip(pred_binary, mask_binary):
            tp = (p * g).sum()
            precision = tp / (p.sum() + 1e-8)
            recall = tp / (g.sum() + 1e-8)
            fm = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
            scores.append(fm)

        return torch.stack(scores).mean().item()

    def _compute_emeasure(self, pred, mask, eps=1e-8):
        """ 计算 E-measure """
        pred = pred.float()
        mask = (mask >= 0.5).float()

        scores = []

        for p, g in zip(pred, mask):
            p = p.squeeze()
            g = g.squeeze()

            # 特殊情况：GT 全背景
            if g.sum() == 0:
                scores.append(1.0 - p.mean())
                continue

            # 特殊情况：GT 全前景
            if g.sum() == g.numel():
                scores.append(p.mean())
                continue

            p_mean = p.mean()
            g_mean = g.mean()

            p_align = p - p_mean
            g_align = g - g_mean

            align_matrix = 2 * p_align * g_align / (
                p_align * p_align + g_align * g_align + eps
            )

            enhanced = ((align_matrix + 1) ** 2) / 4

            scores.append(enhanced.mean())

        return torch.stack(scores).mean().item()

    def _compute_bce_iou_loss(self, pred_logits, masks):
        """ BCE + Soft IoU Loss """
        bce = self.criterion(pred_logits, masks)

        pred = torch.sigmoid(pred_logits)
        inter = (pred * masks).sum(dim=(1, 2, 3))
        union = (pred + masks - pred * masks).sum(dim=(1, 2, 3))
        iou = (inter / (union + 1e-8)).mean()

        return bce + (1.0 - iou)

    def _compute_loss(self, outputs, masks):
        """ 单输出模型与多输出模型损失计算 """
        if not isinstance(outputs, (tuple, list)):
            return self._compute_bce_iou_loss(outputs, masks)

        loss = 0.0
        for i, out in enumerate(outputs):
            w = 2.0 if i == 0 else 1.0
            loss = loss + w * self._compute_bce_iou_loss(out, masks)
        return loss

    def _run_epoch(self, epoch):
        self.model.train()

        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1:02d} [Training]"):
            # 清空梯度
            self.optimizer.zero_grad()

            # 数据上传
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)

            # 模型输出
            outputs = self.model(images)

            # 损失计算
            loss = self._compute_loss(outputs, masks)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            # 梯度反向传播
            loss.backward()
            self.optimizer.step()

        # 该 Epoch 平均样本损失
        avg_loss = total_loss / total_samples

        return avg_loss

    def _evaluate(self, epoch):
        self.model.eval()

        total_loss = 0.0
        total_mae = 0.0
        total_fmeasure = 0.0
        total_emeasure = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch+1:02d} [Eval]"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images)

                loss = self._compute_loss(outputs, masks)

                # 取最终预测图
                preds = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                preds = torch.sigmoid(preds)

                # 计算指标
                mae = self._compute_mae(preds, masks)
                fmeasure = self._compute_fmeasure(preds, masks)
                emeasure = self._compute_emeasure(preds, masks)

                total_loss += loss.item() * images.size(0)
                total_mae += mae * images.size(0)
                total_fmeasure += fmeasure * images.size(0)
                total_emeasure += emeasure * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        avg_fmeasure = total_fmeasure / total_samples
        avg_emeasure = total_emeasure / total_samples

        return avg_loss, avg_mae, avg_fmeasure, avg_emeasure

    def _save_checkpoints(self, val_mae, val_fmeasure, val_emeasure):
        if val_mae < self.best_mae:
            self.best_mae = val_mae
            self.best_fmeasure = val_fmeasure

            save_path = os.path.join(self.output_dir, "best_model.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_mae": self.best_mae,
                    "best_fmeasure": self.best_fmeasure,
                },
                save_path
            )

            print(
                f"✅ Best Model Saved! "
                f"MAE: {val_mae:.4f} | F-measure: {val_fmeasure:.4f} | E-measure: {val_emeasure:.4f}"
            )

    def _save_training_log(self):
        """ 保存训练日志 """
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_maes": self.val_maes,
            "val_fmeasures": self.val_fmeasures,
            "val_emeasures": self.val_emeasures,
            "best_mae": self.best_mae,
            "best_fmeasure": self.best_fmeasure,
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

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-6
        )

        for epoch in range(epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, val_mae, val_fmeasure, val_emeasure = self._evaluate(epoch)

            scheduler.step()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            self.val_fmeasures.append(val_fmeasure)
            self.val_emeasures.append(val_emeasure)

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"MAE: {val_mae:.4f} | "
                f"F-measure: {val_fmeasure:.4f} | "
                f"E-measure: {val_emeasure:.4f}"
            )

            self._save_checkpoints(val_mae, val_fmeasure, val_emeasure)
            self._save_training_log()

        print(f"✅ Model Training Finished! Best MAE: {self.best_mae}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_maes": self.val_maes,
            "val_fmeasures": self.val_fmeasures,
            "val_emeasures": self.val_emeasures,
            "best_mae": self.best_mae,
            "best_fmeasure": self.best_fmeasure,
        }
