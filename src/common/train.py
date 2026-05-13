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

        self.threshold = threshold

    def _compute_mae(self, pred, mask):
        """ 计算 MAE """
        return torch.mean(torch.abs(pred - mask)).item()

    def _compute_fmeasure(self, pred, mask, beta2=0.3):
        """ 计算 F-measure """
        pred_binary = (pred >= self.threshold).float()
        mask_binary = (mask >= 0.5).float()

        tp = torch.sum(pred_binary * mask_binary)
        precision = tp / (torch.sum(pred_binary) + 1e-8)
        recall = tp / (torch.sum(mask_binary) + 1e-8)

        fmeasure = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)

        return fmeasure.item()

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
            _, predicted = torch.max(outputs, 1)

            # 损失计算
            loss = self.criterion(outputs, masks)
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
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc=f"Epoch {epoch+1:02d} [Eval]"):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, masks)

                preds = torch.sigmoid(outputs)
                mae = self._compute_mae(preds, masks)
                fmeasure = self._compute_fmeasure(preds, masks)

                total_loss += loss.item() * images.size(0)
                total_mae += mae * images.size(0)
                total_fmeasure += fmeasure * images.size(0)
                total_samples += images.size(0)
            
        avg_loss = total_loss / total_samples
        avg_mae = total_mae / total_samples
        avg_fmeasure = total_fmeasure / total_samples

        return avg_loss, avg_mae, avg_fmeasure
    
    def _save_checkpoints(self, val_mae, val_fmeasure):
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
                f"MAE: {val_mae:.4f} | F-measure: {val_fmeasure:.4f}"
            )

    def _save_training_log(self):
        """ 保存训练日志 """
        log = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_maes": self.val_maes,
            "val_fmeasures": self.val_fmeasures,
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

        for epoch in range(epochs):
            train_loss = self._run_epoch(epoch)
            val_loss, val_mae, val_fmeasure = self._evaluate(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_maes.append(val_mae)
            self.val_fmeasures.append(val_fmeasure)

            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"MAE: {val_mae:.4f} | "
                f"F-measure: {val_fmeasure:.4f}"
            )

            self._save_checkpoints(val_mae, val_fmeasure)
            self._save_training_log()
        
        print("✅ Model Training Finished!")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_maes": self.val_maes,
            "val_fmeasures": self.val_fmeasures,
            "best_mae": self.best_mae,
            "best_fmeasure": self.best_fmeasure,
        }
