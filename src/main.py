import os
import random
import numpy as np

import torch
import torch.nn as nn

from model import F3Net, PoolNet, CPDResNet
from common import (
    build_saliency_dataloader,
    Trainer,
    plot_training_curves,
    visualize_predictions,
    visualize_predictions_with_error,
)

if __name__ == "__main__":
    # =========================
    # 超参数
    # =========================
    hparams = {
        "root_dir": "../data/ECSSD",
        "image_folder": "images",
        "mask_folder": "masks",

        "val_ratio": 0.3,
        "batch_size": 16,
        "num_workers": 0,

        "epochs": 25,
        "learning_rate": 1e-4,

        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "output_dir": os.path.join(os.path.dirname(__file__), "../outputs"),

        "seed": 42,
    }

    print(f"Using device: {hparams['device']}")

    # =========================
    # 数据准备
    # =========================
    dataset, train_dataset, train_loader, val_dataset, valid_loader = build_saliency_dataloader(
        root_dir=hparams["root_dir"],
        image_folder=hparams["image_folder"],
        mask_folder=hparams["mask_folder"],
        val_ratio=hparams["val_ratio"],
        batch_size=hparams["batch_size"],
        num_workers=hparams["num_workers"],
        seed=hparams["seed"],
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Train size  : {len(train_dataset)}")
    print(f"Valid size  : {len(val_dataset)}")

    # =========================
    # 实例化模型
    # =========================
    model = F3Net().to(hparams["device"])

    # =========================
    # 损失函数与优化器
    # =========================
    criterion = nn.BCEWithLogitsLoss()

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 1e-5},
        {"params": other_params,                "lr": 1e-4},
    ], weight_decay=1e-4)

    # =========================
    # 创建训练器
    # =========================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=hparams["device"],
        output_dir=hparams["output_dir"],
        threshold=0.5
    )

    # =========================
    # 模型训练
    # =========================
    history = trainer.train(
        epochs=hparams["epochs"]
    )

    # =========================
    # 绘制训练曲线
    # =========================
    model_name = model.__class__.__name__
    save_dir = os.path.join(hparams["output_dir"], model_name)

    plot_training_curves(
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        val_maes=history["val_maes"],
        val_fmeasures=history["val_fmeasures"],
        title_prefix=model_name,
        save_path=os.path.join(save_dir, "training_curves.png")
    )

    # =========================
    # 可视化预测结果
    # =========================
    visualize_predictions(
        model=model,
        dataloader=valid_loader,
        device=hparams["device"],
        num_samples=4,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold=0.5,
        save_path=os.path.join(save_dir, "predictions.png"),
        seed=42,
    )

    visualize_predictions_with_error(
        model=model,
        dataloader=valid_loader,
        device=hparams["device"],
        num_samples=4,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        threshold=0.5,
        save_path=os.path.join(save_dir, "predictions_with_error.png"),
    )
