import os
import argparse

import torch
import torch.nn as nn

from common import (
    build_saliency_dataloader,
    Trainer,
    plot_training_curves,
    visualize_predictions,
    visualize_predictions_with_error,
    config,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PoolNetCFM",
                        choices=list(config.MODEL_REGISTRY.keys()),
                        help="选择模型")
    args = parser.parse_args()

    print(f"Using device: {config.DEVICE}")

    # =========================
    # 数据准备
    # =========================
    dataset, train_dataset, train_loader, val_dataset, valid_loader = build_saliency_dataloader(
        root_dir=config.DATA_ROOT,
        image_folder=config.IMAGE_FOLDER,
        mask_folder=config.MASK_FOLDER,
        val_ratio=config.VAL_RATIO,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        seed=config.SEED,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Train size  : {len(train_dataset)}")
    print(f"Valid size  : {len(val_dataset)}")

    # =========================
    # 实例化模型
    # =========================
    model = config.MODEL_REGISTRY[args.model]().to(config.DEVICE)
    print(f"Model       : {model.__class__.__name__}")

    # =========================
    # 损失函数与优化器
    # =========================
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.BACKBONE_LR},
        {"params": other_params,                "lr": config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    # =========================
    # 创建训练器
    # =========================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=config.DEVICE,
        output_dir=config.OUTPUT_DIR,
        threshold=config.THRESHOLD,
    )

    # =========================
    # 模型训练
    # =========================
    history = trainer.train(epochs=config.EPOCHS)

    for name, p in model.named_parameters():
        if "gamma" in name:
            print(name, p.item())
        if "attn_gamma" in name:
            print(name, p.item())

    # =========================
    # 绘制训练曲线
    # =========================
    model_name = model.__class__.__name__
    save_dir = os.path.join(config.OUTPUT_DIR, model_name)

    plot_training_curves(
        train_losses=history["train_losses"],
        val_losses=history["val_losses"],
        val_maes=history["val_maes"],
        val_maxfs = history["val_maxfs"],
        title_prefix=model_name,
        save_path=os.path.join(save_dir, "training_curves.png")
    )

    # =========================
    # 可视化预测结果
    # =========================
    visualize_predictions(
        model=model,
        dataloader=valid_loader,
        device=config.DEVICE,
        num_samples=config.VIZ_NUM_SAMPLES,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
        threshold=config.THRESHOLD,
        save_path=os.path.join(save_dir, "predictions.png"),
        seed=config.VIZ_SEED,
    )

    visualize_predictions_with_error(
        model=model,
        dataloader=valid_loader,
        device=config.DEVICE,
        num_samples=config.VIZ_NUM_SAMPLES,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
        threshold=config.THRESHOLD,
        save_path=os.path.join(save_dir, "predictions_with_error.png"),
    )
