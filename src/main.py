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
from common import distributed as dist_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="PoolNetCFM",
                        choices=list(config.MODEL_REGISTRY.keys()),
                        help="选择模型")
    args = parser.parse_args()

    device = dist_utils.init_distributed(config)
    use_distributed = dist_utils.can_use_distributed(config)

    if dist_utils.is_main_process():
        print(f"Using device: {device}")
        print(f"Multi-GPU training: {use_distributed}")
        if use_distributed:
            print(f"World size: {dist_utils.get_world_size()} | GPU IDs: {config.GPU_IDS}")

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
        distributed=use_distributed,
    )

    if dist_utils.is_main_process():
        print(f"Dataset size: {len(dataset)}")
        print(f"Train size  : {len(train_dataset)}")
        print(f"Valid size  : {len(val_dataset)}")
        print(f"Batch size  : {config.BATCH_SIZE} per process")

    # =========================
    # 实例化模型
    # =========================
    model_cls = config.MODEL_REGISTRY[args.model]
    # 支持 backbone_name 参数的模型传入 BACKBONE 配置
    try:
        raw_model = model_cls(backbone_name=config.BACKBONE).to(device)
    except TypeError:
        raw_model = model_cls().to(device)
    if dist_utils.is_main_process():
        print(f"Model       : {raw_model.__class__.__name__}")

    # =========================
    # 损失函数与优化器
    # =========================
    criterion = nn.BCEWithLogitsLoss(reduction='mean')

    backbone_ids = {id(p) for p in raw_model.backbone.parameters()}
    other_params = [p for p in raw_model.parameters() if id(p) not in backbone_ids]

    optimizer = torch.optim.AdamW([
        {"params": raw_model.backbone.parameters(), "lr": config.BACKBONE_LR},
        {"params": other_params,                "lr": config.LEARNING_RATE},
    ], weight_decay=config.WEIGHT_DECAY)

    model = dist_utils.wrap_model(raw_model, device, config)

    # =========================
    # 创建训练器
    # =========================
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        output_dir=config.OUTPUT_DIR,
        threshold=config.THRESHOLD,
    )

    # =========================
    # 模型训练
    # =========================
    history = trainer.train(epochs=config.EPOCHS)

    if not dist_utils.is_main_process():
        return

    unwrapped_model = dist_utils.unwrap_model(model)
    for name, p in unwrapped_model.named_parameters():
        if "gamma" in name:
            print(name, p.item())
        if "attn_gamma" in name:
            print(name, p.item())

    # =========================
    # 绘制训练曲线
    # =========================
    model_name = unwrapped_model.__class__.__name__
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
        model=unwrapped_model,
        dataloader=valid_loader,
        device=device,
        num_samples=config.VIZ_NUM_SAMPLES,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
        threshold=config.THRESHOLD,
        save_path=os.path.join(save_dir, "predictions.png"),
        seed=config.VIZ_SEED,
    )

    visualize_predictions_with_error(
        model=unwrapped_model,
        dataloader=valid_loader,
        device=device,
        num_samples=config.VIZ_NUM_SAMPLES,
        mean=config.IMAGENET_MEAN,
        std=config.IMAGENET_STD,
        threshold=config.THRESHOLD,
        save_path=os.path.join(save_dir, "predictions_with_error.png"),
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        dist_utils.destroy_distributed()
