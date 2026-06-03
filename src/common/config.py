import os
import torch

from model import (
    F3Net, F3NetASPP, F3NetCBAM,
    PoolNet, PoolNetCFM, PoolNetDS, PoolNetCFMDS, PoolNetFBDA,
    PoolNetCFMDSFBDA, PoolNetCFMFBDA,
    PoolNetRRM, PoolNetCFMRRM,
    PoolNetCA, PoolNetCFMCARRM,
    PoolNetCFMGA,
    CPDResNet,
)

# ──────────────────────────────────────────
# 模型注册表
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "PoolNet": PoolNet,
    "PoolNetCFM": PoolNetCFM,
    "PoolNetDS": PoolNetDS,
    "PoolNetCFMDS": PoolNetCFMDS,
    "PoolNetFBDA": PoolNetFBDA,
    "PoolNetCFMFBDA": PoolNetCFMFBDA,
    "PoolNetCFMDSFBDA": PoolNetCFMDSFBDA,
    "PoolNetRRM": PoolNetRRM,
    "PoolNetCFMRRM": PoolNetCFMRRM,
    "PoolNetCA": PoolNetCA,
    "PoolNetCFMCARRM": PoolNetCFMCARRM,
    "PoolNetCFMGA": PoolNetCFMGA,
    "CPDResNet": CPDResNet,
    "F3Net": F3Net,
    "F3NetCBAM": F3NetCBAM,
    "F3NetASPP": F3NetASPP,
}

# ──────────────────────────────────────────
# 路径
# ──────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

# 训练平台
PLATFORM = "AutoDL"   

# 数据规模
SCALING  = True

# 数据路径
if PLATFORM == "Local":
    DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "ECSSD")
    TEST_DIR  = os.path.join(_SRC_DIR, "../data/test")

elif PLATFORM == "Colab":
    DATA_ROOT = "/content/datasets/ECSSD"
    TEST_DIR  = "/content/datasets/test"

elif PLATFORM == "AutoDL":
    DATA_ROOT = "/root/autodl-tmp/data/ECSSD" if SCALING is not True else "/root/autodl-tmp/data/DUTS-TR"
    TEST_DIR  = "/root/autodl-tmp/data/test"  if SCALING is not True else "/root/autodl-tmp/data/ECSSD"

else:
    raise ValueError(f"Unsupported PLATFORM: {PLATFORM}")


IMAGE_FOLDER  = "images"
MASK_FOLDER   = "masks"
OUTPUT_DIR    = os.path.join(_SRC_DIR, "../outputs")

# ──────────────────────────────────────────
# 数据预处理
# ──────────────────────────────────────────
RESIZE_SIZE  = 256
CROP_SIZE    = 224
FLIP_PROB    = 0.5
MASK_THRESH  = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────
VAL_RATIO    = 0.3 if SCALING is not True else 0.1
BATCH_SIZE   = 16 if SCALING is not True else 64
NUM_WORKERS  = 0
SEED         = 42

# ──────────────────────────────────────────
# 训练
# ──────────────────────────────────────────
EPOCHS          = 25
LEARNING_RATE   = 3e-4 if SCALING is not True else 2e-4
BACKBONE_LR     = 5e-5 if SCALING is not True else 1e-5
WEIGHT_DECAY    = 5e-4
LR_ETA_MIN      = 1e-6

THRESHOLD       = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────
MAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT  = 0.4

# # ──────────────────────────────────────────
# # 评估指标
# # ──────────────────────────────────────────
# FMEASURE_BETA2   = 0.3
# WFMEASURE_BETA2  = 1.0
# MAXF_NUM_THRESH  = 255
# MAXE_NUM_THRESH  = 255
EVAL_BATCH_SIZE  = 8
EPS              = 1e-8

# ──────────────────────────────────────────
# 可视化
# ──────────────────────────────────────────
VIZ_NUM_SAMPLES = 5
VIZ_SEED        = 42
