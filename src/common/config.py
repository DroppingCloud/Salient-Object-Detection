import os
import random
import numpy as np
import torch

from model import (
    F3Net, F3NetASPP, F3NetCBAM,
    PoolNet, PoolNetDS, PoolNetCFM, PoolNetGate, PoolNetGateCFM,
    PoolNetASPP, PoolNetASPP_CFM, PoolNetASPPGate,
    CPDResNet, CPDASPP, CPDCBAM, CPDCFM, CPDGate, CPDDS, CPDResHA, CPDGatedHA, CPDDetailHint,
    GateNet, GateNetCBAM, GateNetDS,
    BASNet,
    ResNet18, ResNet18Pre, ResNet34Pre, ResNet50Pre,
)

# ──────────────────────────────────────────
# 模型注册表
# ──────────────────────────────────────────
MODEL_REGISTRY = {
    "PoolNet": PoolNet,
    "PoolNetDS": PoolNetDS,
    "PoolNetCFM": PoolNetCFM,
    "PoolNetGate": PoolNetGate,
    "PoolNetASPP": PoolNetASPP,
    "PoolNetASPP_CFM": PoolNetASPP_CFM,
    "PoolNetGateCFM": PoolNetGateCFM,
    "PoolNetASPPGate": PoolNetASPPGate,
    "F3Net": F3Net,
    "F3NetCBAM": F3NetCBAM,
    "F3NetASPP": F3NetASPP,
    "CPDResNet": CPDResNet,
    "CPDASPP": CPDASPP,
    "CPDCBAM": CPDCBAM,
    "CPDCFM": CPDCFM,
    "CPDGate": CPDGate,
    "CPDDS": CPDDS,
    "CPDResHA": CPDResHA,
    "CPDGatedHA": CPDGatedHA,
    "CPDDetailHint": CPDDetailHint,
    "GateNet": GateNet,
    "GateNetCBAM": GateNetCBAM,
    "GateNetDS": GateNetDS,
    "BASNet": BASNet,
}

# ──────────────────────────────────────────
# Backbone 配置
# ──────────────────────────────────────────
BACKBONE_REGISTRY = {
    "resnet18": {
        "pretrained": ResNet18Pre,
        "scratch": ResNet18,
        "channels": [64, 128, 256, 512],
    },
    "resnet34": {
        "pretrained": ResNet34Pre,
        "scratch": None,
        "channels": [64, 128, 256, 512],
    },
    "resnet50": {
        "pretrained": ResNet50Pre,
        "scratch": None,
        "channels": [256, 512, 1024, 2048],
    },
}

BACKBONE = "resnet34"   # 切换 backbone: "resnet18" | "resnet34" | "resnet50"

# ──────────────────────────────────────────
# 路径
# ──────────────────────────────────────────
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_gpu_ids(default):
    value = os.environ.get("GPU_IDS")
    if value is None:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]

# 训练平台
PLATFORM = "AutoDL"   

# 数据规模
SCALING  = False

# 数据路径
if PLATFORM == "Local":
    DATA_ROOT = os.path.join(_PROJECT_ROOT, "data", "train")
    TEST_DIR  = os.path.join(_SRC_DIR, "../data/test")

elif PLATFORM == "Colab":
    DATA_ROOT = "/content/datasets/train"
    TEST_DIR  = "/content/datasets/test"

elif PLATFORM == "AutoDL":
    DATA_ROOT = "/root/autodl-tmp/data/ECSSD" if SCALING is not True else "/root/autodl-tmp/data/DUTS-TR"
    TEST_DIR  = "/root/autodl-tmp/data/test"  if SCALING is not True else "/root/autodl-tmp/data/ECSSD"

else:
    raise ValueError(f"Unsupported PLATFORM: {PLATFORM}")


IMAGE_FOLDER  = "images"
MASK_FOLDER   = "masks"
_OUTPUT_BASE = os.path.join(_SRC_DIR, "./outputs/scaling") if SCALING else os.path.join(_SRC_DIR, "./outputs/small")
OUTPUT_DIR = os.path.join(_OUTPUT_BASE, BACKBONE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────
# 数据预处理
# ──────────────────────────────────────────
RESIZE_SIZE  = 320
FLIP_PROB    = 0.5
MASK_THRESH  = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ──────────────────────────────────────────
# 单卡 / 多卡控制
# ──────────────────────────────────────────
MULTI_GPU          = _env_bool("MULTI_GPU", True)
GPU_IDS            = _env_gpu_ids([0, 1])
DIST_BACKEND       = "nccl"
EVAL_USE_MULTI_GPU = _env_bool("EVAL_USE_MULTI_GPU", False)
USE_AMP            = _env_bool("USE_AMP", False)

BATCH_SIZE   = 64 if SCALING else 8

# ──────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────
VAL_RATIO    = 0.1 if SCALING else 0.3
NUM_WORKERS  = 8
SEED         = 42

# ──────────────────────────────────────────
# 固定随机种子
# ──────────────────────────────────────────
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ──────────────────────────────────────────
# 训练
# ──────────────────────────────────────────
EPOCHS          = 25
LEARNING_RATE   = 3e-4 if SCALING is not True else 2e-4
BACKBONE_LR     = 5e-5 if SCALING is not True else 3e-5
WEIGHT_DECAY    = 5e-4
LR_ETA_MIN      = 1e-6

THRESHOLD       = 0.5
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# ──────────────────────────────────────────
# 损失函数
# ──────────────────────────────────────────
MAIN_LOSS_WEIGHT = 1.0
AUX_LOSS_WEIGHT  = [0.4]
EDGE_LOSS_WEIGHT = 0.3

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
