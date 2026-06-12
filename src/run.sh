#!/bin/bash

# ============================================================
# Batch training and evaluation script
# ============================================================
#
# 功能：
#   支持一次传入一个或多个模型名，并按顺序完成：
#     1. 训练模型
#     2. 评估模型
#
# 参数：
#   $1: 模型名列表，默认 PoolNetCFM
#       支持三种形式：
#         - 单模型：
#             PoolNetCFM
#         - 逗号分隔多个模型：
#             PoolNet,PoolNetCFM,PoolNetGateCFM
#         - 空格分隔多个模型：
#             "PoolNet PoolNetCFM PoolNetGateCFM"
#
#   $2: 运行模式，默认 multi
#       可选：
#         - multi  : 多 GPU 分布式训练
#         - single : 单 GPU / 普通训练
#
#   $3: GPU 数量，默认 2
#       例如：
#         - 1 -> 使用 GPU 0
#         - 2 -> 使用 GPU 0,1
#         - 4 -> 使用 GPU 0,1,2,3
#
# 环境变量：
#   PYTHON_BIN:
#       指定 Python 解释器，默认 python
#
# 常见用法：
#   1. 训练并评估单个模型，使用默认 multi + 2 GPUs：
#        bash run.sh PoolNetCFM
#
#   2. 训练并评估单个模型，使用 2 GPUs：
#        bash run.sh PoolNetCFM multi 2
#
#   3. 训练并评估多个模型，逗号分隔：
#        bash run.sh PoolNet,PoolNetCFM,PoolNetGateCFM multi 2
#
#   4. 训练并评估多个模型，空格分隔：
#        bash run.sh "PoolNet PoolNetCFM PoolNetGateCFM" multi 2
#
#   5. 单 GPU 依次训练并评估多个模型：
#        bash run.sh PoolNet,PoolNetCFM single 1
#
# 说明：
#   - 某个模型训练失败后，会跳过该模型的评估，并继续处理下一个模型。
#   - 某个模型评估失败后，不会影响后续模型继续运行。
#
# ============================================================

MODELS_RAW=${1:-PoolNetCFM}
MODE=${2:-multi}
NGPU=${3:-2}
PYTHON_BIN=${PYTHON_BIN:-python}

MODELS_RAW=${MODELS_RAW//,/ }
read -r -a MODELS <<< "$MODELS_RAW"

GPU_IDS=$(seq -s, 0 $((NGPU - 1)))

echo ">>> Models: ${MODELS[*]}"
echo ">>> Mode: $MODE | NGPU: $NGPU | GPU_IDS: $GPU_IDS"
echo

for MODEL in "${MODELS[@]}"; do
  echo "============================================================"
  echo ">>> Training: $MODEL (mode=$MODE, ngpu=$NGPU, gpu_ids=$GPU_IDS)"
  echo "============================================================"

  if [ "$MODE" = "multi" ]; then
    MULTI_GPU=1 GPU_IDS="$GPU_IDS" "$PYTHON_BIN" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="$NGPU" \
      src/main.py \
      --model "$MODEL"
  else
    MULTI_GPU=0 "$PYTHON_BIN" src/main.py \
      --model "$MODEL"
  fi

  TRAIN_STATUS=$?

  if [ $TRAIN_STATUS -ne 0 ]; then
    echo
    echo "!!! Training failed for model: $MODEL"
    echo "!!! Skip evaluation for this model."
    echo
    continue
  fi

  echo
  echo "============================================================"
  echo ">>> Evaluating: $MODEL"
  echo "============================================================"

  MULTI_GPU=0 "$PYTHON_BIN" src/eval.py \
    --model "$MODEL"

  EVAL_STATUS=$?

  if [ $EVAL_STATUS -ne 0 ]; then
    echo
    echo "!!! Evaluation failed for model: $MODEL"
  else
    echo
    echo ">>> Finished: $MODEL"
  fi

  echo
done

echo ">>> All requested models have been processed."