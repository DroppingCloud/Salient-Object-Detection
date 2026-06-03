#!/bin/bash

MODEL=${1:-PoolNetCFM}
MODE=${2:-multi}
PYTHON_BIN=${PYTHON_BIN:-python}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

echo ">>> Training: $MODEL"
if [ "$MODE" = "multi" ]; then
  MULTI_GPU=1 "$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node="$NPROC_PER_NODE" src/main.py --model "$MODEL"
else
  MULTI_GPU=0 "$PYTHON_BIN" src/main.py --model "$MODEL"
fi

echo ">>> Evaluating: $MODEL"
MULTI_GPU=0 "$PYTHON_BIN" src/eval.py --model "$MODEL"
