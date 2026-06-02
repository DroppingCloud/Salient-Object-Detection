#!/bin/bash

MODEL=${1:-PoolNetCFM}

echo ">>> Training: $MODEL"
python src/main.py --model "$MODEL"

echo ">>> Evaluating: $MODEL"
python src/eval.py --model "$MODEL"
