#!/bin/bash
# Full experiment with default settings (10 LRs, 1000 steps)

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 0.1 \
    --lr-max 10.0 \
    --num-lrs 10 \
    --num-steps 1000 \
    --batch-size 128 \
    --output-dir ./results/full_experiment \
    --no-plot

