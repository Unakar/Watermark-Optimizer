#!/bin/bash
# Quick experiment with fewer LRs and steps for testing

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 0.1 \
    --lr-max 5.0 \
    --num-lrs 5 \
    --num-steps 500 \
    --batch-size 128 \
    --output-dir ./results/quick_test \
    --no-plot

