#!/bin/bash
# Full experiment with μP initialization and learning rate scaling
# This uses:
#   - μP init: W = σ * √(d_out/d_in) / ||W'||₂ * W'
#   - μP lr: lr_layer = lr_base * √(d_out/d_in)

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 0.1 \
    --lr-max 10.0 \
    --num-lrs 10 \
    --num-steps 1000 \
    --batch-size 128 \
    --use-mup-init \
    --use-mup-lr \
    --init-sigma 0.02 \
    --output-dir ./results/mup_experiment \
    --no-plot

