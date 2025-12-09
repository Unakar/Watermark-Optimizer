#!/bin/bash
# Run experiment comparing all three optimizers:
# - AdamW: Standard optimizer (no dualization)
# - SpectralBall: With dualization (solves λ)
# - MuonBall: With dualization λ=0 (faster, no solver)

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 0.1 \
    --lr-max 10.0 \
    --num-lrs 10 \
    --num-steps 1000 \
    --batch-size 128 \
    --optimizers adamw spectral_ball muon_ball \
    --output-dir ./results/all_optimizers \
    --no-plot

echo ""
echo "Experiment complete! Results saved to ./results/all_optimizers"

