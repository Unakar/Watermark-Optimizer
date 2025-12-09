#!/bin/bash
# Run experiment comparing all four optimizers:
# - AdamW: Standard optimizer (no dualization)
# - Muon: Orthogonalization only (no spectral constraint)
# - MuonBall: With dualization λ=0 (faster, no solver)
# - SpectralBall: With dualization (solves λ)

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 3e-2 \
    --lr-max 1.3 \
    --num-lrs 4 \
    --num-steps 500 \
    --batch-size 1024 \
    --optimizers muon_ball adamw \
    --output-dir ./results/all_optimizers \
    --use-mup-init \
    --use-mup-lr \
    --init-sigma 0.02 \

echo ""
echo "Experiment complete! Results saved to ./results/all_optimizers"

