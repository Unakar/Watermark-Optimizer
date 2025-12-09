#!/bin/bash
# Run experiment comparing all four optimizers:
# - AdamW: Standard optimizer (no dualization)
# - Muon: Orthogonalization only (no spectral constraint)
# - MuonBall: With dualization λ=0 (faster, no solver)
# - SpectralBall: With dualization (solves λ)

cd "$(dirname "$0")/.."

python3 run_experiment.py \
    --lr-min 0.1 \
    --lr-max 10.0 \
    --num-lrs 10 \
    --num-steps 1000 \
    --batch-size 128 \
    --optimizers adamw muon muon_ball spectral_ball \
    --output-dir ./results/all_optimizers \
    --no-plot

echo ""
echo "Experiment complete! Results saved to ./results/all_optimizers"

