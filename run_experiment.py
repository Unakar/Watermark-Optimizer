#!/usr/bin/env python3
"""Main entry point for the watermark erasure experiment.

This script runs the learning rate sweep experiment comparing:
- AdamW (Without dualization)
- SpectralBall (With dualization)

Usage:
    python run_experiment.py [options]
    
Examples:
    # Run with default settings
    python run_experiment.py
    
    # Run with custom learning rate range
    python run_experiment.py --lr-min 0.01 --lr-max 5.0
    
    # Run quick test with fewer LRs and steps
    python run_experiment.py --num-lrs 5 --num-steps 500
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from experiments.lr_sweep import run_lr_sweep_experiment, print_summary
from utils.visualization import plot_lr_sweep_results, save_experiment_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Watermark Erasure Experiment: Compare AdamW vs SpectralBall",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Learning rate sweep settings
    parser.add_argument(
        "--lr-min", type=float, default=0.1,
        help="Minimum learning rate (log scale)"
    )
    parser.add_argument(
        "--lr-max", type=float, default=10.0,
        help="Maximum learning rate (log scale)"
    )
    parser.add_argument(
        "--num-lrs", type=int, default=10,
        help="Number of learning rates to sweep"
    )
    
    # Training settings
    parser.add_argument(
        "--num-steps", type=int, default=1000,
        help="Number of training steps per experiment"
    )
    parser.add_argument(
        "--batch-size", type=int, default=128,
        help="Batch size for training"
    )
    
    # Optimizer selection
    parser.add_argument(
        "--optimizers", type=str, nargs="+",
        default=["adamw", "spectral_ball"],
        choices=["adamw", "spectral_ball", "muon_ball", "muon"],
        help="Optimizers to compare (muon_ball = SpectralBall with λ=0, muon = no retraction)"
    )
    
    # μP (Maximal Update Parameterization) settings
    parser.add_argument(
        "--use-mup-init", action="store_true",
        help="Use Spectral μP initialization: W = σ * √(d_out/d_in) / ||W'||₂ * W'"
    )
    parser.add_argument(
        "--use-mup-lr", action="store_true",
        help="Use μP learning rate scaling: lr_layer = lr_base * √(d_out/d_in)"
    )
    parser.add_argument(
        "--init-sigma", type=float, default=0.02,
        help="σ parameter for μP initialization"
    )
    
    # Device and reproducibility
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    
    # Data and output
    parser.add_argument(
        "--data-root", type=str, default="./data",
        help="Root directory for CIFAR-10 dataset"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results",
        help="Directory to save results"
    )
    
    # Visualization
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting (useful for headless servers)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("WATERMARK ERASURE EXPERIMENT")
    print("Comparing dualized vs non-dualized gradient descent")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Learning rates: {args.num_lrs} points in [{args.lr_min}, {args.lr_max}]")
    print(f"  Training steps: {args.num_steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Optimizers: {args.optimizers}")
    print(f"  Device: {args.device}")
    print(f"  Seed: {args.seed}")
    print(f"  Output dir: {args.output_dir}")
    print(f"\nμP Settings:")
    print(f"  Use μP init: {args.use_mup_init}")
    print(f"  Use μP lr scaling: {args.use_mup_lr}")
    print(f"  Init σ: {args.init_sigma}")
    if args.use_mup_init:
        print(f"  Init formula: W = σ * √(d_out/d_in) / ||W'||₂ * W'")
    if args.use_mup_lr:
        print(f"  LR formula: lr_layer = lr_base * √(d_out/d_in)")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiments
    results, watermark_info = run_lr_sweep_experiment(
        optimizer_types=args.optimizers,
        lr_range=(args.lr_min, args.lr_max),
        num_lrs=args.num_lrs,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        data_root=args.data_root,
        save_dir=str(output_dir),
        use_mup_init=args.use_mup_init,
        use_mup_lr=args.use_mup_lr,
        init_sigma=args.init_sigma,
    )
    
    # Print summary
    print_summary(results)
    
    # Print watermark info
    print(f"\nWatermark configuration:")
    print(f"  Size: {watermark_info['letter_size']}×{watermark_info['letter_size']} (square)")
    print(f"  Region: {watermark_info['region_ratio']*100:.1f}% of matrix area")
    
    # Save comprehensive results
    save_experiment_summary(results, str(output_dir))
    
    # Plot results
    if not args.no_plot:
        try:
            plot_lr_sweep_results(
                results,
                save_path=str(output_dir / "lr_sweep_results.png"),
                show=False,
            )
            print(f"\nMain figure saved to: {output_dir / 'lr_sweep_results.png'}")
        except Exception as e:
            print(f"\nWarning: Could not create plot: {e}")
    
    # Save numerical results to file
    with open(output_dir / "numerical_results.txt", "w") as f:
        f.write("Watermark Erasure Experiment Results\n")
        f.write("=" * 50 + "\n\n")
        
        for optimizer, lr_results in results.items():
            f.write(f"\n{optimizer.upper()}\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'LR':>12} | {'Accuracy':>10} | {'Visibility':>10}\n")
            f.write("-" * 40 + "\n")
            
            for lr in sorted(lr_results.keys()):
                data = lr_results[lr]
                acc = data.get('accuracy', 0.0)
                vis = data.get('visibility', 1.0)
                f.write(f"{lr:>12.6f} | {acc:>10.4f} | {vis:>10.4f}\n")
    
    print(f"\nExperiment completed! Results saved to: {output_dir}")
    
    # Return results for programmatic use
    return results


if __name__ == "__main__":
    main()

