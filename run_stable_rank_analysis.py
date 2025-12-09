#!/usr/bin/env python3
"""Stable Rank Analysis Script.

This script analyzes the stable rank of raw gradients vs dualized gradients (msign)
at different network widths, replicating the analysis from the weight-erasure.ipynb.

理论预测：
- 普通梯度的 stable rank 很小（~1-10），基本与 width 无关
- Dualized 梯度的 stable rank 随 width 增长，直到 = batch_size 后饱和

这解释了为什么 dualized 优化器（如 Muon, MuonBall, SpectralBall）会擦除水印：
它们的更新在每个元素上的变化更大（因为 stable rank 更高）。

Usage:
    python run_stable_rank_analysis.py
    python run_stable_rank_analysis.py --batch-size 1024
    python run_stable_rank_analysis.py --widths 64 128 256 512 1024 2048 4096
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.lr_sweep import analyze_stable_rank


def main():
    parser = argparse.ArgumentParser(
        description="Analyze stable rank of gradients vs dualized gradients",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--widths", type=int, nargs="+",
        default=[32, 64, 128, 256, 512, 1024, 2048],
        help="Hidden layer widths to test"
    )
    parser.add_argument(
        "--batch-size", type=int, default=512,
        help="Batch size (determines saturation point for dualized gradient stable rank)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--data-root", type=str, default="./data",
        help="Data directory"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./results/stable_rank_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("STABLE RANK ANALYSIS")
    print("=" * 70)
    print(f"\n理论背景：")
    print(f"  Stable Rank = ||M||_F² / ||M||_*²")
    print(f"  - 衡量奇异值的"分散程度"")
    print(f"  - 低 stable rank (~1): 矩阵近似 rank-1，由顶部奇异值主导")
    print(f"  - 高 stable rank (~rank): 奇异值均匀分布")
    print(f"\n预期结果：")
    print(f"  - 普通梯度: stable rank ≈ 1-10 (与 width 无关)")
    print(f"  - Dualized 梯度: stable rank ↑ 随 width，饱和于 batch_size={args.batch_size}")
    print(f"\n这解释了为什么 dualization 会擦除水印：")
    print(f"  每个权重元素的平均更新量 ∝ √(stable_rank / (m×n))")
    print(f"  dualized 梯度的 stable rank 更高 → 每个元素变化更大 → 水印被擦除")
    print("=" * 70 + "\n")
    
    # Run analysis
    results = analyze_stable_rank(
        widths=args.widths,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
        data_root=args.data_root,
        save_path=str(output_dir / "stable_rank_analysis.png"),
    )
    
    # Save numerical results
    with open(output_dir / "stable_rank_results.txt", "w") as f:
        f.write("Stable Rank Analysis Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Batch size: {results['batch_size']}\n\n")
        f.write(f"{'Width':>10} | {'Raw Grad':>12} | {'Dualized':>12}\n")
        f.write("-" * 40 + "\n")
        
        for w, sg, sd in zip(results['widths'], 
                            results['stable_rank_grads'], 
                            results['stable_rank_dualized_grads']):
            f.write(f"{w:>10} | {sg:>12.2f} | {sd:>12.2f}\n")
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()

