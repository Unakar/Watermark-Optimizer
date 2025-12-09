"""Visualization utilities for the watermark experiment."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def visualize_weight_matrix(
    weight: torch.Tensor,
    title: str = "Weight Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = "gray",
    show: bool = False,
) -> np.ndarray:
    """Visualize a weight matrix as a grayscale image.
    
    Args:
        weight: Weight tensor of shape (out_features, in_features)
        title: Title for the plot
        save_path: Path to save the figure (optional)
        figsize: Figure size
        cmap: Colormap to use
        show: Whether to display the plot
        
    Returns:
        Weight matrix as numpy array (normalized to [0, 1])
    """
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        w = weight.detach().cpu().numpy()
    else:
        w = weight
    
    # Normalize to [0, 1] for visualization
    w_min, w_max = w.min(), w.max()
    if w_max - w_min > 1e-8:
        w_normalized = (w - w_min) / (w_max - w_min)
    else:
        w_normalized = np.zeros_like(w)
    
    if save_path or show:
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(w_normalized, cmap=cmap, aspect='auto')
        ax.set_title(title)
        ax.set_xlabel("Input dimension")
        ax.set_ylabel("Output dimension")
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    return w_normalized


def create_weight_thumbnail(
    weight: torch.Tensor,
    size: Tuple[int, int] = (60, 20),
) -> np.ndarray:
    """Create a small thumbnail of the weight matrix.
    
    Args:
        weight: Weight tensor
        size: Target thumbnail size (width, height)
        
    Returns:
        Thumbnail as numpy array
    """
    from PIL import Image
    
    # Get normalized weight
    w = visualize_weight_matrix(weight, show=False)
    
    # Convert to 8-bit grayscale
    w_uint8 = (w * 255).astype(np.uint8)
    
    # Resize using PIL
    img = Image.fromarray(w_uint8, mode='L')
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    return np.array(img_resized)


def plot_lr_sweep_results(
    results: Dict[str, Dict[float, Dict]],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (14, 8),
):
    """Plot learning rate sweep results with weight matrix thumbnails.
    
    Args:
        results: Dict with structure {optimizer_name: {lr: {'accuracy': float, 'weight_image': np.ndarray}}}
        save_path: Path to save the figure
        show: Whether to display the plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = {
        'adamw': '#FF6B35',       # Orange
        'spectral_ball': '#004E89',  # Blue
    }
    labels = {
        'adamw': 'Without dualization (AdamW)',
        'spectral_ball': 'With dualization (SpectralBall)',
    }
    
    for optimizer_name, lr_results in results.items():
        lrs = sorted(lr_results.keys())
        accuracies = [lr_results[lr]['accuracy'] for lr in lrs]
        
        color = colors.get(optimizer_name, 'gray')
        label = labels.get(optimizer_name, optimizer_name)
        
        # Plot the accuracy curve
        ax.plot(lrs, accuracies, 'o-', color=color, label=label, linewidth=2, markersize=8)
        
        # Add weight matrix thumbnails
        for i, lr in enumerate(lrs):
            if 'weight_image' in lr_results[lr]:
                weight_img = lr_results[lr]['weight_image']
                
                # Create thumbnail
                thumbnail = create_weight_thumbnail(
                    torch.from_numpy(weight_img) if isinstance(weight_img, np.ndarray) else weight_img,
                    size=(50, 25)
                )
                
                # Calculate offset for thumbnail placement
                y_offset = 0.08 if optimizer_name == 'spectral_ball' else -0.12
                
                imagebox = OffsetImage(thumbnail, zoom=0.8, cmap='gray')
                ab = AnnotationBbox(
                    imagebox,
                    (lr, accuracies[i] + y_offset),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(edgecolor=color, linewidth=1)
                )
                ax.add_artist(ab)
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Training Accuracy', fontsize=12)
    ax.set_title('Learning Rate Sweep: Watermark Erasure Experiment', fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_weight_comparison(
    initial_weight: torch.Tensor,
    final_weights: Dict[str, torch.Tensor],
    save_path: Optional[str] = None,
    show: bool = True,
):
    """Plot comparison of initial and final weights.
    
    Args:
        initial_weight: Initial weight matrix (with watermark)
        final_weights: Dict of {label: weight_tensor}
        save_path: Path to save the figure
        show: Whether to display
    """
    n_cols = 1 + len(final_weights)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3))
    
    if n_cols == 1:
        axes = [axes]
    
    # Plot initial
    w_init = initial_weight.detach().cpu().numpy()
    w_min, w_max = w_init.min(), w_init.max()
    w_init_norm = (w_init - w_min) / (w_max - w_min + 1e-8)
    
    axes[0].imshow(w_init_norm, cmap='gray', aspect='auto')
    axes[0].set_title('Initial (with watermark)')
    axes[0].axis('off')
    
    # Plot finals
    for i, (label, weight) in enumerate(final_weights.items(), 1):
        w = weight.detach().cpu().numpy()
        w_norm = (w - w_min) / (w_max - w_min + 1e-8)  # Use same scale
        
        axes[i].imshow(w_norm, cmap='gray', aspect='auto')
        axes[i].set_title(f'After training ({label})')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def save_experiment_summary(
    results: Dict[str, Dict[float, Dict]],
    save_dir: str,
):
    """Save a comprehensive summary of the experiment.
    
    Args:
        results: Experiment results
        save_dir: Directory to save outputs
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main figure
    plot_lr_sweep_results(
        results,
        save_path=str(save_dir / "lr_sweep_results.png"),
        show=False,
    )
    
    # Save individual weight matrices
    for optimizer_name, lr_results in results.items():
        opt_dir = save_dir / optimizer_name
        opt_dir.mkdir(exist_ok=True)
        
        for lr, data in lr_results.items():
            if 'weight_image' in data:
                weight_img = data['weight_image']
                if isinstance(weight_img, torch.Tensor):
                    weight_img = weight_img.numpy()
                
                # Save as image
                plt.figure(figsize=(10, 4))
                plt.imshow(weight_img, cmap='gray', aspect='auto')
                plt.title(f'{optimizer_name} - LR={lr:.4f} - Acc={data["accuracy"]:.4f}')
                plt.colorbar(shrink=0.8)
                plt.savefig(opt_dir / f"weight_lr_{lr:.4f}.png", dpi=100, bbox_inches='tight')
                plt.close()
    
    # Save numerical results
    with open(save_dir / "results.txt", 'w') as f:
        f.write("Learning Rate Sweep Results\n")
        f.write("=" * 50 + "\n\n")
        
        for optimizer_name, lr_results in results.items():
            f.write(f"\n{optimizer_name.upper()}\n")
            f.write("-" * 30 + "\n")
            for lr in sorted(lr_results.keys()):
                acc = lr_results[lr]['accuracy']
                f.write(f"LR={lr:.6f}: Accuracy={acc:.4f}\n")
    
    print(f"Experiment summary saved to {save_dir}")

