"""Visualization utilities for the watermark experiment."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from typing import Dict, List, Optional, Tuple
from pathlib import Path


def visualize_watermark_region(
    weight: torch.Tensor,
    watermark_info: dict,
    title: str = "Watermark Region",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8),
    cmap: str = "RdBu_r",
    show: bool = False,
) -> np.ndarray:
    """Visualize ONLY the watermark region (cropped to square).
    
    Args:
        weight: Full weight tensor
        watermark_info: Info dict from create_watermark_setup
        title: Title for the plot
        save_path: Path to save the figure
        figsize: Figure size (square recommended)
        cmap: Colormap
        show: Whether to display
        
    Returns:
        Cropped weight region as numpy array
    """
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        w = weight.detach().cpu().numpy()
    else:
        w = weight
    
    # Extract watermark region
    r_start, c_start = watermark_info['position']
    r_end, c_end = watermark_info['region_end']
    w_region = w[r_start:r_end, c_start:c_end]
    
    if save_path or show:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Center colormap at 0 for watermark visibility
        abs_max = max(abs(w_region.min()), abs(w_region.max()))
        if abs_max < 1e-8:
            abs_max = 1.0
        
        im = ax.imshow(w_region, cmap=cmap, aspect='equal', vmin=-abs_max, vmax=abs_max)
        ax.set_title(f"{title}\n(Size: {w_region.shape[0]}×{w_region.shape[1]}, Watermark='a' at value=0 → white)", fontsize=12)
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Weight value")
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    return w_region


def visualize_watermark_comparison(
    initial_weight: torch.Tensor,
    final_weight: torch.Tensor,
    watermark_info: dict,
    title: str = "Watermark Region: Before vs After Training",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
    cmap: str = "RdBu_r",
    show: bool = False,
):
    """Compare watermark region before and after training.
    
    Args:
        initial_weight: Weight tensor before training
        final_weight: Weight tensor after training
        watermark_info: Info dict from create_watermark_setup
        title: Title for the plot
        save_path: Path to save
        figsize: Figure size
        cmap: Colormap
        show: Whether to display
    """
    # Convert to numpy
    if isinstance(initial_weight, torch.Tensor):
        w_init = initial_weight.detach().cpu().numpy()
    else:
        w_init = initial_weight
    
    if isinstance(final_weight, torch.Tensor):
        w_final = final_weight.detach().cpu().numpy()
    else:
        w_final = final_weight
    
    # Extract regions
    r_start, c_start = watermark_info['position']
    r_end, c_end = watermark_info['region_end']
    
    w_init_region = w_init[r_start:r_end, c_start:c_end]
    w_final_region = w_final[r_start:r_end, c_start:c_end]
    
    # Find common color scale
    all_vals = np.concatenate([w_init_region.flatten(), w_final_region.flatten()])
    abs_max = max(abs(all_vals.min()), abs(all_vals.max()))
    if abs_max < 1e-8:
        abs_max = 1.0
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Initial (with watermark)
    im1 = axes[0].imshow(w_init_region, cmap=cmap, aspect='equal', vmin=-abs_max, vmax=abs_max)
    axes[0].set_title("Before Training\n(Watermark 'a' visible as white)", fontsize=11)
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Final (after training)
    im2 = axes[1].imshow(w_final_region, cmap=cmap, aspect='equal', vmin=-abs_max, vmax=abs_max)
    axes[1].set_title("After Training\n(Watermark erased if colorful)", fontsize=11)
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def visualize_weight_matrix(
    weight: torch.Tensor,
    title: str = "Weight Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    cmap: str = "RdBu_r",  # Red-Blue colormap makes 0 (watermark) clearly visible
    show: bool = False,
    symmetric_clim: bool = True,  # Center colormap at 0 to show watermark clearly
) -> np.ndarray:
    """Visualize a weight matrix as a colored image.
    
    Uses a diverging colormap (RdBu) centered at 0 to make watermarks clearly visible.
    
    Args:
        weight: Weight tensor of shape (out_features, in_features)
        title: Title for the plot
        save_path: Path to save the figure (optional)
        figsize: Figure size
        cmap: Colormap to use (RdBu_r recommended for watermark visibility)
        show: Whether to display the plot
        symmetric_clim: If True, center colormap at 0
        
    Returns:
        Weight matrix as numpy array (normalized to [0, 1])
    """
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        w = weight.detach().cpu().numpy()
    else:
        w = weight
    
    # For return value: normalize to [0, 1]
    w_min, w_max = w.min(), w.max()
    if w_max - w_min > 1e-8:
        w_normalized = (w - w_min) / (w_max - w_min)
    else:
        w_normalized = np.zeros_like(w)
    
    if save_path or show:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use symmetric color limits centered at 0 for better watermark visibility
        if symmetric_clim:
            abs_max = max(abs(w.min()), abs(w.max()))
            vmin, vmax = -abs_max, abs_max
        else:
            vmin, vmax = w.min(), w.max()
        
        im = ax.imshow(w, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Input dimension", fontsize=12)
        ax.set_ylabel("Output dimension", fontsize=12)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Weight value", fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
    
    return w_normalized


def create_weight_thumbnail(
    weight: torch.Tensor,
    size: Tuple[int, int] = (120, 40),  # Larger thumbnails for better visibility
    use_diverging: bool = True,  # Use diverging colormap to show watermark
) -> np.ndarray:
    """Create a thumbnail of the weight matrix with watermark visibility.
    
    Args:
        weight: Weight tensor
        size: Target thumbnail size (width, height)
        use_diverging: If True, use diverging colormap centered at 0
        
    Returns:
        Thumbnail as RGB numpy array
    """
    from PIL import Image
    import matplotlib.cm as cm
    
    # Convert to numpy
    if isinstance(weight, torch.Tensor):
        w = weight.detach().cpu().numpy()
    else:
        w = weight.copy()
    
    if use_diverging:
        # Center at 0 to show watermark (which is 0) as white/neutral
        abs_max = max(abs(w.min()), abs(w.max()))
        if abs_max > 1e-8:
            w_normalized = (w + abs_max) / (2 * abs_max)  # Map to [0, 1] with 0 at 0.5
        else:
            w_normalized = np.full_like(w, 0.5)
        
        # Apply RdBu colormap
        cmap = cm.get_cmap('RdBu_r')
        w_colored = cmap(w_normalized)[:, :, :3]  # RGB only
        w_uint8 = (w_colored * 255).astype(np.uint8)
    else:
        # Simple grayscale normalization
        w_min, w_max = w.min(), w.max()
        if w_max - w_min > 1e-8:
            w_normalized = (w - w_min) / (w_max - w_min)
        else:
            w_normalized = np.zeros_like(w)
        w_uint8 = (w_normalized * 255).astype(np.uint8)
    
    # Resize using PIL
    if use_diverging:
        img = Image.fromarray(w_uint8, mode='RGB')
    else:
        img = Image.fromarray(w_uint8, mode='L')
    img_resized = img.resize(size, Image.Resampling.LANCZOS)
    
    return np.array(img_resized)


def plot_lr_sweep_results(
    results: Dict[str, Dict[float, Dict]],
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (16, 10),  # Larger figure for thumbnails
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
        'adamw': '#FF6B35',          # Orange
        'spectral_ball': '#004E89',  # Blue
        'muon_ball': '#2E8B57',      # Sea Green
        'muon': '#8B008B',           # Dark Magenta
    }
    labels = {
        'adamw': 'Without dualization (AdamW)',
        'spectral_ball': 'With dualization (SpectralBall)',
        'muon_ball': 'With dualization λ=0 (MuonBall)',
        'muon': 'Orthogonalization only (Muon)',
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
            if 'weight_image' in lr_results[lr] and lr_results[lr]['weight_image'] is not None:
                weight_img = lr_results[lr]['weight_image']
                
                # Create larger thumbnail for better visibility
                thumbnail = create_weight_thumbnail(
                    torch.from_numpy(weight_img) if isinstance(weight_img, np.ndarray) else weight_img,
                    size=(100, 35),  # Larger size for watermark visibility
                    use_diverging=True,  # Use colormap to show watermark
                )
                
                # Calculate offset for thumbnail placement (spread out based on optimizer index)
                opt_idx = list(results.keys()).index(optimizer_name)
                y_offset = 0.12 * (opt_idx - len(results) / 2 + 0.5)
                
                # For RGB thumbnail, don't use cmap
                imagebox = OffsetImage(thumbnail, zoom=0.6)
                ab = AnnotationBbox(
                    imagebox,
                    (lr, accuracies[i] + y_offset),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(edgecolor=color, linewidth=2)
                )
                ax.add_artist(ab)
    
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate', fontsize=14)
    ax.set_ylabel('Training Accuracy', fontsize=14)
    ax.set_title('Learning Rate Sweep: Watermark Erasure Experiment\n(Thumbnails show weight matrices - watermark "a" should be visible as white region)', fontsize=14)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.15, 1.25)  # More room for thumbnails
    
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

