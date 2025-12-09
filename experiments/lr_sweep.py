"""Learning rate sweep experiment for watermark erasure."""

import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
from pathlib import Path
import copy

from models import SimpleMLP
from optimizers import SpectralBall, MuonBall
from utils.watermark import create_watermark_setup, apply_watermark, compute_watermark_visibility
from utils.data import get_cifar10_dataloader
from utils.visualization import visualize_weight_matrix


def get_mup_lr_scale(d_out: int, d_in: int, mode: str = "spectral_mup") -> float:
    """Get μP learning rate scale factor.
    
    Args:
        d_out: Output dimension
        d_in: Input dimension
        mode: "spectral_mup" -> √(d_out/d_in), "none" -> 1.0
    """
    if mode == "spectral_mup":
        return math.sqrt(d_out / d_in)
    return 1.0


def create_optimizer(
    model: nn.Module,
    optimizer_type: str,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 0.01,
    use_mup_lr: bool = False,
) -> torch.optim.Optimizer:
    """Create optimizer based on type.
    
    Args:
        model: The model to optimize
        optimizer_type: "adamw" or "spectral_ball"
        lr: Learning rate (base lr if use_mup_lr=True)
        momentum: Momentum coefficient
        weight_decay: Weight decay coefficient
        use_mup_lr: Whether to apply μP learning rate scaling
        
    Returns:
        Optimizer instance
    """
    if use_mup_lr:
        # Create parameter groups with μP lr scaling
        param_groups = []
        for name, param in model.named_parameters():
            if param.dim() == 2:
                d_out, d_in = param.shape
                lr_scale = get_mup_lr_scale(d_out, d_in, mode="spectral_mup")
            else:
                lr_scale = 1.0
            
            param_groups.append({
                'params': [param],
                'lr': lr * lr_scale,
                'weight_decay': weight_decay if param.dim() == 2 else 0.0,
            })
    else:
        param_groups = model.parameters()
    
    if optimizer_type == "adamw":
        if use_mup_lr:
            return torch.optim.AdamW(
                param_groups,
                betas=(momentum, 0.999),
            )
        else:
            return torch.optim.AdamW(
                param_groups,
                lr=lr,
                betas=(momentum, 0.999),
                weight_decay=weight_decay,
            )
    elif optimizer_type == "spectral_ball":
        if use_mup_lr:
            return SpectralBall(
                param_groups,
                momentum_beta=momentum,
                power_iteration_steps=10,
                msign_steps=5,
                radius_mode="spectral_mup",
                scale_mode="spectral_mup",  # Use spectral_mup scale for consistency
            )
        else:
            return SpectralBall(
                param_groups,
                lr=lr,
                momentum_beta=momentum,
                weight_decay=weight_decay,
                power_iteration_steps=10,
                msign_steps=5,
                radius_mode="spectral_mup",
                scale_mode="align_adamw_rms",
            )
    elif optimizer_type == "muon_ball":
        # MuonBall: Spectral Ball with λ=0 (faster, no bisection solver)
        if use_mup_lr:
            return MuonBall(
                param_groups,
                momentum_beta=momentum,
                power_iteration_steps=10,
                msign_steps=5,
                radius_mode="spectral_mup",
                scale_mode="spectral_mup",
            )
        else:
            return MuonBall(
                param_groups,
                lr=lr,
                momentum_beta=momentum,
                weight_decay=weight_decay,
                power_iteration_steps=10,
                msign_steps=5,
                radius_mode="spectral_mup",
                scale_mode="align_adamw_rms",
            )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def train_one_experiment(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_steps: int = 1000,
    device: str = "cuda",
) -> Tuple[float, List[float]]:
    """Train the model for a fixed number of steps.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer to use
        num_steps: Number of training steps
        device: Device to train on
        
    Returns:
        Tuple of (final_accuracy, accuracy_history)
    """
    model = model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    
    # Create infinite iterator
    data_iter = iter(dataloader)
    
    correct = 0
    total = 0
    accuracy_history = []
    
    pbar = tqdm(range(num_steps), desc="Training", leave=False)
    
    for step in pbar:
        # Get batch (with cycling)
        try:
            inputs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            inputs, labels = next(data_iter)
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Check for NaN/Inf loss (training diverged)
        if torch.isnan(loss) or torch.isinf(loss):
            return 0.0, accuracy_history
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track accuracy
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Record accuracy every 100 steps
        if (step + 1) % 100 == 0:
            acc = correct / total
            accuracy_history.append(acc)
            pbar.set_postfix({'acc': f'{acc:.4f}', 'loss': f'{loss.item():.4f}'})
    
    final_accuracy = correct / total if total > 0 else 0.0
    return final_accuracy, accuracy_history


def run_single_experiment(
    optimizer_type: str,
    lr: float,
    watermark_mask: np.ndarray,
    watermark_value: float,
    num_steps: int = 1000,
    batch_size: int = 128,
    device: str = "cuda",
    seed: int = 42,
    data_root: str = "./data",
    use_mup_init: bool = False,
    use_mup_lr: bool = False,
    init_sigma: float = 0.02,
) -> Dict:
    """Run a single experiment with given configuration.
    
    Args:
        optimizer_type: "adamw" or "spectral_ball"
        lr: Learning rate (base lr if use_mup_lr=True)
        watermark_mask: Binary mask for watermark
        watermark_value: Value to set for watermark
        num_steps: Number of training steps
        batch_size: Batch size
        device: Device to use
        seed: Random seed
        data_root: Data directory
        use_mup_init: Whether to use Spectral μP initialization
        use_mup_lr: Whether to use μP learning rate scaling
        init_sigma: σ parameter for μP initialization
        
    Returns:
        Dict with 'accuracy', 'weight_image', 'initial_weight', 'final_weight'
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Create model with appropriate initialization
    model = SimpleMLP(
        input_dim=3072,
        hidden_dim=1024,
        output_dim=10,
        activation="relu",
        init_mode="spectral_mup" if use_mup_init else "xavier",
        init_sigma=init_sigma,
    )
    
    # Apply watermark
    initial_weight = model.get_hidden_weight().clone()
    apply_watermark(initial_weight, watermark_mask, watermark_value)
    model.set_hidden_weight(initial_weight)
    
    # Store initial weight for comparison
    initial_weight_copy = initial_weight.clone()
    
    # Create data loader
    dataloader = get_cifar10_dataloader(
        batch_size=batch_size,
        num_workers=4,
        data_root=data_root,
        train=True,
    )
    
    # Create optimizer
    optimizer = create_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        lr=lr,
        momentum=0.9,
        weight_decay=0.01,
        use_mup_lr=use_mup_lr,
    )
    
    # Train
    final_accuracy, accuracy_history = train_one_experiment(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        num_steps=num_steps,
        device=device,
    )
    
    # Get final weight
    final_weight = model.get_hidden_weight().cpu()
    
    # Compute watermark visibility
    visibility = compute_watermark_visibility(
        final_weight,
        watermark_mask,
        watermark_value,
    )
    
    return {
        'accuracy': final_accuracy,
        'weight_image': final_weight.numpy(),
        'initial_weight': initial_weight_copy.numpy(),
        'final_weight': final_weight.numpy(),
        'visibility': visibility,
        'accuracy_history': accuracy_history,
    }


def run_lr_sweep_experiment(
    optimizer_types: List[str] = ["adamw", "spectral_ball"],
    lr_range: Tuple[float, float] = (0.1, 10.0),
    num_lrs: int = 10,
    num_steps: int = 1000,
    batch_size: int = 128,
    device: str = "cuda",
    seed: int = 42,
    data_root: str = "./data",
    save_dir: Optional[str] = None,
    use_mup_init: bool = False,
    use_mup_lr: bool = False,
    init_sigma: float = 0.02,
) -> Dict[str, Dict[float, Dict]]:
    """Run learning rate sweep experiment.
    
    Args:
        optimizer_types: List of optimizer types to compare
        lr_range: (min_lr, max_lr) in log scale
        num_lrs: Number of learning rates to try
        num_steps: Training steps per experiment
        batch_size: Batch size
        device: Device to use
        seed: Random seed
        data_root: Data directory
        save_dir: Directory to save results
        use_mup_init: Whether to use Spectral μP initialization
        use_mup_lr: Whether to use μP learning rate scaling
        init_sigma: σ parameter for μP initialization
        
    Returns:
        Dict with structure {optimizer: {lr: {results}}}
    """
    # Generate learning rates (log scale)
    lrs = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), num_lrs)
    
    # Create watermark
    watermark_mask, watermark_value = create_watermark_setup(
        weight_shape=(1024, 3072),
        letter="a",
    )
    
    print(f"Watermark created: {watermark_mask.sum()} pixels marked")
    print(f"Learning rates: {lrs}")
    print(f"Optimizers: {optimizer_types}")
    print(f"Training steps: {num_steps}")
    print(f"μP init: {use_mup_init} (σ={init_sigma})")
    print(f"μP lr scaling: {use_mup_lr}")
    print("=" * 60)
    
    results = {}
    
    for optimizer_type in optimizer_types:
        print(f"\n{'='*60}")
        print(f"Running experiments with {optimizer_type.upper()}")
        print(f"{'='*60}")
        
        results[optimizer_type] = {}
        
        for i, lr in enumerate(lrs):
            print(f"\n[{i+1}/{num_lrs}] LR = {lr:.6f}")
            
            try:
                exp_result = run_single_experiment(
                    optimizer_type=optimizer_type,
                    lr=lr,
                    watermark_mask=watermark_mask,
                    watermark_value=watermark_value,
                    num_steps=num_steps,
                    batch_size=batch_size,
                    device=device,
                    seed=seed,
                    data_root=data_root,
                    use_mup_init=use_mup_init,
                    use_mup_lr=use_mup_lr,
                    init_sigma=init_sigma,
                )
                
                results[optimizer_type][lr] = exp_result
                
                print(f"  → Accuracy: {exp_result['accuracy']:.4f}")
                print(f"  → Watermark visibility: {exp_result['visibility']:.4f}")
                
            except RuntimeError as e:
                print(f"  → FAILED: {e}")
                results[optimizer_type][lr] = {
                    'accuracy': 0.0,
                    'weight_image': None,
                    'visibility': 1.0,
                    'error': str(e),
                }
    
    # Save results if directory provided
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save initial watermarked weight
        torch.manual_seed(seed)
        model = SimpleMLP(
            init_mode="spectral_mup" if use_mup_init else "xavier",
            init_sigma=init_sigma,
        )
        initial_weight = model.get_hidden_weight().clone()
        apply_watermark(initial_weight, watermark_mask, watermark_value)
        
        visualize_weight_matrix(
            initial_weight,
            title="Initial Weight (with watermark 'a')",
            save_path=str(save_dir / "initial_weight.png"),
        )
        
        print(f"\nResults saved to {save_dir}")
    
    return results


def print_summary(results: Dict[str, Dict[float, Dict]]):
    """Print a summary of experiment results."""
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for optimizer_type, lr_results in results.items():
        print(f"\n{optimizer_type.upper()}")
        print("-" * 50)
        print(f"{'LR':>12} | {'Accuracy':>10} | {'Visibility':>10}")
        print("-" * 50)
        
        best_acc = 0.0
        best_lr = 0.0
        
        for lr in sorted(lr_results.keys()):
            data = lr_results[lr]
            acc = data.get('accuracy', 0.0)
            vis = data.get('visibility', 1.0)
            
            print(f"{lr:>12.6f} | {acc:>10.4f} | {vis:>10.4f}")
            
            if acc > best_acc:
                best_acc = acc
                best_lr = lr
        
        print("-" * 50)
        print(f"Best: LR={best_lr:.6f}, Accuracy={best_acc:.4f}")
    
    print("\n" + "=" * 70)

