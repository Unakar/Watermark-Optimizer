"""Maximal Update Parameterization (μP) utilities.

Implements:
1. Spectral μP initialization: W = σ * √(d_out/d_in) / ||W'||₂ * W'
2. μP learning rate scaling: lr_layer = lr_base * √(d_out/d_in)

References:
- Spectral MuP: Spectral Control of Feature Learning
- Modular Duality in Deep Learning. arXiv:2410.21265 (2024)
"""

import math
from typing import Optional

import torch
import torch.nn as nn


def get_mup_lr_scale_factor(d_out: int, d_in: int, scale_mode: str = "spectral_mup") -> float:
    """Get the μP learning rate scale factor for a layer.
    
    For spectral μP, the learning rate should scale as √(d_out/d_in) to maintain
    consistent update sizes across different layer shapes.
    
    Args:
        d_out: Output dimension (rows of weight matrix)
        d_in: Input dimension (columns of weight matrix)
        scale_mode: Scaling mode
            - "spectral_mup": √(d_out / d_in) - recommended for spectral methods
            - "mup_standard": 1 / d_in - standard μP scaling
            - "none": 1.0 (no scaling)
            
    Returns:
        Scale factor to multiply with base learning rate.
    """
    if scale_mode == "spectral_mup":
        return math.sqrt(d_out / d_in)
    elif scale_mode == "mup_standard":
        return 1.0 / d_in
    elif scale_mode == "none":
        return 1.0
    else:
        raise ValueError(f"Invalid scale_mode: {scale_mode}")


def spectral_mup_init(tensor: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Apply Spectral μP initialization to a weight tensor.
    
    Spectral μP initialization: W = σ * √(d_out/d_in) / ||W'||₂ * W'
    
    For 2D weight matrices W ∈ R^(d_out × d_in):
    1. Initialize W' ~ N(0, σ)
    2. Compute spectral norm s = ||W'||₂
    3. Apply scaling: W = σ * √(d_out/d_in) / s * W'
    
    This ensures:
    - The spectral norm ||W||₂ = σ * √(d_out/d_in) after initialization
    - Feature learning rates are properly scaled across layers
    
    Args:
        tensor: Weight tensor to initialize (modified in-place)
        sigma: Standard deviation for initial normal distribution
        
    Returns:
        The initialized tensor (same as input, modified in-place)
    """
    # Skip non-2D parameters (bias, layernorm, etc.)
    if len(tensor.shape) != 2:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    
    d_out, d_in = tensor.shape
    
    # Step 1: Initialize W' ~ N(0, σ)
    torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    
    # Step 2: Compute spectral norm s = ||W'||₂
    spectral_norm = torch.linalg.matrix_norm(tensor, ord=2)
    
    # Step 3: Apply μP scaling: W = σ * √(d_out/d_in) / s * W'
    # Target spectral norm = σ * √(d_out/d_in)
    target_spectral_norm = sigma * math.sqrt(d_out / d_in)
    mup_scale = target_spectral_norm / (spectral_norm + 1e-8)
    tensor.data.mul_(mup_scale)
    
    return tensor


def spectral_mup_init_method_normal(sigma: float = 0.02):
    """Returns a Spectral μP initialization function.
    
    Usage:
        init_fn = spectral_mup_init_method_normal(sigma=0.02)
        init_fn(layer.weight)
    
    Args:
        sigma: Standard deviation for initial normal distribution
        
    Returns:
        Initialization function that can be applied to tensors
    """
    def init_(tensor: torch.Tensor) -> torch.Tensor:
        return spectral_mup_init(tensor, sigma=sigma)
    return init_


def apply_mup_init_to_model(
    model: nn.Module,
    sigma: float = 0.02,
    skip_names: Optional[list] = None,
):
    """Apply Spectral μP initialization to all 2D parameters in a model.
    
    Args:
        model: PyTorch model
        sigma: Standard deviation for initialization
        skip_names: List of parameter name patterns to skip (e.g., ["bias", "norm"])
    """
    if skip_names is None:
        skip_names = []
    
    init_fn = spectral_mup_init_method_normal(sigma=sigma)
    
    for name, param in model.named_parameters():
        # Skip if name matches any pattern
        skip = any(pattern in name for pattern in skip_names)
        if skip:
            continue
        
        if param.dim() == 2:
            init_fn(param.data)


def get_mup_param_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    scale_mode: str = "spectral_mup",
    no_decay_names: Optional[list] = None,
) -> list:
    """Create parameter groups with μP learning rate scaling.
    
    Each 2D parameter gets its own learning rate scaled by √(d_out/d_in).
    1D parameters (biases) use the base learning rate.
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate
        weight_decay: Weight decay coefficient
        scale_mode: μP scaling mode
        no_decay_names: Parameter name patterns that should not have weight decay
        
    Returns:
        List of parameter groups for optimizer
    """
    if no_decay_names is None:
        no_decay_names = ["bias", "norm", "layernorm"]
    
    param_groups = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine weight decay
        no_decay = any(nd in name.lower() for nd in no_decay_names)
        wd = 0.0 if no_decay else weight_decay
        
        # Determine learning rate scale
        if param.dim() == 2:
            d_out, d_in = param.shape
            lr_scale = get_mup_lr_scale_factor(d_out, d_in, scale_mode)
        else:
            lr_scale = 1.0
        
        param_groups.append({
            'params': [param],
            'lr': base_lr * lr_scale,
            'weight_decay': wd,
            'name': name,
            'lr_scale': lr_scale,
        })
    
    return param_groups


def print_mup_info(model: nn.Module, scale_mode: str = "spectral_mup"):
    """Print μP scaling information for a model.
    
    Args:
        model: PyTorch model
        scale_mode: μP scaling mode
    """
    print("\n" + "=" * 60)
    print("μP (Maximal Update Parameterization) Configuration")
    print("=" * 60)
    print(f"Scale mode: {scale_mode}")
    print("\nParameter scaling:")
    print(f"{'Name':<40} {'Shape':<20} {'LR Scale':<10}")
    print("-" * 70)
    
    for name, param in model.named_parameters():
        if param.dim() == 2:
            d_out, d_in = param.shape
            lr_scale = get_mup_lr_scale_factor(d_out, d_in, scale_mode)
            shape_str = f"({d_out}, {d_in})"
        else:
            lr_scale = 1.0
            shape_str = str(tuple(param.shape))
        
        print(f"{name:<40} {shape_str:<20} {lr_scale:<10.4f}")
    
    print("=" * 60 + "\n")

