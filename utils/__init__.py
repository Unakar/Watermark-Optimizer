from .watermark import generate_letter_mask, apply_watermark
from .data import get_cifar10_dataloader
from .visualization import visualize_weight_matrix, plot_lr_sweep_results
from .mup import (
    get_mup_lr_scale_factor,
    spectral_mup_init,
    spectral_mup_init_method_normal,
    apply_mup_init_to_model,
    get_mup_param_groups,
)

__all__ = [
    "generate_letter_mask",
    "apply_watermark", 
    "get_cifar10_dataloader",
    "visualize_weight_matrix",
    "plot_lr_sweep_results",
    # μP utilities
    "get_mup_lr_scale_factor",
    "spectral_mup_init",
    "spectral_mup_init_method_normal",
    "apply_mup_init_to_model",
    "get_mup_param_groups",
]

