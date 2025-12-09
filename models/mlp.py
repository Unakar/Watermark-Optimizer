"""MLP models for CIFAR-10 classification."""

import math
import torch
import torch.nn as nn
from typing import Optional


def spectral_mup_init(tensor: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Apply Spectral μP initialization: W = σ * √(d_out/d_in) / ||W'||₂ * W'
    
    Args:
        tensor: Weight tensor to initialize
        sigma: Standard deviation for initial distribution
    """
    if len(tensor.shape) != 2:
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    
    d_out, d_in = tensor.shape
    
    # Step 1: Initialize W' ~ N(0, σ)
    torch.nn.init.normal_(tensor, mean=0.0, std=sigma)
    
    # Step 2: Compute spectral norm s = ||W'||₂
    spectral_norm = torch.linalg.matrix_norm(tensor, ord=2)
    
    # Step 3: Apply μP scaling: W = σ * √(d_out/d_in) / s * W'
    target_spectral_norm = sigma * math.sqrt(d_out / d_in)
    mup_scale = target_spectral_norm / (spectral_norm + 1e-8)
    tensor.data.mul_(mup_scale)
    
    return tensor


class SimpleMLP(nn.Module):
    """3-layer MLP for CIFAR-10.
    
    Architecture:
        Input (3072) → fc1 → ReLU → fc2 → ReLU → fc3 → Output (10)
        
    水印打在 fc1 的权重上 (shape: hidden_dim × input_dim)
    
    Supports two initialization modes:
        - "xavier": Standard Xavier/Glorot initialization
        - "spectral_mup": Spectral μP initialization
    """
    
    def __init__(
        self,
        input_dim: int = 3072,      # 32 * 32 * 3 = 3072
        hidden_dim: int = 1024,      # 第一个隐藏层
        hidden_dim2: int = 1024,     # 第二个隐藏层
        output_dim: int = 10,
        activation: str = "relu",
        init_mode: str = "xavier",   # "xavier" or "spectral_mup"
        init_sigma: float = 0.02,    # σ for spectral_mup init
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.output_dim = output_dim
        self.init_mode = init_mode
        self.init_sigma = init_sigma
        
        # 3 层结构
        # fc1: 3072 → 1024  (水印打在这里)
        # fc2: 1024 → 1024
        # fc3: 1024 → 10
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights based on init_mode."""
        layers = [self.fc1, self.fc2, self.fc3]
        
        if self.init_mode == "xavier":
            for layer in layers:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
        elif self.init_mode == "spectral_mup":
            for layer in layers:
                spectral_mup_init(layer.weight, sigma=self.init_sigma)
                nn.init.zeros_(layer.bias)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, 3, 32, 32)
            
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # 3 层前向传播
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        
        return x
    
    def get_hidden_weight(self, layer_idx: int = 1) -> torch.Tensor:
        """Get weight matrix from specified layer for watermarking.
        
        Args:
            layer_idx: 1 for fc1, 2 for fc2, 3 for fc3
            
        Returns:
            Weight matrix
        """
        if layer_idx == 1:
            return self.fc1.weight.data  # shape: (1024, 3072)
        elif layer_idx == 2:
            return self.fc2.weight.data  # shape: (1024, 1024)
        elif layer_idx == 3:
            return self.fc3.weight.data  # shape: (10, 1024)
        else:
            raise ValueError(f"layer_idx must be 1, 2, or 3, got {layer_idx}")
    
    def set_hidden_weight(self, weight: torch.Tensor, layer_idx: int = 1):
        """Set weight matrix for specified layer.
        
        Args:
            weight: Weight matrix
            layer_idx: 1 for fc1, 2 for fc2, 3 for fc3
        """
        if layer_idx == 1:
            self.fc1.weight.data = weight
        elif layer_idx == 2:
            self.fc2.weight.data = weight
        elif layer_idx == 3:
            self.fc3.weight.data = weight
        else:
            raise ValueError(f"layer_idx must be 1, 2, or 3, got {layer_idx}")
    
    def get_watermark_layer_shape(self, layer_idx: int = 1) -> tuple:
        """Get the shape of specified layer's weight.
        
        Args:
            layer_idx: 1 for fc1, 2 for fc2, 3 for fc3
        """
        return self.get_hidden_weight(layer_idx).shape
    
    def get_mup_lr_scales(self) -> dict:
        """Get μP learning rate scale factors for each parameter."""
        scales = {}
        for name, param in self.named_parameters():
            if param.dim() == 2:
                d_out, d_in = param.shape
                scales[name] = math.sqrt(d_out / d_in)
            else:
                scales[name] = 1.0
        return scales
    
    def print_architecture(self):
        """Print model architecture with shapes."""
        print("\n" + "=" * 60)
        print("Model Architecture: 3-Layer MLP")
        print("=" * 60)
        print(f"""
Input ({self.input_dim}) ─────┐
                      │
                ┌─────▼─────┐
                │   fc1     │  ← 水印层！权重 shape: ({self.hidden_dim}, {self.input_dim})
                │ {self.input_dim}→{self.hidden_dim} │
                └─────┬─────┘
                      │ ReLU
                ┌─────▼─────┐
                │   fc2     │  权重 shape: ({self.hidden_dim2}, {self.hidden_dim})
                │ {self.hidden_dim}→{self.hidden_dim2} │
                └─────┬─────┘
                      │ ReLU
                ┌─────▼─────┐
                │   fc3     │  权重 shape: ({self.output_dim}, {self.hidden_dim2})
                │ {self.hidden_dim2}→{self.output_dim}  │
                └─────┬─────┘
                      │
Output ({self.output_dim}) ◄─────┘
        """)
        print("=" * 60)
    
    def print_spectral_info(self):
        """Print spectral norm information for weight matrices."""
        print("\nSpectral Norm Information:")
        print("-" * 60)
        for name, param in self.named_parameters():
            if param.dim() == 2:
                d_out, d_in = param.shape
                spectral_norm = torch.linalg.matrix_norm(param.data, ord=2).item()
                target = self.init_sigma * math.sqrt(d_out / d_in) if self.init_mode == "spectral_mup" else "N/A"
                print(f"{name}: shape=({d_out}, {d_in}), ||W||₂={spectral_norm:.4f}, target={target}")
        print("-" * 60)
