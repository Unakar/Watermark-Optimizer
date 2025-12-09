"""Simple MLP model for CIFAR-10 classification."""

import math
import torch
import torch.nn as nn


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
    """Simple MLP with one hidden layer for CIFAR-10.
    
    Architecture:
        Input (3072) -> Hidden (1024) -> ReLU -> Output (10)
    
    Supports two initialization modes:
        - "xavier": Standard Xavier/Glorot initialization
        - "spectral_mup": Spectral μP initialization (W = σ * √(d_out/d_in) / s * W')
    
    Attributes:
        fc1: First fully connected layer (input -> hidden)
        fc2: Second fully connected layer (hidden -> output)
        activation: Activation function (ReLU)
    """
    
    def __init__(
        self,
        input_dim: int = 3072,  # 32 * 32 * 3
        hidden_dim: int = 1024,
        output_dim: int = 10,
        activation: str = "relu",
        init_mode: str = "xavier",  # "xavier" or "spectral_mup"
        init_sigma: float = 0.02,   # σ for spectral_mup init
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.init_mode = init_mode
        self.init_sigma = init_sigma
        
        # Layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
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
        if self.init_mode == "xavier":
            # Xavier/Glorot initialization
            nn.init.xavier_normal_(self.fc1.weight)
            nn.init.xavier_normal_(self.fc2.weight)
        elif self.init_mode == "spectral_mup":
            # Spectral μP initialization: W = σ * √(d_out/d_in) / s * W'
            spectral_mup_init(self.fc1.weight, sigma=self.init_sigma)
            spectral_mup_init(self.fc2.weight, sigma=self.init_sigma)
        else:
            raise ValueError(f"Unknown init_mode: {self.init_mode}")
        
        # Biases always initialized to zero
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
    
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
        
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        
        return x
    
    def get_hidden_weight(self) -> torch.Tensor:
        """Get the hidden layer weight matrix for watermarking.
        
        Returns:
            Weight matrix of shape (hidden_dim, input_dim)
        """
        return self.fc1.weight.data
    
    def set_hidden_weight(self, weight: torch.Tensor):
        """Set the hidden layer weight matrix (for applying watermark).
        
        Args:
            weight: Weight matrix of shape (hidden_dim, input_dim)
        """
        self.fc1.weight.data = weight
    
    def get_mup_lr_scales(self) -> dict:
        """Get μP learning rate scale factors for each parameter.
        
        Returns:
            Dict mapping parameter names to their lr scale factors
        """
        scales = {}
        for name, param in self.named_parameters():
            if param.dim() == 2:
                d_out, d_in = param.shape
                scales[name] = math.sqrt(d_out / d_in)
            else:
                scales[name] = 1.0
        return scales
    
    def print_spectral_info(self):
        """Print spectral norm information for weight matrices."""
        print("\nSpectral Norm Information:")
        print("-" * 50)
        for name, param in self.named_parameters():
            if param.dim() == 2:
                d_out, d_in = param.shape
                spectral_norm = torch.linalg.matrix_norm(param.data, ord=2).item()
                target = self.init_sigma * math.sqrt(d_out / d_in) if self.init_mode == "spectral_mup" else "N/A"
                print(f"{name}: shape=({d_out}, {d_in}), ||W||₂={spectral_norm:.4f}, target={target}")
        print("-" * 50)

