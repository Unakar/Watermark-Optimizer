"""Muon Optimizer: MomentUm Orthogonalized by Newton-schulz.

Muon runs standard SGD-momentum with Nesterov momentum, and then performs an
orthogonalization post-processing step using Newton-Schulz iteration.

Unlike MuonBall/SpectralBall, Muon does NOT have spectral norm constraints
(no retraction). The weights can drift freely.

Algorithm:
1. Compute momentum: M ← β*M + grad
2. Orthogonalize: Φ = msign(M)   # Newton-Schulz orthogonalization
3. Update: W ← W - lr * scale * Φ

References:
- Jordan, K. *Muon Optimizer Implementation.*
  [GitHub](https://github.com/KellerJordan/Muon/blob/master/muon.py)
- *Modular Duality in Deep Learning.* arXiv:2410.21265 (2024).
"""

import math
from typing import Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# Import shared utilities from spectral_ball
from .spectral_ball import msign, get_scale_factor


class Muon(Optimizer):
    """Muon Optimizer: MomentUm Orthogonalized by Newton-schulz.

    Muon performs orthogonalization on momentum using Newton-Schulz iteration,
    without any spectral norm constraints.

    The algorithm:
    1. Compute momentum: M ← β*M + grad
    2. Orthogonalize: Φ = msign(M)
    3. Update: W ← W - lr * scale * Φ

    Key differences from other optimizers:
    - vs AdamW: Uses orthogonalization instead of adaptive learning rates
    - vs MuonBall: No spectral constraint (no retraction)
    - vs SpectralBall: No spectral constraint, no λ solver

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum_beta: Momentum coefficient (default: 0.95)
        weight_decay: Weight decay coefficient (default: 0.01)
        msign_steps: Newton-Schulz iterations for msign (default: 5)
        scale_mode: Scale factor mode for updates
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum_beta: float = 0.95,
        weight_decay: float = 0.01,
        msign_steps: int = 5,
        scale_mode: str = "align_adamw_rms",
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum_beta < 0.0 or momentum_beta >= 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")
        if msign_steps < 1:
            raise ValueError(f"msign_steps must be at least 1, got {msign_steps}")

        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.msign_steps = msign_steps
        self.scale_mode = scale_mode

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum_beta = group['momentum_beta']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                # Handle different parameter dimensions
                if p.dim() == 1:
                    # 1D parameters (biases): use simple SGD with momentum
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad)

                    # Weight decay (decoupled)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    p.add_(buf, alpha=-lr)

                elif p.dim() == 2:
                    # 2D parameters: use Muon update
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    # Update momentum buffer
                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad)

                    # Compute Muon update (just orthogonalization, no retraction)
                    update = self._compute_muon_update(buf)

                    # Weight decay (decoupled)
                    if weight_decay != 0:
                        p.mul_(1 - lr * weight_decay)

                    # Apply update
                    p.add_(update, alpha=-lr)

                else:
                    # >2D parameters: reshape to 2D, apply, reshape back
                    original_shape = p.shape
                    p_2d = p.view(p.shape[0], -1)
                    grad_2d = grad.view(grad.shape[0], -1)

                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p_2d)

                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad_2d)

                    update = self._compute_muon_update(buf)

                    if weight_decay != 0:
                        p_2d.mul_(1 - lr * weight_decay)

                    p_2d.add_(update, alpha=-lr)
                    p.data = p_2d.view(original_shape)

        return loss

    def _compute_muon_update(self, M: torch.Tensor) -> torch.Tensor:
        """Compute Muon update direction.

        Simply orthogonalizes the momentum using msign (via exact SVD).
        No spectral constraints (no retraction).

        Args:
            M: Momentum tensor

        Returns:
            Update direction Φ = msign(M) * scale = U @ V^T * scale
        """
        # Convert M to fp32 and normalize
        M_fp32 = M.to(torch.float32)
        M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2, -1), keepdim=True).clamp_min(1e-8))

        # Orthogonalize: Φ = msign(M) = U @ V^T via exact SVD
        Phi = msign(M_fp32)  # Uses SVD internally

        # Apply scale factor
        scale_factor = get_scale_factor(M.shape[0], M.shape[1], mode=self.scale_mode)

        return Phi * scale_factor

