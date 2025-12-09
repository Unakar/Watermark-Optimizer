"""MuonBall Optimizer: Spectral Ball with λ=0 (simplified version).

MuonBall combines:
- Spectral Ball's hard retraction to spectral sphere
- Muon's simple msign orthogonalization (without λ solver)

This is equivalent to Spectral Ball with λ fixed to 0, which skips the
expensive bisection solver and uses the simpler Muon update direction.

Algorithm:
1. Power iteration: σ, u, v = power_iteration(W, steps)
2. Retract W: W ← (R/σ) * W
3. Orthogonalize: Φ = msign(M)  [no λ, just like Muon]
4. Update: W ← W - lr * scale * Φ

Key difference from Spectral Ball:
- Spectral Ball solves: <Θ, msign(M + λΘ)> = 0 to find λ
- MuonBall simply uses: Φ = msign(M) with λ = 0

This makes MuonBall significantly faster than Spectral Ball while still
maintaining spectral norm constraints through retraction.
"""

import math
from typing import Optional, Callable

import torch
from torch.optim.optimizer import Optimizer

# Import shared utilities from spectral_ball
from .spectral_ball import (
    msign,
    power_iteration,
    compute_target_radius,
    get_scale_factor,
)


@torch.no_grad()
def compute_muon_ball_update(
    W: torch.Tensor,
    M: torch.Tensor,
    target_radius: float,
    power_iteration_steps: int,
    msign_steps: int,
    retract_mode: str = 'hard',
    retract_alpha: float = 0.05,
    current_lr: Optional[float] = None,
) -> tuple[torch.Tensor, float, float]:
    """Compute MuonBall update (Spectral Ball with λ=0).

    Algorithm:
    1. Power iteration to get σ, u, v
    2. Retract W to spectral sphere: W ← (R/σ)W
    3. Orthogonalize: Φ = msign(M)  [simplified, no λ]

    Args:
        W: Current weight matrix (modified in-place for retraction)
        M: Momentum tensor
        target_radius: Target spectral norm R
        power_iteration_steps: Number of power iteration steps
        msign_steps: Number of Newton-Schulz iterations
        retract_mode: Retraction mode ('hard' or 'dynamic')
        retract_alpha: Alpha parameter for dynamic retraction
        current_lr: Current learning rate (for dynamic retraction)

    Returns:
        Tuple of (Phi, retract_bias, sigma_value) where:
        - Phi: Update direction to be applied as W ← W - lr * Φ
        - retract_bias: Retraction bias (0.0 for hard mode)
        - sigma_value: Spectral norm before retraction
    """
    # Convert M to fp32 and normalize
    M_fp32 = M.to(torch.float32)
    M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2, -1), keepdim=True).clamp_min(1e-8))

    # 1. Power iteration (returns fp32)
    sigma, u, v = power_iteration(W, steps=power_iteration_steps)
    sigma_value = sigma.item()

    # 2. Retract W to spectral sphere
    retract_bias = 0.0
    if retract_mode == 'hard':
        # Hard retraction: W ← (R/σ) * W
        if sigma_value > 1e-8:
            scale = target_radius / sigma_value
            W.mul_(scale)
    elif retract_mode == 'dynamic':
        # Dynamic retraction: W *= (1 + α * lr * bias) where bias = -sign(σ - R)
        bias = -1.0 if sigma_value > target_radius else 1.0
        effective_alpha = retract_alpha * current_lr if current_lr is not None else retract_alpha
        W.mul_(1.0 + effective_alpha * bias)
        retract_bias = bias
    else:
        raise ValueError(f"Unknown retract_mode: {retract_mode}")

    # 3. MuonBall: Simply msign(M) without λ solver
    # This is the KEY simplification: λ = 0
    Phi = msign(M_fp32, steps=msign_steps)

    return Phi, retract_bias, sigma_value


class MuonBall(Optimizer):
    """MuonBall Optimizer: Spectral Ball with λ=0.

    MuonBall simplifies Spectral Ball by fixing λ=0, which eliminates the need for
    the expensive bisection solver while still maintaining spectral norm constraints.

    The algorithm:
    1. Power iteration to compute spectral norm σ and top singular vectors (u, v)
    2. Retraction to spectral sphere: W ← (R/σ) * W
    3. Orthogonalize momentum: Φ = msign(M)  [no λ, unlike Spectral Ball]
    4. Update: W ← W - lr * Φ

    Comparison to Spectral Ball:
    - Spectral Ball: Solves for λ such that <Θ, msign(M + λΘ)> = 0
    - MuonBall: Uses λ = 0, so Φ = msign(M) directly

    Comparison to Muon:
    - Muon: No spectral constraint, weights can drift
    - MuonBall: Hard constraint via retraction, weights stay on spectral sphere

    This optimizer is useful for:
    - Faster training than Spectral Ball (no λ solver overhead)
    - More constrained than Muon (spectral norm control)
    - Validating whether λ solver is necessary in practice

    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum_beta: Momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0.01)
        power_iteration_steps: Steps for spectral norm computation (default: 10)
        msign_steps: Newton-Schulz iterations for msign (default: 5)
        radius_mode: Target radius mode ("spectral_mup" or "identity")
        scale_mode: Scale factor mode for updates
        retract_mode: Retraction mode ("hard" or "dynamic")
        retract_alpha: Alpha for dynamic retraction
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum_beta: float = 0.9,
        weight_decay: float = 0.01,
        power_iteration_steps: int = 10,
        msign_steps: int = 5,
        radius_mode: str = "spectral_mup",
        scale_mode: str = "align_adamw_rms",
        retract_mode: str = "hard",
        retract_alpha: float = 0.05,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum_beta < 0.0 or momentum_beta >= 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")
        if retract_mode not in ("hard", "dynamic"):
            raise ValueError(f"Invalid retract_mode: {retract_mode}")

        defaults = dict(
            lr=lr,
            momentum_beta=momentum_beta,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        self.power_iteration_steps = power_iteration_steps
        self.msign_steps = msign_steps
        self.radius_mode = radius_mode
        self.scale_mode = scale_mode
        self.retract_mode = retract_mode
        self.retract_alpha = retract_alpha

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
                    # 2D parameters: use MuonBall update
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    # Update momentum buffer
                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad)

                    # Compute MuonBall update
                    update = self._compute_muon_ball_update(p, buf, lr)

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

                    update = self._compute_muon_ball_update(p_2d, buf, lr)

                    if weight_decay != 0:
                        p_2d.mul_(1 - lr * weight_decay)

                    p_2d.add_(update, alpha=-lr)
                    p.data = p_2d.view(original_shape)

        return loss

    def _compute_muon_ball_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        lr: float,
    ) -> torch.Tensor:
        """Compute MuonBall constrained update direction.

        Args:
            W: Weight matrix (modified in-place for retraction)
            M: Momentum tensor
            lr: Current learning rate

        Returns:
            Update direction Φ
        """
        # Compute target radius
        target_radius = compute_target_radius(W.shape, self.radius_mode)

        # Compute MuonBall update (Spectral Ball with λ=0)
        Phi, bias, sigma = compute_muon_ball_update(
            W=W,
            M=M,
            target_radius=target_radius,
            power_iteration_steps=self.power_iteration_steps,
            msign_steps=self.msign_steps,
            retract_mode=self.retract_mode,
            retract_alpha=self.retract_alpha,
            current_lr=lr,
        )

        # Apply scale factor
        scale_factor = get_scale_factor(W.shape[0], W.shape[1], mode=self.scale_mode)

        return Phi * scale_factor

