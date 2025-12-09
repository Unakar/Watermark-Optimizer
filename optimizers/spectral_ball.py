"""Simplified Spectral Ball Optimizer for watermark experiment.

This is a standalone version without TP/QKV split complexity.
Core algorithm:
1. Power iteration to get σ, u, v
2. Retract W to spectral sphere: W ← (R/σ)W
3. Form Θ = uv^T
4. Solve for λ: <Θ, msign(M + λΘ)> = 0
5. Update: W ← W - lr * msign(M + λΘ)
"""

import math
from typing import Optional, Tuple, Callable, List

import torch
from torch.optim.optimizer import Optimizer


# ============================================================================
# Core utility functions
# ============================================================================

def _newton_schulz_step(X: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    """One Newton-Schulz iteration: X ← a·X + X·(b·A + c·A²) where A = X·X^T."""
    A = X @ X.mT
    B = torch.addmm(A, A, A, alpha=c, beta=b)
    X = torch.addmm(X, B, X, alpha=1.0, beta=a)
    return X


@torch.compile
def msign(G: torch.Tensor, steps: int) -> torch.Tensor:
    """Matrix sign via Newton-Schulz with Polar-Express coefficients."""
    if G.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    
    orig_dtype = G.dtype
    if G.dtype != torch.float32:
        G = G.to(torch.float32)
    
    transpose = G.size(-2) > G.size(-1)
    X = G.mT if transpose else G
    X = torch.nn.functional.normalize(X, p=2, dim=(-2, -1), eps=1e-7)
    X = X.to(torch.bfloat16)
    
    # Polar-Express coefficients for fast convergence
    coeffs = [
        (8.2051, -22.9019, 16.4607),
        (4.0664, -2.8612, 0.5184),
        (3.9096, -2.8234, 0.5250),
        (3.2856, -2.4153, 0.4853),
        (2.2779, -1.6198, 0.3985),
        (1.8726, -1.2307, 0.3585),
        (1.8564, -1.2132, 0.3568),
        (1.8750, -1.2500, 0.3750),
    ]
    
    for i in range(steps):
        if i < 8:
            a, b, c = coeffs[i]
        else:
            a, b, c = coeffs[-1]
        X = _newton_schulz_step(X, a, b, c)
    
    result = X.mT if transpose else X
    return result.to(orig_dtype)


@torch.no_grad()
def power_iteration(w: torch.Tensor, steps: int = 10, eps: float = 1e-20) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Leading singular triplet (σ, u, v) via bilateral power iteration."""
    if w.ndim < 2:
        raise ValueError("Input tensor must have at least 2 dimensions.")
    
    w = w.to(torch.float32)
    v = torch.randn_like(w[..., :1, :].transpose(-2, -1))
    v = torch.nn.functional.normalize(v, dim=-2)
    
    for _ in range(steps):
        v = torch.nn.functional.normalize(w.transpose(-2, -1) @ (w @ v), dim=-2)
    u = torch.nn.functional.normalize(w @ v, dim=-2)
    s = (u.transpose(-2, -1) @ w @ v).squeeze(-1).squeeze(-1)
    
    return s, u, v


@torch.no_grad()
def inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Frobenius inner product <a, b>."""
    return (a * b).sum()


@torch.no_grad()
def compute_f(G: torch.Tensor, Theta: torch.Tensor, lambda_value: float, msign_steps: int = 5) -> float:
    """f(λ) = <Θ, msign(G + λΘ)>."""
    z = G + lambda_value * Theta
    Phi = msign(z, steps=msign_steps)
    f_value = float(inner_product(Theta, Phi).item())
    return f_value


@torch.no_grad()
def find_bracket(
    G: torch.Tensor,
    Theta: torch.Tensor,
    initial_guess: float = 0.0,
    initial_step: float = 1e-3,
    max_expansions: int = 10,
    msign_steps: int = 5,
    tolerance_f: float = 1e-8,
) -> Tuple[Optional[float], Optional[float], float, float]:
    """Find λ_L < λ_R such that f(λ_L) <= 0 <= f(λ_R)."""
    
    λ0 = initial_guess
    f0 = compute_f(G, Theta, λ0, msign_steps)
    
    if abs(f0) < tolerance_f:
        return λ0, λ0, f0, f0
    
    step = initial_step if f0 < 0 else -initial_step
    λ_prev, f_prev = λ0, f0
    
    for _ in range(max_expansions):
        λ_new = λ_prev + step
        f_new = compute_f(G, Theta, λ_new, msign_steps)
        
        sign_prev = f_prev <= 0.0
        sign_new = f_new <= 0.0
        
        if sign_prev != sign_new:
            if f_prev <= 0 and f_new >= 0:
                return λ_prev, λ_new, f_prev, f_new
            elif f_new <= 0 and f_prev >= 0:
                return λ_new, λ_prev, f_new, f_prev
            else:
                if abs(f_prev) <= abs(f_new):
                    return λ_prev, λ_prev, f_prev, f_prev
                else:
                    return λ_new, λ_new, f_new, f_new
        
        step *= 2.0
        λ_prev, f_prev = λ_new, f_new
    
    return None, None, f0, f0


@torch.no_grad()
def solve_lambda_bisection(
    G: torch.Tensor,
    Theta: torch.Tensor,
    tolerance_f: float = 1e-6,
    max_iterations: int = 20,
    msign_steps: int = 5,
) -> float:
    """Solve λ such that f(λ) = <Θ, msign(G + λΘ)> = 0 using bisection."""
    
    λ_L, λ_R, f_L, f_R = find_bracket(
        G, Theta,
        initial_guess=0.0,
        initial_step=1e-3,
        max_expansions=10,
        msign_steps=msign_steps,
        tolerance_f=tolerance_f,
    )
    
    if λ_L is None:
        return 0.0  # Fallback to lambda=0 (degenerates to Muon)
    
    best_λ = λ_L if abs(f_L) < abs(f_R) else λ_R
    best_f = f_L if abs(f_L) < abs(f_R) else f_R
    
    if abs(best_f) <= tolerance_f:
        return best_λ
    
    for _ in range(max_iterations):
        λ_mid = 0.5 * (λ_L + λ_R)
        f_mid = compute_f(G, Theta, λ_mid, msign_steps)
        
        if abs(f_mid) < abs(best_f):
            best_λ, best_f = λ_mid, f_mid
        
        if abs(f_mid) <= tolerance_f:
            return λ_mid
        
        if f_mid < 0:
            λ_L, f_L = λ_mid, f_mid
        else:
            λ_R, f_R = λ_mid, f_mid
    
    return best_λ


def compute_target_radius(shape: Tuple[int, int], radius_mode: str = "spectral_mup") -> float:
    """Compute target radius R."""
    if radius_mode == "spectral_mup":
        n_out, n_in = shape
        return math.sqrt(n_out / n_in)
    elif radius_mode == "identity":
        return 1.0
    else:
        raise ValueError(f"Invalid radius_mode: {radius_mode}")


def get_scale_factor(size_out: int, size_in: int, mode: str = "align_adamw_rms") -> float:
    """Get the scale factor for the update."""
    if mode == "shape_scaling":
        return max(1, size_out / size_in) ** 0.5
    elif mode == "align_adamw_rms":
        return 0.2 * max(size_out, size_in) ** 0.5
    elif mode == "spectral_mup":
        return (size_out / size_in) ** 0.5
    else:
        raise ValueError(f"Invalid scale mode: {mode}")


# ============================================================================
# SpectralBall Optimizer
# ============================================================================

class SpectralBall(Optimizer):
    """Spectral Ball Optimizer - simplified version for experiments.
    
    This optimizer constrains weight matrices to lie on a spectral sphere of fixed radius R.
    
    Args:
        params: Iterable of parameters to optimize
        lr: Learning rate
        momentum_beta: Momentum coefficient (default: 0.9)
        weight_decay: Weight decay coefficient (default: 0.01)
        power_iteration_steps: Steps for spectral norm computation (default: 10)
        msign_steps: Newton-Schulz iterations for msign (default: 5)
        radius_mode: Target radius mode ("spectral_mup" or "identity")
        scale_mode: Scale factor mode for updates
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
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum_beta < 0.0 or momentum_beta >= 1.0:
            raise ValueError(f"Invalid momentum_beta: {momentum_beta}")
        
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
                    # 2D parameters: use spectral ball update
                    state = self.state[p]
                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    
                    # Update momentum buffer
                    buf = state['momentum_buffer']
                    buf.mul_(momentum_beta).add_(grad)
                    
                    # Compute spectral ball update
                    update = self._compute_spectral_ball_update(p, buf, lr)
                    
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
                    
                    update = self._compute_spectral_ball_update(p_2d, buf, lr)
                    
                    if weight_decay != 0:
                        p_2d.mul_(1 - lr * weight_decay)
                    
                    p_2d.add_(update, alpha=-lr)
                    p.data = p_2d.view(original_shape)
        
        return loss
    
    def _compute_spectral_ball_update(
        self,
        W: torch.Tensor,
        M: torch.Tensor,
        lr: float,
    ) -> torch.Tensor:
        """Compute spectral ball constrained update direction.
        
        Args:
            W: Weight matrix (modified in-place for retraction)
            M: Momentum tensor
            lr: Current learning rate
            
        Returns:
            Update direction Φ
        """
        # Convert M to fp32 and normalize
        M_fp32 = M.to(torch.float32)
        M_fp32 = M_fp32 / (torch.linalg.norm(M_fp32, dim=(-2, -1), keepdim=True).clamp_min(1e-8))
        
        # 1. Power iteration to get σ, u, v
        sigma, u, v = power_iteration(W, steps=self.power_iteration_steps)
        sigma_value = sigma.item()
        
        # 2. Retract W to spectral sphere: W ← (R/σ)W
        target_radius = compute_target_radius(W.shape, self.radius_mode)
        if sigma_value > 1e-8:
            scale = target_radius / sigma_value
            W.mul_(scale)
        
        # 3. Form Θ = u @ v^T
        Theta = u @ v.transpose(-2, -1)
        
        # 4. Solve for λ: <Θ, msign(M + λΘ)> = 0
        lambda_value = solve_lambda_bisection(
            G=M_fp32,
            Theta=Theta,
            tolerance_f=1e-8,
            max_iterations=100,
            msign_steps=self.msign_steps,
        )
        
        # 5. Compute Φ = msign(M + λΘ)
        Z = M_fp32 + lambda_value * Theta
        Phi = msign(Z, steps=self.msign_steps)
        
        # Apply scale factor
        scale_factor = get_scale_factor(W.shape[0], W.shape[1], mode=self.scale_mode)
        
        return Phi * scale_factor

