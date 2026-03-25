"""
Parabolic-Cycloidal Structured Projection (Donut3)
====================================================

Replaces the Kronecker transform from Donut2 with a structured low-rank
projection whose columns have physical/geometric meaning:

  - Parabolic columns: encode linear + quadratic (constant-acceleration) modes
  - Cycloidal columns: encode sin/cos (rotational/periodic) modes

The projection factorizes a dense d×d transform into:
    W_down (d × r)  @  W_up (r × d)
where r = 2 (parabolic) + 2k (cycloidal modes) << d.

This gives O(d·r) cost instead of O(d²) for the Kronecker transform,
while providing inductive bias from the structured basis.

Connection to PVA paper (Section 4, Gram Matrix Reduction):
When this projection is used before a bilinear operation (like attention),
the effective Gram matrix is r×r instead of d×d.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ParabolicCycloidalProjection(nn.Module):
    """
    Structured low-rank projection with parabolic + cycloidal basis columns.
    
    The down-projection W_down ∈ R^{d × r} has columns partitioned as:
      [p_lin | p_quad | c₁_cos | c₁_sin | c₂_cos | c₂_sin | ...]
       ←── parabolic ──→ ←────────── cycloidal ──────────────→
    
    The up-projection W_up ∈ R^{r × d} is dense but small (r << d).
    
    The parabolic columns are initialized to capture velocity/acceleration
    structure. The cycloidal columns are initialized with frequency-scaled
    sinusoidal patterns. Both are learnable.
    
    Args:
        dim: Input/output dimension
        num_cycloidal_modes: Number of sin/cos pairs (k). Total rank = 2 + 2k.
        scale: Initialization scale
        bias: Whether to include a bias term
    """

    def __init__(
        self,
        dim: int,
        num_cycloidal_modes: int = 3,
        scale: float = 0.02,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_cycloidal_modes = num_cycloidal_modes
        self.rank = 2 + 2 * num_cycloidal_modes  # parabolic(2) + cycloidal(2k)

        # === Build structured W_down ===
        # Parabolic columns: initialized as random directions (learned to represent
        # velocity and acceleration modes of the data)
        p_lin = torch.randn(dim) * scale
        p_quad = torch.randn(dim) * scale

        # Cycloidal columns: initialized with sinusoidal patterns at different
        # frequencies, giving the basis rotational/periodic structure
        cyc_cols = []
        for m in range(num_cycloidal_modes):
            freq = math.pi * (m + 1) / dim  # spread frequencies across dimension
            indices = torch.arange(dim, dtype=torch.float32)
            cyc_cols.append(torch.cos(freq * indices) * scale)
            cyc_cols.append(torch.sin(freq * indices) * scale)

        W_down_init = torch.stack([p_lin, p_quad] + cyc_cols, dim=1)  # (dim, rank)
        self.W_down = nn.Parameter(W_down_init)

        # Dense up-projection (small: rank × dim)
        self.W_up = nn.Parameter(torch.randn(self.rank, dim) * scale)

        if bias:
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.bias = None

        # Learnable per-mode scaling (lets model adjust importance of each mode)
        self.mode_scales = nn.Parameter(torch.ones(self.rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, dim) → (B, N, dim)
        
        Applies: x @ diag(mode_scales) @ W_down @ W_up + bias
        """
        # Scale modes before projection
        W = self.W_down * self.mode_scales.unsqueeze(0)  # (dim, rank) * (1, rank)

        # Two small matmuls instead of one large one
        h = x @ W          # (B, N, dim) @ (dim, rank) → (B, N, rank)
        out = h @ self.W_up   # (B, N, rank) @ (rank, dim) → (B, N, dim)

        if self.bias is not None:
            out = out + self.bias

        return out

    @property
    def param_count(self) -> int:
        """Total learnable parameters."""
        total = self.W_down.numel() + self.W_up.numel() + self.mode_scales.numel()
        if self.bias is not None:
            total += self.bias.numel()
        return total

    @property
    def compression_ratio(self) -> float:
        """Ratio of PVA params to equivalent dense d×d matrix."""
        dense_params = self.dim * self.dim
        return dense_params / self.param_count

    def effective_weight(self) -> torch.Tensor:
        """The implicit d×d matrix (for analysis/debugging only)."""
        W = self.W_down * self.mode_scales.unsqueeze(0)
        return W @ self.W_up


class ResidualPCProjection(nn.Module):
    """
    Parabolic-Cycloidal projection with residual connection.
    
    Computes: x + alpha * PCP(x)
    
    The residual ensures the transform starts near identity and gradually
    learns structured perturbations. This is the recommended replacement
    for KroneckerTransform in Donut3.
    """

    def __init__(
        self,
        dim: int,
        num_cycloidal_modes: int = 3,
        scale: float = 0.02,
        alpha_init: float = 0.1,
    ):
        super().__init__()
        self.pcp = ParabolicCycloidalProjection(dim, num_cycloidal_modes, scale)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.alpha * self.pcp(x)
