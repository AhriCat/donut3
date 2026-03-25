"""
Parabolic-Cycloidal Positional Bias (Donut3)
=============================================

Combines two families of position curves:
  - Cycloidal modes: x = r(t - sin t), y = r(1 - cos t)
    Captures periodic, rolling dynamics. Wraps on the torus.
  - Parabolic modes: x = v₀t, y = ½at²
    Captures accelerating, quadratic dynamics. Encodes recency bias.

The attention bias is computed as a Gaussian kernel over pairwise
distances on these curves. The Gram matrix reduction from PVA
makes this efficient: all pairwise distances reduce to operations
on small Gram matrices regardless of embedding dimension.

The key insight: language has BOTH periodic structure (syntax cycles,
dialogue turns, verse) AND accelerating structure (narrative momentum,
argument building, recency). One curve type alone misses half the signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParabolicCycloidalBias(nn.Module):
    """
    Positional attention bias from parabolic + cycloidal curves.

    Each position i is mapped to a point on a parabolic curve AND a point
    on a cycloidal curve. The attention bias between positions i and j is
    a learned combination of Gaussian kernels over the distances on each curve.
    
    Parabolic component:  p(t) = (v*t, ½*a*t²)
    Cycloidal component:  c(t) = (r*(t - sin t), r*(1 - cos t))
    
    Parameters are learnable so the model can adjust the relative importance
    and shape of each curve family.
    """

    def __init__(
        self,
        max_seq_len: int = 2048,
        learnable: bool = True,
        # Cycloidal params
        init_r: float = 1.0,
        init_alpha: float = 0.4,
        # Parabolic params
        init_v: float = 1.0,
        init_a: float = 0.5,
        # Shared
        init_sigma: float = 1.0,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len

        # Cycloidal curve parameters
        self.r = nn.Parameter(torch.tensor(init_r), requires_grad=learnable)
        self.alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=learnable)
        self.cyc_phase = nn.Parameter(torch.zeros(1), requires_grad=learnable)

        # Parabolic curve parameters (PVA: velocity = linear col, accel = quadratic col)
        self.v = nn.Parameter(torch.tensor(init_v), requires_grad=learnable)
        self.a = nn.Parameter(torch.tensor(init_a), requires_grad=learnable)
        self.para_phase = nn.Parameter(torch.zeros(1), requires_grad=learnable)

        # Kernel widths (separate for each curve family)
        self.sigma_cyc = nn.Parameter(torch.tensor(init_sigma), requires_grad=learnable)
        self.sigma_para = nn.Parameter(torch.tensor(init_sigma), requires_grad=learnable)

        # Learnable blend between parabolic and cycloidal bias
        self.blend_logit = nn.Parameter(torch.tensor(0.0), requires_grad=learnable)

    def _cycloidal_points(self, t: torch.Tensor) -> torch.Tensor:
        """Map parameter t to points on the cycloid curve. Returns (N, 2)."""
        x = self.r * (t - torch.sin(t))
        y = self.r * (1.0 - torch.cos(t))
        return torch.stack([x, y], dim=-1)

    def _parabolic_points(self, t: torch.Tensor) -> torch.Tensor:
        """Map parameter t to points on the parabolic curve. Returns (N, 2).
        
        This is the PVA trajectory: r(t) = P @ [t, t²]ᵀ
        where P = [[v, 0], [0, a/2]] (diagonal parabolic vector).
        """
        x = self.v * t
        y = 0.5 * self.a * t * t
        return torch.stack([x, y], dim=-1)

    def forward(
        self,
        seq_len: int,
        device: torch.device = None,
        window: int = None,
    ) -> torch.Tensor:
        """
        Compute attention bias matrix.

        Args:
            seq_len: Sequence length
            device: Target device
            window: If set, use sparse windowed bias (for efficiency)

        Returns:
            Bias tensor of shape (seq_len, seq_len)
        """
        device = device or next(self.parameters()).device
        i = torch.arange(seq_len, device=device, dtype=torch.float32)

        # Parameterize positions on each curve
        t_cyc = self.alpha * i + self.cyc_phase
        t_para = self.alpha * i + self.para_phase

        if window is not None:
            return self._windowed_bias(t_cyc, t_para, window)

        # Full pairwise distances on each curve
        pts_cyc = self._cycloidal_points(t_cyc)     # (N, 2)
        pts_para = self._parabolic_points(t_para)    # (N, 2)

        # Squared distances via cdist (efficient for 2D points)
        dist2_cyc = torch.cdist(pts_cyc, pts_cyc).pow(2)
        dist2_para = torch.cdist(pts_para, pts_para).pow(2)

        # Gaussian kernels with separate bandwidths
        bias_cyc = -dist2_cyc / (2.0 * self.sigma_cyc.pow(2) + 1e-8)
        bias_para = -dist2_para / (2.0 * self.sigma_para.pow(2) + 1e-8)

        # Learnable blend
        blend = torch.sigmoid(self.blend_logit)
        return blend * bias_cyc + (1.0 - blend) * bias_para

    def _windowed_bias(
        self,
        t_cyc: torch.Tensor,
        t_para: torch.Tensor,
        window: int,
    ) -> torch.Tensor:
        """Sparse windowed version for long sequences."""
        seq_len = t_cyc.size(0)
        bias = torch.zeros(seq_len, seq_len, device=t_cyc.device)

        for offset in range(-window, window + 1):
            if offset == 0:
                continue
            valid_i = torch.arange(
                max(0, -offset), min(seq_len, seq_len - offset),
                device=t_cyc.device,
            )
            valid_j = valid_i + offset

            # Cycloidal distance for this offset
            pts_i_c = self._cycloidal_points(t_cyc[valid_i])
            pts_j_c = self._cycloidal_points(t_cyc[valid_j])
            d2_cyc = (pts_i_c - pts_j_c).pow(2).sum(-1)

            # Parabolic distance for this offset
            pts_i_p = self._parabolic_points(t_para[valid_i])
            pts_j_p = self._parabolic_points(t_para[valid_j])
            d2_para = (pts_i_p - pts_j_p).pow(2).sum(-1)

            blend = torch.sigmoid(self.blend_logit)
            val = blend * (-d2_cyc / (2.0 * self.sigma_cyc.pow(2) + 1e-8)) \
                + (1.0 - blend) * (-d2_para / (2.0 * self.sigma_para.pow(2) + 1e-8))

            bias[valid_i, valid_j] = val

        return bias
