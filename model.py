"""
Donut3: Parabolic-Cycloidal Toroidal Transformer
=================================================

Evolution from Donut2 with two key changes:

1. Positional encoding: ParabolicCycloidalBias replaces CycloidPositionalBias.
   Positions are mapped onto BOTH a cycloid (periodic/rolling dynamics) and
   a parabola (accelerating/recency dynamics). The model learns which curve
   family matters more via a blend parameter.

2. Structured projection: ResidualPCProjection replaces KroneckerTransform.
   A low-rank (2 + 2k) structured basis with parabolic and cycloidal columns
   replaces the Kronecker-factorized dense transform, cutting parameters from
   O(d²) to O(d·r) while adding physics-motivated inductive bias.

Everything else (mHC, HyMBA, focused attention, logic bias, ternary tokenizer)
is retained from Donut2.

Variant summary:
  - Donut3           : Base model with scalar residual scales
  - Donut3_mHC       : With manifold-constrained hyper-connections
  - Donut3_mHC_Simple: Shared H_res, separate H_post (fewer params)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from parabolic_cycloidal_pos import ParabolicCycloidalBias
from pva_projection import ResidualPCProjection
from attn import FocusedAttentionGroup
from logic import LogicBias
from hymba import HyMBA_Block
from mhc import mHCResidual, mHCDualPathResidual, sinkhorn_knopp, check_doubly_stochastic, compute_composite_gain


# =============================================================================
# Donut3 Base
# =============================================================================

class Donut3(nn.Module):
    """
    Base Donut3 model with parabolic-cycloidal geometry.
    Uses scalar residual scales (like Donut2 base).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        # PVA projection params
        num_cycloidal_modes: int = 3,
        pcp_alpha_init: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Embedding
        self.embed = nn.Embedding(vocab_size, dim)

        # NEW: Parabolic-Cycloidal positional bias (replaces CycloidPositionalBias)
        self.pos_bias = ParabolicCycloidalBias(max_seq_len)

        # NEW: Structured PVA projection (replaces KroneckerTransform)
        self.projection = ResidualPCProjection(
            dim, num_cycloidal_modes=num_cycloidal_modes,
            alpha_init=pcp_alpha_init,
        )

        # Logic bias (retained)
        self.logic_bias = LogicBias(dim, strength=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            })
            for _ in range(depth)
        ])
        self.res_scales = nn.Parameter(torch.ones(depth))

        # Output
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

        # Weight tying
        try:
            self.out.weight = self.embed.weight
        except Exception:
            import warnings
            warnings.warn("Weight tying failed; using separate output weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N] token ids → logits [B, N, V]
        """
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, N = x.shape
        bias = self.pos_bias(N, device=x.device)

        x = self.embed(x)          # [B, N, D]
        x = self.projection(x)     # Structured PVA projection
        x = self.logic_bias(x)     # Soft logical inductive bias

        for i, layer in enumerate(self.layers):
            attn_out = layer["attn"](layer["norm"](x), bias)
            ssm_out = layer["hybrid"](x)
            scale = self.res_scales[i]
            x = x + scale * (attn_out + ssm_out)

        return self.out(self.norm(x))

    @torch.no_grad()
    def generate(
        self,
        prompt,
        tokenizer,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token: Optional[int] = None,
        device: Optional[torch.device] = None,
        return_ids: bool = False,
    ):
        device = device or next(self.parameters()).device
        if isinstance(prompt, str):
            ids = tokenizer.encode(prompt, return_torch=True, device=device)
        elif isinstance(prompt, list):
            ids = torch.tensor(prompt, dtype=torch.long, device=device)
        elif isinstance(prompt, torch.Tensor):
            ids = prompt.to(device)
        else:
            raise TypeError("Expected str, list[int], or torch.Tensor")

        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        generated = ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None and top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
            else:
                probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break

        gen_ids = generated.squeeze(0).tolist()
        decoded = tokenizer.decode(gen_ids)
        return (decoded, gen_ids) if return_ids else decoded


# =============================================================================
# Donut3 with mHC
# =============================================================================

class Donut3_mHC(nn.Module):
    """
    Donut3 with manifold-constrained hyper-connections.
    
    Key equation per layer:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
    where H_res is doubly stochastic (Sinkhorn-projected).
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        # PVA params
        num_cycloidal_modes: int = 3,
        pcp_alpha_init: float = 0.1,
        # mHC params
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        super().__init__()
        assert dim % n_streams == 0, f"dim ({dim}) must be divisible by n_streams ({n_streams})"

        self.dim = dim
        self.depth = depth
        self.n_streams = n_streams
        self.stream_dim = dim // n_streams

        # Embeddings and preprocessing
        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_bias = ParabolicCycloidalBias(max_seq_len)
        self.projection = ResidualPCProjection(
            dim, num_cycloidal_modes=num_cycloidal_modes,
            alpha_init=pcp_alpha_init,
        )
        self.logic_bias = LogicBias(dim, strength=0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            })
            for _ in range(depth)
        ])

        # mHC residuals (one per layer)
        self.mhc_residuals = nn.ModuleList([
            mHCResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])

        # H_pre matrices (applied before layer computation)
        self.H_pre_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(n_streams, n_streams))
            for _ in range(depth)
        ])

        # Output
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        try:
            self.out.weight = self.embed.weight
        except Exception:
            import warnings
            warnings.warn("Weight tying failed; using separate output weights.")

    def get_H_pre(self, layer_idx: int) -> torch.Tensor:
        return F.softmax(self.H_pre_logits[layer_idx], dim=-1)

    def _apply_H_pre(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        B, N, D = x.shape
        x_streams = x.view(B, N, self.n_streams, self.stream_dim)
        H_pre = self.get_H_pre(layer_idx)
        x_mixed = torch.einsum('bnsd,st->bntd', x_streams, H_pre)
        return x_mixed.reshape(B, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, N = x.shape
        bias = self.pos_bias(N, device=x.device)

        x = self.embed(x)
        x = self.projection(x)
        x = self.logic_bias(x)

        for i, layer in enumerate(self.layers):
            # H_pre before attention
            x_pre = self._apply_H_pre(layer["norm"](x), i)
            attn_out = layer["attn"](x_pre, bias)

            # H_pre before hybrid (without norm, matching Donut2 convention)
            x_pre_hyb = self._apply_H_pre(x, i)
            ssm_out = layer["hybrid"](x_pre_hyb)

            # mHC residual
            layer_output = attn_out + ssm_out
            x = self.mhc_residuals[i](x, layer_output)

        return self.out(self.norm(x))

    def get_mhc_diagnostics(self) -> dict:
        diagnostics = {'per_layer': {}, 'composite_gain': None}
        H_res_list = []
        for i, mhc in enumerate(self.mhc_residuals):
            H_res = mhc.get_H_res()
            H_res_list.append(H_res)
            diagnostics['per_layer'][f'layer_{i}'] = {
                'H_res': H_res.detach().cpu().tolist(),
                'doubly_stochastic_check': check_doubly_stochastic(H_res),
                'forward_gain': H_res.abs().sum(dim=-1).max().item(),
            }
        diagnostics['composite_gain'] = compute_composite_gain(H_res_list)
        return diagnostics

    @torch.no_grad()
    def generate(self, prompt, tokenizer, **kwargs):
        device = kwargs.pop('device', None) or next(self.parameters()).device
        if isinstance(prompt, str):
            ids = tokenizer.encode(prompt, return_torch=True, device=device)
        elif isinstance(prompt, list):
            ids = torch.tensor(prompt, dtype=torch.long, device=device)
        elif isinstance(prompt, torch.Tensor):
            ids = prompt.to(device)
        else:
            raise TypeError("Expected str, list[int], or torch.Tensor")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        generated = ids.clone()
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 40)
        eos_token = kwargs.get('eos_token', None)
        return_ids = kwargs.get('return_ids', False)

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k and top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
            else:
                probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break

        gen_ids = generated.squeeze(0).tolist()
        decoded = tokenizer.decode(gen_ids)
        return (decoded, gen_ids) if return_ids else decoded


# =============================================================================
# Donut3 with Simple mHC (shared H_res, dual H_post)
# =============================================================================

class Donut3_mHC_Simple(nn.Module):
    """
    Simplified mHC variant: shared H_res for both paths, separate H_post.
    Fewer params than full dual-path mHC, good for smaller models.
    """

    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        groups: int = 4,
        rank: int = 32,
        ssm_dim: int = 64,
        rnn_dim: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        num_cycloidal_modes: int = 3,
        pcp_alpha_init: float = 0.1,
        n_streams: int = 4,
        sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.1,
    ):
        super().__init__()
        assert dim % n_streams == 0

        self.dim = dim
        self.depth = depth

        self.embed = nn.Embedding(vocab_size, dim)
        self.pos_bias = ParabolicCycloidalBias(max_seq_len)
        self.projection = ResidualPCProjection(
            dim, num_cycloidal_modes=num_cycloidal_modes,
            alpha_init=pcp_alpha_init,
        )
        self.logic_bias = LogicBias(dim, strength=0.02)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm": nn.LayerNorm(dim),
                "attn": FocusedAttentionGroup(dim, heads, groups, rank, dropout),
                "hybrid": HyMBA_Block(dim, ssm_dim, rnn_dim, dropout),
            })
            for _ in range(depth)
        ])

        # Unified mHC with shared H_res but separate H_post
        self.mhc = nn.ModuleList([
            mHCDualPathResidual(dim, n_streams, sinkhorn_iters, mhc_alpha_init)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)
        try:
            self.out.weight = self.embed.weight
        except Exception:
            import warnings
            warnings.warn("Weight tying failed; using separate output weights.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.long, device=next(self.parameters()).device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        B, N = x.shape
        bias = self.pos_bias(N, device=x.device)

        x = self.embed(x)
        x = self.projection(x)
        x = self.logic_bias(x)

        for i, layer in enumerate(self.layers):
            normed = layer["norm"](x)
            attn_out = layer["attn"](normed, bias)
            ssm_out = layer["hybrid"](x)
            x = self.mhc[i](x, attn_out, ssm_out)

        return self.out(self.norm(x))

    @torch.no_grad()
    def generate(self, prompt, tokenizer, **kwargs):
        device = kwargs.pop('device', None) or next(self.parameters()).device
        if isinstance(prompt, str):
            ids = tokenizer.encode(prompt, return_torch=True, device=device)
        elif isinstance(prompt, list):
            ids = torch.tensor(prompt, dtype=torch.long, device=device)
        elif isinstance(prompt, torch.Tensor):
            ids = prompt.to(device)
        else:
            raise TypeError("Expected str, list[int], or torch.Tensor")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)

        generated = ids.clone()
        max_new_tokens = kwargs.get('max_new_tokens', 50)
        temperature = kwargs.get('temperature', 0.8)
        top_k = kwargs.get('top_k', 40)
        eos_token = kwargs.get('eos_token', None)
        return_ids = kwargs.get('return_ids', False)

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k and top_k > 0:
                top_vals, top_idx = torch.topk(next_logits, top_k)
                probs = torch.zeros_like(next_logits).scatter_(
                    -1, top_idx, F.softmax(top_vals, dim=-1)
                )
            else:
                probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if eos_token is not None and (next_token == eos_token).all():
                break

        gen_ids = generated.squeeze(0).tolist()
        decoded = tokenizer.decode(gen_ids)
        return (decoded, gen_ids) if return_ids else decoded
