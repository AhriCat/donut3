# Donut3 🍩³

**Parabolic-Cycloidal Toroidal Transformer**

Evolution of Donut2 integrating Parabolic Vector Algebra (PVA) for structured geometric projections.

## What Changed from Donut2

Two components replaced, everything else retained:

| Donut2 | Donut3 | Why |
|--------|--------|-----|
| `CycloidPositionalBias` | `ParabolicCycloidalBias` | Adds parabolic (recency/acceleration) modes alongside cycloidal (periodic) modes. Learnable blend. |
| `KroneckerTransform` | `ResidualPCProjection` | Structured low-rank projection (rank 2+2k) with physics-motivated basis columns. 5.3× compression vs dense at dim=64. |

Everything else (mHC, HyMBA, focused attention, logic bias, ternary tokenizer, weight tying) is unchanged.

## Architecture

```
Input → Ternary Tokenizer → Embedding (tied weights)
                               ↓
                  Parabolic-Cycloidal Projection   ← NEW (replaces Kronecker)
                               ↓
                         Logic Bias
                               ↓
                    ┌──────────┴──────────┐
                    ↓                     ↓
            Focused Attention         HyMBA (SSM+RNN)
              (with parabolic-          (parallel path)
               cycloidal bias)  ← NEW bias
                    ↓                     ↓
                    └──────────┬──────────┘
                               ↓
                      mHC Residual Connection
                      (doubly-stochastic mixing)
                               ↓
                         [repeat × depth]
                               ↓
                      Output Projection (tied weights)
```

## Key Ideas

### Parabolic-Cycloidal Positional Bias

Language has both **periodic** structure (syntax cycles, dialogue turns, verse patterns) and **accelerating** structure (narrative momentum, argument building, recency effects). A single curve type misses half the signal.

The positional encoding maps each position onto two curves simultaneously:
- **Cycloid**: `x = r(t − sin t)`, `y = r(1 − cos t)` — rolling/periodic dynamics
- **Parabola**: `x = v·t`, `y = ½·a·t²` — constant-acceleration dynamics (PVA)

A learnable blend parameter lets the model discover which curve family matters more for each task.

### PVA Structured Projection

The projection layer factorizes a `d×d` dense transform into `W_down (d×r) @ W_up (r×d)` where `r = 2 + 2k`:
- 2 **parabolic columns**: linear + quadratic (velocity/acceleration modes)
- 2k **cycloidal columns**: sin/cos pairs at different frequencies (rotational modes)

This gives O(d·r) cost instead of O(d²) with physics-motivated inductive bias. Each mode has a learnable scale, and the whole thing is wrapped in a residual connection starting near identity.

## Variants

| Variant | Class | Description |
|---------|-------|-------------|
| Donut3 | `Donut3` | Base with scalar residual scales |
| Donut3-mHC | `Donut3_mHC` | Manifold-constrained hyper-connections |
| Donut3-mHC-Simple | `Donut3_mHC_Simple` | Shared H_res, separate H_post |

## Usage

```python
from model import Donut3, Donut3_mHC
from tokenizer import TernaryTokenizer

tokenizer = TernaryTokenizer(vocab_size=50000)
tokenizer.train(your_texts, min_frequency=2)
tokenizer.freeze()

model = Donut3_mHC(
    vocab_size=len(tokenizer.token_to_id),
    dim=512,
    depth=6,
    heads=8,
    groups=4,
    rank=32,
    ssm_dim=64,
    rnn_dim=128,
    max_seq_len=512,
    # PVA params
    num_cycloidal_modes=3,   # 2 parabolic + 6 cycloidal = rank 8
    pcp_alpha_init=0.1,
    # mHC params
    n_streams=4,
    sinkhorn_iters=20,
    mhc_alpha_init=0.1,
)

output = model.generate("The meaning of life is", tokenizer, max_new_tokens=50)
```

## File Structure

```
donut3/
├── model.py                    # Donut3, Donut3_mHC, Donut3_mHC_Simple
├── parabolic_cycloidal_pos.py  # NEW: hybrid positional bias
├── pva_projection.py           # NEW: structured PVA projection
├── attn.py                     # Focused attention (from Donut2)
├── hymba.py                    # HyMBA block (from Donut2)
├── mhc.py                      # Manifold-constrained hyper-connections (from Donut2)
├── logic.py                    # Logic bias (fixed imports)
├── tokenizer.py                # Ternary tokenizer (from Donut2)
├── tokenizer_toroidal.py       # Toroidal tokenizer (from Donut2)
├── train.py                    # Training script (bugs fixed)
└── README.md
```

## Smoke Test Results

```
[1] ParabolicCycloidalBias      — PASS (blend=0.5, windowed sparsity 77%)
[2] ResidualPCProjection        — PASS (rank 8, 5.3× compression)
[3] Donut3 Base forward         — PASS (127,390 params at dim=64)
[4] Donut3_mHC forward + diag   — PASS (composite gain ~1.0, doubly stochastic ✓)
[5] Donut3_mHC_Simple forward   — PASS
[6] Gradient flow               — PASS (all 13 PVA params receive gradients, no NaN)
[7] Param comparison            — PASS (PVA 774 vs dense 4096 vs Kronecker 128)
```

## Connection to PVA Paper

The PVA paper's Gram Matrix Reduction Theorem (Theorem 4.1) applies directly to the positional bias computation: pairwise distances on parabolic curves are degree-4 polynomials determined by 2×2 Gram matrices, making the bias computation efficient regardless of embedding dimension. The structured projection columns encode the same velocity/acceleration decomposition formalized in the paper's casting operator.

## Bugs Fixed from Donut2

- `logic.py`: Added missing `torch` / `nn` imports
- `mhc.py`: `.view()` → `.reshape()` after `einsum` (non-contiguous tensor fix)
- `train.py`: Fixed malformed imports, undefined variables, redundant dataset slicing
- `hymba.py`: Dropout now actually applied in forward pass
- Weight tying: Silent `except` replaced with warning

## License

CC BY-NC 4.0 (inherited from Donut2)

## Citation

```bibtex
@software{donut3-2026,
  title={Donut3: Parabolic-Cycloidal Toroidal Transformer},
  author={Ahri Steele},
  year={2026},
  url={https://github.com/AhriCat/donut3}
}
```
