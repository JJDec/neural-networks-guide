# N-HiTS Architecture & Interpretability Guide

## Overview

**N-HiTS** (Neural Hierarchical Interpolation for Time Series Forecasting) is a state-of-the-art multi-rate hierarchical neural network architecture designed for long-term time series forecasting (Challu et al., AAAI 2023).

This implementation is tailored for **24-hour electricity demand forecasting** using a **168-hour (7-day) look-back window** with calendar covariates (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).

---

## Key Characteristics & Advantages

N-HiTS was designed to overcome three fundamental weaknesses of earlier neural time-series models (N-BEATS, Transformers):

| Challenge | N-HiTS Solution |
|-----------|----------------|
| **Quadratic cost with horizon** | Hierarchical interpolation produces only $\lceil H/k^{fwd} \rceil$ coefficients per block — dramatically fewer parameters for long horizons |
| **Frequency entanglement** | Each stack receives a differently sub-sampled view of the look-back, so coarse and fine patterns are learned by dedicated stacks |
| **Gradient diffusion** | Doubly-residual stacking keeps individual block gradients well-conditioned throughout training |

---

## Architectural Principles

N-HiTS addresses long-horizon forecasting challenges through two core mechanisms:

1. **Multi-Rate Input Subsampling (Max Pooling)**
   - Input sequences (lookback target and historical covariates) are pooled along the time dimension at different rates per stack ($k_1^{in}=8, k_2^{in}=4, k_3^{in}=1$).
   - Coarse stacks learn long-term low-frequency trends.
   - Fine stacks learn high-frequency details.

2. **Hierarchical Basis Interpolation**
   - Each block outputs small coefficient vectors for backcast ($L_{bck} = \lceil L / k^{bck} \rceil$) and forecast ($H_{fwd} = \lceil H / k^{fwd} \rceil$).
   - Coefficients are synthesized into full time-length backcasts $\hat{x}_b \in \mathbb{R}^L$ and forecasts $\hat{y}_b \in \mathbb{R}^H$ via 1D linear interpolation (`torch.nn.functional.interpolate`).

3. **Doubly Residual Stacking**
   - Sequential residual target refinement: $x_b = x_{b-1} - \hat{x}_{b-1}$.
   - Total forecast synthesis: $\hat{y} = \sum_{b=1}^N \hat{y}_b$.

```
Input (168h + Covariates)
   │
   ├─► Stack 1 (Coarse, Pool=8, Downsample=8) ──► Backcast 1 (Interpolated 168h) ─► Target Residual
   │                                           └─► Forecast 1 (Interpolated 24h)  ─┐
   │                                                                               │
   ├─► Stack 2 (Medium, Pool=4, Downsample=4) ──► Backcast 2 (Interpolated 168h) ─► Target Residual
   │                                           └─► Forecast 2 (Interpolated 24h)  ─┼─► Sum = Final Forecast (24h)
   │                                                                               │
   └─► Stack 3 (Fine, Pool=1, Downsample=1)   ──► Backcast 3 (Interpolated 168h)   │
                                               └─► Forecast 3 (Interpolated 24h)  ─┘
```
---

## Architecture & Pipeline

The full forward pass from raw batch to 24-hour forecast:

```
Batch dict ─────────────────────────────────────────────────────────────────
  past_target   (B, 168)        ← scaled demand look-back
  hist_covs     (B, 168, 4)     ← hour_sin/cos, dow_sin/cos (historical)
  future_covs   (B,  24, 4)     ← same features, known ahead for horizon

NHiTSModel.forward()
│
│  target_residual = past_target          # x₀ = raw look-back
│  forecast        = zeros(B, 24)         # ŷ  = accumulator
│
├─ Block 0  [Stack 0 — Coarse, pool=8, downsample=8]
│   ├─ MaxPool1d(k=8) on target + hist_covs  → pooled_len = ⌈168/8⌉ = 21
│   ├─ flat_dim = 21 + 21×4 + 24×4 = 21 + 84 + 96 = 201
│   ├─ MLP(201 → 512 → 512)
│   ├─ backcast_head → 21 coeffs → interpolate → (B, 168)
│   ├─ forecast_head →  3 coeffs → interpolate → (B,  24)
│   ├─ target_residual -= backcast           # residual subtraction
│   └─ forecast        += forecast_b         # forecast accumulation
│
├─ Block 1  [Stack 1 — Medium, pool=4, downsample=4]
│   ├─ MaxPool1d(k=4) → pooled_len = ⌈168/4⌉ = 42
│   ├─ flat_dim = 42 + 42×4 + 24×4 = 42 + 168 + 96 = 306
│   ├─ MLP(306 → 512 → 512)
│   ├─ backcast_head → 42 coeffs → interpolate → (B, 168)
│   ├─ forecast_head →  6 coeffs → interpolate → (B,  24)
│   ├─ target_residual -= backcast
│   └─ forecast        += forecast_b
│
└─ Block 2  [Stack 2 — Fine, pool=1, downsample=1]
    ├─ Identity (no pooling) → pooled_len = 168
    ├─ flat_dim = 168 + 168×4 + 24×4 = 168 + 672 + 96 = 936
    ├─ MLP(936 → 512 → 512)
    ├─ backcast_head → 168 coeffs → (B, 168)  [no interp needed]
    ├─ forecast_head →  24 coeffs → (B,  24)  [no interp needed]
    ├─ target_residual -= backcast
    └─ forecast        += forecast_b

Output: forecast  (B, 24)   ← sum of all three block forecasts
```

---

## Component Breakdown

### `NHiTSBlock` — the atomic unit

Each block executes five sub-steps in `forward()`:

1. **Input subsampling** (`nn.MaxPool1d` or `nn.Identity`)  
   Temporal compression rate is set by `pooling_kernel_size`. The same pooling kernel is applied identically to the target sequence and each historical covariate channel.

2. **Feature concatenation**  
   Pooled target + pooled historical covariates + flattened future covariates are concatenated into a single flat vector of size `flat_dim`.

3. **MLP stack**  
   `num_mlp_layers` fully-connected layers, each followed by `ReLU`/`GELU`, optional `Dropout(p)`, and optional `LayerNorm`. Output: `(B, hidden_size)`.

4. **Basis projection heads**  
   Two independent `nn.Linear` layers project the shared hidden state:
   - `backcast_head` → `⌈L / n_freq_downsample⌉` coefficients
   - `forecast_head` → `⌈H / n_freq_downsample⌉` coefficients

5. **1D linear interpolation** (`F.interpolate`, `mode="linear"`)  
   Coefficients are upsampled back to full length `L` (backcast) and `H` (forecast). When the coefficient size already matches the target length (Stack 2 with downsample=1), interpolation is skipped.

### `NHiTSModel` — orchestrator

Owns an `nn.ModuleList` of all blocks and implements the **doubly-residual** accumulation loop:

```python
target_residual = past_target
forecast = torch.zeros(B, H)
for block in self.blocks:
    backcast_b, forecast_b = block(target_residual, hist_covs, future_covs)
    target_residual = target_residual - backcast_b   # residual update
    forecast = forecast + forecast_b                 # forecast accumulation
```

### `Trainer` — training infrastructure

| Component | Detail |
|-----------|--------|
| **Optimiser** | `AdamW(lr=1e-3, weight_decay=1e-4)` |
| **LR scheduler** | `ReduceLROnPlateau(factor=0.5, patience=patience//2)` — halves LR when val loss stalls |
| **Early stopping** | `EarlyStopping(patience=20, min_delta=1e-6)` — stops when val loss fails to improve |
| **Gradient clipping** | `clip_grad_norm_(parameters, max_norm=1.0)` applied every batch |
| **Checkpoint** | Full state dict (model + optimiser + scheduler + epoch) saved to `checkpoints/nhits/best_model.pt` |
| **W&B** | Per-epoch `train/loss`, `val/loss`, `train/lr`; final checkpoint logged as a W&B Artifact |

---

## Model Interpretability

N-HiTS is more interpretable than Transformer-based forecasters because its outputs are **additive** and **decomposed by frequency band**. Three complementary techniques are supported by this implementation:

### 1. Per-Stack Forecast Decomposition

Each block contributes an individual forecast component `forecast_b`. By capturing these separately you obtain a natural signal decomposition — coarse stack captures trend, medium captures weekly seasonality, fine stack captures hourly residuals.

```python
import torch
from nhits.models.nhits import NHiTSModel

model.eval()
with torch.no_grad():
    target_residual = past_target.clone()
    components = []
    for block in model.blocks:
        backcast_b, forecast_b = block(target_residual, hist_covs, future_covs)
        target_residual = target_residual - backcast_b
        components.append(forecast_b.cpu())          # per-stack component

# components[0] → coarse trend  (Stack 0, pool=8)
# components[1] → medium freq   (Stack 1, pool=4)
# components[2] → fine details  (Stack 2, pool=1)
total_forecast = torch.stack(components).sum(dim=0)  # should match model output
```

### 2. Basis Coefficient Inspection

The raw coefficients before interpolation expose the signal at compressed resolution. A flat coefficient vector (Stack 0 → 3 values for a 24-step horizon) is the model's latent "sketch" of the future.

```python
model.eval()
with torch.no_grad():
    # Manually run one block's forward to intercept coefficients
    block = model.blocks[0]   # coarse stack
    B = past_target.size(0)

    target_3d = past_target.unsqueeze(1)
    target_pooled = block.pool(target_3d).squeeze(1)
    hist_3d = hist_covs.transpose(1, 2)
    hist_pooled = block.pool(hist_3d).reshape(B, -1)
    fut_flat = future_covs.reshape(B, -1)

    x_in = torch.cat([target_pooled, hist_pooled, fut_flat], dim=-1)
    feat = block.mlp(x_in)

    bck_coeffs = block.backcast_head(feat)   # (B, ⌈L/k⌉) — compressed backcast
    fwd_coeffs = block.forecast_head(feat)   # (B, ⌈H/k⌉) — compressed forecast
```

The ratio `fwd_coeff_size / horizon` tells you the **compression ratio** per stack:

| Stack | `n_freq_downsample` | `fwd_coeff_size` | Compression |
|-------|--------------------|-----------------:|------------:|
| Coarse | 8 | 3 | 8× |
| Medium | 4 | 6 | 4× |
| Fine   | 1 | 24 | 1× (none) |

### 3. Residual Tracking

Monitoring how much signal remains in `target_residual` after each block reveals which stack is doing most of the modelling work. A near-zero residual after Stack 0 indicates a highly trend-dominated series.

```python
model.eval()
with torch.no_grad():
    residual = past_target.clone()
    for i, block in enumerate(model.blocks):
        backcast_b, _ = block(residual, hist_covs, future_covs)
        residual = residual - backcast_b
        residual_energy = residual.pow(2).mean().item()
        print(f"Stack {i} residual MSE: {residual_energy:.6f}")
```

### 4. Covariate Sensitivity (Gradient-Based)

Since the MLP stack processes past target and covariates as a flat concatenated vector, input-gradient sensitivity can be used to measure feature importance:

```python
model.eval()
past_target_g = past_target.requires_grad_(True)
hist_covs_g   = hist_covs.requires_grad_(True)

forecast = model(past_target_g, hist_covs_g, future_covs)
forecast.sum().backward()

target_sensitivity = past_target_g.grad.abs().mean(dim=0)   # (L,) — per timestep
covariate_sensitivity = hist_covs_g.grad.abs().mean(dim=0)  # (L, C_hist) — per feature
```

---

## PyTorch Implementation Reference

Key implementation patterns used in [nhits.py](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/models/nhits.py):

### Interpolation helper

```python
def _interpolate_1d(x: Tensor, target_len: int) -> Tensor:
    """Upsample coefficient vector (B, S) → (B, target_len) via linear interp."""
    if x.size(1) == target_len:
        return x
    if x.size(1) == 1:
        return x.expand(-1, target_len)          # broadcast scalar coefficient
    x_3d  = x.unsqueeze(1)                       # (B, 1, S) — required by F.interpolate
    out_3d = F.interpolate(x_3d, size=target_len, mode="linear", align_corners=True)
    return out_3d.squeeze(1)                     # (B, target_len)
```

> **Why `align_corners=True`?**  Ensures the first and last coefficients are pinned to the first and last output positions, avoiding boundary artefacts in short coefficient sequences (e.g. 3 → 24).

### MaxPool1d on multi-channel covariates

```python
# hist_covs: (B, L, C_hist)
hist_3d   = hist_covs.transpose(1, 2)           # (B, C_hist, L) — pool expects (B, C, L)
hist_pooled = self.pool(hist_3d).reshape(B, -1) # (B, C_hist × L_pool) — flattened
```

### Doubly-residual accumulation (NHiTSModel)

```python
target_residual = past_target                   # initialise from raw input
forecast        = torch.zeros(B, H, ...)        # zero accumulator
for block in self.blocks:
    backcast_b, forecast_b = block(target_residual, hist_covs, future_covs)
    target_residual = target_residual - backcast_b   # remove explained variance
    forecast        = forecast        + forecast_b   # add block contribution
return forecast
```

### Building the model from config

```python
from nhits.config import NHiTSConfig
from nhits.models.nhits import NHiTSModel

cfg   = NHiTSConfig()
model = NHiTSModel.from_config(cfg)             # reads all arch params from config
```

---

## Quickstart Commands

### 1. Run Quick Training Smoke Test
```bash
uv run python nhits/train.py --max_epochs 5 --batch_size 32
```

### 2. Run Full Training (Default Synthetic Dataset)
```bash
uv run python nhits/train.py --max_epochs 100 --batch_size 64
```

### 3. Run Training with Real Dataset CSV
```bash
uv run python nhits/train.py --data_path data/electricity.csv --target_col demand_mw
```

## Recommended Workflow & Hyperparameters

### Training workflow

```
1. Smoke test (5 epochs, batch=32)
   └─ uv run python nhits/train.py --max_epochs 5 --batch_size 32

2. Baseline run (default config, synthetic data)
   └─ uv run python nhits/train.py --max_epochs 100 --batch_size 64

3. Real-data run
   └─ uv run python nhits/train.py --data_path data/electricity.csv \
        --target_col demand_mw --max_epochs 200

4. Hyperparameter search (tune hidden_size, dropout, lr)
   └─ Modify NHiTSConfig fields and re-run; compare val loss across runs.

5. Inspect interpretability
   └─ Use per-stack decomposition and residual tracking (see Model Interpretability).
```

---

## Configuration & Hyperparameters

Key settings in `NHiTSConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_len` | `168` | Lookback window (hours) |
| `horizon` | `24` | Forecast horizon (hours) |
| `n_stacks` | `3` | Number of hierarchical stacks |
| `n_blocks_per_stack` | `1` | Blocks per stack |
| `pooling_kernel_sizes` | `[8, 4, 1]` | Input max-pooling kernel sizes per stack |
| `n_freq_downsample` | `[8, 4, 1]` | Basis downsampling rates for interpolation per stack |
| `hidden_size` | `512` | Dense layer width |
| `num_mlp_layers` | `2` | Dense layers per block |
| `dropout` | `0.1` | Dropout rate |
| `lr` | `1e-3` | Initial learning rate (AdamW) |
| `patience` | `20` | Early stopping patience |
| `loss` | `"mae"` | Training loss objective (`mae`, `mse`, `huber`) |

---
### Hyperparameter guidance

| Parameter | Conservative | Default | Aggressive |
|-----------|-------------|---------|------------|
| `hidden_size` | 256 | **512** | 1024 |
| `num_mlp_layers` | 1 | **2** | 3–4 |
| `dropout` | 0.0 | **0.1** | 0.3 |
| `lr` | 5e-4 | **1e-3** | 3e-3 |
| `weight_decay` | 0.0 | **1e-4** | 1e-3 |
| `patience` | 10 | **20** | 50 |
| `batch_size` | 32 | **64** | 256 |

---

### Tuning the hierarchical structure

| Scenario | Recommended change |
|----------|--------------------|
| Prominent weekly cycle | Keep `pooling_kernel_sizes=[8, 4, 1]` (default) |
| Very noisy data | Increase coarse pooling, e.g. `[16, 8, 1]` |
| Short horizon (≤12 h) | Reduce `n_freq_downsample` to `[4, 2, 1]` |
| Long horizon (>48 h) | Increase `n_freq_downsample` to `[16, 8, 1]` |
| Underfitting | Add `n_blocks_per_stack=2` or increase `hidden_size` |
| Overfitting | Raise `dropout` (0.2–0.3) or lower `hidden_size` |

### Loss function selection

| Loss | When to use |
|------|-------------|
| `mae` (default) | Robust to demand spikes; preferred for electricity |
| `mse` | Penalises large errors harder; useful if peak accuracy matters |
| `huber` | Compromise — smooth near zero, linear for outliers |

---

## References

* **Paper Reference:** Challu et al. (2023), *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*, AAAI 2023.
* **Skill Reference:** `neural-networks-forecasting` skill (`architectures/nhits.md`).
