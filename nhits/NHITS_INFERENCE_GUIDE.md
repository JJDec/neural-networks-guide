# N-HiTS Model Inference & Output Explainability Guide

This guide details the complete operational lifecycle for performing inference with the **N-HiTS (Neural Hierarchical Interpolation for Time Series)** model. It covers setup prerequisites, input data transformation pipelines, execution procedures, inverse scaling, and real-time output explainability techniques.

---

## 1. Prerequisites & Environment Requirements

Before running inference with a trained N-HiTS model, ensure the following requirements are met:

### 1.1 Trained Artifacts & Configuration
1. **Model Checkpoint File (`best_model.pt`):** Saved PyTorch model state dictionary containing parameters for all hierarchical blocks (MLP stacks, pooling operators, backcast/forecast heads).
2. **Model Configuration (`NHiTSConfig`):** Hyperparameter definitions matching the trained checkpoint:
   * Look-back window length $L$ (e.g., $168$ hours)
   * Forecast horizon $H$ (e.g., $24$ hours)
   * Covariate counts: $C_{\text{hist}}$ (historical) and $C_{\text{fut}}$ (future known)
   * Architecture sizes: `hidden_size`, `num_mlp_layers`, `n_stacks`, `pooling_kernel_sizes`, `n_freq_downsample`
3. **Scaling Parameters (`scale_stats.json` or tuple):** Mean $\mu$ and standard deviation $\sigma$ computed **strictly on the training dataset target values** used to standardise input target series and inverse-scale model outputs.

### 1.2 Execution Environment
* **Python Runtime:** Python 3.10+
* **Framework Dependencies:** PyTorch (`torch >= 2.0`), NumPy (`numpy >= 1.22`)
* **Inference Mode Settings:**
  * Model set to `model.eval()` to freeze LayerNorm stats and disable Dropout.
  * Inference block executed within `with torch.no_grad():` to disable autograd memory overhead.

---

## 2. Input Data Preparation Pipeline

N-HiTS expects inputs organized into three distinct tensors with a batch dimension $B$. For single-sample real-time inference, $B = 1$.

```
Raw Past Target Data (L,)         Historical Covariates (L, C_hist)        Future Covariates (H, C_fut)
       │                                       │                                      │
       ▼                                       ▼                                      ▼
[Standardize (y - μ)/σ]              [Verify Shape & NaNs]                 [Verify Shape & NaNs]
       │                                       │                                      │
       ▼                                       ▼                                      ▼
Tensor (1, L)                         Tensor (1, L, C_hist)                 Tensor (1, H, C_fut)
       └───────────────────────────────────────┼──────────────────────────────────────┘
                                               ▼
                                      To Device (CPU/CUDA)
```

### 2.1 Preparing Past Target Values ($y_{1:L}$)
* **Input Window:** Extract exactly $L$ consecutive past target observations (e.g., the last 7 days of hourly electricity demand = 168 hours).
* **Missing Value Validation:** Ensure no `NaN` or `Inf` values exist (impute with forward fill or linear interpolation if required).
* **Standardization:**
  $$y_{\text{scaled}} = \frac{y_{\text{raw}} - \mu}{\sigma + 1e-8}$$
* **Tensor Formatting:** Convert array to `float32` and add batch dimension: shape `(1, L)`.

### 2.2 Preparing Historical Covariates ($\mathbf{X}_{\text{hist}}$)
* **Look-back Window Matching:** Align historical covariates with the exact same time indices as $y_{1:L}$.
* **Features:** Calendar sine/cosine encodings (e.g., `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
* **Note:** Each stack applies `MaxPool1d` to both the target and historical covariates independently using its pooling kernel — shape validation must pass before pooling.
* **Tensor Formatting:** Convert array to `float32` with shape `(1, L, C_hist)`.

### 2.3 Preparing Future Covariates ($\mathbf{X}_{\text{fut}}$)
* **Horizon Window Matching:** Extract known future covariates for the forecast horizon steps $t \in [L+1, L+H]$ (e.g., future hour-of-day, day-of-week).
* **Tensor Formatting:** Convert array to `float32` with shape `(1, H, C_fut)`.

---

## 3. End-to-End Inference Execution

The standard inference workflow consists of model instantiation, parameter loading, forward pass execution, and output inverse-scaling back to original domain units.

### 3.1 Simple 24-Hour Forecast in Original MW Units

Use `predict_next_24h` for one-step inference from a historical dataframe window:

```python
import numpy as np
import pandas as pd

from nhits.config import NHiTSConfig
from nhits.datasets.electricity_dataset import generate_synthetic_electricity
from nhits.inference.predict import load_model, predict_next_24h

# 1. Initialize config and load model from checkpoint
cfg = NHiTSConfig()
model = load_model(cfg, checkpoint_path="checkpoints/nhits/best_model.pt", device="cpu")

# 2. Prepare recent data window (168 hours look-back)
df = generate_synthetic_electricity()
covariate_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]

past_window = df.iloc[-168:]
past_demand_mw = past_window["demand_mw"].values
past_covariates = past_window[covariate_cols].values

# 3. Future covariates for the next 24 hours (known ahead of time)
future_covariates = df[covariate_cols].iloc[-24:].values

# 4. Target scaling statistics (obtained during training or from dataset loader)
scale_mean = 3000.0  # MW
scale_std = 600.0   # MW

# 5. Predict next 24 hours in MW
forecast_mw = predict_next_24h(
    model=model,
    past_demand_mw=past_demand_mw,
    past_covariates=past_covariates,
    future_covariates=future_covariates,
    scale_mean=scale_mean,
    scale_std=scale_std,
    device="cpu",
)

print("24-Hour Forecast (MW):")
for hour, mw in enumerate(forecast_mw, start=1):
    print(f"  Hour {hour:2d}: {mw:.1f} MW")
```

### 3.2 Python Reference Implementation (Low-Level)

```python
import numpy as np
import torch
from nhits.config import NHiTSConfig
from nhits.models.nhits import NHiTSModel

def load_nhits_checkpoint(cfg: NHiTSConfig, checkpoint_path: str, device: str = "cpu") -> NHiTSModel:
    """Load model checkpoint in evaluation mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = NHiTSModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_nhits_inference(
    model: NHiTSModel,
    past_target_raw: np.ndarray,      # Shape: (L,) in original units (e.g., MW)
    hist_covariates: np.ndarray,      # Shape: (L, C_hist)
    future_covariates: np.ndarray,    # Shape: (H, C_fut)
    scale_mean: float,
    scale_std: float,
    device: str = "cpu",
) -> np.ndarray:
    """Execute forward pass and return predictions in original physical units."""
    # 1. Standardize target
    past_scaled = (past_target_raw - scale_mean) / (scale_std + 1e-8)

    # 2. Convert to PyTorch Tensors (B=1)
    past_t = torch.tensor(past_scaled, dtype=torch.float32, device=device).unsqueeze(0)       # (1, L)
    hist_c = torch.tensor(hist_covariates, dtype=torch.float32, device=device).unsqueeze(0)   # (1, L, C_hist)
    fut_c  = torch.tensor(future_covariates, dtype=torch.float32, device=device).unsqueeze(0) # (1, H, C_fut)

    # 3. Model Forward Pass (doubly-residual accumulation across all blocks)
    preds_scaled = model(past_t, hist_c, fut_c).cpu().numpy().squeeze(0)  # (H,)

    # 4. Inverse Scaling to original units (e.g. MW)
    preds_raw = (preds_scaled * scale_std) + scale_mean
    return preds_raw
```

### 3.3 Batch Tensor Inference

For batch inference on PyTorch tensors:

```python
import torch

from nhits.config import NHiTSConfig
from nhits.inference.predict import load_model, predict

cfg = NHiTSConfig()
model = load_model(cfg, device="cuda" if torch.cuda.is_available() else "cpu")

# Batch shapes:
# past_target: (B, L=168)
# hist_covs:   (B, L=168, C_hist=4)
# future_covs: (B, H=24,  C_fut=4)

past_target = torch.randn(32, 168)
hist_covs = torch.randn(32, 168, 4)
future_covs = torch.randn(32, 24, 4)

preds_scaled = predict(model, past_target, hist_covs, future_covs)
# preds_scaled shape: (32, 24)
```

---

## 4. Model Output Explainability During Inference

Explainability transforms raw numeric forecasts into actionable insights. Because N-HiTS features an explicit **additive hierarchical structure**, inference outputs can be decomposed in real time across frequency bands.

```
                                         ┌──► Stack 0 — Coarse (pool=8) : Trend contribution    ─┐
Total Inference Output (y_hat) = Sum of  ├──► Stack 1 — Medium (pool=4) : Seasonal contribution ─┼──► Additive Sum
                                         └──► Stack 2 — Fine   (pool=1) : Residual contribution ─┘
```

### 4.1 Per-Stack Forecast Decomposition

N-HiTS's total output is a sum of per-block forecasts via doubly-residual accumulation:
$$\hat{y}_{\text{final}} = \sum_{b=0}^{N-1} \hat{y}_b$$

During inference, we can capture each block's individual contribution to explain **how much of the forecast is driven by coarse trends vs. medium seasonality vs. fine-grained residuals**:

```python
@torch.no_grad()
def explain_stack_components(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
    scale_mean: float,
    scale_std: float,
) -> dict[str, np.ndarray]:
    """Decompose forecast into per-stack frequency components."""
    target_residual = past_t.clone()
    components_scaled = []

    for block in model.blocks:
        backcast_b, forecast_b = block(target_residual, hist_c, fut_c)
        target_residual = target_residual - backcast_b
        components_scaled.append(forecast_b.cpu().numpy().squeeze(0))  # (H,)

    # Rescale additive components to original units
    # NOTE: only the mean offset applies to the total forecast, not individual components
    total_scaled = sum(components_scaled)
    total_mw = total_scaled * scale_std + scale_mean

    return {
        "total_forecast":     total_mw,
        "coarse_trend":       components_scaled[0] * scale_std,  # Stack 0 (pool=8)
        "medium_seasonality": components_scaled[1] * scale_std,  # Stack 1 (pool=4)
        "fine_residual":      components_scaled[2] * scale_std,  # Stack 2 (pool=1)
    }
```

### 4.2 Basis Coefficient Inspection

The raw forecast coefficients before interpolation expose the signal at compressed resolution. A Stack 0 coefficient vector of only 3 values (for a 24-step horizon at downsample=8) is the model's latent "sketch" of the future:

```python
@torch.no_grad()
def inspect_basis_coefficients(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
) -> list[dict]:
    """Intercept compressed forecast coefficients from each block before interpolation."""
    target_residual = past_t.clone()
    coeff_info = []

    for i, block in enumerate(model.blocks):
        B = past_t.size(0)

        # Apply pooling to target
        target_3d = target_residual.unsqueeze(1)
        target_pooled = block.pool(target_3d).squeeze(1)

        # Apply pooling to historical covariates
        hist_3d = hist_c.transpose(1, 2)
        hist_pooled = block.pool(hist_3d).reshape(B, -1)

        # Flatten future covariates
        fut_flat = fut_c.reshape(B, -1)

        # MLP forward
        x_in = torch.cat([target_pooled, hist_pooled, fut_flat], dim=-1)
        feat = block.mlp(x_in)

        fwd_coeffs = block.forecast_head(feat)   # (B, ⌈H/k⌉) — compressed coefficients
        bck_coeffs = block.backcast_head(feat)   # (B, ⌈L/k⌉) — compressed backcast

        coeff_info.append({
            "stack": i,
            "forecast_coeffs":  fwd_coeffs.cpu().numpy().squeeze(0),
            "backcast_coeffs":  bck_coeffs.cpu().numpy().squeeze(0),
            "compression_ratio": f"{fwd_coeffs.size(1)}→24",
        })

        # Advance residual
        backcast_b, _ = block(target_residual, hist_c, fut_c)
        target_residual = target_residual - backcast_b

    return coeff_info
```

Compression ratio per stack:

| Stack | `n_freq_downsample` | `fwd_coeff_size` | Compression |
| :---- | :------------------ | ---------------: | ----------: |
| Coarse (0) | 8 | 3 | 8× |
| Medium (1) | 4 | 6 | 4× |
| Fine   (2) | 1 | 24 | 1× (none) |

### 4.3 Residual Energy Tracking

Monitoring how much signal remains in `target_residual` after each block reveals which stack is doing most of the modelling work. A near-zero residual after Stack 0 indicates a strongly trend-dominated series:

```python
@torch.no_grad()
def track_residual_energy(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
) -> list[dict]:
    """Track residual MSE energy after each block to reveal modelling effort distribution."""
    residual = past_t.clone()
    residual_log = []

    for i, block in enumerate(model.blocks):
        backcast_b, _ = block(residual, hist_c, fut_c)
        residual = residual - backcast_b
        residual_energy = residual.pow(2).mean().item()
        residual_log.append({"stack": i, "residual_mse": residual_energy})
        print(f"  Stack {i} residual MSE: {residual_energy:.6f}")

    return residual_log
```

### 4.4 Look-Back Timestep Importance via Gradient Saliency

To explain *which specific past hours* influenced a particular horizon prediction $\hat{y}_{h}$, compute the gradient of the predicted value with respect to the input target:

$$\text{Attribution}(t) = \left| \frac{\partial \hat{y}_h}{\partial y_t} \cdot y_t \right|$$

```python
def explain_input_saliency(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L) requires_grad=True
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
    target_horizon_step: int = 0,
) -> np.ndarray:
    """Compute gradient-based attribution for a specific forecast horizon step."""
    past_t = past_t.clone().detach().requires_grad_(True)

    # Forward pass without torch.no_grad() for saliency calculation
    preds = model(past_t, hist_c, fut_c)
    target_val = preds[0, target_horizon_step]

    # Compute gradients
    target_val.backward()

    saliency = (past_t.grad.abs() * past_t.abs()).cpu().numpy().squeeze(0)
    # Normalize saliency to sum to 1
    saliency /= (saliency.sum() + 1e-8)
    return saliency
```

### 4.5 Summary Table of Inference & Explainability Steps

| Step | Operation | Key Function | Output Shape | Explainability Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Validation** | Check missing values & ranges | `np.isnan(data).any()` | `(L,)` | Input quality validation |
| **2. Standardize** | Scale target window | `(y - mean) / std` | `(1, L)` | Zero-centered standard inputs |
| **3. Forward Pass** | Run N-HiTS doubly-residual loop | `model(past, hist, fut)` | `(1, H)` | Scaled horizon prediction |
| **4. Rescaling** | Inverse scaling to original units | `(pred * std) + mean` | `(H,)` | Physical forecast (MW) |
| **5. Decomposition** | Split per-stack contributions | Block-by-block loop | `(H,)` × 3 | Trend / seasonal / residual ratio |
| **6. Coefficients** | Intercept compressed basis sketches | `block.forecast_head(feat)` | `(⌈H/k⌉,)` | Latent signal at each frequency band |
| **7. Residual energy** | Track modelling effort per stack | `residual.pow(2).mean()` | scalar × 3 | Stack-wise signal absorption |
| **8. Saliency** | Gradient backpropagation | `grad(y_hat / y_past)` | `(L,)` | Hourly input importance weights |

---

## 5. References

* **Model Architecture Guide:** [NHITS_MODEL_GUIDE.md](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/NHITS_MODEL_GUIDE.md)
* **Code Reference:** `nhits/inference/predict.py`
* **Paper Reference:** Challu et al. (2023), *N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting*, AAAI 2023.
