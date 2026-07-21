# TiDE Model Inference & Output Explainability Guide

This guide details the complete operational lifecycle for performing inference with the **TiDE (Time-series Dense Encoder)** model. It covers setup prerequisites, input data transformation pipelines, execution procedures, inverse scaling, and real-time output explainability techniques.

---

## 1. Prerequisites & Environment Requirements

Before running inference with a trained TiDE model, ensure the following requirements are met:

### 1.1 Trained Artifacts & Configuration
1. **Model Checkpoint File (`best_model.pt`):** Saved PyTorch model state dictionary containing parameters for all dense residual blocks, feature projection layers, temporal decoders, and global skip matrices.
2. **Model Configuration (`TiDEConfig`):** Hyperparameter definitions matching the trained checkpoint:
   * Look-back window length $L$ (e.g., $168$ hours)
   * Forecast horizon $H$ (e.g., $24$ hours)
   * Covariate counts: $C_{\text{hist}}$ (historical) and $C_{\text{fut}}$ (future known)
   * Architecture sizes: `hidden_size`, `num_encoder_layers`, `num_decoder_layers`, `temporal_decoder_hidden`
3. **Scaling Parameters (`scale_stats.json` or tuple):** Mean $\mu$ and standard deviation $\sigma$ computed **strictly on the training dataset target values** used to standardise input target series and inverse-scale model outputs.

### 1.2 Execution Environment
* **Python Runtime:** Python 3.10+
* **Framework Dependencies:** PyTorch (`torch >= 2.0`), NumPy (`numpy >= 1.22`)
* **Inference Mode Settings:**
  * Model set to `model.eval()` to freeze LayerNorm stats and disable Dropout.
  * Inference block executed within `with torch.no_grad():` to disable autograd memory overhead.

---

## 2. Input Data Preparation Pipeline

TiDE expects inputs organized into three distinct tensors with a batch dimension $B$. For single-sample real-time inference, $B = 1$.

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
* **Features:** Exogenous measurements (e.g., historical temperature, pressure) or calendar sine/cosine encodings.
* **Tensor Formatting:** Convert array to `float32` with shape `(1, L, C_hist)`.

### 2.3 Preparing Future Covariates ($\mathbf{X}_{\text{fut}}$)
* **Horizon Window Matching:** Extract known future covariates for the forecast horizon steps $t \in [L+1, L+H]$ (e.g., future hour-of-day, day-of-week, holiday indicators).
* **Tensor Formatting:** Convert array to `float32` with shape `(1, H, C_fut)`.

---

## 3. End-to-End Inference Execution

The standard inference workflow consists of model instantiation, parameter loading, forward pass execution, and output inverse-scaling back to original domain units.

### 3.1 Python Reference Implementation

```python
import path
import numpy as np
import torch
from tide.config import TiDEConfig
from tide.models.tide import TiDEModel

def load_tide_checkpoint(cfg: TiDEConfig, checkpoint_path: str, device: str = "cpu") -> TiDEModel:
    """Load model checkpoint in evaluation mode."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TiDEModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def run_tide_inference(
    model: TiDEModel,
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
    past_t = torch.tensor(past_scaled, dtype=torch.float32, device=device).unsqueeze(0)      # (1, L)
    hist_c = torch.tensor(hist_covariates, dtype=torch.float32, device=device).unsqueeze(0)  # (1, L, C_hist)
    fut_c = torch.tensor(future_covariates, dtype=torch.float32, device=device).unsqueeze(0) # (1, H, C_fut)
    
    # 3. Model Forward Pass
    preds_scaled = model(past_t, hist_c, fut_c).cpu().numpy().squeeze(0)  # (H,)
    
    # 4. Inverse Scaling to original units (e.g. MW)
    preds_raw = (preds_scaled * scale_std) + scale_mean
    return preds_raw
```

---

## 4. Model Output Explainability During Inference

Explainability transforms raw numeric forecasts into actionable insights. Because TiDE features an explicit additive structure, inference outputs can be decomposed in real time.

```
                               ┌──> Linear Skip Forecast  (y_skip) ──┐
Total Inference Output (y_hat) ┤                                     ├──> Additive Contribution Split
                               └──> Deep Temporal Forecast (y_deep) ──┘
```

### 4.1 Real-Time Linear vs. Deep Component Decomposition

TiDE's total output is computed as:
$$\hat{y}_{\text{final}} = \hat{y}_{\text{skip}} + \hat{y}_{\text{deep}}$$

During inference, we can compute both branches separately to explain **how much of the forecast is driven by baseline auto-regressive trends vs. deep nonlinear pattern learning**:

```python
@torch.no_grad()
def explain_forecast_components(
    model: TiDEModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
    scale_mean: float,
    scale_std: float,
) -> dict[str, np.ndarray]:
    """Decompose forecast into linear baseline trend and deep non-linear adjustments."""
    # Linear Skip Prediction
    skip_out_scaled = model.skip(past_t).cpu().numpy().squeeze(0)  # (H,)
    
    # Full Model Prediction
    full_out_scaled = model(past_t, hist_c, fut_c).cpu().numpy().squeeze(0)  # (H,)
    
    # Deep Network Contribution
    deep_out_scaled = full_out_scaled - skip_out_scaled
    
    # Rescale to original units
    skip_mw = skip_out_scaled * scale_std + scale_mean
    deep_mw = deep_out_scaled * scale_std
    total_mw = full_out_scaled * scale_std + scale_mean
    
    return {
        "total_forecast": total_mw,
        "linear_skip_baseline": skip_mw,
        "deep_nonlinear_adjustment": deep_mw,
    }
```

### 4.2 Look-Back Timestep Importance via Gradient Saliency

To explain *which specific past hours* influenced a particular horizon prediction $\hat{y}_{h}$, compute the gradient of the predicted value with respect to the input target:

$$\text{Attribution}(t) = \left| \frac{\partial \hat{y}_h}{\partial y_t} \cdot y_t \right|$$

```python
def explain_input_saliency(
    model: TiDEModel,
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

### 4.3 Summary Table of Inference & Explainability Steps

| Step | Operation | Key Function | Output Shape | Explainability Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Validation** | Check missing values & ranges | `np.isnan(data).any()` | `(L,)` | Input quality validation |
| **2. Standardize** | Scale target window | `(y - mean) / std` | `(1, L)` | Zero-centered standard inputs |
| **3. Forward Pass** | Run TiDE model | `model(past, hist, fut)` | `(1, H)` | Scaled horizon prediction |
| **4. Rescaling** | Inverse scaling to original units | `(pred * std) + mean` | `(H,)` | Physical forecast (MW, $, etc.) |
| **5. Decomposition** | Split linear skip vs deep MLP | `model.skip(past)` | `(H,)` | Baseline vs deep interaction ratio |
| **6. Saliency** | Gradient backpropagation | `grad(y_hat / y_past)` | `(L,)` | Hourly input importance weights |

---

## 5. References

* **Model Architecture Guide:** [TIDE_MODEL_GUIDE.md](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/TIDE_MODEL_GUIDE.md)
* **Code Reference:** `tide/inference/predict.py`
* **Paper Reference:** Das et al. (2023), *Long-term Forecasting with TiDE*, arXiv:2304.08424.
