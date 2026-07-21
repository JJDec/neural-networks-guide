# TiDE (Time-series Dense Encoder) Architecture & Interpretability Guide

## 1. Overview of TiDE

**TiDE (Time-series Dense Encoder)** is a lightweight, high-performance deep learning architecture for long-term multivariate time-series forecasting, introduced by Das et al. (2023) in *Long-term Forecasting with TiDE* ([arXiv:2304.08424](https://arxiv.org/abs/2304.08424)).

Unlike Transformer-based architectures (e.g., Autoformer, Informer, PatchTST) that rely on self-attention mechanisms with quadratic temporal complexity, TiDE is composed **entirely of dense residual Multi-Layer Perceptron (MLP) blocks**. 

### Key Characteristics & Advantages
* **No Self-Attention, Recurrence, or Convolutions:** Replaces self-attention with dense residual blocks, achieving competitive or superior accuracy with significantly lower computational complexity ($10\times$ to $100\times$ faster training and inference).
* **Direct Multi-Step Forecasting:** Predicts the entire forecast horizon $H$ simultaneously in a single forward pass, avoiding auto-regressive error accumulation.
* **Unified Covariate Integration:** Efficiently incorporates past target values, historical covariates (e.g., ambient temperature, solar radiation), future known covariates (e.g., day-of-week, hour-of-day), and static metadata.
* **Explicit Linear-Nonlinear Decomposition:** Combines a deep nonlinear encoder-decoder pipeline with an explicit linear global skip connection, ensuring strong baseline performance while capturing complex nonlinear relationships.

---

## 2. Architecture & Pipeline

The TiDE pipeline transforms a look-back window of length $L$ into a future prediction window of length $H$.

```
Past Target (B, L) ──┐
Hist. Covs  (B, L, C_hist) ──┼──> [Feature Projection] ──> [Dense Encoder] ──> Latent Vector z (B, hidden_size)
Future Covs (B, H, C_fut) ──┘                                                      │
                                                                                   ▼
                                                                           [Dense Decoder]
                                                                                   │
                                                                                   ▼
                                                                           (B, H, temporal_hidden)
                                                                                   │
                                                                                   ▼
Past Target (B, L) ──> [Global Skip Connection] ───────────────────────> [Temporal Decoder]
                                │                                                  │
                                ▼                                                  ▼
                        Skip Output (B, H) ───────────────────────────> Forecast Output (B, H)
```

### Component Breakdown

#### 1. Residual Block (`ResidualBlock`)
The fundamental building block of TiDE:
$$\text{Output} = \text{LayerNorm}(\text{Linear}_2(\text{Dropout}(\text{GELU}(\text{Linear}_1(x)))) + \text{Skip}(x))$$
* **Linear Projection:** Expands and contracts hidden dimensions.
* **Activation & Regularization:** GELU activation with optional Dropout and LayerNorm.
* **Residual Connection:** Maps input to output directly (with linear dimension matching if input/output dims differ).

#### 2. Feature Projection (`FeatureProjection`)
Flattens and concatenates all inputs across time and feature dimensions into a single vector of size:
$`\text{flat\_dim} = L + (L \times C_{\text{hist}}) + (H \times C_{\text{fut}})`$

Passes the concatenated vector through a `ResidualBlock` to project it into the model's core hidden dimension `hidden_size`.

#### 3. Dense Encoder (`TiDEEncoder`)
A stack of $N_e$ `ResidualBlock` modules operating on the projected representation to extract a compact, global latent vector $`z \in \mathbb{R}^{\text{hidden\_size}}`$.

#### 4. Dense Decoder (`TiDEDecoder`)
A stack of $N_d$ `ResidualBlock` modules that decodes the global latent vector $z$ into per-step temporal features of shape $`(B, H, \text{temporal\_hidden})`$.

#### 5. Temporal Decoder (`TemporalDecoder`)
Applies a shared dense layer across each forecast horizon step $h \in [1, H]$, combining the decoder output at step $h$ with the future known covariates at step $h$:
$`\hat{y}_{\text{deep}, h} = \text{Linear}(\text{DecoderOut}_h \,||\, \text{FutureCov}_h)`$

#### 6. Global Linear Skip Connection (`Linear`)
A direct linear transformation mapping the raw look-back target values directly to the horizon:
$`\hat{y}_{\text{skip}} = W_{\text{skip}} \, y_{1:L} + b_{\text{skip}}`$
The final forecast is the sum of the linear skip prediction and the deep temporal decoder output:
$`\hat{y}_{\text{final}} = \hat{y}_{\text{skip}} + \hat{y}_{\text{deep}}`$

---

## 3. Model Interpretability in TiDE

Model interpretability is a core strength of TiDE. Because TiDE relies on linear skip layers and feed-forward MLP blocks rather than dynamic attention mechanisms, interpretability can be extracted cleanly through structural decomposition, weight inspection, and gradient attribution.

### 3.1 Global Linear Skip Weight Matrix Analysis ($W_{\text{skip}}$)

The global skip connection is a learnable matrix $W_{\text{skip}} \in \mathbb{R}^{H \times L}$. Each element $W_{\text{skip}}[h, t]$ represents the direct linear weight assigned to past timestep $t \in [1, L]$ when forecasting horizon step $h \in [1, H]$.

* **Temporal Lag Attribution:** Plotting $W_{\text{skip}}$ as a heatmap directly reveals periodic dependencies (e.g., daily 24h peaks, weekly 168h peaks).
* **Autoregressive Pattern Extraction:** The rows of $W_{\text{skip}}$ expose how the model relies on immediate past values vs. seasonal past values for short-term vs. long-term horizon steps.

#### Code Snippet: Visualizing Skip Weights
```python
import matplotlib.pyplot as plt
import torch

def plot_skip_weights(model, save_path="outputs/tide/skip_weights.png"):
    # Extract linear skip weights (shape: H, L)
    w_skip = model.skip.weight.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.imshow(w_skip, aspect="auto", cmap="Blues", origin="lower")
    plt.colorbar(label="Weight Intensity")
    plt.xlabel("Past Look-back Timestep (t)")
    plt.ylabel("Forecast Horizon Step (h)")
    plt.title("TiDE Global Skip Weight Matrix (W_skip)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
```

### 3.2 Linear vs. Non-linear Component Output Decomposition

TiDE naturally decomposes forecasts into two distinct additive components:
1. **Linear Skip Component ($`\hat{y}_{\text{skip}}`$):** Captures auto-regressive baseline trends, seasonality, and persistent levels.
2. **Deep Residual Component ($`\hat{y}_{\text{deep}}`$):** Captures complex nonlinear feature interactions, covariate effects, and non-stationary pattern shifts.

By measuring the relative norm or variance of $`\hat{y}_{\text{skip}}`$ vs. $`\hat{y}_{\text{deep}}`$, practitioners can quantify the proportion of the forecast driven by simple trend/seasonality versus complex feature interactions:
$`\text{Linear Contribution Ratio} = \frac{\|\hat{y}_{\text{skip}}\|_2}{\|\hat{y}_{\text{skip}}\|_2 + \|\hat{y}_{\text{deep}}\|_2}`$

### 3.3 Feature Projection Input Attribution

In `FeatureProjection`, the input layer receives concatenated vectors of:
* Past Targets: $[y_1, y_2, \dots, y_L]$
* Historical Covariates: $`[\mathbf{X}_{\text{hist}, 1}, \dots, \mathbf{X}_{\text{hist}, L}]`$
* Future Covariates: $`[\mathbf{X}_{\text{fut}, 1}, \dots, \mathbf{X}_{\text{fut}, H}]`$

By computing the mean absolute weight ($\text{MAW}$) or Frobenius norm of the weights corresponding to each feature channel in the first linear layer (`feature_proj.proj.fc1`), we obtain global feature importance scores:

$`\text{Importance}(f) = \frac{1}{K_f} \sum_{i \in \text{indices}(f)} \|W_{\text{proj}}[:, i]\|_1`$

This identifies whether historical targets, calendar features, or ambient environmental covariates contribute most to the deep representation.

### 3.4 Gradient-Based Attribution (Saliency Maps & Integrated Gradients)

Because TiDE uses smooth, fully differentiable activations (GELU) and linear projections, input attribution can be computed using standard gradient backpropagation:
$`\text{Saliency}(x_i) = \left| \frac{\partial \hat{y}_h}{\partial x_i} \cdot x_i \right|`$

Unlike Transformer attention matrices which suffer from attention-weight unreliability, gradient attributions in dense residual networks accurately reflect causal input sensitivity.

---

## 4. PyTorch Implementation Reference

Below is the standard modular PyTorch implementation of TiDE, following the guidelines from the `neural-networks-forecasting` skill.

```python
import torch
import torch.nn as nn
from torch import Tensor

class ResidualBlock(nn.Module):
    """Dense residual block: Linear -> GELU -> Dropout -> Linear + Skip + LayerNorm."""
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim, bias=False) if input_dim != output_dim else nn.Identity()
        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        out = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.norm(out + residual)

class FeatureProjection(nn.Module):
    """Projects concatenated raw target and covariate features to hidden_size."""
    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        hidden_size: int,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        flat_dim = input_len + (input_len * num_hist_covariates) + (horizon * num_future_covariates)
        self.proj = ResidualBlock(flat_dim, hidden_size, dropout=dropout, use_layer_norm=use_layer_norm)

    def forward(self, past_target: Tensor, hist_covs: Tensor, future_covs: Tensor) -> Tensor:
        B = past_target.size(0)
        flat = torch.cat([
            past_target,
            hist_covs.reshape(B, -1),
            future_covs.reshape(B, -1)
        ], dim=-1)
        return self.proj(flat)

class TiDEModel(nn.Module):
    """Full TiDE Architecture: FeatureProjection -> Encoder -> Decoder -> TemporalDecoder + Skip."""
    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        hidden_size: int = 512,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        temporal_decoder_hidden: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.feature_proj = FeatureProjection(
            input_len, horizon, num_hist_covariates, num_future_covariates, hidden_size, dropout, use_layer_norm
        )
        self.encoder = nn.Sequential(*[
            ResidualBlock(hidden_size, hidden_size, dropout=dropout, use_layer_norm=use_layer_norm)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder maps hidden_size -> (horizon * temporal_decoder_hidden)
        dec_layers = []
        in_dim = hidden_size
        out_dim = horizon * temporal_decoder_hidden
        for i in range(num_decoder_layers):
            layer_out = out_dim if i == num_decoder_layers - 1 else hidden_size
            dec_layers.append(ResidualBlock(in_dim, layer_out, dropout=dropout, use_layer_norm=use_layer_norm))
            in_dim = layer_out
        self.decoder = nn.Sequential(*dec_layers)
        
        self.temporal_decoder = nn.Linear(temporal_decoder_hidden + num_future_covariates, 1)
        self.skip = nn.Linear(input_len, horizon)
        self.horizon = horizon
        self.temporal_decoder_hidden = temporal_decoder_hidden

    def forward(self, past_target: Tensor, hist_covs: Tensor, future_covs: Tensor) -> Tensor:
        # 1. Projection & Encoding
        projected = self.feature_proj(past_target, hist_covs, future_covs)
        latent = self.encoder(projected)
        
        # 2. Decoding
        dec_out = self.decoder(latent).view(-1, self.horizon, self.temporal_decoder_hidden)
        
        # 3. Temporal Decoding (per step forecast)
        temporal_in = torch.cat([dec_out, future_covs], dim=-1)
        deep_forecast = self.temporal_decoder(temporal_in).squeeze(-1)
        
        # 4. Global Skip Additive Combination
        skip_forecast = self.skip(past_target)
        return deep_forecast + skip_forecast
```

---

## 5. Recommended Workflow & Hyperparameters

### Hyperparameter Recommendations

| Parameter | Recommended Range | Description |
| :--- | :--- | :--- |
| `hidden_size` | $256 - 1024$ | Width of dense residual layers |
| `num_encoder_layers` | $2 - 4$ | Depth of global encoder stack |
| `num_decoder_layers` | $1 - 3$ | Depth of decoder stack |
| `temporal_decoder_hidden` | $32 - 128$ | Per-step temporal MLP hidden size |
| `dropout` | $0.0 - 0.2$ | Dropout rate within residual blocks |
| `learning_rate` | $1e-4 - 1e-3$ | AdamW initial learning rate |
| `loss` | MAE / MSE / Huber | Huber loss offers robustness to outliers |

### Common Pitfalls to Avoid

1. **Random Train/Test Splitting:** Time-series data must be split chronologically (e.g., 70% train, 15% validation, 15% test) to prevent temporal data leakage.
2. **Missing Feature Normalization:** Target values and non-cyclical covariates should be normalized (e.g., `StandardScaler` fitted only on training data).
3. **Autoregressive Rollout at Inference:** Predict the entire horizon $H$ simultaneously rather than feeding predictions step-by-step.
4. **Disabling Global Skip Layer:** Leaving out `self.skip` removes the baseline linear trend mapping, causing degradation on linear/seasonal series.

---

## 6. Summary & References

TiDE provides a powerful alternative to Transformer models for time-series forecasting. By combining dense residual networks, unified covariate processing, and additive linear skip connections, TiDE achieves state-of-the-art accuracy with fast execution and clear model interpretability.

* **Paper:** Das, A., Kong, W., Sen, R., & Rajan, A. (2023). *Long-term Forecasting with TiDE: Time-series Dense Encoder*. arXiv preprint arXiv:2304.08424.
* **Skill Reference:** `neural-networks-forecasting` skill (architectures/tide.md).
