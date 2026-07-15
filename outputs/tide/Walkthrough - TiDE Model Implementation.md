# TiDE Model for 24-Hour Electricity Demand Forecasting

Build a production-quality **TiDE (Time-series Dense Encoder)** implementation in PyTorch for forecasting the next 24 hours of electricity demand. The model uses dense residual MLP blocks — no attention, no recurrence, no convolutions.

## Problem Analysis

| Property | Value |
|---|---|
| Task | Multivariate → univariate (electricity demand) |
| Horizon | 24 steps (hours) |
| Input window | 168 steps (7 days of hourly history) |
| Historical covariates | Hour-of-day, day-of-week, month, temperature (if available) |
| Future covariates | Hour-of-day, day-of-week, month (calendar features are always known) |
| Static covariates | None (single series dataset) |
| Forecasting type | Deterministic |
| Loss | MAE (robust to outliers in demand data) |

## Proposed Changes

---

### Synthetic Dataset (electricity demand)

Uses a synthetic dataset that mimics realistic electricity demand patterns (daily seasonality + weekly seasonality + trend + noise), so the model runs out-of-the-box without external data downloads.

#### [NEW] `tide/datasets/electricity_dataset.py`
- `ElectricityDemandDataset` — generates synthetic hourly demand or loads a CSV
- `WindowDataset` — converts a time series into supervised sliding windows
- `build_dataloaders()` — returns train / val / test `DataLoader`s with chronological splits

---

### TiDE Model

#### [NEW] `tide/models/tide.py`
- `ResidualBlock` — Linear → GELU → Linear + skip connection + optional LayerNorm
- `FeatureProjection` — projects concatenated inputs to a fixed hidden dimension
- `TiDEEncoder` — stack of `ResidualBlock`s
- `TiDEDecoder` — stack of `ResidualBlock`s
- `TemporalDecoder` — per-step dense layer mapping latent → horizon outputs
- `TiDEModel` — top-level `nn.Module` assembling all components

---

### Metrics

#### [NEW] `tide/metrics/forecasting_metrics.py`
- `mae`, `rmse`, `mse`, `mape`, `smape`, `mase`
- `compute_all_metrics()` — returns a dict of all metrics

---

### Trainer

#### [NEW] `tide/trainer/trainer.py`
- `EarlyStopping` — monitors validation loss with configurable patience
- `Trainer` — training loop, validation loop, checkpointing (`best_model.pt`)
- AdamW optimizer + `ReduceLROnPlateau` scheduler

---

### Config

#### [NEW] `tide/config.py`
- `TiDEConfig` dataclass — all hyperparameters in one place (hidden_size, num_encoder_layers, num_decoder_layers, dropout, lr, batch_size, max_epochs, patience, seed, etc.)

---

### Inference

#### [NEW] `tide/inference/predict.py`
- `load_model()` — loads checkpoint
- `predict()` — runs a forward pass with `torch.no_grad()`
- `predict_next_24h()` — convenience function for the most common use case

---

### Evaluation & Visualization

#### [NEW] `tide/evaluation/evaluate.py`
- Runs the model on the test set
- Prints a metrics summary table
- Generates and saves matplotlib plots:
  - Predicted vs actual demand
  - Residuals
  - Error by forecast horizon step

---

### Entry Point

#### [NEW] `tide/train.py`
- Parses CLI args (or uses defaults from `TiDEConfig`)
- Calls `build_dataloaders()` → `TiDEModel` → `Trainer` → evaluation
- Prints final metrics

---

### Dependencies

#### [MODIFY] `pyproject.toml`
- Add `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Verification Plan

### Automated Tests
```bash
uv run python tide/train.py
```
- Should complete training without errors
- Should print MAE, RMSE, MAPE on the test set
- Should save `checkpoints/best_model.pt`
- Should save plots to `outputs/`

### Manual Verification
- Inspect the "Predicted vs Actual" plot to confirm the model learns daily and weekly patterns.
