"""Dataset and DataLoader utilities for hourly electricity demand forecasting.

Provides:
  - Synthetic electricity demand generation (realistic daily + weekly patterns)
  - CSV loading with automatic covariate engineering
  - ``WindowDataset`` — converts a time series to supervised sliding windows
  - ``build_dataloaders`` — returns train / val / test DataLoaders
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tide.config import TiDEConfig


# ---------------------------------------------------------------------------
# Covariate engineering
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic calendar features to *df* in-place and return it.

    Four features are created:
      - hour_sin / hour_cos  — captures the 24-h daily cycle
      - dow_sin  / dow_cos   — captures the 7-day weekly cycle

    Using sin/cos encoding preserves the cyclical structure (hour 23 is
    close to hour 0) in a way that plain integer encoding cannot.
    """
    hour = df.index.hour
    dow = df.index.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

def generate_synthetic_electricity(
    n_hours: int = 8760,   # 1 year of hourly data
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic synthetic hourly electricity demand series.

    The signal combines:
      - Daily seasonality  (peak around 18:00, trough around 04:00)
      - Weekly seasonality (weekends ~20 % lower)
      - Slow upward trend
      - Gaussian noise

    Parameters
    ----------
    n_hours:
        Number of hourly observations to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with a DatetimeIndex and columns:
        ``demand_mw``, ``hour_sin``, ``hour_cos``, ``dow_sin``, ``dow_cos``.
    """
    rng = np.random.default_rng(seed)

    t = np.arange(n_hours)
    hours = t % 24
    days = t // 24
    dow = days % 7  # 0 = Monday

    # Daily pattern: sinusoidal with morning ramp and evening peak
    daily = (
        3000
        + 800 * np.sin(2 * np.pi * (hours - 6) / 24)
        - 400 * np.cos(2 * np.pi * (hours - 6) / 24)
    )

    # Weekly pattern: weekends (dow 5, 6) are ~20 % lower
    weekend_mask = (dow >= 5).astype(float)
    weekly = 1.0 - 0.20 * weekend_mask

    # Slow upward trend (≈ 5 % over the full year)
    trend = 1.0 + 0.05 * (t / n_hours)

    # White noise (σ ≈ 3 % of baseline)
    noise = rng.normal(0, 90, size=n_hours)

    demand = daily * weekly * trend + noise

    start = pd.Timestamp("2023-01-01 00:00:00")
    idx = pd.date_range(start, periods=n_hours, freq="h")

    df = pd.DataFrame({"demand_mw": demand}, index=idx)
    df = _add_calendar_features(df)
    return df


def load_electricity_csv(path: str, target_col: str) -> pd.DataFrame:
    """Load electricity demand data from a CSV file.

    The CSV must have a parseable datetime column as its first column (or
    index) and at least one numeric column named *target_col*.

    Parameters
    ----------
    path:
        Path to the CSV file.
    target_col:
        Name of the column containing hourly electricity demand values.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by time, with calendar covariates added.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()
    df = df[[target_col]].rename(columns={target_col: "demand_mw"})

    # Resample to hourly and forward-fill any gaps
    df = df.resample("h").mean().ffill()

    df = _add_calendar_features(df)
    return df


# ---------------------------------------------------------------------------
# WindowDataset
# ---------------------------------------------------------------------------

class WindowDataset(Dataset):
    """PyTorch Dataset that generates sliding windows from a time series.

    Each sample consists of:
      - ``past_target``   — shape ``(input_len,)``  target values
      - ``hist_covs``     — shape ``(input_len, C_hist)``  historical covariates
      - ``future_covs``   — shape ``(horizon, C_fut)``  future known covariates
      - ``target``        — shape ``(horizon,)``  ground-truth forecast values

    Parameters
    ----------
    data:
        Scaled numpy array of shape ``(T, 1 + C_hist)`` where column 0 is
        the target and columns 1: are historical / future covariates.
    input_len:
        Look-back window length.
    horizon:
        Forecast horizon.
    num_future_covariates:
        Number of columns from *data* that are also valid future covariates.
        These columns are sliced from the horizon window.
    stride:
        Step between consecutive windows (default: 1 for maximum data use).
    """

    def __init__(
        self,
        data: np.ndarray,
        input_len: int,
        horizon: int,
        num_future_covariates: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)
        self.input_len = input_len
        self.horizon = horizon
        self.num_future_covariates = num_future_covariates
        self.stride = stride

        self._total = data.shape[0]
        # Total window size = look-back + forecast horizon
        self._window = input_len + horizon

        # Pre-compute valid start indices
        self._starts = list(range(0, self._total - self._window + 1, stride))

    def __len__(self) -> int:
        return len(self._starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self._starts[idx]
        end = start + self._window

        window = self.data[start:end]  # (input_len + horizon, 1 + C)

        past_window = window[: self.input_len]    # (input_len, 1+C)
        future_window = window[self.input_len :]  # (horizon, 1+C)

        past_target = past_window[:, 0]           # (input_len,)
        hist_covs = past_window[:, 1:]            # (input_len, C)
        future_covs = future_window[:, 1: 1 + self.num_future_covariates]  # (horizon, C_fut)
        target = future_window[:, 0]              # (horizon,)

        return {
            "past_target": past_target,
            "hist_covs": hist_covs,
            "future_covs": future_covs,
            "target": target,
        }


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: TiDEConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from config.

    Splitting is strictly chronological — no shuffling of time order.
    The StandardScaler is fit only on training data and applied to val/test.

    Parameters
    ----------
    cfg:
        ``TiDEConfig`` instance.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, test loaders.
    """
    # ── Load data ────────────────────────────────────────────────────────
    if cfg.data_path is None:
        df = generate_synthetic_electricity(seed=cfg.seed)
    else:
        df = load_electricity_csv(str(cfg.data_path), cfg.target_col)

    # Column order: [demand_mw, hour_sin, hour_cos, dow_sin, dow_cos]
    covariate_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    all_cols = ["demand_mw"] + covariate_cols
    data = df[all_cols].values  # (T, 5)

    T = len(data)
    n_train = int(T * cfg.train_frac)
    n_val = int(T * cfg.val_frac)

    train_raw = data[:n_train]
    val_raw = data[n_train: n_train + n_val]
    test_raw = data[n_train + n_val:]

    # ── Scale only the target column using training statistics ────────────
    train_mean = train_raw[:, 0].mean()
    train_std = train_raw[:, 0].std() + 1e-8   # avoid div-by-zero

    def _scale(arr: np.ndarray) -> np.ndarray:
        """Scale the target column (col 0) using training mean/std."""
        scaled = arr.copy()
        scaled[:, 0] = (arr[:, 0] - train_mean) / train_std
        return scaled

    train_data = _scale(train_raw)
    val_data = _scale(val_raw)
    test_data = _scale(test_raw)

    # Expose scaling stats on the returned loaders (attached as attributes)
    scale_stats = {"mean": float(train_mean), "std": float(train_std)}

    # ── Build PyTorch Datasets ────────────────────────────────────────────
    train_ds = WindowDataset(
        train_data, cfg.input_len, cfg.horizon, cfg.num_future_covariates
    )
    val_ds = WindowDataset(
        val_data, cfg.input_len, cfg.horizon, cfg.num_future_covariates
    )
    test_ds = WindowDataset(
        test_data, cfg.input_len, cfg.horizon, cfg.num_future_covariates, stride=cfg.horizon
    )

    # ── Build DataLoaders ─────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,          # shuffle windows (not time steps!) within split
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Attach scaling stats so the caller can inverse-transform predictions
    for loader in (train_loader, val_loader, test_loader):
        loader.dataset.scale_stats = scale_stats  # type: ignore[attr-defined]

    return train_loader, val_loader, test_loader
