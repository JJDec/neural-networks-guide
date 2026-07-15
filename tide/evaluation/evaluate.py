"""Model evaluation on the test set with metrics and visualisations.

Functions
---------
collect_predictions
    Run the model over a DataLoader, collect predictions and targets.
evaluate
    Compute all metrics and produce plots saved to ``cfg.output_dir``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from tide.config import TiDEConfig
from tide.metrics.forecasting_metrics import compute_all_metrics
from tide.models.tide import TiDEModel


# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: TiDEModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over *loader* and collect all predictions and ground-truth values.

    Parameters
    ----------
    model:
        ``TiDEModel`` in eval mode.
    loader:
        DataLoader (typically the test loader).
    device:
        Inference device.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(preds, targets)`` both of shape ``(N * H,)`` — flattened across
        all batches and horizon steps.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        past_target = batch["past_target"].to(device)
        hist_covs = batch["hist_covs"].to(device)
        future_covs = batch["future_covs"].to(device)
        target = batch["target"].cpu().numpy()

        pred = model(past_target, hist_covs, future_covs).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(target)

    return np.concatenate(all_preds).ravel(), np.concatenate(all_targets).ravel()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_forecast_vs_actual(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
    n_samples: int = 7 * 24,
) -> None:
    """Plot the first *n_samples* hours of predictions vs. actuals."""
    n = min(n_samples, len(preds))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(x, targets[:n], label="Actual", linewidth=1.5, color="#2d6a9f")
    ax.plot(x, preds[:n], label="Forecast", linewidth=1.5, linestyle="--", color="#e87040")
    ax.set_title("TiDE — Predicted vs Actual Electricity Demand (test set, scaled)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Demand (scaled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "forecast_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_residuals(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
) -> None:
    """Plot the residual distribution."""
    residuals = targets - preds

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Time series of residuals
    ax = axes[0]
    ax.plot(residuals[:7 * 24], linewidth=0.8, color="#7b5ea7")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Residuals over time (first 7 days)")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Residual (scaled)")
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=60, color="#7b5ea7", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual (scaled)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_error_by_horizon(
    model: TiDEModel,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
) -> None:
    """Compute and plot MAE at each of the 24 horizon steps."""
    model.eval()
    horizon_preds: list[np.ndarray] = []
    horizon_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            pred = model(
                batch["past_target"].to(device),
                batch["hist_covs"].to(device),
                batch["future_covs"].to(device),
            ).cpu().numpy()   # (B, H)
            horizon_preds.append(pred)
            horizon_targets.append(batch["target"].numpy())

    preds_h = np.concatenate(horizon_preds, axis=0)    # (N, H)
    targets_h = np.concatenate(horizon_targets, axis=0)  # (N, H)
    mae_per_step = np.mean(np.abs(preds_h - targets_h), axis=0)  # (H,)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(
        np.arange(1, len(mae_per_step) + 1),
        mae_per_step,
        color="#2d6a9f",
        alpha=0.8,
    )
    ax.set_title("MAE by Forecast Horizon Step")
    ax.set_xlabel("Horizon step (hours ahead)")
    ax.set_ylabel("MAE (scaled)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "error_by_horizon.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_training_history(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Plot train/val loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss", color="#2d6a9f")
    ax.plot(epochs, history["val_loss"], label="Val loss", color="#e87040")
    ax.set_title("Training History")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = output_dir / "training_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    model: TiDEModel,
    test_loader: DataLoader,
    cfg: TiDEConfig,
    device: torch.device,
    history: dict[str, list[float]] | None = None,
    train_targets: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate the model on the test set and save plots.

    Parameters
    ----------
    model:
        Trained ``TiDEModel`` in eval mode.
    test_loader:
        Test DataLoader.
    cfg:
        ``TiDEConfig`` instance (used for ``output_dir``).
    device:
        Inference device.
    history:
        Training history dict (optional; used to plot loss curves).
    train_targets:
        Raw training target values for MASE computation.

    Returns
    -------
    dict[str, float]
        Computed metrics.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Evaluation on test set")
    print("=" * 60)

    preds, targets = collect_predictions(model, test_loader, device)

    metrics = compute_all_metrics(targets, preds, y_train=train_targets)

    # ── Print metrics table ────────────────────────────────────────────────
    print(f"\n  {'Metric':<10} {'Value':>12}")
    print(f"  {'-'*24}")
    for name, value in metrics.items():
        unit = "%" if name in ("mape", "smape") else ""
        print(f"  {name.upper():<10} {value:>11.4f}{unit}")

    # ── Save plots ─────────────────────────────────────────────────────────
    print("\n  Saving plots …")
    _plot_forecast_vs_actual(preds, targets, cfg.output_dir)
    _plot_residuals(preds, targets, cfg.output_dir)
    _plot_error_by_horizon(model, test_loader, device, cfg.output_dir)

    if history is not None:
        _plot_training_history(history, cfg.output_dir)

    print(f"\n  All outputs saved to: {cfg.output_dir.resolve()}")
    return metrics
