"""Forecasting evaluation metrics.

All functions operate on plain numpy arrays and return scalar floats.
Use ``compute_all_metrics`` to get a dict of every metric at once.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Parameters
    ----------
    eps:
        Small constant added to the denominator to avoid division by zero.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    sMAPE is bounded in [0, 200] and avoids the asymmetry of MAPE.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(numerator / denominator) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 24,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample naive seasonal forecast error, making
    the metric unit-free and comparable across series.

    Parameters
    ----------
    y_train:
        Training target values used to compute the scale (naive forecast).
    seasonality:
        Seasonal lag for the naive baseline (default: 24 h for daily cycle).
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) + 1e-8
    return float(mae(y_true, y_pred) / scale)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 24,
) -> dict[str, float]:
    """Compute all forecasting metrics and return as a dict.

    Parameters
    ----------
    y_true:
        Ground-truth values, shape ``(N,)`` or ``(N, H)``.
    y_pred:
        Predicted values, same shape as *y_true*.
    y_train:
        Training targets required for MASE; if ``None`` MASE is omitted.
    seasonality:
        Seasonal period passed to ``mase``.

    Returns
    -------
    dict[str, float]
        Keys: ``mae``, ``rmse``, ``mse``, ``mape``, ``smape``, ``r2``,
        and optionally ``mase``.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    if y_train is not None:
        metrics["mase"] = mase(y_true, y_pred, y_train.ravel(), seasonality)

    return metrics
