"""Inference utilities for the TiDE electricity demand forecasting model.

Functions
---------
load_model
    Load a ``TiDEModel`` from a saved checkpoint.
predict
    Run a single forward pass with ``torch.no_grad()``.
predict_next_24h
    Convenience wrapper: given the last ``input_len`` hours of data,
    return the forecast for the next 24 hours in **original MW units**.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tide.config import TiDEConfig
from tide.models.tide import TiDEModel


def load_model(
    cfg: TiDEConfig,
    checkpoint_path: Path | str | None = None,
    device: str | torch.device = "cpu",
) -> TiDEModel:
    """Load a TiDEModel from a checkpoint file.

    Parameters
    ----------
    cfg:
        Configuration used to construct the model architecture.
    checkpoint_path:
        Path to the ``.pt`` file.  Defaults to ``cfg.checkpoint_dir / "best_model.pt"``.
    device:
        Target device for inference.

    Returns
    -------
    TiDEModel
        Model in eval mode on *device*.
    """
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_dir / "best_model.pt"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TiDEModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: TiDEModel,
    past_target: Tensor,   # (B, L)
    hist_covs: Tensor,     # (B, L, C_hist)
    future_covs: Tensor,   # (B, H, C_fut)
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run a forward pass and return predictions as a numpy array.

    Parameters
    ----------
    model:
        Loaded ``TiDEModel`` in eval mode.
    past_target:
        Scaled look-back target tensor, shape ``(B, L)``.
    hist_covs:
        Historical covariates, shape ``(B, L, C_hist)``.
    future_covs:
        Future covariates for the horizon, shape ``(B, H, C_fut)``.
    device:
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Predictions in scaled space, shape ``(B, H)``.
    """
    device = torch.device(device)
    pred = model(
        past_target.to(device),
        hist_covs.to(device),
        future_covs.to(device),
    )
    return pred.cpu().numpy()


def predict_next_24h(
    model: TiDEModel,
    past_demand_mw: np.ndarray,       # (L,)  in original MW units
    past_covariates: np.ndarray,      # (L, C_hist)
    future_covariates: np.ndarray,    # (24, C_fut)
    scale_mean: float,
    scale_std: float,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Predict the next 24 hours of electricity demand in MW.

    This convenience function handles scaling and inverse-scaling so
    the caller works entirely in the original unit (MW).

    Parameters
    ----------
    model:
        Loaded ``TiDEModel`` in eval mode.
    past_demand_mw:
        Last ``input_len`` hourly demand values in MW, shape ``(L,)``.
    past_covariates:
        Historical covariates corresponding to the look-back window,
        shape ``(L, C_hist)``.
    future_covariates:
        Known future covariates for the 24-hour horizon,
        shape ``(24, C_fut)``.
    scale_mean:
        Training set mean of the target column (from ``scale_stats``).
    scale_std:
        Training set standard deviation of the target column.
    device:
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Forecast in MW, shape ``(24,)``.
    """
    # Scale the past target
    past_scaled = (past_demand_mw - scale_mean) / (scale_std + 1e-8)

    # Build tensors with a batch dimension of 1
    past_t = torch.tensor(past_scaled, dtype=torch.float32).unsqueeze(0)      # (1, L)
    hist_c = torch.tensor(past_covariates, dtype=torch.float32).unsqueeze(0)  # (1, L, C)
    fut_c = torch.tensor(future_covariates, dtype=torch.float32).unsqueeze(0) # (1, 24, C)

    pred_scaled = predict(model, past_t, hist_c, fut_c, device)  # (1, 24)

    # Inverse-scale back to MW
    pred_mw = pred_scaled[0] * scale_std + scale_mean
    return pred_mw
