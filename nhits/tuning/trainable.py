"""Ray Tune trainable function for N-HiTS.

Each Ray Tune *trial* calls ``tune_nhits(config)`` once.  The function:

1. Merges the trial's ``config`` dict into a base ``NHiTSConfig``.
2. Builds data loaders, model, and trainer.
3. Runs the training loop, reporting ``val_loss`` to Tune after every epoch
   (enables ASHA to prune poorly-performing trials early).

Usage
-----
This module is not called directly — it is passed to ``ray.tune.Tuner``
as the *trainable* argument by ``run_tuning.py``.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


def _resolve_device() -> str:
    """Pick the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tune_nhits(config: dict[str, Any]) -> None:
    """Ray Tune trainable: train one NHiTS trial and report metrics.

    Parameters
    ----------
    config:
        Dictionary produced by Ray Tune from the search space.  Expected
        keys (all optional — missing keys fall back to ``NHiTSConfig``
        defaults):

        * ``lr``, ``weight_decay``
        * ``batch_size``
        * ``hidden_size``, ``num_mlp_layers``, ``dropout``
        * ``loss``, ``activation``
        * ``max_epochs``, ``patience``
        * ``data_path`` (str path or None)
        * ``seed``

    Tune integration
    ----------------
    ``tune.report(val_loss=..., train_loss=...)`` is called after every
    epoch via the ``on_epoch_end`` callback so that ASHA can prune trials.
    """
    # Deferred imports keep startup fast when Ray workers initialise.
    from ray import tune

    from nhits.config import NHiTSConfig
    from nhits.datasets.electricity_dataset import build_dataloaders
    from nhits.models.nhits import NHiTSModel
    from nhits.trainer.trainer import Trainer

    # ── Build NHiTSConfig from trial config ──────────────────────────────
    data_path_raw = config.get("data_path", None)
    data_path = Path(data_path_raw) if data_path_raw else None

    # n_stacks is fixed; derive per-stack lists to keep config consistent.
    n_stacks: int = config.get("n_stacks", 3)
    default_pooling = {1: [1], 2: [4, 1], 3: [8, 4, 1]}
    default_freq_ds = {1: [1], 2: [4, 1], 3: [8, 4, 1]}
    pooling = config.get("pooling_kernel_sizes", default_pooling.get(n_stacks, [8, 4, 1]))
    freq_ds = config.get("n_freq_downsample", default_freq_ds.get(n_stacks, [8, 4, 1]))

    # Use a per-trial checkpoint dir so workers don't clobber each other.
    trial_id = os.environ.get("TUNE_TRIAL_ID", "local")
    checkpoint_dir = Path("checkpoints") / "tune" / trial_id
    output_dir = Path("outputs") / "tune" / trial_id

    cfg = NHiTSConfig(
        data_path=data_path,
        # model
        n_stacks=n_stacks,
        n_blocks_per_stack=config.get("n_blocks_per_stack", 1),
        pooling_kernel_sizes=pooling,
        n_freq_downsample=freq_ds,
        hidden_size=config.get("hidden_size", 512),
        num_mlp_layers=config.get("num_mlp_layers", 2),
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "relu"),
        # training
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
        batch_size=config.get("batch_size", 64),
        max_epochs=config.get("max_epochs", 50),
        patience=config.get("patience", 10),
        loss=config.get("loss", "mae"),
        grad_clip=config.get("grad_clip", 1.0),
        # output — isolated per trial
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="nhits",
        seed=config.get("seed", 42),
        # W&B is handled by WandbLoggerCallback in run_tuning.py
        wandb_enabled=False,
    )

    device = _resolve_device()

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────
    model = NHiTSModel.from_config(cfg)

    # ── Epoch-end callback: report to Tune after every epoch ─────────────
    def _on_epoch_end(epoch: int, train_loss: float, val_loss: float) -> None:
        tune.report({"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch})

    # ── Training ─────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device=device, on_epoch_end=_on_epoch_end)
    trainer.fit(train_loader, val_loader)
