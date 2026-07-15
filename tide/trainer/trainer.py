"""Training loop, early stopping, and checkpointing for TiDE.

Classes
-------
EarlyStopping
    Monitors validation loss and raises a flag after *patience* epochs
    without improvement.

Trainer
    Orchestrates the full training loop:
      - AdamW optimiser
      - ReduceLROnPlateau scheduler
      - Validation every epoch
      - Checkpointing the best model
      - Early stopping
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from tide.config import TiDEConfig


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def _build_loss(name: str) -> nn.Module:
    """Return the requested loss module.

    Parameters
    ----------
    name:
        One of ``"mae"``, ``"mse"``, ``"huber"``.
    """
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unknown loss '{name}'.")


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience:
        Number of epochs to wait for improvement before stopping.
    min_delta:
        Minimum change in monitored value to qualify as an improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss.

        Returns
        -------
        bool
            ``True`` if training should stop.
        """
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    @property
    def best_loss(self) -> float:
        """Best validation loss seen so far."""
        return self._best_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop for TiDE.

    Parameters
    ----------
    model:
        The ``TiDEModel`` instance to train.
    cfg:
        ``TiDEConfig`` containing all hyperparameters.
    device:
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
    on_epoch_end:
        Optional callback called at the end of every epoch with
        ``(epoch, train_loss, val_loss)``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: TiDEConfig,
        device: str | torch.device = "cpu",
        on_epoch_end: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.on_epoch_end = on_epoch_end

        self.criterion = _build_loss(cfg.loss)
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=cfg.patience // 2,
        )
        self.early_stopping = EarlyStopping(patience=cfg.patience)

        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = cfg.checkpoint_dir / "best_model.pt"

    # ── Internal helpers ──────────────────────────────────────────────────

    def _batch_to_device(
        self, batch: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """Move all tensors in a batch dict to the target device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _train_epoch(self, loader: DataLoader) -> float:
        """Run one training epoch and return the average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)

            self.optimizer.zero_grad()
            pred = self.model(
                batch["past_target"],
                batch["hist_covs"],
                batch["future_covs"],
            )
            loss: Tensor = self.criterion(pred, batch["target"])
            loss.backward()

            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        """Run one validation epoch and return the average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            pred = self.model(
                batch["past_target"],
                batch["hist_covs"],
                batch["future_covs"],
            )
            loss = self.criterion(pred, batch["target"])
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save the full model state to ``best_model.pt``."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
            },
            self._ckpt_path,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """Train the model for up to ``cfg.max_epochs`` epochs.

        Parameters
        ----------
        train_loader:
            DataLoader for the training set.
        val_loader:
            DataLoader for the validation set.

        Returns
        -------
        dict[str, list[float]]
            Training history with keys ``"train_loss"`` and ``"val_loss"``.
        """
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val = float("inf")

        print(f"\n{'='*60}")
        print(f"  TiDE training — device: {self.device}")
        print(f"  Max epochs: {self.cfg.max_epochs}  |  Patience: {self.cfg.patience}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            elapsed = time.perf_counter() - t0

            self.scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            # Checkpoint best model
            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, val_loss)
                ckpt_marker = " ✓"
            else:
                ckpt_marker = ""

            lr_now = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:4d}/{self.cfg.max_epochs} | "
                f"train {train_loss:.5f} | val {val_loss:.5f} | "
                f"lr {lr_now:.2e} | {elapsed:.1f}s{ckpt_marker}"
            )

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, train_loss, val_loss)

            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"\nBest val loss: {best_val:.5f}")
        print(f"Checkpoint: {self._ckpt_path}\n")
        return history
