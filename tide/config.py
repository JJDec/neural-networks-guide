"""Configuration dataclass for the TiDE electricity-demand forecasting model.

All hyperparameters live here so that experiments are reproducible and
nothing is hardcoded inside model or trainer modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TiDEConfig:
    """Centralised hyperparameter container for TiDE.

    Attributes
    ----------
    # ── Data ──────────────────────────────────────────────────────────────
    data_path:
        Path to a CSV file with an hourly electricity-demand column.
        If ``None`` the built-in synthetic dataset is used.
    target_col:
        Name of the target column when loading from CSV.
    input_len:
        Look-back window length (hours).  7 days × 24 h = 168.
    horizon:
        Forecast horizon (hours).  24 h ahead.
    train_frac:
        Fraction of data used for training (chronological split).
    val_frac:
        Fraction of data used for validation.
    # ── Covariates ────────────────────────────────────────────────────────
    num_hist_covariates:
        Number of historical covariates fed alongside past targets.
        Default: 4 (hour sin, hour cos, dow sin, dow cos).
    num_future_covariates:
        Number of future covariates available for the forecast horizon.
        Default: 4 (same calendar features, always known ahead of time).
    # ── Model ─────────────────────────────────────────────────────────────
    hidden_size:
        Width of every dense layer inside residual blocks.
    num_encoder_layers:
        Number of residual blocks in the encoder.
    num_decoder_layers:
        Number of residual blocks in the decoder.
    temporal_decoder_hidden:
        Hidden size of the per-step temporal decoder MLP.
    dropout:
        Dropout probability applied inside residual blocks.
    use_layer_norm:
        Whether to apply LayerNorm after each residual connection.
    # ── Training ──────────────────────────────────────────────────────────
    lr:
        Initial learning rate for AdamW.
    weight_decay:
        L2 regularisation coefficient.
    batch_size:
        Mini-batch size.
    max_epochs:
        Maximum number of training epochs.
    patience:
        Early-stopping patience (epochs without val-loss improvement).
    grad_clip:
        Maximum gradient norm; ``None`` disables gradient clipping.
    loss:
        Loss function — ``"mae"`` | ``"mse"`` | ``"huber"``.
    # ── Output ────────────────────────────────────────────────────────────
    checkpoint_dir:
        Directory where ``best_model.pt`` is saved.
    output_dir:
        Directory where evaluation plots are saved.
    seed:
        Global random seed for reproducibility.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: Path | None = None
    target_col: str = "demand_mw"
    input_len: int = 168          # 7-day look-back
    horizon: int = 24             # 24-hour forecast
    train_frac: float = 0.70
    val_frac: float = 0.15        # test_frac = 1 - train_frac - val_frac

    # ── Covariates ────────────────────────────────────────────────────────
    num_hist_covariates: int = 4   # hour_sin, hour_cos, dow_sin, dow_cos
    num_future_covariates: int = 4 # same four features, known ahead of time

    # ── Model ─────────────────────────────────────────────────────────────
    hidden_size: int = 512
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    temporal_decoder_hidden: int = 64
    dropout: float = 0.1
    use_layer_norm: bool = True

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 20
    grad_clip: float | None = 1.0
    loss: str = "mae"             # "mae" | "mse" | "huber"

    # ── Output ────────────────────────────────────────────────────────────
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_name: str = "tide"
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate config values after initialisation."""
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        if self.loss not in {"mae", "mse", "huber"}:
            raise ValueError(f"Unknown loss '{self.loss}'. Choose mae | mse | huber.")
        self.checkpoint_dir = Path(self.checkpoint_dir) / self.model_name
        self.output_dir = Path(self.output_dir) / self.model_name

    @property
    def test_frac(self) -> float:
        """Derived fraction of data reserved for the test set."""
        return 1.0 - self.train_frac - self.val_frac
