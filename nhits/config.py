"""Configuration dataclass for the N-HiTS electricity-demand forecasting model.

All hyperparameters live here so that experiments are reproducible and
nothing is hardcoded inside model or trainer modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NHiTSConfig:
    """Centralised hyperparameter container for N-HiTS.

    Attributes
    ----------
    # ── Data ──────────────────────────────────────────────────────────────
    data_path:
        Path to a CSV file with an hourly electricity-demand column.
        If ``None`` the built-in synthetic dataset is used.
    target_col:
        Name of the target column when loading from CSV.
    input_len:
        Look-back window length (hours). 7 days × 24 h = 168.
    horizon:
        Forecast horizon (hours). 24 h ahead.
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
    n_stacks:
        Number of hierarchical stacks in the model (e.g. Coarse, Medium, Fine).
    n_blocks_per_stack:
        Number of blocks in each stack.
    pooling_kernel_sizes:
        Max-pooling kernel/stride for input lookback subsampling in each stack.
    n_freq_downsample:
        Basis downsampling factor for backcast/forecast interpolation in each stack.
    hidden_size:
        Width of every dense layer in MLP blocks.
    num_mlp_layers:
        Number of dense layers inside each N-HiTS block.
    dropout:
        Dropout probability applied inside MLP layers.
    activation:
        Activation function ("relu" | "gelu").
    use_layer_norm:
        Whether to apply LayerNorm after dense layers.
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
    model_name:
        Subdirectory identifier for outputs and checkpoints.
    seed:
        Global random seed for reproducibility.
    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb_enabled:
        Whether to log this run to Weights & Biases.
    wandb_project:
        W&B project name (e.g. ``"nhits-forecasting"``).
    wandb_entity:
        W&B entity (username or team, e.g. ``"j95-jaworska-na"``).
    wandb_run_name:
        Optional human-readable name for this W&B run.
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
    n_stacks: int = 3
    n_blocks_per_stack: int = 1
    pooling_kernel_sizes: list[int] = field(default_factory=lambda: [8, 4, 1])
    n_freq_downsample: list[int] = field(default_factory=lambda: [8, 4, 1])
    hidden_size: int = 512
    num_mlp_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"
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
    model_name: str = "nhits"
    seed: int = 42

    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb_enabled: bool = False
    wandb_project: str = "nhits-forecasting"
    wandb_entity: str = "j95-jaworska-na"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        """Validate config values after initialisation."""
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        if self.loss not in {"mae", "mse", "huber"}:
            raise ValueError(f"Unknown loss '{self.loss}'. Choose mae | mse | huber.")
        if len(self.pooling_kernel_sizes) != self.n_stacks:
            raise ValueError(
                f"Length of pooling_kernel_sizes ({len(self.pooling_kernel_sizes)}) "
                f"must equal n_stacks ({self.n_stacks})"
            )
        if len(self.n_freq_downsample) != self.n_stacks:
            raise ValueError(
                f"Length of n_freq_downsample ({len(self.n_freq_downsample)}) "
                f"must equal n_stacks ({self.n_stacks})"
            )
        self.checkpoint_dir = Path(self.checkpoint_dir) / self.model_name
        self.output_dir = Path(self.output_dir) / self.model_name

    @property
    def test_frac(self) -> float:
        """Derived fraction of data reserved for the test set."""
        return 1.0 - self.train_frac - self.val_frac
