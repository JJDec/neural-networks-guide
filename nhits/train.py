"""N-HiTS training entry point.

Usage
-----
# Train with defaults (synthetic dataset)
uv run python nhits/train.py

# Train with a real CSV
uv run python nhits/train.py --data_path data/electricity.csv --target_col demand_mw

# Quick smoke-test (few epochs)
uv run python nhits/train.py --max_epochs 5 --batch_size 32
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from nhits.config import NHiTSConfig
from nhits.datasets.electricity_dataset import build_dataloaders
from nhits.evaluation.evaluate import evaluate
from nhits.inference.predict import load_model
from nhits.models.nhits import NHiTSModel
from nhits.trainer.trainer import Trainer

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path | None = None) -> None:
    """Parse a .env file and inject variables into ``os.environ``.

    Only sets variables that are not already present in the environment,
    so shell exports always take precedence.
    """
    if env_path is None:
        candidate = Path(__file__).resolve()
        for _ in range(5):
            candidate = candidate.parent
            dotenv = candidate / ".env"
            if dotenv.exists():
                env_path = dotenv
                break

    if env_path is None or not env_path.exists():
        return

    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> NHiTSConfig:
    """Parse CLI arguments and return a populated ``NHiTSConfig``."""
    parser = argparse.ArgumentParser(
        description="Train an N-HiTS model for 24-hour electricity demand forecasting."
    )

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV file (default: synthetic dataset)")
    parser.add_argument("--target_col", type=str, default="demand_mw")
    parser.add_argument("--input_len", type=int, default=168)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    # Model
    parser.add_argument("--n_stacks", type=int, default=3)
    parser.add_argument("--n_blocks_per_stack", type=int, default=1)
    parser.add_argument("--pooling_kernel_sizes", type=int, nargs="+", default=[8, 4, 1])
    parser.add_argument("--n_freq_downsample", type=int, nargs="+", default=[8, 4, 1])
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "gelu"])
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable LayerNorm in MLP blocks")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--loss", type=str, default="mae",
                        choices=["mae", "mse", "huber"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="nhits-forecasting",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="j95-jaworska-na",
                        help="W&B entity (username or team)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional W&B run name")

    args = parser.parse_args()

    return NHiTSConfig(
        data_path=Path(args.data_path) if args.data_path else None,
        target_col=args.target_col,
        input_len=args.input_len,
        horizon=args.horizon,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        n_stacks=args.n_stacks,
        n_blocks_per_stack=args.n_blocks_per_stack,
        pooling_kernel_sizes=args.pooling_kernel_sizes,
        n_freq_downsample=args.n_freq_downsample,
        hidden_size=args.hidden_size,
        num_mlp_layers=args.num_mlp_layers,
        dropout=args.dropout,
        activation=args.activation,
        use_layer_norm=not args.no_layer_norm,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        loss=args.loss,
        grad_clip=args.grad_clip,
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training and evaluation pipeline."""
    _load_dotenv()

    cfg = parse_args()
    set_seed(cfg.seed)

    # ── W&B authentication ────────────────────────────────────────────────
    if cfg.wandb_enabled and _WANDB_AVAILABLE:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key and api_key != "your_wandb_api_key_here":
            _wandb.login(key=api_key, relogin=True)
        else:
            if not _wandb.login(anonymous="never"):
                raise RuntimeError(
                    "W&B login failed. Set WANDB_API_KEY in .env or run `wandb login`."
                )

    # ── Device selection ───────────────────────────────────────────────────
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    n_train = len(train_loader.dataset)   # type: ignore[arg-type]
    n_val = len(val_loader.dataset)       # type: ignore[arg-type]
    n_test = len(test_loader.dataset)     # type: ignore[arg-type]
    print(f"  Windows — train: {n_train}  val: {n_val}  test: {n_test}")

    scale_stats = train_loader.dataset.scale_stats  # type: ignore[attr-defined]

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nConstructing NHiTSModel …")
    model = NHiTSModel.from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Architecture: {cfg.n_stacks} stacks, {cfg.n_blocks_per_stack} blocks/stack")
    print(f"  Pooling kernels:  {cfg.pooling_kernel_sizes}")
    print(f"  Basis downsample: {cfg.n_freq_downsample}")
    print(f"  Hidden size: {cfg.hidden_size} | Dropout: {cfg.dropout}")
    print(f"  Input len:   {cfg.input_len} h | Horizon: {cfg.horizon} h")

    # ── Training ──────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device=device_str)
    history = trainer.fit(train_loader, val_loader)

    # ── Load best checkpoint for evaluation ───────────────────────────────
    print("\nLoading best checkpoint …")
    model = load_model(cfg, device=device_str)

    # ── Evaluation ────────────────────────────────────────────────────────────
    evaluate(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        history=history,
        train_targets=train_loader.dataset.data[:, 0].cpu().numpy(),
    )

    if cfg.wandb_enabled and _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.finish()
        print("  W&B run finished.")

    # ── Inference demo ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Inference demo: predicting next 24 h from last window")
    print("=" * 60)

    from nhits.inference.predict import predict_next_24h
    from nhits.datasets.electricity_dataset import generate_synthetic_electricity

    demo_df = generate_synthetic_electricity(seed=cfg.seed)
    covariate_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    demo_window = demo_df.iloc[-cfg.input_len:]

    forecast_mw = predict_next_24h(
        model=model,
        past_demand_mw=demo_window["demand_mw"].values,
        past_covariates=demo_window[covariate_cols].values,
        future_covariates=demo_df[covariate_cols].iloc[
            -cfg.horizon:
        ].values,
        scale_mean=scale_stats["mean"],
        scale_std=scale_stats["std"],
        device=device_str,
    )

    print("\n  Next 24-hour electricity demand forecast:")
    print(f"  {'Hour':>4}  {'Forecast (MW)':>14}")
    print(f"  {'-'*20}")
    for h, mw in enumerate(forecast_mw, start=1):
        print(f"  {h:>4}  {mw:>14.1f}")


if __name__ == "__main__":
    main()
