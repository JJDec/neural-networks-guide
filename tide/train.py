"""TiDE training entry point.

Usage
-----
# Train with defaults (synthetic dataset)
uv run python tide/train.py

# Train with a real CSV
uv run python tide/train.py --data_path data/electricity.csv --target_col demand_mw

# Quick smoke-test (few epochs)
uv run python tide/train.py --max_epochs 5 --batch_size 32
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from tide.config import TiDEConfig
from tide.datasets.electricity_dataset import build_dataloaders
from tide.evaluation.evaluate import evaluate
from tide.inference.predict import load_model
from tide.models.tide import TiDEModel
from tide.trainer.trainer import Trainer


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

def parse_args() -> TiDEConfig:
    """Parse CLI arguments and return a populated ``TiDEConfig``."""
    parser = argparse.ArgumentParser(
        description="Train a TiDE model for 24-hour electricity demand forecasting."
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
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=2)
    parser.add_argument("--temporal_decoder_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable LayerNorm in residual blocks")

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

    args = parser.parse_args()

    return TiDEConfig(
        data_path=Path(args.data_path) if args.data_path else None,
        target_col=args.target_col,
        input_len=args.input_len,
        horizon=args.horizon,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        temporal_decoder_hidden=args.temporal_decoder_hidden,
        dropout=args.dropout,
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
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training and evaluation pipeline."""
    cfg = parse_args()
    set_seed(cfg.seed)

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

    # Keep raw training targets for MASE computation (scaled space)
    scale_stats = train_loader.dataset.scale_stats  # type: ignore[attr-defined]

    # ── Model ─────────────────────────────────────────────────────────────
    print("\nConstructing TiDEModel …")
    model = TiDEModel.from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Architecture: {cfg.num_encoder_layers} encoder + {cfg.num_decoder_layers} decoder blocks")
    print(f"  Hidden size:  {cfg.hidden_size}  |  Dropout: {cfg.dropout}")
    print(f"  Input len:    {cfg.input_len} h  |  Horizon: {cfg.horizon} h")

    # ── Training ──────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device=device_str)
    history = trainer.fit(train_loader, val_loader)

    # ── Load best checkpoint for evaluation ───────────────────────────────
    print("\nLoading best checkpoint …")
    model = load_model(cfg, device=device_str)

    # ── Evaluation ────────────────────────────────────────────────────────
    evaluate(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        history=history,
        train_targets=train_loader.dataset.data[:, 0].cpu().numpy(),
    )

    # ── Inference demo ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Inference demo: predicting next 24 h from last window")
    print("=" * 60)

    from tide.inference.predict import predict_next_24h
    from tide.datasets.electricity_dataset import generate_synthetic_electricity
    import pandas as pd

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
