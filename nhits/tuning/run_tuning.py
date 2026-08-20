"""Entry-point for N-HiTS hyperparameter search with Ray Tune + W&B.

Usage
-----
# Quick smoke-test (4 trials, 5 epochs each)
uv run python nhits/tuning/run_tuning.py --num_samples 4 --max_epochs 5

# Full search (20 trials) with W&B logging
uv run python nhits/tuning/run_tuning.py --num_samples 20 --wandb

# Custom data, GPU
uv run python nhits/tuning/run_tuning.py \\
    --data_path data/electricity.csv \\
    --num_samples 30 --max_epochs 50 \\
    --cpus_per_trial 2 --gpus_per_trial 0.5 \\
    --wandb --wandb_project nhits-tune

Algorithm
---------
* Sampler : Optuna TPE (Tree-structured Parzen Estimator)
* Scheduler: ASHA (Async Successive Halving) — prunes under-performing trials
  after ``grace_period`` epochs.
* W&B      : one run per trial via ``WandbLoggerCallback``; all trials share
  the same W&B group so they appear in a single parallel-coordinates view.

After the search the best configuration is printed and (optionally) re-trained
for the full number of epochs with a final checkpoint saved under
``checkpoints/tune/best/``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# .env loader (mirror of nhits/train.py — keeps the script self-contained)
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path | None = None) -> None:
    """Parse a .env file and inject variables into ``os.environ``."""
    if env_path is None:
        candidate = Path(__file__).resolve()
        for _ in range(6):
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
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for N-HiTS using Ray Tune + W&B."
    )
    # Search budget
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of hyperparameter trials to run (default: 20).")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum epochs per trial (default: 50).")
    parser.add_argument("--grace_period", type=int, default=5,
                        help="ASHA minimum epochs before pruning a trial (default: 5).")

    # Resources
    parser.add_argument("--cpus_per_trial", type=float, default=1.0,
                        help="CPUs allocated per trial (default: 1.0).")
    parser.add_argument("--gpus_per_trial", type=float, default=0.0,
                        help="GPUs allocated per trial (default: 0.0).")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV file (default: built-in synthetic dataset).")

    # Re-train best
    parser.add_argument("--retrain_best", action="store_true",
                        help="Re-train best config for full max_epochs after search.")
    parser.add_argument("--retrain_epochs", type=int, default=100,
                        help="Epochs for best-config re-train (default: 100).")

    # Ray Tune storage
    parser.add_argument("--storage_path", type=str, default=None,
                        help="Ray Tune storage path (default: ./ray_results).")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging for all trials.")
    parser.add_argument("--wandb_project", type=str, default="nhits-forecasting",
                        help="W&B project name (default: nhits-forecasting).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team). Reads WANDB_ENTITY env var if unset.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# W&B login helper
# ---------------------------------------------------------------------------

def _wandb_login() -> None:
    """Authenticate with W&B using WANDB_API_KEY or interactive login."""
    import wandb  # noqa: PLC0415

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key and api_key != "your_wandb_api_key_here":
        wandb.login(key=api_key, relogin=True)
    else:
        if not wandb.login(anonymous="never"):
            raise RuntimeError(
                "W&B login failed. Set WANDB_API_KEY in .env or run `wandb login`."
            )


# ---------------------------------------------------------------------------
# Best-config re-train
# ---------------------------------------------------------------------------

def _retrain_best(
    best_config: dict,
    max_epochs: int,
    data_path: str | None,
) -> None:
    """Re-train the winning configuration for the full number of epochs."""
    import torch  # noqa: PLC0415

    from nhits.config import NHiTSConfig  # noqa: PLC0415
    from nhits.datasets.electricity_dataset import build_dataloaders  # noqa: PLC0415
    from nhits.models.nhits import NHiTSModel  # noqa: PLC0415
    from nhits.trainer.trainer import Trainer  # noqa: PLC0415

    n_stacks = best_config.get("n_stacks", 3)
    default_pooling = {1: [1], 2: [4, 1], 3: [8, 4, 1]}
    default_freq_ds = {1: [1], 2: [4, 1], 3: [8, 4, 1]}

    cfg = NHiTSConfig(
        data_path=Path(data_path) if data_path else None,
        n_stacks=n_stacks,
        n_blocks_per_stack=best_config.get("n_blocks_per_stack", 1),
        pooling_kernel_sizes=best_config.get(
            "pooling_kernel_sizes", default_pooling.get(n_stacks, [8, 4, 1])
        ),
        n_freq_downsample=best_config.get(
            "n_freq_downsample", default_freq_ds.get(n_stacks, [8, 4, 1])
        ),
        hidden_size=best_config.get("hidden_size", 512),
        num_mlp_layers=best_config.get("num_mlp_layers", 2),
        dropout=best_config.get("dropout", 0.1),
        activation=best_config.get("activation", "relu"),
        lr=best_config.get("lr", 1e-3),
        weight_decay=best_config.get("weight_decay", 1e-4),
        batch_size=best_config.get("batch_size", 64),
        max_epochs=max_epochs,
        patience=max_epochs // 5,
        loss=best_config.get("loss", "mae"),
        grad_clip=best_config.get("grad_clip", 1.0),
        checkpoint_dir=Path("checkpoints") / "tune" / "best",
        output_dir=Path("outputs") / "tune" / "best",
        model_name="nhits",
        seed=best_config.get("seed", 42),
        wandb_enabled=False,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = NHiTSModel.from_config(cfg)
    trainer = Trainer(model, cfg, device=device)
    trainer.fit(train_loader, val_loader)
    print(f"\n  Best model checkpoint: {cfg.checkpoint_dir / 'best_model.pt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full hyperparameter search."""
    _load_dotenv()
    args = _parse_args()

    # ── W&B auth (before Ray workers start) ──────────────────────────────
    if args.wandb:
        try:
            _wandb_login()
        except Exception as exc:
            print(f"[WARNING] W&B login failed: {exc}. Disabling W&B.")
            args.wandb = False

    # ── Ray Tune imports ──────────────────────────────────────────────────
    import ray  # noqa: PLC0415
    from ray import tune  # noqa: PLC0415
    from ray.tune.schedulers import ASHAScheduler  # noqa: PLC0415
    from ray.tune.search.optuna import OptunaSearch  # noqa: PLC0415

    from nhits.tuning.search_space import default_search_space  # noqa: PLC0415
    from nhits.tuning.trainable import tune_nhits  # noqa: PLC0415

    # ── Search space: inject fixed/CLI params ─────────────────────────────
    param_space = default_search_space()
    param_space["max_epochs"] = args.max_epochs
    param_space["patience"] = max(args.grace_period, args.max_epochs // 5)
    if args.data_path:
        param_space["data_path"] = args.data_path

    # ── Scheduler: ASHA ───────────────────────────────────────────────────
    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=2,
    )

    # ── Sampler: Optuna TPE ───────────────────────────────────────────────
    search_alg = OptunaSearch()

    # ── W&B callback ──────────────────────────────────────────────────────
    callbacks = []
    wandb_group = f"nhits-tune-{ray.util.get_node_ip_address()}"
    if args.wandb:
        from ray.air.integrations.wandb import WandbLoggerCallback  # noqa: PLC0415

        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        wb_kwargs: dict = dict(
            project=args.wandb_project,
            group=wandb_group,
            tags=["nhits", "tune", "asha", "optuna"],
            log_config=True,
        )
        if entity:
            wb_kwargs["entity"] = entity

        callbacks.append(WandbLoggerCallback(**wb_kwargs))
        print(f"\n  W&B project : {args.wandb_project}")
        print(f"  W&B group   : {wandb_group}")

    # ── Storage path ──────────────────────────────────────────────────────
    storage_path = (
        args.storage_path
        if args.storage_path
        else str(Path("ray_results").resolve())
    )

    # ── Tuner ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  N-HiTS Hyperparameter Search — Ray Tune")
    print(f"  Trials     : {args.num_samples}")
    print(f"  Max epochs : {args.max_epochs}")
    print(f"  Grace period: {args.grace_period}")
    print(f"  Resources  : {args.cpus_per_trial} CPU, {args.gpus_per_trial} GPU per trial")
    print(f"{'='*60}\n")

    tuner = tune.Tuner(
        tune.with_resources(
            tune_nhits,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.num_samples,
            trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        ),
        run_config=tune.RunConfig(
            name="nhits_tune",
            storage_path=storage_path,
            callbacks=callbacks,
            verbose=1,
        ),
    )

    results = tuner.fit()

    # ── Results ───────────────────────────────────────────────────────────
    best_result = results.get_best_result(metric="val_loss", mode="min")

    print(f"\n{'='*60}")
    print("  Hyperparameter Search Complete")
    print(f"{'='*60}")
    print(f"\n  Best val_loss : {best_result.metrics.get('val_loss', 'N/A'):.5f}")
    print("\n  Best config:")
    for k, v in sorted(best_result.config.items()):
        # Skip injected fixed params that aren't HPs
        if k in {"max_epochs", "patience", "data_path"}:
            continue
        print(f"    {k:<22} = {v}")

    # ── Optional re-train ─────────────────────────────────────────────────
    if args.retrain_best:
        print(f"\n  Re-training best config for {args.retrain_epochs} epochs …")
        _retrain_best(
            best_config=best_result.config,
            max_epochs=args.retrain_epochs,
            data_path=args.data_path,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
