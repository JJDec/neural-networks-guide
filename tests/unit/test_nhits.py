"""Unit test suite for N-HiTS electricity demand forecasting model."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch

from nhits.config import NHiTSConfig
from nhits.datasets.electricity_dataset import WindowDataset, build_dataloaders, generate_synthetic_electricity
from nhits.evaluation.evaluate import collect_predictions
from nhits.inference.predict import load_model, predict, predict_next_24h
from nhits.metrics.forecasting_metrics import compute_all_metrics
from nhits.models.nhits import NHiTSBlock, NHiTSModel
from nhits.trainer.trainer import EarlyStopping, Trainer


def test_config_validation() -> None:
    """Test NHiTSConfig default values and post-init validation."""
    cfg = NHiTSConfig()
    assert cfg.input_len == 168
    assert cfg.horizon == 24
    assert cfg.n_stacks == 3
    assert cfg.pooling_kernel_sizes == [8, 4, 1]
    assert cfg.n_freq_downsample == [8, 4, 1]
    assert cfg.test_frac == pytest.approx(0.15)

    with pytest.raises(ValueError, match="train_frac \\+ val_frac must be < 1.0"):
        NHiTSConfig(train_frac=0.8, val_frac=0.3)

    with pytest.raises(ValueError, match="Unknown loss"):
        NHiTSConfig(loss="invalid_loss")

    with pytest.raises(ValueError, match="must equal n_stacks"):
        NHiTSConfig(n_stacks=3, pooling_kernel_sizes=[8, 4])


def test_synthetic_dataset_and_window_dataset() -> None:
    """Test synthetic data generation and sliding window dataset creation."""
    df = generate_synthetic_electricity(n_hours=500, seed=42)
    assert len(df) == 500
    expected_cols = {"demand_mw", "hour_sin", "hour_cos", "dow_sin", "dow_cos"}
    assert expected_cols.issubset(df.columns)

    data = df[["demand_mw", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]].values
    dataset = WindowDataset(data=data, input_len=168, horizon=24, num_future_covariates=4)

    assert len(dataset) == 500 - (168 + 24) + 1
    sample = dataset[0]

    assert sample["past_target"].shape == (168,)
    assert sample["hist_covs"].shape == (168, 4)
    assert sample["future_covs"].shape == (24, 4)
    assert sample["target"].shape == (24,)


def test_nhits_block_forward() -> None:
    """Test single NHiTSBlock forward pass shape."""
    block = NHiTSBlock(
        input_len=168,
        horizon=24,
        num_hist_covariates=4,
        num_future_covariates=4,
        pooling_kernel_size=8,
        n_freq_downsample=8,
        hidden_size=128,
        num_mlp_layers=2,
    )

    B = 4
    past_target = torch.randn(B, 168)
    hist_covs = torch.randn(B, 168, 4)
    future_covs = torch.randn(B, 24, 4)

    backcast, forecast = block(past_target, hist_covs, future_covs)
    assert backcast.shape == (B, 168)
    assert forecast.shape == (B, 24)


def test_nhits_model_forward_and_gradient() -> None:
    """Test full NHiTSModel forward pass and gradient flow."""
    model = NHiTSModel(
        input_len=168,
        horizon=24,
        num_hist_covariates=4,
        num_future_covariates=4,
        n_stacks=3,
        n_blocks_per_stack=1,
        pooling_kernel_sizes=[8, 4, 1],
        n_freq_downsample=[8, 4, 1],
        hidden_size=64,
        num_mlp_layers=2,
    )

    B = 4
    past_target = torch.randn(B, 168)
    hist_covs = torch.randn(B, 168, 4)
    future_covs = torch.randn(B, 24, 4)
    target = torch.randn(B, 24)

    pred = model(past_target, hist_covs, future_covs)
    assert pred.shape == (B, 24)

    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None


def test_metrics_computation() -> None:
    """Test forecasting metrics calculation."""
    y_true = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 310.0, 390.0])
    y_train = np.array([100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0] * 5)

    metrics = compute_all_metrics(y_true, y_pred, y_train=y_train, seasonality=4)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "smape" in metrics
    assert "r2" in metrics
    assert "mase" in metrics
    assert metrics["mae"] == pytest.approx(10.0)


def test_early_stopping() -> None:
    """Test early stopping logic."""
    es = EarlyStopping(patience=2)
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.95)  # patience 1
    assert es.step(0.96)      # patience 2 -> stop
    assert es.best_loss == pytest.approx(0.9)


def test_trainer_and_inference_pipeline(tmp_path: Path) -> None:
    """Smoke test training, checkpointing, and predict_next_24h inference."""
    cfg = NHiTSConfig(
        max_epochs=2,
        batch_size=16,
        hidden_size=32,
        n_stacks=2,
        pooling_kernel_sizes=[4, 1],
        n_freq_downsample=[4, 1],
        checkpoint_dir=tmp_path / "checkpoints",
        output_dir=tmp_path / "outputs",
    )

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = NHiTSModel.from_config(cfg)

    trainer = Trainer(model, cfg, device="cpu")
    history = trainer.fit(train_loader, val_loader)

    assert len(history["train_loss"]) == 2
    ckpt_file = cfg.checkpoint_dir / "best_model.pt"
    assert ckpt_file.exists()

    loaded_model = load_model(cfg, checkpoint_path=ckpt_file, device="cpu")
    preds, targets = collect_predictions(loaded_model, test_loader, torch.device("cpu"))
    assert len(preds) > 0
    assert len(targets) > 0

    # Demo 24h inference test
    past_demand = np.random.randn(168) * 100 + 3000
    past_covs = np.random.randn(168, 4)
    fut_covs = np.random.randn(24, 4)

    forecast_mw = predict_next_24h(
        model=loaded_model,
        past_demand_mw=past_demand,
        past_covariates=past_covs,
        future_covariates=fut_covs,
        scale_mean=3000.0,
        scale_std=500.0,
        device="cpu",
    )
    assert forecast_mw.shape == (24,)
