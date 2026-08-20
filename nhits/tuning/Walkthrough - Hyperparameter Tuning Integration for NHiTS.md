# Ray Tune + W&B Hyperparameter Tuning Integration for NHiTS

We have successfully integrated **Ray Tune** (with **Optuna TPE** sampling & **ASHA** early stopping) into the PyTorch NHiTS training workflow, fully integrated with **Weights & Biases (W&B)** experiment tracking.

---

## Accomplishments

### 1. New Hyperparameter Tuning Package (`nhits/tuning/`)
- [`nhits/tuning/__init__.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/tuning/__init__.py): Exposes `default_search_space` and `tune_nhits`.
- [`nhits/tuning/search_space.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/tuning/search_space.py): Configures a 8-hyperparameter search space (`lr`, `weight_decay`, `batch_size`, `hidden_size`, `num_mlp_layers`, `dropout`, `loss`, `activation`).
- [`nhits/tuning/trainable.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/tuning/trainable.py): Ray Tune trainable function that links `Trainer` epoch metrics (`val_loss`, `train_loss`) to `tune.report()` for early trial pruning.
- [`nhits/tuning/run_tuning.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/nhits/tuning/run_tuning.py): Complete entry point supporting CLI options, ASHA scheduler, Optuna search, W&B logger callback, Windows MAX_PATH length protection, and optional re-training of the best config.

### 2. Dependency & Configuration Updates
- Added `ray[tune]>=2.10.0` and `optuna>=3.6.0` to [`pyproject.toml`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/pyproject.toml).

### 3. Unit Test Suite
- [`tests/unit/test_tuning_search_space.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/tests/unit/test_tuning_search_space.py): Verifies hyperparameter distribution types and search space structure (9 unit tests passed).
- [`tests/unit/test_tuning_trainable.py`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/tests/unit/test_tuning_trainable.py): Integration smoke test verifying trial execution and checkpoint directory isolation (2 unit tests passed).

---

## Verification & Tuning Results

### Search Execution
Executed run command:
```bash
uv run python nhits/tuning/run_tuning.py --num_samples 6 --max_epochs 10 --grace_period 3 --retrain_best --retrain_epochs 15 --wandb
```

### Best Hyperparameters Found
- **Loss Function**: `huber`
- **Activation**: `gelu`
- **Learning Rate**: `3.34e-4`
- **Weight Decay**: `1.49e-4`
- **Batch Size**: `64`
- **Hidden Size**: `256`
- **Num MLP Layers**: `4`
- **Dropout**: `0.0597`

### Performance Metrics
- **Best Tuning Trial `val_loss`**: `0.01290`
- **Re-trained Best Model `val_loss`** (15 epochs): **`0.01143`**
- **Saved Best Checkpoint**: [`checkpoints/tune/best/nhits/best_model.pt`](file:///c:/Users/Joanna/agy2-projects/neural-networks-project/checkpoints/tune/best/nhits/best_model.pt)

### W&B Experiment Tracking
- Every trial is logged to the `nhits-forecasting` project under entity `j95-jaworska-na` on W&B.
- Dashboard URL: [https://wandb.ai/j95-jaworska-na/nhits-forecasting](https://wandb.ai/j95-jaworska-na/nhits-forecasting)
