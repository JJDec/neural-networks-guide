"""Ray Tune search-space definition for N-HiTS hyperparameters.

The search space is exposed as a plain ``dict`` so it can be:

* passed directly to ``ray.tune.Tuner`` as ``param_space``.
* serialised / logged without any Ray dependencies at import time
  (all ``tune.*`` calls are deferred inside the factory function).

Usage
-----
>>> from nhits.tuning.search_space import default_search_space
>>> space = default_search_space()
>>> tuner = Tuner(tune_nhits, param_space=space, ...)
"""

from __future__ import annotations

from typing import Any


def default_search_space() -> dict[str, Any]:
    """Return the default Ray Tune search space for N-HiTS.

    Search dimensions
    -----------------
    lr : log-uniform in [1e-4, 1e-2]
        Learning rate for AdamW.
    weight_decay : log-uniform in [1e-5, 1e-3]
        L2 regularisation strength.
    batch_size : choice of [32, 64, 128]
        Mini-batch size.
    hidden_size : choice of [128, 256, 512]
        Width of all MLP layers.
    num_mlp_layers : choice of [2, 3, 4]
        Number of dense layers per N-HiTS block.
    dropout : uniform in [0.0, 0.4]
        Dropout probability inside MLP blocks.
    loss : choice of ["mae", "mse", "huber"]
        Training objective.
    activation : choice of ["relu", "gelu"]
        Non-linearity applied after each linear layer.

    Fixed during search
    -------------------
    n_stacks=3, pooling_kernel_sizes=[8, 4, 1],
    n_freq_downsample=[8, 4, 1], input_len=168, horizon=24.
    """
    # Import here so the module can be imported even if Ray is not installed
    # (e.g. in unit-test environments that mock the search space dict).
    from ray import tune

    return {
        # ── Optimiser ─────────────────────────────────────────────────────
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        # ── Data loading ──────────────────────────────────────────────────
        "batch_size": tune.choice([32, 64, 128]),
        # ── Model architecture ────────────────────────────────────────────
        "hidden_size": tune.choice([128, 256, 512]),
        "num_mlp_layers": tune.choice([2, 3, 4]),
        "dropout": tune.uniform(0.0, 0.4),
        # ── Objective & activation ────────────────────────────────────────
        "loss": tune.choice(["mae", "mse", "huber"]),
        "activation": tune.choice(["relu", "gelu"]),
    }
