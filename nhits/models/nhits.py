"""N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting in PyTorch.

Architecture (Challu et al., AAAI 2023):
  1. Multi-Rate Input Subsampling — Max-pooling applied to the input lookback window
     at stack-dependent temporal frequencies (coarse, medium, fine).
  2. Block MLP Architecture — Dense neural network per block with ReLU/GELU activations,
     dropout, and optional LayerNorm.
  3. Multi-Rate Hierarchical Interpolation — Linear projections generate low-rate backcast
     and forecast basis coefficients, which are interpolated back to target sequence lengths
     (L and H) using 1D linear interpolation.
  4. Doubly Residual Stacking — Blocks are arranged sequentially. Each block subtracts its
     backcast prediction from the lookback target sequence and adds its forecast prediction
     to the accumulated forecast.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nhits.config import NHiTSConfig


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def _interpolate_1d(x: Tensor, target_len: int) -> Tensor:
    """Interpolate 1D coefficient sequence of shape ``(B, S)`` to ``(B, target_len)``.

    Parameters
    ----------
    x:
        Tensor of coefficients, shape ``(B, S)``.
    target_len:
        Target sequence length to interpolate to.

    Returns
    -------
    Tensor
        Interpolated tensor of shape ``(B, target_len)``.
    """
    if x.size(1) == target_len:
        return x
    if x.size(1) == 1:
        return x.expand(-1, target_len)

    # 1D linear interpolation expects (B, C, S)
    x_3d = x.unsqueeze(1)
    out_3d = F.interpolate(x_3d, size=target_len, mode="linear", align_corners=True)
    return out_3d.squeeze(1)


# ---------------------------------------------------------------------------
# N-HiTS Block
# ---------------------------------------------------------------------------

class NHiTSBlock(nn.Module):
    """Single N-HiTS block with multi-rate input pooling and hierarchical basis interpolation.

    Parameters
    ----------
    input_len:
        Look-back window length L.
    horizon:
        Forecast horizon H.
    num_hist_covariates:
        Number of historical covariate features.
    num_future_covariates:
        Number of future covariate features.
    pooling_kernel_size:
        Max-pooling kernel and stride for input subsampling.
    n_freq_downsample:
        Downsampling factor for basis coefficients (determines coefficient vector lengths).
    hidden_size:
        Width of dense layers inside the MLP block.
    num_mlp_layers:
        Number of dense layers in the MLP stack.
    dropout:
        Dropout probability.
    activation:
        Activation function ("relu" or "gelu").
    use_layer_norm:
        Whether to apply LayerNorm after dense layers.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        pooling_kernel_size: int = 1,
        n_freq_downsample: int = 1,
        hidden_size: int = 512,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.num_hist_covariates = num_hist_covariates
        self.num_future_covariates = num_future_covariates
        self.pooling_kernel_size = pooling_kernel_size
        self.n_freq_downsample = n_freq_downsample

        # ── Input max-pooling layer ───────────────────────────────────────
        if pooling_kernel_size > 1:
            self.pool = nn.MaxPool1d(
                kernel_size=pooling_kernel_size,
                stride=pooling_kernel_size,
                ceil_mode=True,
            )
        else:
            self.pool = nn.Identity()

        # Compute pooled dimensions
        self.pooled_input_len = math.ceil(input_len / pooling_kernel_size)

        flat_dim = (
            self.pooled_input_len                                # pooled past target
            + self.pooled_input_len * num_hist_covariates         # pooled historical covariates
            + horizon * num_future_covariates                    # future covariates
        )

        # ── MLP Stack ─────────────────────────────────────────────────────
        act_layer = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        layers: list[nn.Module] = []
        in_dim = flat_dim
        for _ in range(num_mlp_layers):
            layers.append(nn.Linear(in_dim, hidden_size))
            layers.append(act_layer)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            in_dim = hidden_size

        self.mlp = nn.Sequential(*layers)

        # ── Basis Coefficient Heads ───────────────────────────────────────
        # Number of backcast/forecast coefficients before interpolation
        self.bck_coeff_size = max(1, math.ceil(input_len / n_freq_downsample))
        self.fwd_coeff_size = max(1, math.ceil(horizon / n_freq_downsample))

        self.backcast_head = nn.Linear(hidden_size, self.bck_coeff_size)
        self.forecast_head = nn.Linear(hidden_size, self.fwd_coeff_size)

    def forward(
        self,
        past_target: Tensor,   # (B, L)
        hist_covs: Tensor,     # (B, L, C_hist)
        future_covs: Tensor,   # (B, H, C_fut)
    ) -> tuple[Tensor, Tensor]:
        """Run block forward pass.

        Parameters
        ----------
        past_target:
            Target residual sequence of shape ``(B, L)``.
        hist_covs:
            Historical covariates of shape ``(B, L, C_hist)``.
        future_covs:
            Future covariates of shape ``(B, H, C_fut)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            Synthesized backcast ``(B, L)`` and synthesized forecast ``(B, H)``.
        """
        B = past_target.size(0)

        # 1. Pool target and historical covariates along time dimension
        # (B, L) -> (B, 1, L) -> pool -> (B, 1, L_pool) -> (B, L_pool)
        target_3d = past_target.unsqueeze(1)
        target_pooled = self.pool(target_3d).squeeze(1)

        if self.num_hist_covariates > 0:
            # (B, L, C_hist) -> (B, C_hist, L) -> pool -> (B, C_hist, L_pool) -> (B, L_pool * C_hist)
            hist_3d = hist_covs.transpose(1, 2)
            hist_pooled = self.pool(hist_3d).reshape(B, -1)
        else:
            hist_pooled = torch.empty((B, 0), device=past_target.device)

        fut_flat = future_covs.reshape(B, -1)  # (B, H * C_fut)

        # 2. Concatenate all pooled/flattened inputs
        x_in = torch.cat([target_pooled, hist_pooled, fut_flat], dim=-1)  # (B, flat_dim)

        # 3. Pass through MLP stack
        feat = self.mlp(x_in)  # (B, hidden_size)

        # 4. Generate basis coefficients
        bck_coeffs = self.backcast_head(feat)  # (B, bck_coeff_size)
        fwd_coeffs = self.forecast_head(feat)  # (B, fwd_coeff_size)

        # 5. Synthesize backcast and forecast via 1D linear interpolation
        backcast = _interpolate_1d(bck_coeffs, self.input_len)  # (B, L)
        forecast = _interpolate_1d(fwd_coeffs, self.horizon)    # (B, H)

        return backcast, forecast


# ---------------------------------------------------------------------------
# N-HiTS Model
# ---------------------------------------------------------------------------

class NHiTSModel(nn.Module):
    """N-HiTS: Neural Hierarchical Interpolation for electricity demand forecasting.

    Combines multi-stack hierarchical blocks into a doubly-residual architecture.

    Parameters
    ----------
    input_len:
        Look-back window length (hours).
    horizon:
        Forecast horizon (hours).
    num_hist_covariates:
        Number of historical covariate channels.
    num_future_covariates:
        Number of future covariate channels.
    n_stacks:
        Number of hierarchical stacks.
    n_blocks_per_stack:
        Number of blocks in each stack.
    pooling_kernel_sizes:
        List of max-pooling kernel sizes for each stack.
    n_freq_downsample:
        List of downsampling factors for basis interpolation in each stack.
    hidden_size:
        Width of every dense layer in MLP blocks.
    num_mlp_layers:
        Number of dense layers per block.
    dropout:
        Dropout probability.
    activation:
        Activation function ("relu" or "gelu").
    use_layer_norm:
        Whether to use LayerNorm in MLP blocks.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        n_stacks: int = 3,
        n_blocks_per_stack: int = 1,
        pooling_kernel_sizes: list[int] | None = None,
        n_freq_downsample: list[int] | None = None,
        hidden_size: int = 512,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "relu",
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.num_hist_covariates = num_hist_covariates
        self.num_future_covariates = num_future_covariates

        if pooling_kernel_sizes is None:
            pooling_kernel_sizes = [8, 4, 1]
        if n_freq_downsample is None:
            n_freq_downsample = [8, 4, 1]

        self.blocks = nn.ModuleList()
        for stack_idx in range(n_stacks):
            pool_kernel = pooling_kernel_sizes[stack_idx]
            downsample = n_freq_downsample[stack_idx]
            for _ in range(n_blocks_per_stack):
                self.blocks.append(
                    NHiTSBlock(
                        input_len=input_len,
                        horizon=horizon,
                        num_hist_covariates=num_hist_covariates,
                        num_future_covariates=num_future_covariates,
                        pooling_kernel_size=pool_kernel,
                        n_freq_downsample=downsample,
                        hidden_size=hidden_size,
                        num_mlp_layers=num_mlp_layers,
                        dropout=dropout,
                        activation=activation,
                        use_layer_norm=use_layer_norm,
                    )
                )

    def forward(
        self,
        past_target: Tensor,   # (B, L)
        hist_covs: Tensor,     # (B, L, C_hist)
        future_covs: Tensor,   # (B, H, C_fut)
    ) -> Tensor:
        """Run full N-HiTS forward pass.

        Parameters
        ----------
        past_target:
            Scaled past target values, shape ``(B, L)``.
        hist_covs:
            Historical covariates, shape ``(B, L, C_hist)``.
        future_covs:
            Future known covariates, shape ``(B, H, C_fut)``.

        Returns
        -------
        Tensor
            Forecast of shape ``(B, H)`` in scaled space.
        """
        target_residual = past_target
        forecast = torch.zeros(
            (past_target.size(0), self.horizon),
            dtype=past_target.dtype,
            device=past_target.device,
        )

        # Doubly residual stacking across blocks
        for block in self.blocks:
            backcast_b, forecast_b = block(target_residual, hist_covs, future_covs)
            target_residual = target_residual - backcast_b
            forecast = forecast + forecast_b

        return forecast

    @classmethod
    def from_config(cls, cfg: NHiTSConfig) -> NHiTSModel:
        """Construct an NHiTSModel from an ``NHiTSConfig`` instance."""
        from nhits.config import NHiTSConfig  # local import at runtime if needed
        return cls(
            input_len=cfg.input_len,
            horizon=cfg.horizon,
            num_hist_covariates=cfg.num_hist_covariates,
            num_future_covariates=cfg.num_future_covariates,
            n_stacks=cfg.n_stacks,
            n_blocks_per_stack=cfg.n_blocks_per_stack,
            pooling_kernel_sizes=cfg.pooling_kernel_sizes,
            n_freq_downsample=cfg.n_freq_downsample,
            hidden_size=cfg.hidden_size,
            num_mlp_layers=cfg.num_mlp_layers,
            dropout=cfg.dropout,
            activation=cfg.activation,
            use_layer_norm=cfg.use_layer_norm,
        )
