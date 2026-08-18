# N-HiTS

Reference

Long-term Forecasting with N-HiTS

## Summary

N-HiTS (Neural Hierarchical Interpolation for Time Series) is a deep learning architecture designed for long-horizon forecasting. The key idea is that different stacks predict at different temporal resolutions. A coarse stack can efficiently learn long-term/low-frequency structure, while later stacks refine shorter-term/high-frequency behavior.



No attention.



No recurrent layers.



No convolutions.



N-HiTS = stacked residual MLPs + multi-rate hierarchical interpolation.

## Pipeline

N-HiTS pipeline is a multi-scale residual forecasting pipeline built from MLP blocks.



Historical target

&#x20;     │

&#x20;     ├── past window: y\[t-L : t]

&#x20;     │

&#x20;     ├── optional historical/future covariates

&#x20;     │

&#x20;     ▼

┌─────────────────────┐

│   N-HiTS Stack 1    │

│  coarse resolution  │

│       MLP           │

└─────────┬───────────┘

&#x20;         │

&#x20;         ├── backcast ──► subtract from input

&#x20;         │

&#x20;         └── forecast ──► hierarchical interpolation

&#x20;                             │

&#x20;                             ▼

&#x20;                        coarse forecast

&#x20;                             │

&#x20;                             ▼

&#x20;                      accumulated forecast

&#x20;                             │



Residual history = Target − Previous Forecast

&#x20;     │

&#x20;     ▼

┌─────────────────────┐

│   N-HiTS Stack 2    │

│ different resolution│

│       MLP           │

└─────────┬───────────┘

&#x20;         │

&#x20;         ▼

&#x20;      ...

&#x20;         │

&#x20;         ▼

┌─────────────────────┐

│   N-HiTS Stack N    │

│ fine resolution     │

└─────────┬───────────┘

&#x20;         │

&#x20;         ▼

&#x20;    Final forecast = Σ BlockForecasts

&#x20;  y\[t+1 : t+H]

## Inputs

Past target values

Historical covariates

Future covariates

Static covariates (optional, implementation-dependent)

## Residual block

Linear

↓

Activation

↓

Linear

↓

Residual connection

↓

LayerNorm (optional)

## Recommendations

Hidden size

128–512

Residual blocks

2–3

Dropout

0.0–0.2

Activation

ReLU

GELU

## Loss

MSE

or

MAE

or

Huber

or

Quantile Loss

## Advantages

Interpretable

Scales well

Excellent long-horizon forecasting

Effective at handling time series with complex, non-linear patterns

## Common mistakes

Random train/test split

No normalization

Incorrect window generation

Forgetting backcast residuals

Incorrect lookback and horizon design

Missing timestamps

## Implementation requirements

Implement

Windowing

Normalization

Multi-stack N-HiTS blocks

MLP

backcast head

forecast head + interpolation

ResidualBlock

HierarchicalInterpolation

Residual subtraction + forecast aggregation

H-step prediction

Loss + training/validation

N-HiTSModel

Do not replace dense layers with transformers.

Do not replace residual blocks with LSTMs.

