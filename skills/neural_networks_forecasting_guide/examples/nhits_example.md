# N-HiTS Example

Architecture

Input / Residual History

↓

Multi-rate Downsampling

↓

MLP Block

↓

├── Backcast

│     ↓

│   Residual History

│

└── Forecast Coefficients

&#x20;     ↓

&#x20;  Interpolation

&#x20;     ↓

&#x20;  Block Forecast

↓

Stacked Multi-resolution Blocks

↓

Forecast Aggregation

↓

Forecast



Inputs

history

historical covariates

future covariates

static features (optional, implementation-dependent)



Recommended

window

168

forecast

24

hidden dimension

256

stacks

3

blocks per stack

2

pooling / downsampling

4, 2, 1

dropout

0.1

optimizer

AdamW

loss

MSE

scheduler

ReduceLROnPlateau



Generate

NHiTSBlock

MLP

BackcastHead

ForecastHead

Interpolation

NHiTSStack

NHiTSModel

Trainer

Evaluation script

Inference script

