# TiDE

Reference

Long-term Forecasting with TiDE

## Summary

TiDE is an encoder-decoder architecture composed entirely of dense
residual MLP blocks.

No attention.

No recurrent layers.

No convolutions.

## Pipeline

History

↓

Feature projection

↓

Dense encoder

↓

Latent vector

↓

Dense decoder

↓

Temporal decoder

↓

Forecast

## Inputs

Past target values

Historical covariates

Future covariates

Static covariates

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

256–1024

Residual blocks

2–6

Dropout

0.0–0.3

Activation

ReLU

GELU

## Loss

MSE

or

MAE

## Advantages

Very fast

Simple implementation

Scales well

Excellent long-horizon forecasting

## Common mistakes

Random train/test split

No normalization

Ignoring future covariates

Incorrect window generation

Predicting autoregressively instead of simultaneously

## Implementation requirements

Implement

ResidualBlock

FeatureProjection

Encoder

Decoder

TemporalDecoder

TiDEModel

Do not replace dense layers with transformers.

Do not replace residual blocks with LSTMs.