# TiDE Example

Architecture

Feature Projection

↓

Dense Encoder

↓

Latent Representation

↓

Dense Decoder

↓

Temporal Decoder

↓

Forecast

Inputs

history

historical covariates

future covariates

static features

Recommended

window

96

forecast

24

hidden dimension

512

residual blocks

4

dropout

0.1

optimizer

AdamW

loss

MSE

scheduler

ReduceLROnPlateau

Generate

ResidualBlock

Encoder

Decoder

TemporalDecoder

TiDEModel

Trainer

Evaluation script

Inference script