# Data Preparation

## Dataset inspection

Always inspect

- missing values
- duplicated timestamps
- irregular sampling
- outliers
- feature types

## Time ordering

Sort chronologically.

Never shuffle observations.

## Splitting

Use

Train

↓

Validation

↓

Test

Chronological order only.

Never randomly split time series.

## Scaling

Fit scaler only on training data.

Apply to

Validation

Test

Future inference

Supported scalers

- StandardScaler
- MinMaxScaler
- RobustScaler

## Window generation

Convert the series into supervised learning examples.

Input window

↓

Forecast horizon

Example

History

96 observations

↓

Forecast

24 observations

Support

- overlapping windows
- configurable stride

## Covariates

Separate

Historical covariates

Future known covariates

Static covariates

## PyTorch Dataset

Implement

Dataset

DataLoader

Batching

Automatic tensor conversion