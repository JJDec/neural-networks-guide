# Example

Task

Forecast one variable.

Dataset

timestamp,value

Pipeline

CSV

↓

Sort timestamps

↓

Normalize

↓

Window generation

↓

Dataset

↓

DataLoader

↓

Model

↓

Training

↓

Evaluation

Example

History

96 observations

↓

Forecast

24 observations

Metrics

MAE

RMSE

MAPE

Plot

Prediction vs Target