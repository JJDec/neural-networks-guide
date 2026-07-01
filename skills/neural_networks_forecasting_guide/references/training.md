# Training

Always use

PyTorch

## Optimizer

Default

AdamW

Allow configuration.

## Loss

Regression

- MSE
- MAE
- Huber

Probabilistic

- Quantile
- Gaussian NLL

## Scheduler

Preferred

ReduceLROnPlateau

or

CosineAnnealingLR

## Training loop

Forward

↓

Loss

↓

Backward

↓

Optimizer step

↓

Validation

↓

Checkpoint

## Early stopping

Monitor

Validation loss

Default patience

20 epochs

## Checkpoints

Save

best_model.pt

Include

optimizer state

scheduler state

epoch

loss