---
name: neural-network-forecasting
description: >
  Generate production-quality Python and PyTorch implementations for
  neural network forecasting models. Supports data preparation,
  architecture-specific implementation, training, evaluation,
  hyperparameter tuning, visualization, and inference.

---

# Neural Network Forecasting Skill

## Purpose

Use this skill whenever the user asks to:

- build a neural network for forecasting
- implement a paper
- implement a specific architecture
- explain an architecture
- generate PyTorch code
- prepare data
- train or evaluate a forecasting model

## Workflow

Always follow this sequence.

1. Analyze the problem
2. Prepare the dataset
3. Select architecture
4. Design the model
5. Generate implementation
6. Train
7. Evaluate
8. Produce inference example

Never skip steps.

## Problem analysis

Determine:

- univariate or multivariate
- forecasting horizon
- input window
- static covariates
- historical covariates
- future covariates
- regression or forecasting
- deterministic or probabilistic forecasting

## Data preparation

Read:

references/data_preparation.md

## Training

Read:

references/training.md

## Evaluation

Read:

references/evaluation.md

## PyTorch implementation

Read:

references/pytorch_guidelines.md

## Architecture

If the user specifies an architecture, read:

architectures/<architecture>.md

Examples

architectures/tide.md

architectures/lstm.md

architectures/tcn.md

architectures/patchtst.md

## Output requirements

Generate modular projects.

Separate

- datasets
- models
- trainers
- metrics
- utilities
- inference

Do not generate one huge script unless explicitly requested.

Always include:

- comments
- docstrings
- type hints
- reproducible random seeds
- GPU support
- checkpointing
- early stopping