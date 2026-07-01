# Dataset Template

Every generated project should implement a Dataset class.

Responsibilities

• read data
• validate columns
• sort chronologically
• create windows
• convert to tensors

Dataset should expose

__len__()

__getitem__()

Support

• univariate
• multivariate
• future covariates
• historical covariates
• static features

Window generation

History
↓

Forecast Horizon

Return

(
    history,
    historical_covariates,
    future_covariates,
    static_features,
    target
)

when available.

Otherwise omit unused tensors.

Use float32 tensors.

Never perform normalization inside __getitem__().