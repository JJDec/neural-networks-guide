# Architecture Selection

If user specifies architecture

↓

Use requested architecture.

Otherwise recommend.

Small dataset

↓

LSTM

GRU

TCN

Long horizon

↓

TiDE

PatchTST

N-HiTS

Large multivariate

↓

TiDE

PatchTST

TimesNet

Very long sequences

↓

PatchTST

Informer

iTransformer

Need interpretability

↓

N-BEATS

Linear

DLinear

Fast inference

↓

TiDE

DLinear

TCN

Always explain

why the architecture was selected.