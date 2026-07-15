"""Inference subpackage for TiDE."""
from tide.inference.predict import load_model, predict, predict_next_24h

__all__ = ["load_model", "predict", "predict_next_24h"]
