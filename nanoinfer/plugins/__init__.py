"""Optimization plugins for NanoInfer."""

from .optimizer import optimize_model, enable_kv_cache

__all__ = ["optimize_model", "enable_kv_cache"]
