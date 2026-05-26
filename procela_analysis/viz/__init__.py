"""
Visualization tools for Procela hypothesis memory analysis.

Provides publication-ready plotting functions for dominance,
error, epistemic signals, regime detection, diversity, and
turnover. All functions return matplotlib Figure objects
suitable for embedding in reports or saving to disk.
"""

from .plots import (
    coverage_timeline,
    diversity_timeline,
    dominance_heatmap,
    dominance_timeline,
    error_timeline,
    fragility_timeline,
    per_mechanism_error,
    regime_bands,
    turnover_timeline,
)

__all__ = [
    "dominance_timeline",
    "dominance_heatmap",
    "error_timeline",
    "per_mechanism_error",
    "fragility_timeline",
    "coverage_timeline",
    "regime_bands",
    "diversity_timeline",
    "turnover_timeline",
]
