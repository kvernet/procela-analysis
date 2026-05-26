"""
Regime detection and transition analysis for Procela simulations.

Provides unsupervised detection of structural breaks in mechanism
error patterns and characterization of what changes at regime
boundaries. RegimeDetector finds periods of stable mechanism
performance. TransitionAnalyzer quantifies dominance shifts,
error changes, and transition abruptness.
"""

from .detector import RegimeDetector
from .transitions import TransitionAnalyzer

__all__ = [
    "RegimeDetector",
    "TransitionAnalyzer",
]
