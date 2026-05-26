"""
Procela analysis tools for epistemic simulation audit.

Provides memory reading, mechanism profiling, regime detection,
policy analysis, ecological modeling of competing theories,
visualization, and self-contained HTML audit report generation.
"""

__version__ = "0.1.0"

from .mechanisms import MechanismEcology, MechanismProfiler
from .memory import (
    ERRORS_SCHEMA,
    HYPOTHESES_SCHEMA,
    RESOLUTIONS_SCHEMA,
    MemoryReader,
    confidence_spread,
    coverage,
    disagreement_index,
    errors_frame,
    fragility,
    hypotheses_frame,
    resolutions_frame,
    rolling_error,
    validate_errors,
    validate_hypotheses,
    validate_resolutions,
)
from .policies import PolicyStability
from .regimes import RegimeDetector, TransitionAnalyzer
from .reports import AuditReport
from .viz import (
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
    # Mechanisms
    "MechanismEcology",
    "MechanismProfiler",
    # Memory
    "hypotheses_frame",
    "resolutions_frame",
    "errors_frame",
    "validate_hypotheses",
    "validate_resolutions",
    "validate_errors",
    "HYPOTHESES_SCHEMA",
    "RESOLUTIONS_SCHEMA",
    "ERRORS_SCHEMA",
    "rolling_error",
    "coverage",
    "fragility",
    "disagreement_index",
    "confidence_spread",
    "MemoryReader",
    # Policies
    "PolicyStability",
    # Regimes
    "RegimeDetector",
    "TransitionAnalyzer",
    # Reports
    "AuditReport",
    # Viz
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
