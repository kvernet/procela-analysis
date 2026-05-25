"""
Memory analysis module for Procela hypothesis data.

Provides DataFrame schemas, memory reading, and epistemic signal
computations over Procela variable histories. This module is the
primary entry point for converting raw Procela memory into
structured data ready for mechanism profiling, regime detection,
policy analysis, and audit reporting.

Key components
--------------
MemoryReader : Convert Procela variables to DataFrames
frames : Canonical DataFrame schemas and validation
metrics : Generic epistemic signal computations
"""

from .frames import (
    ERRORS_SCHEMA,
    HYPOTHESES_SCHEMA,
    RESOLUTIONS_SCHEMA,
    errors_frame,
    hypotheses_frame,
    resolutions_frame,
    validate_errors,
    validate_hypotheses,
    validate_resolutions,
)
from .metrics import (
    confidence_spread,
    coverage,
    disagreement_index,
    fragility,
    rolling_error,
)
from .reader import MemoryReader

__all__ = [
    # Frames
    "hypotheses_frame",
    "resolutions_frame",
    "errors_frame",
    "validate_hypotheses",
    "validate_resolutions",
    "validate_errors",
    "HYPOTHESES_SCHEMA",
    "RESOLUTIONS_SCHEMA",
    "ERRORS_SCHEMA",
    # Metrics
    "rolling_error",
    "coverage",
    "fragility",
    "disagreement_index",
    "confidence_spread",
    # Reader
    "MemoryReader",
]
