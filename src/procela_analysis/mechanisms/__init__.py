"""
Mechanism analysis for Procela hypothesis memory.

Provides per-mechanism performance profiling and population-level
ecological analysis of competing theories. MechanismProfiler focuses
on individual accuracy, falsifiability, and redundancy.
MechanismEcology studies dominance, diversity, extinction, and
turnover dynamics across the mechanism population.
"""

from .ecology import MechanismEcology
from .profiler import MechanismProfiler

__all__ = [
    "MechanismEcology",
    "MechanismProfiler",
]
