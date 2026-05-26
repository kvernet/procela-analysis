"""
Audit report generation for Procela simulations.

Provides self-contained HTML reports that document the complete
epistemic history of a simulation run, including mechanism dominance,
regime detection, governance actions, and falsifiability analysis.
Designed as supplementary material for simulation-based research.
"""

from .audit import AuditReport

__all__ = ["AuditReport"]
