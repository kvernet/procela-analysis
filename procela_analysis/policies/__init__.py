"""
Policy stability analysis for Procela resolution policies.

Provides per-step comparison of alternative resolution policies
without simulating counterfactual trajectories.
"""

from .stability import PolicyStability

__all__ = ["PolicyStability"]
