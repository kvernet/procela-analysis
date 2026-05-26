"""
Generic epistemic signal computations over hypothesis memory.

These functions compute standard signals (rolling error, coverage,
fragility, disagreement) from the DataFrames produced by
MemoryReader. They are domain-agnostic: no Procela core dependency,
no built-in thresholds, no governance logic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal validation
# ---------------------------------------------------------------------------

_ERRORS_REQUIRED = {"step", "variable", "mechanism", "absolute_error"}
_HYPOTHESES_REQUIRED = {"step", "variable", "mechanism", "proposed", "confidence"}


def _validate_columns(
    df: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    """
    Validate that a DataFrame contains required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required : set of str
        Set of required column names.
    label : str
        Human-readable label for error messages.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    if df.empty:
        raise ValueError(f"{label} frame is empty.")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. " f"Expected: {required}."
        )


def _validate_window(window: int) -> None:
    """
    Validate rolling window parameter.

    Parameters
    ----------
    window : int
        Rolling window size.

    Raises
    ------
    ValueError
        If window is less than 1.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")


def _validate_sigma(sigma: float) -> None:
    """
    Validate coverage kernel parameter.

    Parameters
    ----------
    sigma : float
        Exponential kernel scaling parameter.

    Raises
    ------
    ValueError
        If sigma is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}.")


def _validate_alpha(alpha: float) -> None:
    """
    Validate EWMA smoothing parameter.

    Parameters
    ----------
    alpha : float
        Smoothing factor.

    Raises
    ------
    ValueError
        If alpha is not in (0, 1].
    """
    if not (0 < alpha <= 1):
        raise ValueError(f"alpha must be in (0, 1], got {alpha}.")


# ---------------------------------------------------------------------------
# Signal functions
# ---------------------------------------------------------------------------


def rolling_error(
    errors: pd.DataFrame,
    variable: str,
    window: int = 10,
) -> pd.DataFrame:
    """
    Compute rolling mean absolute error across all mechanisms.

    Aggregates per-mechanism errors into a single per-step error
    signal by taking the mean across mechanisms, then applies a
    rolling window.

    Parameters
    ----------
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    variable : str
        Variable name to filter on.
    window : int
        Rolling window size in steps. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``step``, ``mean_error``, ``rolling_error``.

    Raises
    ------
    ValueError
        If the errors frame is missing required columns, is empty,
        or if ``window < 1``.
    """
    _validate_columns(errors, _ERRORS_REQUIRED, "errors")
    _validate_window(window)

    err = errors[errors["variable"] == variable].copy()

    if err.empty:
        raise ValueError(f"No errors found for variable '{variable}'.")

    step_error = (
        err.groupby("step")["absolute_error"]
        .mean()
        .reset_index()
        .rename(columns={"absolute_error": "mean_error"})
        .sort_values("step")
    )

    step_error["rolling_error"] = (
        step_error["mean_error"].rolling(window=window, min_periods=1).mean()
    )

    return step_error.reset_index(drop=True)


def coverage(
    hypotheses: pd.DataFrame,
    errors: pd.DataFrame,
    variable: str,
    mechanism: str,
    sigma: float = 5.0,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """
    Compute exponentially smoothed coverage for a single mechanism.

    Coverage measures how close a mechanism's proposals are to the
    resolved value, transformed through an exponential kernel and
    smoothed with EWMA.

    At each step::

        cov(t) = exp(-|proposed - resolved| / sigma)
        smoothed(t) = alpha * cov(t) + (1 - alpha) * smoothed(t-1)

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    variable : str
        Variable name to filter on.
    mechanism : str
        Mechanism name to compute coverage for.
    sigma : float
        Scaling parameter for the exponential kernel.
        Higher = more tolerant of large errors. Must be > 0.
    alpha : float
        EWMA smoothing factor in (0, 1].
        Higher = more weight on recent observations.

    Returns
    -------
    pd.DataFrame
        Columns: ``step``, ``raw_coverage``, ``smoothed_coverage``.

    Raises
    ------
    ValueError
        If required columns are missing, frames are empty,
        or parameters are invalid.
    """
    _validate_columns(hypotheses, _HYPOTHESES_REQUIRED, "hypotheses")
    _validate_columns(errors, _ERRORS_REQUIRED, "errors")
    _validate_sigma(sigma)
    _validate_alpha(alpha)

    err = errors[
        (errors["variable"] == variable) & (errors["mechanism"] == mechanism)
    ].copy()

    if err.empty:
        raise ValueError(
            f"No errors found for variable '{variable}', " f"mechanism '{mechanism}'."
        )

    err = err.sort_values("step")

    err["raw_coverage"] = np.exp(-err["absolute_error"] / sigma)

    err["smoothed_coverage"] = err["raw_coverage"].ewm(alpha=alpha, adjust=False).mean()

    return err[["step", "raw_coverage", "smoothed_coverage"]].reset_index(drop=True)


def fragility(
    hypotheses: pd.DataFrame,
    variable: str,
    window: int = 5,
) -> pd.DataFrame:
    """
    Compute policy fragility.

    Policy fragility is computed as the normalized range of proposed
    values among competing hypotheses.

    Fragility measures how much mechanisms disagree about what
    the variable's value should be. High fragility indicates
    the epistemic state is not consolidated — different ontologies
    imply different outcomes.

    At each step::

        frag(t) = (max(proposed) - min(proposed)) / max_range
        smoothed(t) = mean of frag over window

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    variable : str
        Variable name to filter on.
    window : int
        Smoothing window size in steps. Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Columns: ``step``, ``num_hypotheses``, ``proposed_min``,
        ``proposed_max``, ``proposed_range``, ``raw_fragility``,
        ``smoothed_fragility``.

    Raises
    ------
    ValueError
        If required columns are missing, frame is empty,
        or ``window < 1``.

    Notes
    -----
    Fragility values are comparable only within the same simulation
    realization. The normalizer is the maximum observed range across
    the full run, making values path-dependent. For cross-run
    comparisons, normalize externally using known domain bounds.
    """
    _validate_columns(hypotheses, _HYPOTHESES_REQUIRED, "hypotheses")
    _validate_window(window)

    hyp = hypotheses[hypotheses["variable"] == variable].copy()

    if hyp.empty:
        raise ValueError(f"No hypotheses found for variable '{variable}'.")

    step_stats = (
        hyp.groupby("step")
        .agg(
            num_hypotheses=("proposed", "count"),
            proposed_min=("proposed", "min"),
            proposed_max=("proposed", "max"),
        )
        .reset_index()
        .sort_values("step")
    )

    step_stats["proposed_range"] = (
        step_stats["proposed_max"] - step_stats["proposed_min"]
    )

    max_range = step_stats["proposed_range"].max()
    if max_range > 0:
        step_stats["raw_fragility"] = step_stats["proposed_range"] / max_range
    else:
        step_stats["raw_fragility"] = 0.0

    step_stats["smoothed_fragility"] = (
        step_stats["raw_fragility"].rolling(window=window, min_periods=1).mean()
    )

    return step_stats[
        [
            "step",
            "num_hypotheses",
            "proposed_min",
            "proposed_max",
            "proposed_range",
            "raw_fragility",
            "smoothed_fragility",
        ]
    ].reset_index(drop=True)


def disagreement_index(
    hypotheses: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """
    Compute a normalized disagreement index among mechanisms.

    Uses tanh normalization of the coefficient of variation to
    produce a bounded score in [0, 1) that preserves ordering.

    At each step::

        cv = std(proposed) / (|mean(proposed)| + epsilon)
        disagreement = tanh(cv)

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    variable : str
        Variable name to filter on.

    Returns
    -------
    pd.DataFrame
        Columns: ``step``, ``mean_proposed``, ``std_proposed``,
        ``disagreement_index``.

    Raises
    ------
    ValueError
        If required columns are missing or frame is empty.
    """
    _validate_columns(hypotheses, _HYPOTHESES_REQUIRED, "hypotheses")

    hyp = hypotheses[hypotheses["variable"] == variable].copy()

    if hyp.empty:
        raise ValueError(f"No hypotheses found for variable '{variable}'.")

    step_stats = (
        hyp.groupby("step")
        .agg(
            mean_proposed=("proposed", "mean"),
            std_proposed=("proposed", "std"),
        )
        .reset_index()
        .sort_values("step")
    )

    # Handle singleton groups where std is NaN
    step_stats["std_proposed"] = step_stats["std_proposed"].fillna(0.0)

    step_stats["disagreement_index"] = np.tanh(
        step_stats["std_proposed"] / (step_stats["mean_proposed"].abs() + 1e-10)
    )

    return step_stats.reset_index(drop=True)


def confidence_spread(
    hypotheses: pd.DataFrame,
    variable: str,
) -> pd.DataFrame:
    """
    Compute the spread of confidence scores among mechanisms.

    High spread = some mechanisms are much more confident than
    others. Low spread = mechanisms have similar confidence.

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    variable : str
        Variable name to filter on.

    Returns
    -------
    pd.DataFrame
        Columns: ``step``, ``mean_confidence``, ``min_confidence``,
        ``max_confidence``, ``confidence_spread``.

    Raises
    ------
    ValueError
        If required columns are missing or frame is empty.
    """
    _validate_columns(hypotheses, _HYPOTHESES_REQUIRED, "hypotheses")

    hyp = hypotheses[hypotheses["variable"] == variable].copy()

    if hyp.empty:
        raise ValueError(f"No hypotheses found for variable '{variable}'.")

    step_stats = (
        hyp.groupby("step")
        .agg(
            mean_confidence=("confidence", "mean"),
            min_confidence=("confidence", "min"),
            max_confidence=("confidence", "max"),
        )
        .reset_index()
        .sort_values("step")
    )

    step_stats["confidence_spread"] = (
        step_stats["max_confidence"] - step_stats["min_confidence"]
    )

    return step_stats.reset_index(drop=True)
