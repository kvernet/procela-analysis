"""
DataFrame schemas and construction utilities for Procela hypothesis memory.

This module defines the canonical DataFrame layouts used throughout
procela_analysis. All analysis modules consume these schemas, making
them the contract between memory reading and analytical computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Schema definitions (documentation, not runtime enforcement)
# ---------------------------------------------------------------------------

HYPOTHESES_SCHEMA: dict[str, str] = {
    "step": "int64",
    "variable": "str",
    "mechanism": "str",
    "proposed": "float64",
    "confidence": "float64",
}

RESOLUTIONS_SCHEMA: dict[str, str] = {
    "step": "int64",
    "variable": "str",
    "resolved": "float64",
    "confidence": "float64",
    "policy": "str",
    "num_hypotheses": "int64",
}

ERRORS_SCHEMA: dict[str, str] = {
    "step": "int64",
    "variable": "str",
    "mechanism": "str",
    "absolute_error": "float64",
    "squared_error": "float64",
}


# ---------------------------------------------------------------------------
# Frame construction
# ---------------------------------------------------------------------------


def hypotheses_frame(
    records: list[dict[str, int | str | float]],
) -> pd.DataFrame:
    """
    Construct a hypotheses DataFrame from raw memory records.

    Parameters
    ----------
    records : list of dict
        Each dict must contain the keys defined in `HYPOTHESES_SCHEMA`.
        Typically produced by `MemoryReader` when iterating over a
        Procela variable's hypothesis history.

    Returns
    -------
    pd.DataFrame
        Columns as defined in `HYPOTHESES_SCHEMA`. One row per
        hypothesis ever proposed to any variable.

    Raises
    ------
    ValueError
        If `records` is empty or any record is missing required keys.
    """
    if not records:
        raise ValueError("Cannot construct hypotheses frame from empty records.")

    required = set(HYPOTHESES_SCHEMA.keys())
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Records missing required keys: {missing}. " f"Expected keys: {required}."
        )

    df = pd.DataFrame(records)
    df = df.astype(HYPOTHESES_SCHEMA)
    # Normalize string columns to object dtype for consistency
    _normalize_str_columns(df, HYPOTHESES_SCHEMA)
    return df


def resolutions_frame(
    records: list[dict[str, int | str | float]],
) -> pd.DataFrame:
    """
    Construct a resolutions DataFrame from raw resolution records.

    Parameters
    ----------
    records : list of dict
        Each dict must contain the keys defined in `RESOLUTIONS_SCHEMA`.
        One record per simulation step per variable.

    Returns
    -------
    pd.DataFrame
        Columns as defined in `RESOLUTIONS_SCHEMA`. One row per
        resolution event.

    Raises
    ------
    ValueError
        If `records` is empty or any record is missing required keys.
    """
    if not records:
        raise ValueError("Cannot construct resolutions frame from empty records.")

    required = set(RESOLUTIONS_SCHEMA.keys())
    missing = required - set(records[0].keys())
    if missing:
        raise ValueError(
            f"Records missing required keys: {missing}. " f"Expected keys: {required}."
        )

    df = pd.DataFrame(records)
    df = df.astype(RESOLUTIONS_SCHEMA)
    _normalize_str_columns(df, RESOLUTIONS_SCHEMA)
    return df


def errors_frame(
    hypotheses: pd.DataFrame,
    resolutions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute per-hypothesis prediction errors.

    Computed by joining hypotheses with their corresponding resolutions.

    Each hypothesis is compared to the resolved value at the same
    step for the same variable. The error is the absolute difference
    between what a mechanism proposed and what was ultimately resolved.

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by `hypotheses_frame`. Must contain columns
        ``step``, ``variable``, ``mechanism``, ``proposed``.
    resolutions : pd.DataFrame
        As produced by `resolutions_frame`. Must contain columns
        ``step``, ``variable``, ``resolved``.

    Returns
    -------
    pd.DataFrame
        Columns as defined in `ERRORS_SCHEMA`. One row per hypothesis
        with computed ``absolute_error`` and ``squared_error``.

    Raises
    ------
    ValueError
        If either DataFrame is empty.
    """
    if hypotheses.empty:
        raise ValueError("Hypotheses frame is empty.")
    if resolutions.empty:
        raise ValueError("Resolutions frame is empty.")

    merged = hypotheses.merge(
        resolutions[["step", "variable", "resolved"]],
        how="left",
        on=["step", "variable"],
    )

    merged["absolute_error"] = (merged["proposed"] - merged["resolved"]).abs()
    merged["squared_error"] = (merged["proposed"] - merged["resolved"]) ** 2

    result = merged[list(ERRORS_SCHEMA.keys())]
    result = result.astype(ERRORS_SCHEMA)
    _normalize_str_columns(result, ERRORS_SCHEMA)
    return result


# ---------------------------------------------------------------------------
# Frame validation
# ---------------------------------------------------------------------------


def validate_hypotheses(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame conforms to the hypotheses schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required columns are missing or dtypes do not match.
    """
    _validate_schema(df, HYPOTHESES_SCHEMA, "hypotheses")


def validate_resolutions(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame conforms to the resolutions schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required columns are missing or dtypes do not match.
    """
    _validate_schema(df, RESOLUTIONS_SCHEMA, "resolutions")


def validate_errors(df: pd.DataFrame) -> None:
    """
    Validate that a DataFrame conforms to the errors schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If required columns are missing or dtypes do not match.
    """
    _validate_schema(df, ERRORS_SCHEMA, "errors")


def _validate_schema(
    df: pd.DataFrame,
    schema: dict[str, str],
    name: str,
) -> None:
    """
    Validate DataFrame columns and dtypes against a schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    schema : dict of str to str
        Mapping of column name to expected numpy dtype string.
    name : str
        Human-readable name for error messages.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If columns are missing or dtypes are incompatible.
    """
    if df.empty:
        raise ValueError(f"Cannot validate empty {name} frame.")

    missing_cols = set(schema.keys()) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{name} frame missing required columns: {missing_cols}. "
            f"Expected: {set(schema.keys())}."
        )

    for col, expected_dtype in schema.items():
        actual_dtype = df[col].dtype
        if not _dtype_compatible(actual_dtype, expected_dtype):
            raise ValueError(
                f"{name} frame column '{col}' has dtype {actual_dtype}, "
                f"expected {expected_dtype}."
            )


def _dtype_compatible(actual: np.dtype, expected: str) -> bool:
    """
    Check if a pandas dtype is compatible with an expected numpy dtype string.

    Parameters
    ----------
    actual : numpy.dtype
        The actual dtype of the column.
    expected : str
        Expected dtype string (e.g., 'int64', 'float64', 'str').

    Returns
    -------
    bool
        True if types are compatible.
    """
    expected_normalized = expected.lower()

    if expected_normalized == "str":
        return actual is object or pd.api.types.is_string_dtype(actual)

    return str(actual) == expected_normalized


def _normalize_str_columns(df: pd.DataFrame, schema: dict[str, str]) -> None:
    """
    Normalize string columns to object dtype for consistency.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to normalize.
    schema : dict of str to str
        Schema mapping of column name.
    """
    for col, expected_dtype in schema.items():
        if col in df.columns and expected_dtype == "str":
            df[col] = df[col].astype(object)
