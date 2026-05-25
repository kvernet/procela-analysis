"""Tests for procela_analysis.memory.frames."""

import numpy as np
import pandas as pd
import pytest

from procela_analysis.memory.frames import (
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

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_hypotheses_records():
    return [
        {
            "step": 0,
            "variable": "X",
            "mechanism": "contact",
            "proposed": 10.5,
            "confidence": 0.8,
        },
        {
            "step": 0,
            "variable": "X",
            "mechanism": "environmental",
            "proposed": 12.0,
            "confidence": 0.6,
        },
        {
            "step": 1,
            "variable": "X",
            "mechanism": "contact",
            "proposed": 11.0,
            "confidence": 0.9,
        },
    ]


@pytest.fixture
def valid_resolutions_records():
    return [
        {
            "step": 0,
            "variable": "X",
            "resolved": 11.0,
            "policy": "weighted_voting",
            "num_hypotheses": 2,
        },
        {
            "step": 1,
            "variable": "X",
            "resolved": 11.0,
            "policy": "weighted_voting",
            "num_hypotheses": 1,
        },
    ]


@pytest.fixture
def hypotheses_df(valid_hypotheses_records):
    return hypotheses_frame(valid_hypotheses_records)


@pytest.fixture
def resolutions_df(valid_resolutions_records):
    return resolutions_frame(valid_resolutions_records)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestSchemas:
    """Schema constants define the expected columns and types."""

    def test_hypotheses_schema_columns(self):
        assert set(HYPOTHESES_SCHEMA.keys()) == {
            "step",
            "variable",
            "mechanism",
            "proposed",
            "confidence",
        }

    def test_resolutions_schema_columns(self):
        assert set(RESOLUTIONS_SCHEMA.keys()) == {
            "step",
            "variable",
            "resolved",
            "policy",
            "num_hypotheses",
        }

    def test_errors_schema_columns(self):
        assert set(ERRORS_SCHEMA.keys()) == {
            "step",
            "variable",
            "mechanism",
            "absolute_error",
            "squared_error",
        }


# ---------------------------------------------------------------------------
# hypotheses_frame tests
# ---------------------------------------------------------------------------


class TestHypothesesFrame:
    """Tests for hypotheses_frame construction."""

    def test_constructs_from_valid_records(self, valid_hypotheses_records):
        df = hypotheses_frame(valid_hypotheses_records)
        assert len(df) == 3
        assert list(df.columns) == list(HYPOTHESES_SCHEMA.keys())
        assert df["step"].dtype == np.int64
        assert df["proposed"].dtype == np.float64
        assert df["confidence"].dtype == np.float64

    def test_raises_on_empty_records(self):
        with pytest.raises(ValueError, match="empty records"):
            hypotheses_frame([])

    def test_raises_on_missing_keys(self):
        bad_records = [{"step": 0, "variable": "X"}]  # missing most keys
        with pytest.raises(ValueError, match="missing required keys"):
            hypotheses_frame(bad_records)

    def test_preserves_record_order(self, valid_hypotheses_records):
        df = hypotheses_frame(valid_hypotheses_records)
        assert df.iloc[0]["mechanism"] == "contact"
        assert df.iloc[1]["mechanism"] == "environmental"
        assert df.iloc[2]["mechanism"] == "contact"

    def test_string_columns_are_string_dtype(self, valid_hypotheses_records):
        df = hypotheses_frame(valid_hypotheses_records)
        assert pd.api.types.is_string_dtype(df["variable"])
        assert pd.api.types.is_string_dtype(df["mechanism"])


# ---------------------------------------------------------------------------
# resolutions_frame tests
# ---------------------------------------------------------------------------


class TestResolutionsFrame:
    """Tests for resolutions_frame construction."""

    def test_constructs_from_valid_records(self, valid_resolutions_records):
        df = resolutions_frame(valid_resolutions_records)
        assert len(df) == 2
        assert list(df.columns) == list(RESOLUTIONS_SCHEMA.keys())
        assert df["step"].dtype == np.int64
        assert df["resolved"].dtype == np.float64
        assert df["num_hypotheses"].dtype == np.int64

    def test_raises_on_empty_records(self):
        with pytest.raises(ValueError, match="empty records"):
            resolutions_frame([])

    def test_raises_on_missing_keys(self):
        bad_records = [{"step": 0, "variable": "X"}]
        with pytest.raises(ValueError, match="missing required keys"):
            resolutions_frame(bad_records)

    def test_policy_column_is_string(self, valid_resolutions_records):
        df = resolutions_frame(valid_resolutions_records)
        assert pd.api.types.is_string_dtype(df["policy"])
        assert df.iloc[0]["policy"] == "weighted_voting"


# ---------------------------------------------------------------------------
# errors_frame tests
# ---------------------------------------------------------------------------


class TestErrorsFrame:
    """Tests for errors_frame computation."""

    def test_computes_errors_correctly(self, hypotheses_df, resolutions_df):
        df = errors_frame(hypotheses_df, resolutions_df)

        # Step 0: two hypotheses
        step0 = df[df["step"] == 0]
        assert len(step0) == 2

        # contact: |10.5 - 11.0| = 0.5
        contact = step0[step0["mechanism"] == "contact"]
        assert contact["absolute_error"].iloc[0] == pytest.approx(0.5)
        assert contact["squared_error"].iloc[0] == pytest.approx(0.25)

        # environmental: |12.0 - 11.0| = 1.0
        env = step0[step0["mechanism"] == "environmental"]
        assert env["absolute_error"].iloc[0] == pytest.approx(1.0)
        assert env["squared_error"].iloc[0] == pytest.approx(1.0)

        # Step 1: one hypothesis, |11.0 - 11.0| = 0.0
        step1 = df[df["step"] == 1]
        assert len(step1) == 1
        assert step1["absolute_error"].iloc[0] == pytest.approx(0.0)
        assert step1["squared_error"].iloc[0] == pytest.approx(0.0)

    def test_schema_columns_match(self, hypotheses_df, resolutions_df):
        df = errors_frame(hypotheses_df, resolutions_df)
        assert list(df.columns) == list(ERRORS_SCHEMA.keys())

    def test_dtypes_match_schema(self, hypotheses_df, resolutions_df):
        df = errors_frame(hypotheses_df, resolutions_df)
        assert df["step"].dtype == np.int64
        assert df["absolute_error"].dtype == np.float64
        assert df["squared_error"].dtype == np.float64

    def test_raises_on_empty_hypotheses(self, resolutions_df):
        empty_h = pd.DataFrame(columns=list(HYPOTHESES_SCHEMA.keys()))
        with pytest.raises(ValueError, match="Hypotheses frame is empty"):
            errors_frame(empty_h, resolutions_df)

    def test_raises_on_empty_resolutions(self, hypotheses_df):
        empty_r = pd.DataFrame(columns=list(RESOLUTIONS_SCHEMA.keys()))
        with pytest.raises(ValueError, match="Resolutions frame is empty"):
            errors_frame(hypotheses_df, empty_r)

    def test_handles_multiple_variables(self):
        """Errors should be computed per variable correctly."""
        hyp = hypotheses_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "mechanism": "m1",
                    "proposed": 5.0,
                    "confidence": 0.8,
                },
                {
                    "step": 0,
                    "variable": "Y",
                    "mechanism": "m2",
                    "proposed": 20.0,
                    "confidence": 0.7,
                },
            ]
        )
        res = resolutions_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "resolved": 5.0,
                    "policy": "highest",
                    "num_hypotheses": 1,
                },
                {
                    "step": 0,
                    "variable": "Y",
                    "resolved": 25.0,
                    "policy": "highest",
                    "num_hypotheses": 1,
                },
            ]
        )
        df = errors_frame(hyp, res)
        x_err = df[df["variable"] == "X"]["absolute_error"].iloc[0]
        y_err = df[df["variable"] == "Y"]["absolute_error"].iloc[0]
        assert x_err == pytest.approx(0.0)
        assert y_err == pytest.approx(5.0)

    def test_does_not_modify_input_frames(self, hypotheses_df, resolutions_df):
        hyp_before = hypotheses_df.copy()
        res_before = resolutions_df.copy()
        errors_frame(hypotheses_df, resolutions_df)
        pd.testing.assert_frame_equal(hypotheses_df, hyp_before)
        pd.testing.assert_frame_equal(resolutions_df, res_before)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateHypotheses:
    """Tests for validate_hypotheses."""

    def test_passes_on_valid_frame(self, hypotheses_df):
        validate_hypotheses(hypotheses_df)  # Should not raise

    def test_raises_on_empty_frame(self):
        empty = pd.DataFrame(columns=list(HYPOTHESES_SCHEMA.keys()))
        with pytest.raises(ValueError, match="empty hypotheses frame"):
            validate_hypotheses(empty)

    def test_raises_on_wrong_dtype(self, hypotheses_df):
        bad = hypotheses_df.copy()
        bad["step"] = bad["step"].astype(np.float64)
        with pytest.raises(ValueError, match="has dtype float64"):
            validate_hypotheses(bad)


class TestValidateResolutions:
    """Tests for validate_resolutions."""

    def test_passes_on_valid_frame(self, resolutions_df):
        validate_resolutions(resolutions_df)

    def test_raises_on_empty_frame(self):
        empty = pd.DataFrame(columns=list(RESOLUTIONS_SCHEMA.keys()))
        with pytest.raises(ValueError, match="empty resolutions frame"):
            validate_resolutions(empty)

    def test_raises_on_missing_column(self, resolutions_df):
        bad = resolutions_df.drop(columns=["policy"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_resolutions(bad)


class TestValidateErrors:
    """Tests for validate_errors."""

    def test_passes_on_valid_frame(self, hypotheses_df, resolutions_df):
        errors = errors_frame(hypotheses_df, resolutions_df)
        validate_errors(errors)

    def test_raises_on_empty_frame(self):
        empty = pd.DataFrame(columns=list(ERRORS_SCHEMA.keys()))
        with pytest.raises(ValueError, match="empty errors frame"):
            validate_errors(empty)

    def test_raises_on_missing_column(self, hypotheses_df, resolutions_df):
        errors = errors_frame(hypotheses_df, resolutions_df)
        bad = errors.drop(columns=["absolute_error"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_errors(bad)

    def test_raises_on_wrong_dtype(self, hypotheses_df, resolutions_df):
        errors = errors_frame(hypotheses_df, resolutions_df)
        bad = errors.copy()
        bad["absolute_error"] = bad["absolute_error"].astype(np.int64)
        with pytest.raises(ValueError, match="has dtype int64"):
            validate_errors(bad)


# ---------------------------------------------------------------------------
# Edge case: _dtype_compatible behavior
# ---------------------------------------------------------------------------


class TestDtypeCompatible:
    """Tests for internal _dtype_compatible via validation functions."""

    def test_string_dtype_via_object(self):
        """String columns stored as object should validate."""
        df = hypotheses_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "mechanism": "m1",
                    "proposed": 1.0,
                    "confidence": 0.5,
                }
            ]
        )
        # variable column is object dtype, should pass
        validate_hypotheses(df)

    def test_string_dtype_via_stringdtype(self):
        """String columns stored as StringDtype should validate."""
        df = hypotheses_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "mechanism": "m1",
                    "proposed": 1.0,
                    "confidence": 0.5,
                }
            ]
        )
        df["variable"] = df["variable"].astype("string")
        validate_hypotheses(df)  # Should not raise with modern pandas

    def test_int64_matches_int64(self):
        """Exact dtype match should pass."""
        df = hypotheses_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "mechanism": "m1",
                    "proposed": 1.0,
                    "confidence": 0.5,
                }
            ]
        )
        # step is int64, should pass
        validate_hypotheses(df)

    def test_float64_matches_float64(self):
        """Exact float dtype match should pass."""
        df = hypotheses_frame(
            [
                {
                    "step": 0,
                    "variable": "X",
                    "mechanism": "m1",
                    "proposed": 1.0,
                    "confidence": 0.5,
                }
            ]
        )
        validate_hypotheses(df)  # proposed and confidence are float64
