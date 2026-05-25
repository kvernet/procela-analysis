"""Tests for procela_analysis.memory.metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from procela_analysis.memory.metrics import (
    confidence_spread,
    coverage,
    disagreement_index,
    fragility,
    rolling_error,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_errors():
    """3 steps, 2 mechanisms, variable X."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2],
            "variable": ["X"] * 6,
            "mechanism": ["m1", "m2"] * 3,
            "absolute_error": [0.1, 1.0, 0.2, 0.9, 0.1, 1.1],
            "squared_error": [0.01, 1.0] * 3,
        }
    )


@pytest.fixture
def simple_hypotheses():
    """3 steps, 2 mechanisms, variable X."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2],
            "variable": ["X"] * 6,
            "mechanism": ["m1", "m2"] * 3,
            "proposed": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
            "confidence": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
        }
    )


@pytest.fixture
def single_mechanism_errors():
    """One mechanism only."""
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "mechanism": ["m1"] * 3,
            "absolute_error": [0.5, 0.6, 0.4],
            "squared_error": [0.25, 0.36, 0.16],
        }
    )


@pytest.fixture
def single_mechanism_hypotheses():
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "mechanism": ["m1"] * 3,
            "proposed": [5.0, 6.0, 4.0],
            "confidence": [0.9, 0.8, 0.7],
        }
    )


@pytest.fixture
def multi_variable_errors():
    """Two variables."""
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 0, 1, 1, 1, 1],
            "variable": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            "mechanism": ["m1", "m2"] * 4,
            "absolute_error": [0.1, 1.0, 0.5, 0.5, 0.2, 0.9, 0.5, 0.5],
            "squared_error": [0.01, 1.0, 0.25, 0.25] * 2,
        }
    )


# ---------------------------------------------------------------------------
# rolling_error tests
# ---------------------------------------------------------------------------


class TestRollingError:
    """Tests for rolling_error()."""

    def test_returns_correct_columns(self, simple_errors):
        df = rolling_error(simple_errors, "X")
        assert list(df.columns) == ["step", "mean_error", "rolling_error"]

    def test_correct_number_of_rows(self, simple_errors):
        df = rolling_error(simple_errors, "X")
        assert len(df) == 3

    def test_mean_error_aggregates_mechanisms(self, simple_errors):
        """Step 0: (0.1 + 1.0) / 2 = 0.55."""
        df = rolling_error(simple_errors, "X")
        step0 = df[df["step"] == 0]
        assert step0["mean_error"].iloc[0] == pytest.approx(0.55)

    def test_rolling_error_with_window_1_equals_mean_error(self, simple_errors):
        df = rolling_error(simple_errors, "X", window=1)
        for _, row in df.iterrows():
            assert row["rolling_error"] == pytest.approx(row["mean_error"])

    def test_rolling_error_averages_over_window(self, simple_errors):
        """Window=3: all three steps averaged together."""
        df = rolling_error(simple_errors, "X", window=3)
        # Step 2 rolling error = mean of [0.55, 0.55, 0.6] ≈ 0.567
        step2 = df[df["step"] == 2]
        expected = (0.55 + 0.55 + 0.60) / 3
        assert step2["rolling_error"].iloc[0] == pytest.approx(expected)

    def test_sorted_by_step(self, simple_errors):
        df = rolling_error(simple_errors, "X")
        assert df["step"].is_monotonic_increasing

    def test_raises_on_missing_variable(self, simple_errors):
        with pytest.raises(ValueError, match="No errors found"):
            rolling_error(simple_errors, "nonexistent")

    def test_raises_on_invalid_window(self, simple_errors):
        with pytest.raises(ValueError, match="window must be >= 1"):
            rolling_error(simple_errors, "X", window=0)

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"step": [0], "variable": ["X"]})
        with pytest.raises(ValueError, match="missing required columns"):
            rolling_error(bad, "X")

    def test_raises_on_empty_frame(self):
        empty = pd.DataFrame(
            columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
        )
        with pytest.raises(ValueError, match="empty"):
            rolling_error(empty, "X")

    def test_works_with_single_mechanism(self, single_mechanism_errors):
        df = rolling_error(single_mechanism_errors, "X")
        assert len(df) == 3
        assert df.iloc[0]["mean_error"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# coverage tests
# ---------------------------------------------------------------------------


class TestCoverage:
    """Tests for coverage()."""

    def test_returns_correct_columns(self, simple_hypotheses, simple_errors):
        df = coverage(simple_hypotheses, simple_errors, "X", "m1")
        assert list(df.columns) == ["step", "raw_coverage", "smoothed_coverage"]

    def test_raw_coverage_range(self, simple_hypotheses, simple_errors):
        """Coverage should be in (0, 1]."""
        df = coverage(simple_hypotheses, simple_errors, "X", "m1")
        assert (df["raw_coverage"] > 0).all()
        assert (df["raw_coverage"] <= 1).all()

    def test_zero_error_gives_coverage_one(self):
        """When error is 0, coverage should be 1.0."""
        hyp = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "mechanism": ["m1"],
                "proposed": [5.0],
                "confidence": [0.5],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "mechanism": ["m1"],
                "absolute_error": [0.0],
                "squared_error": [0.0],
            }
        )
        df = coverage(hyp, err, "X", "m1")
        assert df["raw_coverage"].iloc[0] == pytest.approx(1.0)

    def test_large_error_gives_low_coverage(self, simple_hypotheses, simple_errors):
        """m2 has errors ~1.0 → coverage should be lower than m1."""
        df_m1 = coverage(simple_hypotheses, simple_errors, "X", "m1")
        df_m2 = coverage(simple_hypotheses, simple_errors, "X", "m2")
        # m2 mean raw_coverage should be lower than m1
        assert df_m2["raw_coverage"].mean() < df_m1["raw_coverage"].mean()

    def test_smoothed_coverage_is_ewma(self, simple_hypotheses, simple_errors):
        """With alpha=1.0, smoothed should equal raw."""
        df = coverage(simple_hypotheses, simple_errors, "X", "m1", alpha=1.0)
        for _, row in df.iterrows():
            assert row["smoothed_coverage"] == pytest.approx(row["raw_coverage"])

    def test_raises_on_missing_mechanism(self, simple_hypotheses, simple_errors):
        with pytest.raises(ValueError, match="No errors found"):
            coverage(simple_hypotheses, simple_errors, "X", "nonexistent")

    def test_raises_on_invalid_sigma(self, simple_hypotheses, simple_errors):
        with pytest.raises(ValueError, match="sigma must be > 0"):
            coverage(simple_hypotheses, simple_errors, "X", "m1", sigma=0)

    def test_raises_on_invalid_alpha(self, simple_hypotheses, simple_errors):
        with pytest.raises(ValueError, match="alpha must be in"):
            coverage(simple_hypotheses, simple_errors, "X", "m1", alpha=0)

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"step": [0]})
        err = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "mechanism": ["m1"],
                "absolute_error": [0.1],
                "squared_error": [0.01],
            }
        )
        with pytest.raises(ValueError, match="missing required columns"):
            coverage(bad, err, "X", "m1")


# ---------------------------------------------------------------------------
# fragility tests
# ---------------------------------------------------------------------------


class TestFragility:
    """Tests for fragility()."""

    def test_returns_correct_columns(self, simple_hypotheses):
        df = fragility(simple_hypotheses, "X")
        assert list(df.columns) == [
            "step",
            "num_hypotheses",
            "proposed_min",
            "proposed_max",
            "proposed_range",
            "raw_fragility",
            "smoothed_fragility",
        ]

    def test_correct_number_of_rows(self, simple_hypotheses):
        df = fragility(simple_hypotheses, "X")
        assert len(df) == 3

    def test_range_is_max_minus_min(self, simple_hypotheses):
        """Step 0: proposed 10, 20 → range = 10."""
        df = fragility(simple_hypotheses, "X")
        step0 = df[df["step"] == 0]
        assert step0["proposed_range"].iloc[0] == pytest.approx(10.0)

    def test_fragility_normalized_by_max_range(self, simple_hypotheses):
        """Max range across all steps is 10, so fragility at step 0 = 10/10 = 1.0."""
        df = fragility(simple_hypotheses, "X")
        step0 = df[df["step"] == 0]
        assert step0["raw_fragility"].iloc[0] == pytest.approx(1.0)

    def test_zero_range_gives_zero_fragility(self):
        """All proposals identical → zero fragility."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "proposed": [5.0, 5.0, 5.0, 5.0],
                "confidence": [0.5] * 4,
            }
        )
        df = fragility(hyp, "X")
        assert (df["raw_fragility"] == 0.0).all()

    def test_smoothed_fragility_is_rolling_mean(self, simple_hypotheses):
        df = fragility(simple_hypotheses, "X", window=3)
        # With window=3, step 2 smoothed = mean of raw at steps 0,1,2
        expected = df["raw_fragility"].mean()
        assert df["smoothed_fragility"].iloc[-1] == pytest.approx(expected)

    def test_num_hypotheses_correct(self, simple_hypotheses):
        df = fragility(simple_hypotheses, "X")
        assert (df["num_hypotheses"] == 2).all()

    def test_raises_on_missing_variable(self, simple_hypotheses):
        with pytest.raises(ValueError, match="No hypotheses found"):
            fragility(simple_hypotheses, "nonexistent")

    def test_raises_on_invalid_window(self, simple_hypotheses):
        with pytest.raises(ValueError, match="window must be >= 1"):
            fragility(simple_hypotheses, "X", window=0)

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"step": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            fragility(bad, "X")

    def test_single_mechanism_gives_zero_range(self, single_mechanism_hypotheses):
        df = fragility(single_mechanism_hypotheses, "X")
        assert (df["proposed_range"] == 0.0).all()
        assert (df["raw_fragility"] == 0.0).all()


# ---------------------------------------------------------------------------
# disagreement_index tests
# ---------------------------------------------------------------------------


class TestDisagreementIndex:
    """Tests for disagreement_index()."""

    def test_returns_correct_columns(self, simple_hypotheses):
        df = disagreement_index(simple_hypotheses, "X")
        assert list(df.columns) == [
            "step",
            "mean_proposed",
            "std_proposed",
            "disagreement_index",
        ]

    def test_correct_number_of_rows(self, simple_hypotheses):
        df = disagreement_index(simple_hypotheses, "X")
        assert len(df) == 3

    def test_disagreement_in_range(self, simple_hypotheses):
        """tanh output should be in [0, 1)."""
        df = disagreement_index(simple_hypotheses, "X")
        assert (df["disagreement_index"] >= 0).all()
        assert (df["disagreement_index"] < 1).all()

    def test_zero_disagreement_when_all_same(self):
        """Identical proposals → std=0 → disagreement=0."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [10.0, 10.0],
                "confidence": [0.5, 0.5],
            }
        )
        df = disagreement_index(hyp, "X")
        assert df["disagreement_index"].iloc[0] == pytest.approx(0.0)

    def test_high_disagreement_when_large_spread(self):
        """Very different proposals → high disagreement."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [0.1, 100.0],
                "confidence": [0.5, 0.5],
            }
        )
        df = disagreement_index(hyp, "X")
        assert df["disagreement_index"].iloc[0] > 0.85

    def test_single_mechanism_gives_zero_disagreement(
        self, single_mechanism_hypotheses
    ):
        df = disagreement_index(single_mechanism_hypotheses, "X")
        assert (df["disagreement_index"] == 0.0).all()

    def test_raises_on_missing_variable(self, simple_hypotheses):
        with pytest.raises(ValueError, match="No hypotheses found"):
            disagreement_index(simple_hypotheses, "nonexistent")

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"step": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            disagreement_index(bad, "X")


# ---------------------------------------------------------------------------
# confidence_spread tests
# ---------------------------------------------------------------------------


class TestConfidenceSpread:
    """Tests for confidence_spread()."""

    def test_returns_correct_columns(self, simple_hypotheses):
        df = confidence_spread(simple_hypotheses, "X")
        assert list(df.columns) == [
            "step",
            "mean_confidence",
            "min_confidence",
            "max_confidence",
            "confidence_spread",
        ]

    def test_correct_number_of_rows(self, simple_hypotheses):
        df = confidence_spread(simple_hypotheses, "X")
        assert len(df) == 3

    def test_spread_is_max_minus_min(self, simple_hypotheses):
        """Step 0: confidences 0.8, 0.2 → spread = 0.6."""
        df = confidence_spread(simple_hypotheses, "X")
        step0 = df[df["step"] == 0]
        assert step0["confidence_spread"].iloc[0] == pytest.approx(0.6)

    def test_mean_confidence_correct(self, simple_hypotheses):
        """Step 0: (0.8 + 0.2) / 2 = 0.5."""
        df = confidence_spread(simple_hypotheses, "X")
        step0 = df[df["step"] == 0]
        assert step0["mean_confidence"].iloc[0] == pytest.approx(0.5)

    def test_zero_spread_when_same_confidence(self):
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [1.0, 2.0],
                "confidence": [0.5, 0.5],
            }
        )
        df = confidence_spread(hyp, "X")
        assert df["confidence_spread"].iloc[0] == pytest.approx(0.0)

    def test_single_mechanism_gives_zero_spread(self, single_mechanism_hypotheses):
        df = confidence_spread(single_mechanism_hypotheses, "X")
        assert (df["confidence_spread"] == 0.0).all()

    def test_raises_on_missing_variable(self, simple_hypotheses):
        with pytest.raises(ValueError, match="No hypotheses found"):
            confidence_spread(simple_hypotheses, "nonexistent")

    def test_raises_on_missing_columns(self):
        bad = pd.DataFrame({"step": [0]})
        with pytest.raises(ValueError, match="missing required columns"):
            confidence_spread(bad, "X")


# ---------------------------------------------------------------------------
# Multi-variable tests
# ---------------------------------------------------------------------------


class TestMultiVariable:
    """Tests that variable filtering works correctly."""

    def test_rolling_error_filters_by_variable(self, multi_variable_errors):
        df_x = rolling_error(multi_variable_errors, "X")
        df_y = rolling_error(multi_variable_errors, "Y")
        assert len(df_x) > 0
        assert len(df_y) > 0
        # X and Y have different error patterns
        assert not df_x["mean_error"].equals(df_y["mean_error"])
