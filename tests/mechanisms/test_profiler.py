"""Tests for procela_analysis.mechanisms.profiler."""

from __future__ import annotations

import pandas as pd
import pytest

from procela_analysis.mechanisms.profiler import MechanismProfiler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_hypotheses():
    """3 steps, 2 mechanisms, one variable X."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2],
            "variable": ["X"] * 6,
            "mechanism": ["m1", "m2", "m1", "m2", "m1", "m2"],
            "proposed": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0],
            "confidence": [0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
            "source_key": ["k1", "k2", "k1", "k2", "k1", "k2"],
        }
    )


@pytest.fixture
def simple_errors():
    """Errors matching simple_hypotheses. Resolutions: 15.0 at each step."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2],
            "variable": ["X"] * 6,
            "mechanism": ["m1", "m2", "m1", "m2", "m1", "m2"],
            "absolute_error": [5.0, 5.0, 4.0, 4.0, 3.0, 3.0],
            "squared_error": [25.0, 25.0, 16.0, 16.0, 9.0, 9.0],
        }
    )


@pytest.fixture
def profiler(simple_hypotheses, simple_errors):
    return MechanismProfiler(simple_hypotheses, simple_errors)


@pytest.fixture
def multi_variable_hypotheses():
    """Two variables, two mechanisms each."""
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 0, 1, 1, 1, 1],
            "variable": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            "mechanism": ["m1", "m2", "m1", "m2", "m1", "m2", "m1", "m2"],
            "proposed": [1.0, 2.0, 10.0, 20.0, 1.5, 2.5, 11.0, 19.0],
            "confidence": [0.6, 0.4, 0.7, 0.3, 0.5, 0.5, 0.8, 0.2],
            "source_key": ["a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )


@pytest.fixture
def single_mechanism_hypotheses():
    """Only one mechanism to test edge cases."""
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "mechanism": ["m1"] * 3,
            "proposed": [10.0, 11.0, 12.0],
            "confidence": [0.9, 0.8, 0.7],
            "source_key": ["k1"] * 3,
        }
    )


@pytest.fixture
def single_mechanism_errors():
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "mechanism": ["m1"] * 3,
            "absolute_error": [1.0, 2.0, 3.0],
            "squared_error": [1.0, 4.0, 9.0],
        }
    )


# ---------------------------------------------------------------------------
# dominance tests
# ---------------------------------------------------------------------------


class TestDominance:
    """Tests for MechanismProfiler.dominance()."""

    def test_returns_correct_columns(self, profiler):
        df = profiler.dominance("X")
        assert list(df.columns) == ["step", "mechanism", "confidence_share"]

    def test_confidence_shares_sum_to_one_per_step(self, profiler):
        df = profiler.dominance("X")
        step_sums = df.groupby("step")["confidence_share"].sum()
        for s in step_sums:
            assert s == pytest.approx(1.0)

    def test_high_confidence_gets_high_share(self, profiler):
        df = profiler.dominance("X")
        step0 = df[df["step"] == 0]
        m1_share = step0[step0["mechanism"] == "m1"]["confidence_share"].iloc[0]
        m2_share = step0[step0["mechanism"] == "m2"]["confidence_share"].iloc[0]
        # m1 confidence 0.8, m2 0.2 → shares 0.8 and 0.2
        assert m1_share == pytest.approx(0.8)
        assert m2_share == pytest.approx(0.2)

    def test_sorted_by_step_then_share_descending(self, profiler):
        df = profiler.dominance("X")
        for step in df["step"].unique():
            step_df = df[df["step"] == step]
            shares = step_df["confidence_share"].values
            assert all(shares[i] >= shares[i + 1] for i in range(len(shares) - 1))

    def test_filters_by_variable(self, multi_variable_hypotheses, simple_errors):
        prof = MechanismProfiler(multi_variable_hypotheses, simple_errors)
        df_x = prof.dominance("X")
        assert (df_x["mechanism"].isin(["m1", "m2"])).all()
        # Should not contain Y data
        df_y = prof.dominance("Y")
        assert len(df_y) > 0

    def test_raises_on_missing_variable(self, profiler):
        with pytest.raises(ValueError, match="No hypotheses found"):
            profiler.dominance("nonexistent")

    def test_raises_on_empty_hypotheses(self):
        empty_hyp = pd.DataFrame(
            columns=[
                "step",
                "variable",
                "mechanism",
                "proposed",
                "confidence",
                "source_key",
            ]
        )
        empty_err = pd.DataFrame(
            columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
        )
        prof = MechanismProfiler(empty_hyp, empty_err)
        with pytest.raises(ValueError, match="No hypotheses found"):
            prof.dominance("X")


# ---------------------------------------------------------------------------
# rolling_mae tests
# ---------------------------------------------------------------------------


class TestRollingMAE:
    """Tests for MechanismProfiler.rolling_mae()."""

    def test_returns_correct_columns(self, profiler):
        df = profiler.rolling_mae("X")
        assert list(df.columns) == ["step", "mechanism", "rolling_mae"]

    def test_rolling_mae_with_window_1(self, profiler):
        """Window=1: rolling_mae equals absolute_error."""
        df = profiler.rolling_mae("X", window=1)
        for _, row in df.iterrows():
            original = profiler._errors[
                (profiler._errors["step"] == row["step"])
                & (profiler._errors["mechanism"] == row["mechanism"])
            ]["absolute_error"].iloc[0]
            assert row["rolling_mae"] == pytest.approx(original)

    def test_rolling_mae_decays_for_improving_mechanism(self, profiler):
        """m1 errors: 5.0, 4.0, 3.0 → rolling MAE should decrease."""
        df = profiler.rolling_mae("X", window=3)
        m1 = df[df["mechanism"] == "m1"]
        # Step 2 with window=3: mean of 5.0, 4.0, 3.0 = 4.0
        assert m1[m1["step"] == 2]["rolling_mae"].iloc[0] == pytest.approx(4.0)

    def test_default_window_is_10(self, profiler):
        df = profiler.rolling_mae("X")
        # With only 3 steps and window=10, min_periods=1 means all steps have values
        assert len(df) == 6

    def test_raises_on_missing_variable(self, profiler):
        with pytest.raises(ValueError, match="No errors found"):
            profiler.rolling_mae("nonexistent")


# ---------------------------------------------------------------------------
# influence tests
# ---------------------------------------------------------------------------


class TestInfluence:
    """Tests for MechanismProfiler.influence()."""

    def test_returns_correct_columns(self, profiler):
        df = profiler.influence("X")
        assert list(df.columns) == ["step", "mechanism", "is_influential"]

    def test_high_share_is_influential(self, profiler):
        df = profiler.influence("X", threshold=0.5)
        # m1 has share 0.8 at step 0 → influential
        step0_m1 = df[(df["step"] == 0) & (df["mechanism"] == "m1")]
        assert step0_m1["is_influential"].iloc[0]

    def test_low_share_is_not_influential(self, profiler):
        df = profiler.influence("X", threshold=0.5)
        # m2 has share 0.2 at step 0 → not influential
        step0_m2 = df[(df["step"] == 0) & (df["mechanism"] == "m2")]
        assert not step0_m2["is_influential"].iloc[0]

    def test_default_threshold_is_0_1(self, profiler):
        df = profiler.influence("X")
        # With threshold 0.1, even m2 (share 0.1-0.3) should be influential
        step2_m2 = df[(df["step"] == 2) & (df["mechanism"] == "m2")]
        assert step2_m2["is_influential"].iloc[0]

    def test_very_high_threshold_excludes_all(self, profiler):
        df = profiler.influence("X", threshold=0.95)
        # m1 max share is 0.9 → still below 0.95
        assert not df["is_influential"].all()


# ---------------------------------------------------------------------------
# redundancy tests
# ---------------------------------------------------------------------------


class TestRedundancy:
    """Tests for MechanismProfiler.redundancy()."""

    def test_returns_correct_columns(self, profiler):
        df = profiler.redundancy("X")
        assert list(df.columns) == ["mechanism_a", "mechanism_b", "pearson_r"]

    def test_one_pair_for_two_mechanisms(self, profiler):
        df = profiler.redundancy("X")
        assert len(df) == 1

    def test_negatively_correlated_mechanisms(self, profiler):
        """m1: 10, 11, 12 (increasing); m2: 20, 19, 18 (decreasing) → negative r."""
        df = profiler.redundancy("X")
        assert df.iloc[0]["pearson_r"] < 0.0

    def test_nan_for_insufficient_data(self):
        """Less than 3 overlapping steps should produce NaN."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2", "m1", "m2"],
                "proposed": [1.0, 2.0, 3.0, 4.0],
                "confidence": [0.5] * 4,
                "source_key": ["a", "b", "a", "b"],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2", "m1", "m2"],
                "absolute_error": [0.1] * 4,
                "squared_error": [0.01] * 4,
            }
        )
        prof = MechanismProfiler(hyp, err)
        df = prof.redundancy("X")
        assert pd.isna(df.iloc[0]["pearson_r"])

    def test_raises_on_single_mechanism(
        self, single_mechanism_hypotheses, single_mechanism_errors
    ):
        prof = MechanismProfiler(single_mechanism_hypotheses, single_mechanism_errors)
        with pytest.raises(ValueError, match="at least two mechanisms"):
            prof.redundancy("X")

    def test_raises_on_missing_variable(self, profiler):
        with pytest.raises(ValueError, match="No hypotheses found"):
            profiler.redundancy("nonexistent")

    def test_perfectly_correlated_mechanisms(self):
        """Two mechanisms proposing identical values → r = 1.0."""
        hyp = pd.DataFrame(
            {
                "step": [0, 1, 2, 0, 1, 2],
                "variable": ["X"] * 6,
                "mechanism": ["m1", "m1", "m1", "m2", "m2", "m2"],
                "proposed": [5.0, 10.0, 15.0, 5.0, 10.0, 15.0],
                "confidence": [0.5] * 6,
                "source_key": ["a"] * 3 + ["b"] * 3,
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 1, 2, 0, 1, 2],
                "variable": ["X"] * 6,
                "mechanism": ["m1", "m1", "m1", "m2", "m2", "m2"],
                "absolute_error": [0.0] * 6,
                "squared_error": [0.0] * 6,
            }
        )
        prof = MechanismProfiler(hyp, err)
        df = prof.redundancy("X")
        assert df.iloc[0]["pearson_r"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# falsifiability tests
# ---------------------------------------------------------------------------


class TestFalsifiability:
    """Tests for MechanismProfiler.falsifiability()."""

    def test_returns_correct_columns(self, profiler):
        df = profiler.falsifiability("X")
        assert list(df.columns) == [
            "mechanism",
            "mean_error",
            "error_volatility",
            "steps_active",
            "falsifiability_score",
        ]

    def test_one_row_per_mechanism(self, profiler):
        df = profiler.falsifiability("X")
        assert len(df) == 2

    def test_mean_error_correct(self, profiler):
        """m1 errors: 5.0, 4.0, 3.0 → mean = 4.0."""
        df = profiler.falsifiability("X")
        m1 = df[df["mechanism"] == "m1"]
        assert m1["mean_error"].iloc[0] == pytest.approx(4.0)

    def test_steps_active_correct(self, profiler):
        df = profiler.falsifiability("X")
        m1 = df[df["mechanism"] == "m1"]
        assert m1["steps_active"].iloc[0] == 3

    def test_constant_error_gives_low_volatility(self):
        """Mechanism with identical errors → volatility near 0 → high score."""
        hyp = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "variable": ["X"] * 3,
                "mechanism": ["steady"] * 3,
                "proposed": [5.0, 6.0, 7.0],
                "confidence": [0.5] * 3,
                "source_key": ["k"] * 3,
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "variable": ["X"] * 3,
                "mechanism": ["steady"] * 3,
                "absolute_error": [2.0, 2.0, 2.0],
                "squared_error": [4.0, 4.0, 4.0],
            }
        )
        prof = MechanismProfiler(hyp, err)
        df = prof.falsifiability("X")
        assert df["error_volatility"].iloc[0] == pytest.approx(0.0)
        # mean_error / epsilon → very large score
        assert df["falsifiability_score"].iloc[0] > 100.0

    def test_erratic_error_gives_high_volatility(self):
        """Mechanism with highly variable errors → high volatility → low score."""
        hyp = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "variable": ["X"] * 3,
                "mechanism": ["erratic"] * 3,
                "proposed": [1.0, 100.0, 50.0],
                "confidence": [0.5] * 3,
                "source_key": ["k"] * 3,
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 1, 2],
                "variable": ["X"] * 3,
                "mechanism": ["erratic"] * 3,
                "absolute_error": [0.1, 50.0, 25.0],
                "squared_error": [0.01, 2500.0, 625.0],
            }
        )
        prof = MechanismProfiler(hyp, err)
        df = prof.falsifiability("X")
        # Volatility should be substantial
        assert df["error_volatility"].iloc[0] > 10.0
        # Score should be lower than the steady mechanism
        assert df["falsifiability_score"].iloc[0] < 5.0

    def test_raises_on_missing_variable(self, profiler):
        with pytest.raises(ValueError, match="No errors found"):
            profiler.falsifiability("nonexistent")

    def test_single_mechanism_works(
        self, single_mechanism_hypotheses, single_mechanism_errors
    ):
        prof = MechanismProfiler(single_mechanism_hypotheses, single_mechanism_errors)
        df = prof.falsifiability("X")
        assert len(df) == 1
        assert df.iloc[0]["mechanism"] == "m1"


# ---------------------------------------------------------------------------
# Multi-variable tests
# ---------------------------------------------------------------------------


class TestMultiVariable:
    """Tests that filtering by variable works correctly."""

    def test_dominance_filters_variables(
        self, multi_variable_hypotheses, simple_errors
    ):
        prof = MechanismProfiler(multi_variable_hypotheses, simple_errors)
        df = prof.dominance("X")
        # All steps in result should have exactly 2 mechanisms (m1, m2 for X)
        step_counts = df.groupby("step").size()
        assert (step_counts == 2).all()
