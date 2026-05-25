"""Tests for procela_analysis.policies.stability."""

from __future__ import annotations

import pandas as pd
import pytest

from procela_analysis.policies.stability import PolicyStability

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def simple_resolutions():
    """Resolutions matching simple_hypotheses."""
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "resolved": [12.0, 13.0, 13.0],
            "confidence": [0.8, 0.7, 0.9],
            "policy": ["weighted_voting"] * 3,
            "num_hypotheses": [2, 2, 2],
        }
    )


@pytest.fixture
def stability(simple_resolutions, simple_hypotheses):
    return PolicyStability(simple_resolutions, simple_hypotheses)


@pytest.fixture
def policy_switch_hypotheses():
    """6 steps with a policy switch at step 3."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "variable": ["X"] * 12,
            "mechanism": ["m1", "m2"] * 6,
            "proposed": [
                5.0,
                15.0,
                6.0,
                14.0,
                7.0,
                13.0,  # steps 0-2
                8.0,
                12.0,
                9.0,
                11.0,
                10.0,
                10.0,  # steps 3-5
            ],
            "confidence": [0.8, 0.2] * 6,
        }
    )


@pytest.fixture
def policy_switch_resolutions():
    """Resolutions with policy switch at step 3."""
    return pd.DataFrame(
        {
            "step": [0, 1, 2, 3, 4, 5],
            "variable": ["X"] * 6,
            "resolved": [7.0, 7.6, 8.2, 10.0, 10.0, 10.0],
            "confidence": [0.8, 0.7, 0.6, 0.9, 0.8, 0.7],
            "policy": [
                "weighted_voting",
                "weighted_voting",
                "weighted_voting",
                "highest_confidence",
                "highest_confidence",
                "highest_confidence",
            ],
            "num_hypotheses": [2] * 6,
        }
    )


@pytest.fixture
def single_policy_hypotheses():
    """Only one mechanism — no disagreement possible."""
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "mechanism": ["m1"] * 3,
            "proposed": [10.0, 11.0, 12.0],
            "confidence": [0.9, 0.8, 0.7],
        }
    )


@pytest.fixture
def single_policy_resolutions():
    return pd.DataFrame(
        {
            "step": [0, 1, 2],
            "variable": ["X"] * 3,
            "resolved": [10.0, 11.0, 12.0],
            "confidence": [0.9, 0.8, 0.7],
            "policy": ["highest_confidence"] * 3,
            "num_hypotheses": [1, 1, 1],
        }
    )


# ---------------------------------------------------------------------------
# timeline() tests
# ---------------------------------------------------------------------------


class TestTimeline:
    """Tests for PolicyStability.timeline()."""

    def test_returns_dataframe_with_correct_columns(self, stability):
        df = stability.timeline("X")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "step",
            "policy",
            "resolved_value",
            "resolved_confidence",
            "num_hypotheses",
            "confidence_range",
        ]

    def test_correct_number_of_rows(self, stability):
        df = stability.timeline("X")
        assert len(df) == 3

    def test_confidence_range_is_max_minus_min(self, stability):
        """Step 0: confidences 0.8, 0.2 → range = 0.6."""
        df = stability.timeline("X")
        step0 = df[df["step"] == 0]
        assert step0["confidence_range"].iloc[0] == pytest.approx(0.6)

    def test_resolved_confidence_present(self, stability):
        df = stability.timeline("X")
        assert df.iloc[0]["resolved_confidence"] == pytest.approx(0.8)

    def test_sorted_by_step(self, stability):
        df = stability.timeline("X")
        assert df["step"].is_monotonic_increasing

    def test_raises_on_missing_variable(self, stability):
        with pytest.raises(ValueError, match="No resolutions found"):
            stability.timeline("nonexistent")

    def test_empty_hypotheses_gives_zero_range(self):
        """When no hypotheses, confidence_range should be 0.0."""
        res = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "resolved": [5.0],
                "confidence": [0.5],
                "policy": ["test"],
                "num_hypotheses": [0],
            }
        )
        hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        stab = PolicyStability(res, hyp)
        df = stab.timeline("X")
        assert df["confidence_range"].iloc[0] == 0.0


# ---------------------------------------------------------------------------
# disagreement() tests
# ---------------------------------------------------------------------------


class TestDisagreement:
    """Tests for PolicyStability.disagreement()."""

    def test_returns_dataframe_with_correct_columns(self, stability):
        df = stability.disagreement("X")
        assert isinstance(df, pd.DataFrame)
        expected = [
            "step",
            "resolved_value",
            "resolved_confidence",
            "actual_policy",
            "highest_conf_value",
            "highest_conf_confidence",
            "median_value",
            "median_confidence",
            "would_highest_differ",
            "would_median_differ",
        ]
        assert list(df.columns) == expected

    def test_highest_conf_selects_highest_confidence_proposal(self, stability):
        """Step 0: m1 confidence 0.8 vs m2 0.2 → highest = m1 = 10.0."""
        df = stability.disagreement("X")
        step0 = df[df["step"] == 0]
        assert step0["highest_conf_value"].iloc[0] == pytest.approx(10.0)
        assert step0["highest_conf_confidence"].iloc[0] == pytest.approx(0.8)

    def test_median_selects_median_proposal(self, stability):
        """Step 0: proposals 10.0, 20.0 → median = 15.0."""
        df = stability.disagreement("X")
        step0 = df[df["step"] == 0]
        assert step0["median_value"].iloc[0] == pytest.approx(15.0)

    def test_median_confidence_is_median_of_confidences(self, stability):
        """Step 0: confidences 0.8, 0.2 → median = 0.5."""
        df = stability.disagreement("X")
        step0 = df[df["step"] == 0]
        assert step0["median_confidence"].iloc[0] == pytest.approx(0.5)

    def test_would_highest_differ_when_different(self, stability):
        """Step 0: resolved=12.0, highest_conf=10.0 → differ=True."""
        df = stability.disagreement("X")
        step0 = df[df["step"] == 0]
        assert step0["would_highest_differ"].iloc[0]

    def test_would_median_differ_when_different(self, stability):
        """Step 0: resolved=12.0, median=15.0 → differ=True."""
        df = stability.disagreement("X")
        step0 = df[df["step"] == 0]
        assert step0["would_median_differ"].iloc[0]

    def test_would_not_differ_when_values_match(self):
        """When resolved = highest_conf = median, no difference."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [10.0, 10.0],
                "confidence": [0.5, 0.5],
            }
        )
        res = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "resolved": [10.0],
                "confidence": [0.5],
                "policy": ["weighted_voting"],
                "num_hypotheses": [2],
            }
        )
        stab = PolicyStability(res, hyp)
        df = stab.disagreement("X")
        assert not df["would_highest_differ"].iloc[0]
        assert not df["would_median_differ"].iloc[0]

    def test_single_mechanism_matches_all_policies(
        self, single_policy_hypotheses, single_policy_resolutions
    ):
        """With one mechanism, all policies should select same value."""
        stab = PolicyStability(single_policy_resolutions, single_policy_hypotheses)
        df = stab.disagreement("X")
        assert (
            not (df["would_highest_differ"]).all()
            or (not df["would_highest_differ"]).all()
        )
        assert (
            not (df["would_median_differ"]).all()
            or (not df["would_median_differ"]).all()
        )

    def test_raises_on_missing_variable(self, stability):
        with pytest.raises(ValueError, match="No hypotheses found"):
            stability.disagreement("nonexistent")

    def test_raises_on_empty_hypotheses(self, simple_resolutions):
        empty_hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        stab = PolicyStability(simple_resolutions, empty_hyp)
        with pytest.raises(ValueError, match="No hypotheses found"):
            stab.disagreement("X")


# ---------------------------------------------------------------------------
# switch_impact() tests
# ---------------------------------------------------------------------------


class TestSwitchImpact:
    """Tests for PolicyStability.switch_impact()."""

    def test_returns_dataframe_with_correct_columns(
        self, policy_switch_hypotheses, policy_switch_resolutions
    ):
        stab = PolicyStability(policy_switch_resolutions, policy_switch_hypotheses)
        df = stab.switch_impact("X")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == [
            "switch_step",
            "from_policy",
            "to_policy",
            "mean_error_before",
            "mean_error_after",
        ]

    def test_detects_policy_switch(
        self, policy_switch_hypotheses, policy_switch_resolutions
    ):
        stab = PolicyStability(policy_switch_resolutions, policy_switch_hypotheses)
        df = stab.switch_impact("X")
        assert len(df) == 1
        assert df.iloc[0]["switch_step"] == 3
        assert df.iloc[0]["from_policy"] == "weighted_voting"
        assert df.iloc[0]["to_policy"] == "highest_confidence"

    def test_no_switches_returns_empty(self, stability):
        df = stability.switch_impact("X")
        assert len(df) == 0
        assert list(df.columns) == [
            "switch_step",
            "from_policy",
            "to_policy",
            "mean_error_before",
            "mean_error_after",
        ]

    def test_mean_error_before_and_after_are_floats(
        self, policy_switch_hypotheses, policy_switch_resolutions
    ):
        stab = PolicyStability(policy_switch_resolutions, policy_switch_hypotheses)
        df = stab.switch_impact("X")
        assert isinstance(df["mean_error_before"].iloc[0], float)
        assert isinstance(df["mean_error_after"].iloc[0], float)

    def test_single_step_returns_empty(self):
        """Only one step — no switch possible."""
        res = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "resolved": [5.0],
                "confidence": [0.5],
                "policy": ["test"],
                "num_hypotheses": [1],
            }
        )
        hyp = pd.DataFrame(
            {
                "step": [0],
                "variable": ["X"],
                "mechanism": ["m1"],
                "proposed": [5.0],
                "confidence": [0.5],
            }
        )
        stab = PolicyStability(res, hyp)
        df = stab.switch_impact("X")
        assert len(df) == 0

    def test_raises_on_missing_variable(self, stability):
        with pytest.raises(ValueError, match="No resolutions found"):
            stability.switch_impact("nonexistent")

    def test_empty_hypotheses_returns_empty(self, policy_switch_resolutions):
        empty_hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        stab = PolicyStability(policy_switch_resolutions, empty_hyp)
        df = stab.switch_impact("X")
        assert len(df) == 0


# ---------------------------------------------------------------------------
# Multi-variable tests
# ---------------------------------------------------------------------------


class TestMultiVariable:
    """Tests that variable filtering works correctly."""

    def test_timeline_filters_by_variable(self):
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 0, 0],
                "variable": ["X", "X", "Y", "Y"],
                "mechanism": ["m1", "m2", "m1", "m2"],
                "proposed": [1.0, 2.0, 10.0, 20.0],
                "confidence": [0.8, 0.2, 0.7, 0.3],
            }
        )
        res = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X", "Y"],
                "resolved": [1.5, 15.0],
                "confidence": [0.8, 0.7],
                "policy": ["test"] * 2,
                "num_hypotheses": [2, 2],
            }
        )
        stab = PolicyStability(res, hyp)
        df_x = stab.timeline("X")
        df_y = stab.timeline("Y")
        assert len(df_x) == 1
        assert len(df_y) == 1

    def test_disagreement_filters_by_variable(self):
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 0, 0],
                "variable": ["X", "X", "Y", "Y"],
                "mechanism": ["m1", "m2", "m1", "m2"],
                "proposed": [1.0, 2.0, 10.0, 20.0],
                "confidence": [0.8, 0.2, 0.7, 0.3],
            }
        )
        res = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X", "Y"],
                "resolved": [1.5, 15.0],
                "confidence": [0.8, 0.7],
                "policy": ["test"] * 2,
                "num_hypotheses": [2, 2],
            }
        )
        stab = PolicyStability(res, hyp)
        df = stab.disagreement("X")
        assert len(df) == 1
        assert df.iloc[0]["actual_policy"] == "test"
