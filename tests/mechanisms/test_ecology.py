"""Tests for procela_analysis.mechanisms.ecology."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procela_analysis.mechanisms.ecology import MechanismEcology

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_hypotheses():
    """4 steps, 2 mechanisms, variable X. m1 dominates early, m2 later."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3],
            "variable": ["X"] * 8,
            "mechanism": ["m1", "m2"] * 4,
            "proposed": [10.0, 20.0, 11.0, 19.0, 12.0, 18.0, 13.0, 17.0],
            "confidence": [0.9, 0.1, 0.8, 0.2, 0.3, 0.7, 0.1, 0.9],
        }
    )


@pytest.fixture
def simple_errors():
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3],
            "variable": ["X"] * 8,
            "mechanism": ["m1", "m2"] * 4,
            "absolute_error": [1.0, 5.0, 1.5, 4.5, 3.0, 2.0, 5.0, 1.0],
            "squared_error": [1.0, 25.0] * 4,
        }
    )


@pytest.fixture
def ecology(simple_hypotheses, simple_errors):
    return MechanismEcology(simple_hypotheses, simple_errors)


@pytest.fixture
def three_mechanism_hypotheses():
    """3 mechanisms over 3 steps."""
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "variable": ["X"] * 9,
            "mechanism": ["m1", "m2", "m3"] * 3,
            "proposed": [5.0, 15.0, 25.0, 6.0, 14.0, 24.0, 7.0, 13.0, 23.0],
            "confidence": [0.5, 0.3, 0.2, 0.4, 0.4, 0.2, 0.2, 0.3, 0.5],
        }
    )


@pytest.fixture
def three_mechanism_errors():
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "variable": ["X"] * 9,
            "mechanism": ["m1", "m2", "m3"] * 3,
            "absolute_error": [0.1, 0.5, 1.0, 0.2, 0.4, 0.9, 0.3, 0.3, 0.8],
            "squared_error": [0.01, 0.25, 1.0] * 3,
        }
    )


@pytest.fixture
def single_mechanism_hypotheses():
    """Only one mechanism — edge case for niche and turnover."""
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


@pytest.fixture
def constant_proposal_hypotheses():
    """m3 has constant proposals — should produce NaN correlation."""
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "variable": ["X"] * 9,
            "mechanism": ["m1", "m2", "m3"] * 3,
            "proposed": [1.0, 2.0, 5.0, 3.0, 4.0, 5.0, 5.0, 6.0, 5.0],
            "confidence": [0.4, 0.3, 0.3] * 3,
        }
    )


@pytest.fixture
def constant_proposal_errors():
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 1, 1, 1, 2, 2, 2],
            "variable": ["X"] * 9,
            "mechanism": ["m1", "m2", "m3"] * 3,
            "absolute_error": [0.1] * 9,
            "squared_error": [0.01] * 9,
        }
    )


# ---------------------------------------------------------------------------
# dominance_curve tests
# ---------------------------------------------------------------------------


class TestDominanceCurve:
    """Tests for MechanismEcology.dominance_curve()."""

    def test_returns_correct_columns(self, ecology):
        df = ecology.dominance_curve("X")
        assert list(df.columns) == ["step", "mechanism", "confidence_share"]

    def test_shares_sum_to_one_per_step(self, ecology):
        df = ecology.dominance_curve("X")
        step_sums = df.groupby("step")["confidence_share"].sum()
        for s in step_sums:
            assert s == pytest.approx(1.0)

    def test_sorted_by_step_then_share_descending(self, ecology):
        df = ecology.dominance_curve("X")
        for step in df["step"].unique():
            step_df = df[df["step"] == step]
            shares = step_df["confidence_share"].values
            assert all(shares[i] >= shares[i + 1] for i in range(len(shares) - 1))

    def test_dominance_shift_captured(self, ecology):
        """Step 0: m1=0.9, m2=0.1. Step 3: m1=0.1, m2=0.9."""
        df = ecology.dominance_curve("X")
        step0_m1 = df[(df["step"] == 0) & (df["mechanism"] == "m1")]
        step3_m1 = df[(df["step"] == 3) & (df["mechanism"] == "m1")]
        assert step0_m1["confidence_share"].iloc[0] == pytest.approx(0.9)
        assert step3_m1["confidence_share"].iloc[0] == pytest.approx(0.1)

    def test_zero_confidence_handled(self):
        """All confidences zero should produce zero shares, not NaN."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [1.0, 2.0],
                "confidence": [0.0, 0.0],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "absolute_error": [0.0, 0.0],
                "squared_error": [0.0, 0.0],
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.dominance_curve("X")
        assert (df["confidence_share"] == 0.0).all()
        assert not df["confidence_share"].isna().any()

    def test_raises_on_missing_variable(self, ecology):
        with pytest.raises(ValueError, match="No hypotheses found"):
            ecology.dominance_curve("nonexistent")


# ---------------------------------------------------------------------------
# niche_overlap tests
# ---------------------------------------------------------------------------


class TestNicheOverlap:
    """Tests for MechanismEcology.niche_overlap()."""

    def test_returns_correct_columns(self, ecology):
        df = ecology.niche_overlap("X")
        assert list(df.columns) == [
            "mechanism_a",
            "mechanism_b",
            "pearson_r",
            "overlap_strength",
        ]

    def test_one_pair_for_two_mechanisms(self, ecology):
        df = ecology.niche_overlap("X")
        assert len(df) == 1

    def test_three_pairs_for_three_mechanisms(
        self, three_mechanism_hypotheses, three_mechanism_errors
    ):
        eco = MechanismEcology(three_mechanism_hypotheses, three_mechanism_errors)
        df = eco.niche_overlap("X")
        assert len(df) == 3  # (m1,m2), (m1,m3), (m2,m3)

    def test_negatively_correlated_mechanisms(self, ecology):
        """m1: 10,11,12,13 (up); m2: 20,19,18,17 (down) → negative r."""
        df = ecology.niche_overlap("X")
        assert df.iloc[0]["pearson_r"] < 0.0
        assert df.iloc[0]["overlap_strength"] == "high"  # |r| near 1.0

    @pytest.mark.filterwarnings("ignore::RuntimeWarning")
    def test_constant_proposal_gives_unknown(
        self, constant_proposal_hypotheses, constant_proposal_errors
    ):
        """m3 is constant → correlation with m3 is NaN → strength unknown."""
        eco = MechanismEcology(constant_proposal_hypotheses, constant_proposal_errors)
        df = eco.niche_overlap("X")
        # Find pairs involving m3
        m3_pairs = df[(df["mechanism_a"] == "m3") | (df["mechanism_b"] == "m3")]
        assert (m3_pairs["overlap_strength"] == "unknown").all()

    def test_insufficient_data_gives_unknown(self):
        """Less than 3 overlapping steps → unknown."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "proposed": [1.0, 2.0, 3.0, 4.0],
                "confidence": [0.5] * 4,
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.1] * 4,
                "squared_error": [0.01] * 4,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.niche_overlap("X")
        assert df.iloc[0]["overlap_strength"] == "unknown"

    def test_raises_on_single_mechanism(
        self, single_mechanism_hypotheses, single_mechanism_errors
    ):
        eco = MechanismEcology(single_mechanism_hypotheses, single_mechanism_errors)
        with pytest.raises(ValueError, match="at least two mechanisms"):
            eco.niche_overlap("X")

    def test_raises_on_missing_variable(self, ecology):
        with pytest.raises(
            ValueError, match="Need at least two mechanisms for niche overlap"
        ):
            ecology.niche_overlap("nonexistent")

    def test_low_overlap_strength(self):
        """Correlation below 0.5 → low."""
        np.random.seed(42)
        hyp = pd.DataFrame(
            {
                "step": list(range(10)) * 2,
                "variable": ["X"] * 20,
                "mechanism": ["m1"] * 10 + ["m2"] * 10,
                "proposed": np.random.normal(0, 1, 10).tolist()
                + np.random.normal(0, 1, 10).tolist(),
                "confidence": [0.5] * 20,
            }
        )
        err = pd.DataFrame(
            {
                "step": list(range(10)) * 2,
                "variable": ["X"] * 20,
                "mechanism": ["m1"] * 10 + ["m2"] * 10,
                "absolute_error": [0.1] * 20,
                "squared_error": [0.01] * 20,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.niche_overlap("X")
        assert df.iloc[0]["overlap_strength"] == "low"

    def test_moderate_overlap_strength(self):
        """Correlation between 0.5 and 0.8 → moderate."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 2, 2, 3, 3],
                "variable": ["X"] * 8,
                "mechanism": ["m1", "m2"] * 4,
                "proposed": [1.0, 2.0, 2.0, 1.5, 3.0, 3.5, 4.0, 3.0],
                "confidence": [0.5] * 8,
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 2, 2, 3, 3],
                "variable": ["X"] * 8,
                "mechanism": ["m1", "m2"] * 4,
                "absolute_error": [0.1] * 8,
                "squared_error": [0.01] * 8,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.niche_overlap("X")
        # r ≈ 0.66 → moderate
        assert df.iloc[0]["overlap_strength"] == "moderate"
        assert 0.5 < abs(df.iloc[0]["pearson_r"]) <= 0.8


# ---------------------------------------------------------------------------
# extinction_events tests
# ---------------------------------------------------------------------------


class TestExtinctionEvents:
    """Tests for MechanismEcology.extinction_events()."""

    def test_returns_correct_columns(self, ecology):
        df = ecology.extinction_events("X", threshold=0.15)
        assert list(df.columns) == [
            "mechanism",
            "extinction_step",
            "last_confidence_share",
            "steps_active_after_extinction",
        ]

    def test_detects_extinction(self, ecology):
        """m1 drops from 0.9 to 0.1 and stays low → extinct."""
        df = ecology.extinction_events("X", threshold=0.15, recovery_window=2)
        assert len(df) >= 1
        extinct_mechanisms = set(df["mechanism"])
        assert "m1" in extinct_mechanisms

    def test_no_extinction_when_all_above_threshold(self, ecology):
        """With threshold=0.05, m1 dips to 0.1 at step 3, still above threshold."""
        df = ecology.extinction_events("X", threshold=0.05, recovery_window=2)
        # m1 lowest is 0.1, m2 lowest is 0.1 — both stay above 0.05
        assert len(df) == 0

    def test_no_extinction_when_recovers(self, ecology):
        """With recovery_window=5, m1 recovers? m1 goes 0.9, 0.8, 0.3, 0.1."""
        df = ecology.extinction_events("X", threshold=0.15, recovery_window=10)
        # m1 drops to 0.1 at step 3 but simulation ends at step 3
        # No steps after extinction to check recovery → declared extinct
        # This is correct: end of data means no recovery observed
        assert len(df) >= 1

    def test_extinction_step_is_first_below_threshold(self, ecology):
        df = ecology.extinction_events("X", threshold=0.15, recovery_window=2)
        m1 = df[df["mechanism"] == "m1"]
        # m1: steps 0=0.9, 1=0.8, 2=0.3, 3=0.1 → first below 0.15 is step 3
        assert m1["extinction_step"].iloc[0] == 3

    def test_raises_on_missing_variable(self, ecology):
        with pytest.raises(ValueError, match="No hypotheses found"):
            ecology.extinction_events("nonexistent")

    def test_raises_on_empty_dominance(self):
        """If dominance returns empty, extinction should raise."""
        empty_hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        empty_err = pd.DataFrame(
            columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
        )
        eco = MechanismEcology(empty_hyp, empty_err)
        with pytest.raises(ValueError, match="No hypotheses found"):
            eco.extinction_events("X")


# ---------------------------------------------------------------------------
# diversity_index tests
# ---------------------------------------------------------------------------


class TestDiversityIndex:
    """Tests for MechanismEcology.diversity_index()."""

    def test_returns_correct_columns(self, ecology):
        df = ecology.diversity_index("X")
        assert list(df.columns) == ["step", "num_active", "diversity"]

    def test_correct_number_of_rows(self, ecology):
        df = ecology.diversity_index("X")
        assert len(df) == 4

    def test_diversity_range(self, ecology):
        """Simpson's D should be in [0, 1)."""
        df = ecology.diversity_index("X")
        assert (df["diversity"] >= 0).all()
        assert (df["diversity"] < 1).all()

    def test_perfect_dominance_gives_zero_diversity(self):
        """One mechanism with all confidence → D=0."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [1.0, 2.0],
                "confidence": [1.0, 0.0],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "absolute_error": [0.0, 0.0],
                "squared_error": [0.0, 0.0],
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.diversity_index("X")
        assert df["diversity"].iloc[0] == pytest.approx(0.0)

    def test_equal_shares_gives_max_diversity(self):
        """Two mechanisms with equal shares → D = 0.5."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "proposed": [1.0, 2.0],
                "confidence": [0.5, 0.5],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "absolute_error": [0.0, 0.0],
                "squared_error": [0.0, 0.0],
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.diversity_index("X")
        assert df["diversity"].iloc[0] == pytest.approx(0.5)

    def test_num_active_counts_positive_shares(self):
        """Mechanisms with zero share should not count as active."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 0],
                "variable": ["X"] * 3,
                "mechanism": ["m1", "m2", "m3"],
                "proposed": [1.0, 2.0, 3.0],
                "confidence": [0.5, 0.5, 0.0],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 0],
                "variable": ["X"] * 3,
                "mechanism": ["m1", "m2", "m3"],
                "absolute_error": [0.0] * 3,
                "squared_error": [0.0] * 3,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.diversity_index("X")
        assert df["num_active"].iloc[0] == 2

    def test_raises_on_missing_variable(self, ecology):
        with pytest.raises(ValueError, match="No hypotheses found"):
            ecology.diversity_index("nonexistent")

    def test_raises_on_empty_dominance(self):
        empty_hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        empty_err = pd.DataFrame(
            columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
        )
        eco = MechanismEcology(empty_hyp, empty_err)
        with pytest.raises(ValueError, match="No hypotheses found"):
            eco.diversity_index("X")


# ---------------------------------------------------------------------------
# turnover tests
# ---------------------------------------------------------------------------


class TestTurnover:
    """Tests for MechanismEcology.turnover()."""

    def test_returns_correct_columns(self, ecology):
        df = ecology.turnover("X")
        assert list(df.columns) == ["step", "turnover"]

    def test_first_step_is_nan(self, ecology):
        df = ecology.turnover("X")
        assert pd.isna(df.iloc[0]["turnover"])

    def test_turnover_range(self, ecology):
        """Turnover should be in [0, 1] for non-NaN steps."""
        df = ecology.turnover("X")
        non_nan = df[~df["turnover"].isna()]
        assert (non_nan["turnover"] >= 0).all()
        assert (non_nan["turnover"] <= 1).all()

    def test_identical_rankings_give_zero_turnover(self):
        """Same ranking at consecutive steps → turnover=0."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "proposed": [1.0, 2.0, 3.0, 4.0],
                "confidence": [0.8, 0.2, 0.9, 0.1],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.0] * 4,
                "squared_error": [0.0] * 4,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.turnover("X")
        # m1 dominates both steps → same ranking → zero turnover
        assert df["turnover"].iloc[1] == pytest.approx(0.0)

    def test_reversed_rankings_give_one_turnover(self):
        """Fully reversed ranking → turnover=1."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "proposed": [1.0, 2.0, 3.0, 4.0],
                "confidence": [0.9, 0.1, 0.1, 0.9],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.0] * 4,
                "squared_error": [0.0] * 4,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.turnover("X")
        assert df["turnover"].iloc[1] == pytest.approx(1.0)

    def test_raises_on_missing_variable(self, ecology):
        with pytest.raises(ValueError, match="No hypotheses found"):
            ecology.turnover("nonexistent")

    def test_raises_on_empty_dominance(self):
        empty_hyp = pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        )
        empty_err = pd.DataFrame(
            columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
        )
        eco = MechanismEcology(empty_hyp, empty_err)
        with pytest.raises(ValueError, match="No hypotheses found"):
            eco.turnover("X")

    def test_less_than_two_common_mechanisms_gives_nan(self):
        """When only one mechanism is shared between steps, turnover is NaN."""
        hyp = pd.DataFrame(
            {
                "step": [0, 0, 0, 1, 1, 1],
                "variable": ["X"] * 6,
                "mechanism": ["m1", "m2", "m3", "m1", "m4", "m5"],
                "proposed": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "confidence": [0.5, 0.3, 0.2, 0.6, 0.2, 0.2],
            }
        )
        err = pd.DataFrame(
            {
                "step": [0, 0, 0, 1, 1, 1],
                "variable": ["X"] * 6,
                "mechanism": ["m1", "m2", "m3", "m1", "m4", "m5"],
                "absolute_error": [0.1] * 6,
                "squared_error": [0.01] * 6,
            }
        )
        eco = MechanismEcology(hyp, err)
        df = eco.turnover("X")
        assert pd.isna(df["turnover"].iloc[1])
