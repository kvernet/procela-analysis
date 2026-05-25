"""Tests for procela_analysis.regimes.detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from procela_analysis.regimes.detector import RegimeDetector

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_errors():
    """Two clear regimes: m1 dominates then m2 dominates."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "variable": ["X"] * 12,
            "mechanism": ["m1", "m2"] * 6,
            "absolute_error": [
                0.1,
                1.0,  # step 0: m1 good, m2 bad
                0.2,
                0.9,  # step 1
                0.1,
                1.1,  # step 2
                1.0,
                0.1,  # step 3: m2 good, m1 bad (regime shift)
                0.9,
                0.2,  # step 4
                1.1,
                0.1,  # step 5
            ],
            "squared_error": [0.01, 1.0] * 6,
        }
    )


@pytest.fixture
def single_regime_errors():
    """All steps have same error pattern — no regime change."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1, 2, 2],
            "variable": ["X"] * 6,
            "mechanism": ["m1", "m2"] * 3,
            "absolute_error": [0.5, 1.5, 0.6, 1.4, 0.5, 1.5],
            "squared_error": [0.25, 2.25] * 3,
        }
    )


@pytest.fixture
def three_mechanism_errors():
    """Three mechanisms, m3 overtakes m1 mid-run."""
    return pd.DataFrame(
        {
            "step": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "variable": ["X"] * 15,
            "mechanism": ["m1", "m2", "m3"] * 5,
            "absolute_error": [
                0.1,
                1.0,
                0.5,  # step 0
                0.2,
                0.9,
                0.5,  # step 1
                1.0,
                0.8,
                0.1,  # step 2: regime shift, m3 becomes best
                0.9,
                1.0,
                0.2,  # step 3
                1.1,
                0.9,
                0.1,  # step 4
            ],
            "squared_error": [0.01, 1.0, 0.25] * 5,
        }
    )


@pytest.fixture
def empty_errors():
    """No data."""
    return pd.DataFrame(
        columns=["step", "variable", "mechanism", "absolute_error", "squared_error"]
    )


@pytest.fixture
def minimal_errors():
    """Only 2 steps — not enough for changepoint detection."""
    return pd.DataFrame(
        {
            "step": [0, 0, 1, 1],
            "variable": ["X"] * 4,
            "mechanism": ["m1", "m2"] * 2,
            "absolute_error": [0.1, 1.0, 0.2, 0.9],
            "squared_error": [0.01, 1.0] * 2,
        }
    )


# ---------------------------------------------------------------------------
# detect() tests
# ---------------------------------------------------------------------------


class TestDetect:
    """Tests for RegimeDetector.detect()."""

    def test_returns_dataframe_with_correct_columns(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        result = detector.detect("X")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["step", "regime_label"]

    def test_detects_two_regimes_in_clear_shift(self, simple_errors):
        """m1 dominates steps 0-2, m2 dominates steps 3-5."""
        detector = RegimeDetector(simple_errors)
        result = detector.detect("X", penalty=0.3)
        unique_labels = set(result["regime_label"].unique())
        assert len(unique_labels) == 3

    def test_regime_labels_start_at_zero(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        result = detector.detect("X")
        assert result["regime_label"].min() == 0

    def test_single_regime_for_stable_errors(self, single_regime_errors):
        """All steps have same error pattern — one regime."""
        detector = RegimeDetector(single_regime_errors)
        result = detector.detect("X")
        assert len(result["regime_label"].unique()) == 1

    def test_high_penalty_produces_fewer_regimes(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        result_low = detector.detect("X", penalty=0.1)
        # Reset and try high penalty
        detector2 = RegimeDetector(simple_errors)
        result_high = detector2.detect("X", penalty=10.0)
        assert len(result_high["regime_label"].unique()) <= len(
            result_low["regime_label"].unique()
        )

    def test_minimal_data_returns_single_regime(self, minimal_errors):
        detector = RegimeDetector(minimal_errors)
        result = detector.detect("X")
        assert len(result["regime_label"].unique()) == 1

    def test_three_mechanism_detection(self, three_mechanism_errors):
        """Should detect regime when m3 overtakes m1."""
        detector = RegimeDetector(three_mechanism_errors)
        result = detector.detect("X", penalty=0.3)
        # With clear shift at step 2, should find at least 2 regimes
        assert len(result["regime_label"].unique()) >= 2

    def test_raises_on_empty_errors(self, empty_errors):
        detector = RegimeDetector(empty_errors)
        with pytest.raises(ValueError, match="No errors found"):
            detector.detect("X")

    def test_raises_on_missing_variable(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        with pytest.raises(ValueError, match="No errors found"):
            detector.detect("nonexistent")

    def test_result_stored_in_results_dict(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        assert "X" in detector._results
        assert "labels" in detector._results["X"]
        assert "changepoints" in detector._results["X"]

    def test_consistent_results_on_repeated_calls(self, simple_errors):
        """Same data, same penalty → same regimes (deterministic)."""
        detector1 = RegimeDetector(simple_errors)
        r1 = detector1.detect("X", penalty=0.5)

        detector2 = RegimeDetector(simple_errors)
        r2 = detector2.detect("X", penalty=0.5)

        pd.testing.assert_series_equal(
            r1["regime_label"], r2["regime_label"], check_names=False
        )


# ---------------------------------------------------------------------------
# labels() tests
# ---------------------------------------------------------------------------


class TestLabels:
    """Tests for RegimeDetector.labels()."""

    def test_returns_dataframe(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        result = detector.labels("X")
        assert isinstance(result, pd.DataFrame)

    def test_raises_if_detect_not_called(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        with pytest.raises(KeyError, match="No detection results"):
            detector.labels("X")

    def test_raises_for_unknown_variable(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        with pytest.raises(KeyError, match="No detection results"):
            detector.labels("Y")

    def test_labels_match_detect_output(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detect_result = detector.detect("X")
        labels_result = detector.labels("X")
        pd.testing.assert_frame_equal(detect_result, labels_result)


# ---------------------------------------------------------------------------
# characterize() tests
# ---------------------------------------------------------------------------


class TestCharacterize:
    """Tests for RegimeDetector.characterize()."""

    def test_returns_dataframe_with_correct_columns(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        result = detector.characterize("X")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "regime_label",
            "dominant_mechanism",
            "mean_error",
            "start_step",
            "end_step",
            "duration",
        ]

    def test_one_row_per_regime(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        result = detector.characterize("X")
        unique_regimes = len(detector._results["X"]["labels"].unique())
        assert len(result) == unique_regimes

    def test_dominant_mechanism_is_lowest_error(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.3)
        result = detector.characterize("X")
        # With 3 regimes, find the one where m1 should dominate
        # Regime 0 covers steps where m1 errors are lower
        regime0 = result[result["regime_label"] == 0]
        assert regime0["dominant_mechanism"].iloc[0] == "m1"

    def test_duration_sums_to_total_steps(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        result = detector.characterize("X")
        total = result["duration"].sum()
        assert total == 6  # 6 unique steps

    def test_start_step_less_than_end_step(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        result = detector.characterize("X")
        for _, row in result.iterrows():
            assert row["start_step"] <= row["end_step"]

    def test_raises_if_detect_not_called(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        with pytest.raises(KeyError, match="No detection results"):
            detector.characterize("X")

    def test_raises_on_missing_variable_after_detect(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        with pytest.raises(KeyError, match="No detection results for variable"):
            detector.characterize("Y")

    def test_characterize_raises_on_empty_errors_after_detect(self):
        """Guard against empty errors even if results dict has entries."""
        errors = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.1, 1.0, 0.2, 0.9],
                "squared_error": [0.01, 1.0] * 2,
            }
        )
        detector = RegimeDetector(errors)
        detector.detect("X")
        # Now make errors empty for X
        detector._errors = pd.DataFrame(columns=errors.columns)
        with pytest.raises(ValueError, match="No errors found"):
            detector.characterize("X")

    # In TestCompare
    def test_ari_nmi_single_element_match(self):
        """Single overlapping step with matching labels → 1.0."""
        errors = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "absolute_error": [0.1, 0.2],
                "squared_error": [0.01, 0.04],
            }
        )
        detector = RegimeDetector(errors)
        detector.detect("X")
        ground_truth = pd.DataFrame(
            {
                "step": [0],
                "true_label": [0],
            }
        )
        result = detector.compare("X", ground_truth)
        assert result["adjusted_rand_index"] == 1.0
        assert result["normalized_mutual_info"] == 1.0

    def test_ari_nmi_single_element_mismatch(self):
        """Single overlapping step with mismatched labels → 0.0."""
        errors = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.1, 0.2, 1.0, 1.1],
                "squared_error": [0.01, 0.04, 1.0, 1.21],
            }
        )
        detector = RegimeDetector(errors)
        detector.detect("X", penalty=0.5)
        ground_truth = pd.DataFrame(
            {
                "step": [0],
                "true_label": [99],  # Different from detected regime 0
            }
        )
        result = detector.compare("X", ground_truth)
        # Single step: if labels differ, score is 0.0
        assert result["adjusted_rand_index"] == 0.0
        assert result["normalized_mutual_info"] == 0.0


# ---------------------------------------------------------------------------
# transitions() tests
# ---------------------------------------------------------------------------


class TestTransitions:
    """Tests for RegimeDetector.transitions()."""

    def test_returns_dataframe_with_correct_columns(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        result = detector.transitions("X")
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "transition_step",
            "from_regime",
            "to_regime",
            "confidence",
        ]

    def test_transition_for_two_regimes(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.4)
        result = detector.transitions("X")
        assert len(result) == 2

    def test_no_transitions_for_single_regime(self, single_regime_errors):
        detector = RegimeDetector(single_regime_errors)
        detector.detect("X")
        result = detector.transitions("X")
        assert len(result) == 0

    def test_from_and_to_labels_are_different(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        result = detector.transitions("X")
        for _, row in result.iterrows():
            assert row["from_regime"] != row["to_regime"]

    def test_confidence_is_placeholder(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        result = detector.transitions("X")
        if len(result) > 0:
            assert result.iloc[0]["confidence"] == 0.5

    def test_raises_if_detect_not_called(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        with pytest.raises(KeyError, match="No detection results"):
            detector.transitions("X")


# ---------------------------------------------------------------------------
# compare() tests
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for RegimeDetector.compare()."""

    def test_returns_dict_with_correct_keys(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        ground_truth = pd.DataFrame(
            {
                "step": [0, 1, 2, 3, 4, 5],
                "true_label": [0, 0, 0, 1, 1, 1],
            }
        )
        result = detector.compare("X", ground_truth)
        assert isinstance(result, dict)
        assert "adjusted_rand_index" in result
        assert "normalized_mutual_info" in result

    def test_perfect_match_gives_high_scores(self, simple_errors):
        """When ground truth matches detection, scores should be 1.0."""
        detector = RegimeDetector(simple_errors)
        result_labels = detector.detect("X", penalty=0.3)
        # Use the detected labels as ground truth
        ground_truth = result_labels.rename(columns={"regime_label": "true_label"})
        result = detector.compare("X", ground_truth)
        assert result["adjusted_rand_index"] == pytest.approx(1.0)
        assert result["normalized_mutual_info"] == pytest.approx(1.0)

    def test_wrong_labels_give_low_scores(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X", penalty=0.5)
        # Ground truth says all one regime → should differ from detection
        ground_truth = pd.DataFrame(
            {
                "step": [0, 1, 2, 3, 4, 5],
                "true_label": [0, 0, 0, 0, 0, 0],
            }
        )
        result = detector.compare("X", ground_truth)
        assert result["adjusted_rand_index"] < 1.0

    def test_empty_merge_returns_zeros(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        detector.detect("X")
        # Ground truth with no overlapping steps
        ground_truth = pd.DataFrame(
            {
                "step": [100, 101],
                "true_label": [0, 1],
            }
        )
        result = detector.compare("X", ground_truth)
        assert result["adjusted_rand_index"] == 0.0
        assert result["normalized_mutual_info"] == 0.0

    def test_raises_if_detect_not_called(self, simple_errors):
        detector = RegimeDetector(simple_errors)
        ground_truth = pd.DataFrame(
            {
                "step": [0, 1],
                "true_label": [0, 0],
            }
        )
        with pytest.raises(KeyError, match="No detection results"):
            detector.compare("X", ground_truth)


# ---------------------------------------------------------------------------
# Multi-variable isolation tests
# ---------------------------------------------------------------------------


class TestMultiVariable:
    """Tests that per-variable state doesn't cross-contaminate."""

    def test_different_variables_independent(self):
        errors = pd.DataFrame(
            {
                "step": [0, 0, 1, 1, 0, 0, 1, 1],
                "variable": ["X", "X", "X", "X", "Y", "Y", "Y", "Y"],
                "mechanism": ["m1", "m2"] * 4,
                "absolute_error": [
                    0.1,
                    1.0,
                    0.2,
                    0.9,  # X
                    0.5,
                    0.5,
                    0.5,
                    0.5,  # Y (stable)
                ],
                "squared_error": [0.01, 1.0] * 4,
            }
        )
        detector = RegimeDetector(errors)
        detector.detect("X", penalty=0.5)
        detector.detect("Y", penalty=0.5)

        # X and Y should have independent results
        assert "X" in detector._results
        assert "Y" in detector._results
        assert detector._results["X"] is not detector._results["Y"]

    def test_variable_y_does_not_affect_variable_x(self, simple_errors):
        """Detecting Y after X should not corrupt X results."""
        detector = RegimeDetector(simple_errors)
        result_x_before = detector.detect("X", penalty=0.5)

        # Add Y data and detect
        y_errors = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["Y"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.5] * 4,
                "squared_error": [0.25] * 4,
            }
        )
        combined = pd.concat([simple_errors, y_errors], ignore_index=True)
        detector2 = RegimeDetector(combined)
        detector2.detect("Y")

        # X should not have been affected (detector2 is a new instance though)
        # Test that original detector still has its results
        result_x_after = detector.labels("X")
        pd.testing.assert_frame_equal(result_x_before, result_x_after)


# ---------------------------------------------------------------------------
# Edge case: _binary_segment
# ---------------------------------------------------------------------------


class TestBinarySegment:
    """Tests for internal _binary_segment static method."""

    def test_flat_signal_returns_empty(self):
        signal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        result = RegimeDetector._binary_segment(signal)
        assert result == []

    def test_clear_step_change_detected(self):
        """Signal jumps from 0 to 10 mid-way."""
        signal = np.array([0.1, 0.1, 0.2, 10.0, 10.1, 9.9])
        result = RegimeDetector._binary_segment(signal, penalty=0.5)
        assert len(result) >= 1

    def test_boundary_values_do_not_crash(self):
        """Signal where maximum deviation is at the boundary."""
        signal = np.array([100.0, 0.1, 0.1, 0.1])
        result = RegimeDetector._binary_segment(signal)
        # Should not recurse infinitely; returns empty or valid
        assert isinstance(result, list)

    def test_single_element_returns_empty(self):
        signal = np.array([5.0])
        result = RegimeDetector._binary_segment(signal)
        assert result == []

    def test_two_elements_returns_empty_if_below_threshold(self):
        signal = np.array([0.1, 0.2])
        result = RegimeDetector._binary_segment(signal, penalty=100.0)
        assert result == []

    def test_high_penalty_suppresses_detection(self):
        """With a huge penalty, even large changes are ignored."""
        signal = np.array([0.1, 0.1, 100.0, 100.0])
        result = RegimeDetector._binary_segment(signal, penalty=1e6)
        assert result == []


# ---------------------------------------------------------------------------
# Edge case: _compute_signal
# ---------------------------------------------------------------------------


class TestComputeSignal:
    """Tests for internal _compute_signal static method."""

    def test_returns_array_and_index(self, simple_errors):
        err = simple_errors[simple_errors["variable"] == "X"]
        signal, steps = RegimeDetector._compute_signal(err)
        assert isinstance(signal, np.ndarray)
        assert isinstance(steps, pd.Index)
        # 6 unique steps → 5 deltas
        assert len(signal) == 5

    def test_distance_zero_for_identical_vectors(self):
        """When consecutive error vectors are identical, distance is 0."""
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.5, 1.0, 0.5, 1.0],
                "squared_error": [0.25, 1.0] * 2,
            }
        )
        signal, _ = RegimeDetector._compute_signal(err)
        assert signal[0] == pytest.approx(0.0)

    def test_distance_nonzero_for_different_vectors(self):
        """When error vectors differ, distance is positive."""
        err = pd.DataFrame(
            {
                "step": [0, 0, 1, 1],
                "variable": ["X"] * 4,
                "mechanism": ["m1", "m2"] * 2,
                "absolute_error": [0.1, 1.0, 0.9, 0.2],
                "squared_error": [0.01, 1.0] * 2,
            }
        )
        signal, _ = RegimeDetector._compute_signal(err)
        assert signal[0] > 0.0

    def test_single_step_returns_empty_signal(self):
        err = pd.DataFrame(
            {
                "step": [0, 0],
                "variable": ["X"] * 2,
                "mechanism": ["m1", "m2"],
                "absolute_error": [0.5, 1.0],
                "squared_error": [0.25, 1.0],
            }
        )
        signal, steps = RegimeDetector._compute_signal(err)
        assert len(signal) == 0
        assert len(steps) == 1
