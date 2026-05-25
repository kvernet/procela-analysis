"""
Unsupervised regime detection from mechanism error patterns.

Detects structural breaks in the error matrix (step x mechanism)
by tracking changes in the vector of per-mechanism errors rather
than collapsing to a scalar mean. Regimes are periods where the
relative performance of mechanisms is stable.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


class RegimeDetector:
    """
    Detects regimes from mechanism error structure.

    Regimes are periods where the relative performance of mechanisms
    is stable. A regime shift occurs when the error vector changes
    significantly — e.g., when a previously accurate mechanism
    suddenly fails, indicating the ground truth dynamics have changed.

    Detection uses recursive binary segmentation on step-to-step
    Euclidean distances between mechanism error vectors. The penalty
    parameter controls sensitivity via a heuristic threshold: higher
    values produce fewer regimes.

    Parameters
    ----------
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``. Must contain
        columns ``step``, ``variable``, ``mechanism``, ``absolute_error``.
    """

    def __init__(self, errors: pd.DataFrame) -> None:
        self._errors = errors
        self._results: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        variable: str,
        penalty: float = 1.0,
    ) -> pd.DataFrame:
        """
        Detect regime changepoints and assign regime labels.

        Constructs a step x mechanism error matrix and computes
        Euclidean distance between consecutive error vectors.
        Recursive binary segmentation finds changepoints where
        the mechanism error structure shifts.

        Parameters
        ----------
        variable : str
            Variable name to detect regimes for.
        penalty : float
            Heuristic sensitivity parameter. Higher values produce
            fewer changepoints. Typical range: 0.5 (sensitive) to
            2.0 (conservative).

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``regime_label``.
            Regime labels are integers starting from 0.
        """
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        # Build error matrix and compute step-to-step distances
        signal, steps = self._compute_signal(err)

        if len(signal) < 2:
            # Need at least one delta to segment on
            changepoints: list[int] = []
        else:
            changepoints = self._binary_segment(signal, penalty)

        # Assign regime labels
        n_steps = len(steps)
        labels_array = np.zeros(n_steps, dtype=int)
        for cp in changepoints:
            # cp indexes into signal (deltas), which has length n_steps - 1
            # The changepoint occurs *between* step cp and step cp+1
            # So regime changes starting at step cp+1
            labels_array[cp + 1 :] += 1

        labels = pd.Series(labels_array, index=steps, name="regime_label")

        self._results[variable] = {
            "labels": labels,
            "changepoints": changepoints,
            "steps": steps,
        }

        return self.labels(variable)

    def labels(self, variable: str) -> pd.DataFrame:
        """
        Return detected regime labels.

        Must call ``detect()`` first for this variable.

        Parameters
        ----------
        variable : str
            Variable name.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``regime_label``.

        Raises
        ------
        KeyError
            If ``detect()`` has not been called for this variable.
        """
        if variable not in self._results:
            raise KeyError(
                f"No detection results for variable '{variable}'. "
                f"Call detect('{variable}') first."
            )

        return self._results[variable]["labels"].reset_index()

    def characterize(
        self,
        variable: str,
    ) -> pd.DataFrame:
        """
        Profile each detected regime: dominant mechanism, error stats.

        Parameters
        ----------
        variable : str
            Variable name to characterize regimes for.

        Returns
        -------
        pd.DataFrame
            Columns: ``regime_label``, ``dominant_mechanism``,
            ``mean_error``, ``start_step``, ``end_step``, ``duration``.

        Raises
        ------
        KeyError
            If ``detect()`` has not been called for this variable.
        """
        if variable not in self._results:
            raise KeyError(
                f"No detection results for variable '{variable}'. "
                f"Call detect('{variable}') first."
            )

        result = self._results[variable]
        labels = result["labels"]
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        labeled = err.merge(labels.reset_index(), on="step", how="left")

        regimes: list[dict[str, str | float | int]] = []

        for regime_id in sorted(labels.unique()):
            regime_data = labeled[labeled["regime_label"] == regime_id]

            mech_errors = regime_data.groupby("mechanism")["absolute_error"].mean()
            dominant = str(mech_errors.idxmin()) if not mech_errors.empty else "unknown"

            regimes.append(
                {
                    "regime_label": int(regime_id),
                    "dominant_mechanism": dominant,
                    "mean_error": float(regime_data["absolute_error"].mean()),
                    "start_step": int(regime_data["step"].min()),
                    "end_step": int(regime_data["step"].max()),
                    "duration": int(
                        regime_data["step"].max() - regime_data["step"].min() + 1
                    ),
                }
            )

        return pd.DataFrame(regimes)

    def transitions(self, variable: str) -> pd.DataFrame:
        """
        Extract regime transition points.

        Parameters
        ----------
        variable : str
            Variable name.

        Returns
        -------
        pd.DataFrame
            Columns: ``transition_step``, ``from_regime``, ``to_regime``,
            ``confidence``.

        Raises
        ------
        KeyError
            If ``detect()`` has not been called for this variable.
        """
        if variable not in self._results:
            raise KeyError(
                f"No detection results for variable '{variable}'. "
                f"Call detect('{variable}') first."
            )

        result = self._results[variable]
        changepoints = result["changepoints"]
        labels = result["labels"]
        steps = result["steps"]

        if not changepoints:
            return pd.DataFrame(
                columns=["transition_step", "from_regime", "to_regime", "confidence"]
            )

        transitions_list: list[dict[str, int | float]] = []

        for cp in changepoints:
            # Transition occurs at step cp+1 (the first step of new regime)
            transition_step = int(steps[cp + 1])
            from_regime = int(labels.iloc[cp])
            to_regime = int(labels.iloc[cp + 1])

            transitions_list.append(
                {
                    "transition_step": transition_step,
                    "from_regime": from_regime,
                    "to_regime": to_regime,
                    "confidence": 0.5,  # Heuristic placeholder
                }
            )

        return pd.DataFrame(transitions_list)

    def compare(
        self,
        variable: str,
        ground_truth: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Compare detected regimes against ground truth labels.

        Parameters
        ----------
        variable : str
            Variable name.
        ground_truth : pd.DataFrame
            Columns: ``step``, ``true_label``.

        Returns
        -------
        dict
            Keys: ``adjusted_rand_index``, ``normalized_mutual_info``.
            Values are floats in [0, 1] where 1.0 is perfect agreement.

        Raises
        ------
        KeyError
            If ``detect()`` has not been called for this variable.
        """
        if variable not in self._results:
            raise KeyError(
                f"No detection results for variable '{variable}'. "
                f"Call detect('{variable}') first."
            )

        labels = self._results[variable]["labels"]

        merged = labels.reset_index().merge(
            ground_truth,
            on="step",
            how="inner",
        )

        if merged.empty:
            return {
                "adjusted_rand_index": 0.0,
                "normalized_mutual_info": 0.0,
            }

        ari = self._adjusted_rand_index(
            merged["regime_label"].values,
            merged["true_label"].values,
        )
        nmi = self._normalized_mutual_info(
            merged["regime_label"].values,
            merged["true_label"].values,
        )

        return {
            "adjusted_rand_index": float(ari),
            "normalized_mutual_info": float(nmi),
        }

    # ------------------------------------------------------------------
    # Internal: signal construction
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_signal(
        err: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.Index]:
        """
        Build step x mechanism error matrix and compute step-to-step distance.

        This method computes Euclidean distance between consecutive error vectors.

        Parameters
        ----------
        err : pd.DataFrame
            Error data for a single variable.

        Returns
        -------
        signal : np.ndarray
            1D array of step-to-step distances. Length is n_steps - 1.
        steps : pd.Index
            Sorted step indices (length n_steps).
        """
        pivot = err.pivot_table(
            index="step",
            columns="mechanism",
            values="absolute_error",
            aggfunc="mean",
        ).ffill()

        steps = pivot.index
        X = pivot.values  # (n_steps, n_mechanisms)

        if len(X) < 2:
            return np.array([]), steps

        # Euclidean distance between consecutive rows
        deltas = np.sqrt(np.sum((X[1:] - X[:-1]) ** 2, axis=1))

        return deltas, steps

    # ------------------------------------------------------------------
    # Internal: recursive binary segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _binary_segment(
        signal: np.ndarray,
        penalty: float = 1.0,
    ) -> list[int]:
        """
        Recursively find changepoints via maximum deviation from mean.

        At each recursion, finds the index of maximum cumulative
        deviation and splits the segment if the deviation exceeds
        a heuristic threshold.

        Parameters
        ----------
        signal : np.ndarray
            1D array of step-to-step distances.
        penalty : float
            Heuristic threshold multiplier. Higher values produce
            fewer changepoints.

        Returns
        -------
        list of int
            Indices of changepoints in the signal array.
            Each changepoint i means the regime changes between
            step i and step i+1 in the original step index.
        """
        n = len(signal)
        if n < 2:
            return []

        mean = np.mean(signal)
        cusum = np.cumsum(signal - mean)
        max_deviation = np.max(np.abs(cusum))

        if max_deviation == 0:
            return []

        threshold = penalty * np.std(signal) * np.sqrt(n)

        if max_deviation < threshold:
            return []

        cp = int(np.argmax(np.abs(cusum)))

        # Guard against boundary picks that prevent further splitting
        if cp <= 0 or cp >= n - 1:
            return []

        # Recurse on left and right segments
        left = RegimeDetector._binary_segment(signal[: cp + 1], penalty)
        right = RegimeDetector._binary_segment(signal[cp + 1 :], penalty)
        right = [r + cp + 1 for r in right]

        return sorted(left + [cp] + right)

    # ------------------------------------------------------------------
    # Internal: clustering comparison metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _adjusted_rand_index(
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
    ) -> float:
        """
        Compute Adjusted Rand Index between two label sets.

        Parameters
        ----------
        labels_true : np.ndarray
            Ground truth labels.
        labels_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        float
            ARI score in [-1, 1]. 1.0 is perfect agreement,
            0.0 is random labeling.
        """
        n = len(labels_true)
        if n < 2:
            return 1.0 if np.array_equal(labels_true, labels_pred) else 0.0

        contingency: dict[tuple[int, int], int] = {}
        for t, p in zip(labels_true, labels_pred):
            key = (int(t), int(p))
            contingency[key] = contingency.get(key, 0) + 1

        true_counts = Counter(int(t) for t in labels_true)
        pred_counts = Counter(int(p) for p in labels_pred)

        sum_comb = sum(v * (v - 1) / 2 for v in contingency.values())
        sum_true = sum(v * (v - 1) / 2 for v in true_counts.values())
        sum_pred = sum(v * (v - 1) / 2 for v in pred_counts.values())

        expected = (sum_true * sum_pred) / (n * (n - 1) / 2)
        max_possible = (sum_true + sum_pred) / 2

        if abs(max_possible - expected) < 1e-10:
            return 0.0

        return (sum_comb - expected) / (max_possible - expected)

    @staticmethod
    def _normalized_mutual_info(
        labels_true: np.ndarray,
        labels_pred: np.ndarray,
    ) -> float:
        """
        Compute Normalized Mutual Information between two label sets.

        Uses geometric normalization:
        NMI = I(X; Y) / sqrt(H(X) * H(Y))

        Parameters
        ----------
        labels_true : np.ndarray
            Ground truth labels.
        labels_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        float
            NMI score in [0, 1]. 1.0 is perfect agreement.
        """
        n = len(labels_true)
        if n < 2:
            return 1.0 if np.array_equal(labels_true, labels_pred) else 0.0

        true_counts = Counter(int(t) for t in labels_true)
        pred_counts = Counter(int(p) for p in labels_pred)

        h_true = -sum((c / n) * np.log(c / n) for c in true_counts.values() if c > 0)
        h_pred = -sum((c / n) * np.log(c / n) for c in pred_counts.values() if c > 0)

        if h_true == 0 or h_pred == 0:
            return 0.0

        contingency: dict[tuple[int, int], int] = {}
        for t, p in zip(labels_true, labels_pred):
            key = (int(t), int(p))
            contingency[key] = contingency.get(key, 0) + 1

        mi = 0.0
        for (t, p), count in contingency.items():
            p_t = true_counts[t] / n
            p_p = pred_counts[p] / n
            p_joint = count / n
            if p_joint > 0:
                mi += p_joint * np.log(p_joint / (p_t * p_p))

        return float(mi / np.sqrt(h_true * h_pred))
