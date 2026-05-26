"""
Regime transition analysis for detected regime boundaries.

Characterizes what changes between consecutive regimes: which
mechanisms gained or lost dominance, how error structures shifted,
and whether the transition was gradual or abrupt.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TransitionAnalyzer:
    """
    Analyzes what changes at regime boundaries.

    Takes detected regime labels and mechanism-level data to
    characterize transitions: dominance flips, error magnitude
    changes, and transition abruptness.

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    """

    def __init__(
        self,
        hypotheses: pd.DataFrame,
        errors: pd.DataFrame,
    ) -> None:
        """Regime transition analysis for detected regime boundaries."""
        self._hypotheses = hypotheses
        self._errors = errors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dominance_shift(
        self,
        variable: str,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Identify which mechanisms gained or lost dominance.

        The dominance is computed at each regime transition.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        labels : pd.DataFrame
            As produced by ``RegimeDetector.labels()``.
            Columns: ``step``, ``regime_label``.

        Returns
        -------
        pd.DataFrame
            Columns: ``transition_step``, ``from_regime``,
            ``to_regime``, ``mechanism``, ``old_rank``,
            ``new_rank``, ``rank_change``, ``share_change``.
            Positive ``rank_change`` = improved rank (closer to 1).
            Empty if fewer than two regimes detected.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()

        if hyp.empty:
            raise ValueError(f"No hypotheses found for variable '{variable}'.")

        # Validate and merge labels
        labeled = hyp.merge(
            labels,
            how="inner",
            on="step",
            validate="many_to_one",
        )

        # Temporal ordering: regimes ordered by first appearance
        regimes = (
            labels.groupby("regime_label")["step"].min().sort_values().index.tolist()
        )

        if len(regimes) < 2:
            return pd.DataFrame(
                columns=[
                    "transition_step",
                    "from_regime",
                    "to_regime",
                    "mechanism",
                    "old_rank",
                    "new_rank",
                    "rank_change",
                    "share_change",
                ]
            )

        # Compute confidence share per step
        step_totals = labeled.groupby("step")["confidence"].transform("sum")
        labeled["confidence_share"] = (
            labeled["confidence"] / step_totals.replace(0, float("nan"))
        ).fillna(0.0)

        # Average share per mechanism per regime
        regime_shares = (
            labeled.groupby(["regime_label", "mechanism"])["confidence_share"]
            .mean()
            .reset_index()
        )

        shifts: list[dict[str, str | float | int]] = []

        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            transition_step = self._get_transition_step(labels, to_regime)

            old_shares = regime_shares[
                regime_shares["regime_label"] == from_regime
            ].set_index("mechanism")["confidence_share"]

            new_shares = regime_shares[
                regime_shares["regime_label"] == to_regime
            ].set_index("mechanism")["confidence_share"]

            old_ranks = old_shares.rank(method="first", ascending=False)
            new_ranks = new_shares.rank(method="first", ascending=False)

            all_mechanisms = old_shares.index.union(new_shares.index)

            for mech in all_mechanisms:
                old_rank = old_ranks.get(mech, float("nan"))
                new_rank = new_ranks.get(mech, float("nan"))
                old_share = old_shares.get(mech, 0.0)
                new_share = new_shares.get(mech, 0.0)

                shifts.append(
                    {
                        "transition_step": transition_step,
                        "from_regime": int(from_regime),
                        "to_regime": int(to_regime),
                        "mechanism": mech,
                        "old_rank": float(old_rank),
                        "new_rank": float(new_rank),
                        "rank_change": float(old_rank - new_rank),
                        "share_change": float(new_share - old_share),
                    }
                )

        return pd.DataFrame(shifts)

    def error_shift(
        self,
        variable: str,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compare mean absolute error before and after each regime transition.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        labels : pd.DataFrame
            As produced by ``RegimeDetector.labels()``.

        Returns
        -------
        pd.DataFrame
            Columns: ``transition_step``, ``from_regime``,
            ``to_regime``, ``mechanism``, ``mean_error_before``,
            ``mean_error_after``, ``error_change``.
            Negative ``error_change`` = improvement.

        Raises
        ------
        ValueError
            If no errors found for the variable.
        """
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        # Validate and merge labels
        labeled = err.merge(
            labels,
            on="step",
            how="inner",
            validate="many_to_one",
        )

        regimes = (
            labels.groupby("regime_label")["step"].min().sort_values().index.tolist()
        )

        if len(regimes) < 2:
            return pd.DataFrame(
                columns=[
                    "transition_step",
                    "from_regime",
                    "to_regime",
                    "mechanism",
                    "mean_error_before",
                    "mean_error_after",
                    "error_change",
                ]
            )

        error_shifts: list[dict[str, str | float | int]] = []

        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            transition_step = self._get_transition_step(labels, to_regime)

            before = labeled[labeled["regime_label"] == from_regime]
            after = labeled[labeled["regime_label"] == to_regime]

            before_means = before.groupby("mechanism")["absolute_error"].mean()
            after_means = after.groupby("mechanism")["absolute_error"].mean()

            all_mechanisms = before_means.index.union(after_means.index)

            for mech in all_mechanisms:
                mean_before = before_means.get(mech, float("nan"))
                mean_after = after_means.get(mech, float("nan"))

                error_shifts.append(
                    {
                        "transition_step": transition_step,
                        "from_regime": int(from_regime),
                        "to_regime": int(to_regime),
                        "mechanism": mech,
                        "mean_error_before": float(mean_before),
                        "mean_error_after": float(mean_after),
                        "error_change": (
                            float(mean_after - mean_before)
                            if not (pd.isna(mean_before) or pd.isna(mean_after))
                            else float("nan")
                        ),
                    }
                )

        return pd.DataFrame(error_shifts)

    def abruptness(
        self,
        variable: str,
        labels: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Measure how abrupt each regime transition was.

        Abruptness is the normalized step-to-step distance in the
        error vector at the transition point, relative to the
        average step-to-step distance within the surrounding regimes.
        High values indicate a sudden structural break; low values
        indicate a gradual drift.

        Uses NaN-aware Euclidean distance so mechanisms absent from
        one side of the transition do not fabricate continuity.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        labels : pd.DataFrame
            As produced by ``RegimeDetector.labels()``.

        Returns
        -------
        pd.DataFrame
            Columns: ``transition_step``, ``from_regime``,
            ``to_regime``, ``transition_distance``,
            ``baseline_distance``, ``abruptness``.
            ``abruptness`` > 1 means the transition was larger than
            typical step-to-step variation.

        Raises
        ------
        ValueError
            If no errors found for the variable.
        """
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        regimes = (
            labels.groupby("regime_label")["step"].min().sort_values().index.tolist()
        )

        if len(regimes) < 2:
            return pd.DataFrame(
                columns=[
                    "transition_step",
                    "from_regime",
                    "to_regime",
                    "transition_distance",
                    "baseline_distance",
                    "abruptness",
                ]
            )

        # Build error matrix without forward fill — keep NaN for absent mechanisms
        pivot = err.pivot_table(
            index="step",
            columns="mechanism",
            values="absolute_error",
            aggfunc="mean",
        )

        X = pivot.values
        steps = pivot.index

        # NaN-aware Euclidean distance
        diff = X[1:] - X[:-1]
        deltas = np.sqrt(np.nansum(diff**2, axis=1))

        abrupt_data: list[dict[str, int | float]] = []

        for i in range(len(regimes) - 1):
            from_regime = regimes[i]
            to_regime = regimes[i + 1]

            transition_step = self._get_transition_step(labels, to_regime)

            # Find the transition index in the steps array
            transition_idx = np.where(steps == transition_step)[0]
            if len(transition_idx) == 0:
                abrupt_data.append(
                    {
                        "transition_step": transition_step,
                        "from_regime": int(from_regime),
                        "to_regime": int(to_regime),
                        "transition_distance": float("nan"),
                        "baseline_distance": float("nan"),
                        "abruptness": float("nan"),
                    }
                )
                continue

            idx = int(transition_idx[0])

            # Transition distance: delta at the boundary
            trans_dist = deltas[idx - 1]

            # Baseline: mean delta within surrounding regimes
            from_steps_arr = labels[labels["regime_label"] == from_regime][
                "step"
            ].values
            to_steps_arr = labels[labels["regime_label"] == to_regime]["step"].values

            from_deltas = [
                deltas[j]
                for j in range(len(steps) - 1)
                if steps[j] in from_steps_arr and steps[j + 1] in from_steps_arr
            ]
            to_deltas = [
                deltas[j]
                for j in range(len(steps) - 1)
                if steps[j] in to_steps_arr and steps[j + 1] in to_steps_arr
            ]

            all_baseline = from_deltas + to_deltas
            baseline = np.mean(all_baseline) if all_baseline else float("nan")

            if not pd.isna(trans_dist) and not pd.isna(baseline) and baseline > 0:
                abruptness_val = trans_dist / baseline
            else:
                abruptness_val = float("nan")

            abrupt_data.append(
                {
                    "transition_step": transition_step,
                    "from_regime": int(from_regime),
                    "to_regime": int(to_regime),
                    "transition_distance": float(trans_dist),
                    "baseline_distance": float(baseline),
                    "abruptness": float(abruptness_val),
                }
            )

        return pd.DataFrame(abrupt_data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_transition_step(
        labels: pd.DataFrame,
        to_regime: int,
    ) -> int:
        """
        Return the first step of a target regime.

        Parameters
        ----------
        labels : pd.DataFrame
            Regime labels with columns ``step``, ``regime_label``.
        to_regime : int
            The regime being transitioned into.

        Returns
        -------
        int
            The first step of ``to_regime``.
        """
        to_steps = labels[labels["regime_label"] == to_regime]
        return int(to_steps["step"].min())
