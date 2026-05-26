"""
Policy stability analysis from resolution history.

Analyzes which policies were active, how often alternative policies
would have selected different values from the same hypotheses, and
whether policy switches correlated with error changes.

All analyses are per-step and do NOT simulate counterfactual
trajectories — they only compare selection functions on the
hypotheses that were actually proposed at each step.
"""

from __future__ import annotations

import pandas as pd


class PolicyStability:
    """
    Analyzes resolution policy behavior from hypothesis memory.

    All methods operate on DataFrames produced by MemoryReader.
    Comparisons between policies are per-step: given the same set
    of hypotheses at a step, would a different policy have picked
    a different value? This does NOT simulate what would have
    happened in subsequent steps.

    Parameters
    ----------
    resolutions : pd.DataFrame
        As produced by ``MemoryReader.resolutions()``.
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    """

    def __init__(
        self,
        resolutions: pd.DataFrame,
        hypotheses: pd.DataFrame,
    ) -> None:
        """Policy stability analysis from resolution history."""
        self._resolutions = resolutions
        self._hypotheses = hypotheses

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def timeline(self, variable: str) -> pd.DataFrame:
        """
        Return the policy timeline for a variable.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``policy``, ``resolved_value``,
            ``resolved_confidence``, ``num_hypotheses``, ``confidence_range``.
            ``confidence_range`` is max - min confidence among
            hypotheses at that step.

        Raises
        ------
        ValueError
            If no resolutions found for the variable.
        """
        res = self._resolutions[self._resolutions["variable"] == variable].copy()

        if res.empty:
            raise ValueError(f"No resolutions found for variable '{variable}'.")

        hyp = self._hypotheses[self._hypotheses["variable"] == variable]

        # Compute confidence range per step
        if not hyp.empty:
            conf_range = hyp.groupby("step")["confidence"].agg(
                lambda x: x.max() - x.min()
            )
            res["confidence_range"] = res["step"].map(conf_range).fillna(0.0)
        else:
            res["confidence_range"] = 0.0

        result = res[
            [
                "step",
                "policy",
                "resolved",
                "confidence",
                "num_hypotheses",
                "confidence_range",
            ]
        ].copy()
        result = result.rename(columns={"resolved": "resolved_value"})
        result = result.rename(columns={"confidence": "resolved_confidence"})
        result = result.sort_values("step").reset_index(drop=True)

        return result

    def disagreement(self, variable: str) -> pd.DataFrame:
        """
        Check whether alternative policies would have selected different values.

        Compares the actual resolution to what would have been
        selected by highest-confidence and median policies,
        given the same set of hypotheses at each step.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``resolved_value``, ``resolved_confidence``,
            ``actual_policy``, ``highest_conf_value``, ``highest_conf_confidence``,
            ``median_value``, ``median_confidence``,
            ``would_highest_differ``, ``would_median_differ``.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()
        res = self._resolutions[self._resolutions["variable"] == variable].copy()

        if hyp.empty:
            raise ValueError(f"No hypotheses found for variable '{variable}'.")

        # Highest confidence: select row with max confidence per step
        highest_conf_idx = hyp.groupby("step")["confidence"].idxmax()
        highest_conf = hyp.loc[highest_conf_idx][
            ["step", "proposed", "confidence"]
        ].rename(
            columns={
                "proposed": "highest_conf_value",
                "confidence": "highest_conf_confidence",
            }
        )

        # Median: median of proposed values and confidences per step
        median = (
            hyp.groupby("step")
            .agg(
                median_value=("proposed", "median"),
                median_confidence=("confidence", "median"),
            )
            .reset_index()
        )

        # Merge all together
        result = res[["step", "resolved", "confidence", "policy"]].copy()
        result = result.rename(
            columns={
                "resolved": "resolved_value",
                "confidence": "resolved_confidence",
                "policy": "actual_policy",
            }
        )
        result = result.merge(highest_conf, on="step", how="left")
        result = result.merge(median, on="step", how="left")

        # Would alternatives differ?
        tolerance = 1e-10
        result["would_highest_differ"] = (
            result["resolved_value"] - result["highest_conf_value"]
        ).abs() > tolerance
        result["would_median_differ"] = (
            result["resolved_value"] - result["median_value"]
        ).abs() > tolerance

        column_order = [
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

        return result[column_order].sort_values("step").reset_index(drop=True)

    def switch_impact(self, variable: str) -> pd.DataFrame:
        """
        Switch impact for a variable.

        Identify steps where the resolution policy changed and
        compare error levels before and after the switch.

        This is descriptive, not causal. It reports the mean
        absolute error in windows before and after each switch.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``switch_step``, ``from_policy``, ``to_policy``,
            ``mean_error_before``, ``mean_error_after``.
            Empty DataFrame if the policy never changed.

        Raises
        ------
        ValueError
            If no resolutions found for the variable.
        """
        res = self._resolutions[self._resolutions["variable"] == variable].copy()

        if res.empty:
            raise ValueError(f"No resolutions found for variable '{variable}'.")

        if len(res) < 2:
            return pd.DataFrame(
                columns=[
                    "switch_step",
                    "from_policy",
                    "to_policy",
                    "mean_error_before",
                    "mean_error_after",
                ]
            )

        res = res.sort_values("step").reset_index(drop=True)

        # Find steps where policy changes
        res["policy_prev"] = res["policy"].shift(1)
        switches = res[
            (res["policy"] != res["policy_prev"]) & (res["policy_prev"].notna())
        ].copy()

        if switches.empty:
            return pd.DataFrame(
                columns=[
                    "switch_step",
                    "from_policy",
                    "to_policy",
                    "mean_error_before",
                    "mean_error_after",
                ]
            )

        # Get errors for this variable
        hyp = self._hypotheses[self._hypotheses["variable"] == variable]
        if hyp.empty:
            return pd.DataFrame(
                columns=[
                    "switch_step",
                    "from_policy",
                    "to_policy",
                    "mean_error_before",
                    "mean_error_after",
                ]
            )

        # Mean absolute error per step (across all mechanisms)
        step_error = hyp.groupby("step")["proposed"].mean().reset_index()
        step_error = step_error.merge(res[["step", "resolved"]], on="step")
        step_error["absolute_error"] = (
            step_error["proposed"] - step_error["resolved"]
        ).abs()

        results: list[dict[str, str | float | int]] = []
        window = 10

        for _, switch in switches.iterrows():
            switch_step = int(switch["step"])

            before = step_error[
                (step_error["step"] >= switch_step - window)
                & (step_error["step"] < switch_step)
            ]
            after = step_error[
                (step_error["step"] >= switch_step)
                & (step_error["step"] < switch_step + window)
            ]

            results.append(
                {
                    "switch_step": switch_step,
                    "from_policy": str(switch["policy_prev"]),
                    "to_policy": str(switch["policy"]),
                    "mean_error_before": (
                        float(before["absolute_error"].mean())
                        if not before.empty
                        else float("nan")
                    ),
                    "mean_error_after": (
                        float(after["absolute_error"].mean())
                        if not after.empty
                        else float("nan")
                    ),
                }
            )

        return pd.DataFrame(results)
