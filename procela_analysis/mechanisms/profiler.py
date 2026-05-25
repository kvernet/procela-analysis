"""
Mechanism profiler for per-mechanism performance analysis.

Computes accuracy curves, dominance timelines, influence windows,
redundancy matrices, and falsifiability scores from hypothesis
and error DataFrames produced by MemoryReader.
"""

from __future__ import annotations

import pandas as pd


class MechanismProfiler:
    """
    Analyzes individual mechanism behavior from hypothesis memory.

    All methods operate on DataFrames produced by MemoryReader.

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
        self._hypotheses = hypotheses
        self._errors = errors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def dominance(self, variable: str) -> pd.DataFrame:
        """
        Compute confidence share over time for each mechanism.

        At each step, confidence share is a mechanism's confidence
        divided by the sum of all confidences for that variable.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``mechanism``, ``confidence_share``.
            One row per mechanism per step. Sorted by step, then
            by confidence share descending.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()

        if hyp.empty:
            raise ValueError(f"No hypotheses found for variable '{variable}'.")

        # Sum of confidences per step
        step_totals = hyp.groupby("step")["confidence"].sum()

        # Join and compute share
        hyp["confidence_share"] = hyp.apply(
            lambda row: row["confidence"] / step_totals[row["step"]], axis=1
        )

        result = hyp[["step", "mechanism", "confidence_share"]].copy()
        result = result.sort_values(
            ["step", "confidence_share"], ascending=[True, False]
        )
        result = result.reset_index(drop=True)
        return result

    def rolling_mae(
        self,
        variable: str,
        window: int = 10,
    ) -> pd.DataFrame:
        """
        Compute rolling mean absolute error per mechanism.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        window : int
            Rolling window size in steps. Default 10.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``mechanism``, ``rolling_mae``.
            NaN for steps where window is incomplete.
        """
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        result = err.sort_values(["mechanism", "step"]).copy()
        result["rolling_mae"] = result.groupby("mechanism")["absolute_error"].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )

        return result[["step", "mechanism", "rolling_mae"]]

    def influence(
        self,
        variable: str,
        threshold: float = 0.1,
    ) -> pd.DataFrame:
        """
        Identify steps where each mechanism was epistemically influential.

        A mechanism is influential when its confidence share exceeds
        the threshold. This distinguishes 'enabled' from 'actually
        contributing to the resolved value'.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        threshold : float
            Minimum confidence share to be considered influential.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``mechanism``, ``is_influential``.
        """
        dom = self.dominance(variable)
        dom["is_influential"] = dom["confidence_share"] >= threshold
        return dom[["step", "mechanism", "is_influential"]]

    def redundancy(self, variable: str) -> pd.DataFrame:
        """
        Compute pairwise proposal correlation between mechanisms.

        High correlation suggests mechanisms encode similar theories
        with different noise. Low correlation indicates genuinely
        distinct causal hypotheses.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``mechanism_a``, ``mechanism_b``, ``pearson_r``.
            One row per unique mechanism pair.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()

        if hyp.empty:
            raise ValueError(f"No hypotheses found for variable '{variable}'.")

        # Pivot: step × mechanism → proposed value
        pivot = hyp.pivot_table(
            values="proposed",
            index="step",
            columns="mechanism",
            aggfunc="first",
        )

        mechanisms = pivot.columns.tolist()
        pairs: list[dict[str, str | float]] = []

        for i, ma in enumerate(mechanisms):
            for mb in mechanisms[i + 1 :]:
                # Only compute for steps where both proposed
                valid = pivot[[ma, mb]].dropna()
                if len(valid) < 3:
                    r = float("nan")
                else:
                    r = float(valid[ma].corr(valid[mb]))

                pairs.append(
                    {
                        "mechanism_a": ma,
                        "mechanism_b": mb,
                        "pearson_r": r,
                    }
                )

        if not pairs:
            raise ValueError(
                f"Need at least two mechanisms for redundancy analysis "
                f"on variable '{variable}'."
            )

        return pd.DataFrame(pairs)

    def falsifiability(self, variable: str) -> pd.DataFrame:
        """
        Compute falsifiability scores for each mechanism.

        A mechanism is falsifiable if its errors are consistently
        low within specific regimes and high outside them. This is
        measured as the ratio of mean error to error volatility.
        High ratio = predictable errors = falsifiable theory.
        Low ratio = erratic errors = unfalsifiable in practice.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``mechanism``, ``mean_error``, ``error_volatility``,
            ``steps_active``, ``falsifiability_score``.
        """
        err = self._errors[self._errors["variable"] == variable].copy()

        if err.empty:
            raise ValueError(f"No errors found for variable '{variable}'.")

        stats = (
            err.groupby("mechanism")
            .agg(
                mean_error=("absolute_error", "mean"),
                error_volatility=("absolute_error", "std"),
                steps_active=("step", "count"),
            )
            .reset_index()
        )

        # Falsifiability: mean / (volatility + epsilon)
        # High when error is stable (low volatility relative to mean)
        # Low when error is erratic (high volatility)
        stats["falsifiability_score"] = stats["mean_error"] / (
            stats["error_volatility"] + 1e-8
        )

        return stats
