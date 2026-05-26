"""
Mechanism ecology for population-level analysis of competing theories.

Studies the dynamics of mechanism populations: which mechanisms
dominate, which are redundant, which go extinct after regime shifts.
Treats mechanisms as species competing in an epistemic niche.
"""

from __future__ import annotations

import pandas as pd


class MechanismEcology:
    """
    Analyzes mechanism population dynamics from hypothesis memory.

    Treats mechanisms as competing species. Dominance is measured
    by confidence share. Niche overlap is measured by proposal
    correlation. Extinction occurs when a mechanism's influence
    drops below a threshold and does not recover within a bounded
    window.

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

    def dominance_curve(
        self,
        variable: str,
    ) -> pd.DataFrame:
        """
        Compute confidence share over time for each mechanism.

        At each step, confidence share is a mechanism's confidence
        divided by the sum of all confidences for that variable.
        If all confidences at a step are zero, shares are set to
        zero for all mechanisms at that step.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``mechanism``, ``confidence_share``.
            Sorted by step, then by confidence share descending.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()

        if hyp.empty:
            raise ValueError(f"No hypotheses found for variable '{variable}'.")

        # Vectorized confidence share computation
        step_totals = hyp.groupby("step")["confidence"].transform("sum")
        hyp["confidence_share"] = (
            hyp["confidence"] / step_totals.replace(0, float("nan"))
        ).fillna(0.0)

        result = hyp[["step", "mechanism", "confidence_share"]].copy()
        result = result.sort_values(
            ["step", "confidence_share"], ascending=[True, False]
        ).reset_index(drop=True)

        return result

    def niche_overlap(self, variable: str) -> pd.DataFrame:
        """
        Compute pairwise proposal correlation between mechanisms.

        High correlation suggests mechanisms encode similar theories
        (high niche overlap). Low correlation indicates genuinely
        distinct causal hypotheses (distinct niches).

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``mechanism_a``, ``mechanism_b``, ``pearson_r``,
            ``overlap_strength``.
            ``overlap_strength`` is ``|pearson_r|`` categorized as
            'high' (>0.8), 'moderate' (0.5-0.8), 'low' (<0.5),
            or 'unknown' (insufficient data or constant proposals).

        Raises
        ------
        ValueError
            If fewer than two mechanisms or no hypotheses found.
        """
        hyp = self._hypotheses[self._hypotheses["variable"] == variable].copy()

        pivot = hyp.pivot_table(
            index="step",
            columns="mechanism",
            values="proposed",
            aggfunc="first",
        )

        mechanisms = pivot.columns.tolist()

        if len(mechanisms) < 2:
            raise ValueError(
                f"Need at least two mechanisms for niche overlap analysis "
                f"on variable '{variable}'. Found: {len(mechanisms)}."
            )

        pairs: list[dict[str, str | float]] = []

        for i, ma in enumerate(mechanisms):
            for mb in mechanisms[i + 1 :]:
                valid = pivot[[ma, mb]].dropna()
                if len(valid) < 3:
                    r = float("nan")
                    strength = "unknown"
                else:
                    r = float(valid[ma].corr(valid[mb]))
                    if pd.isna(r):
                        strength = "unknown"
                    else:
                        abs_r = abs(r)
                        if abs_r > 0.8:
                            strength = "high"
                        elif abs_r > 0.5:
                            strength = "moderate"
                        else:
                            strength = "low"

                pairs.append(
                    {
                        "mechanism_a": ma,
                        "mechanism_b": mb,
                        "pearson_r": r,
                        "overlap_strength": strength,
                    }
                )

        return pd.DataFrame(pairs)

    def extinction_events(
        self,
        variable: str,
        threshold: float = 0.05,
        recovery_window: int = 10,
    ) -> pd.DataFrame:
        """
        Detect mechanisms that went functionally extinct.

        A mechanism is considered extinct when its confidence share
        drops below the threshold and does not recover above it
        within ``recovery_window`` steps. The extinction step is
        the first step where the share drops below the threshold.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        threshold : float
            Confidence share below which a mechanism is considered
            at risk of extinction. Default 0.05.
        recovery_window : int
            Number of steps after the initial drop to check for
            recovery. If the mechanism stays below threshold for
            this entire window, it is declared extinct.

        Returns
        -------
        pd.DataFrame
            Columns: ``mechanism``, ``extinction_step``,
            ``last_confidence_share``, ``steps_active_after_extinction``.
            Empty if no extinctions detected.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        dom = self.dominance_curve(variable)

        extinctions: list[dict[str, str | float | int]] = []

        for mechanism in dom["mechanism"].unique():
            mech_data = dom[dom["mechanism"] == mechanism].sort_values("step")

            below = mech_data[mech_data["confidence_share"] < threshold]

            if below.empty:
                continue

            extinction_step = int(below["step"].iloc[0])

            # Check recovery within the bounded window
            after_drop = mech_data[
                (mech_data["step"] > extinction_step)
                & (mech_data["step"] <= extinction_step + recovery_window)
            ]

            recovered = (after_drop["confidence_share"] >= threshold).any()

            if not recovered:
                steps_after = int(
                    mech_data[mech_data["step"] > extinction_step]["step"].count()
                )
                extinctions.append(
                    {
                        "mechanism": mechanism,
                        "extinction_step": extinction_step,
                        "last_confidence_share": float(
                            below["confidence_share"].iloc[0]
                        ),
                        "steps_active_after_extinction": steps_after,
                    }
                )

        return pd.DataFrame(extinctions)

    def diversity_index(
        self,
        variable: str,
    ) -> pd.DataFrame:
        """
        Compute Simpson's diversity index over time for mechanisms.

        D = 1 - sum(share_i^2) for all mechanisms at each step.
        D = 0 means one mechanism dominates completely.
        D approaches 1 - 1/N when all mechanisms have equal share.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``num_active``, ``diversity``.
            ``num_active`` is count of mechanisms with share > 0.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        dom = self.dominance_curve(variable)

        diversity = (
            dom.groupby("step")
            .agg(
                num_active=("confidence_share", lambda x: (x > 0).sum()),
                simpson_sum=("confidence_share", lambda x: (x**2).sum()),
            )
            .reset_index()
        )

        diversity["diversity"] = 1.0 - diversity["simpson_sum"]
        diversity = diversity.drop(columns=["simpson_sum"])

        return diversity.sort_values("step").reset_index(drop=True)

    def turnover(
        self,
        variable: str,
    ) -> pd.DataFrame:
        """
        Compute mechanism rank turnover at each step.

        Measures how much the dominance ranking changes between
        consecutive steps using normalized pairwise discordance.
        High turnover indicates the epistemic landscape is shifting
        rapidly.

        At each step after the first, mechanisms are ranked by
        confidence share (descending). The fraction of mechanism
        pairs whose relative ordering changed from the previous
        step is computed. Ties are broken by first occurrence.

        Parameters
        ----------
        variable : str
            Variable name to filter on.

        Returns
        -------
        pd.DataFrame
            Columns: ``step``, ``turnover``.
            ``turnover`` is the fraction of rank pairs that changed
            from the previous step. 0 = identical ranking, 1 = fully
            reversed ranking. NaN for the first step.

        Raises
        ------
        ValueError
            If no hypotheses found for the variable.
        """
        dom = self.dominance_curve(variable)

        steps = sorted(dom["step"].unique())
        turnover_data: list[dict[str, int | float]] = []

        for i, step in enumerate(steps):
            if i == 0:
                turnover_data.append({"step": int(step), "turnover": float("nan")})
                continue

            prev_step = steps[i - 1]

            prev_ranks = (
                dom[dom["step"] == prev_step]
                .set_index("mechanism")["confidence_share"]
                .rank(method="first", ascending=False)
            )
            curr_ranks = (
                dom[dom["step"] == step]
                .set_index("mechanism")["confidence_share"]
                .rank(method="first", ascending=False)
            )

            # Align on shared mechanisms
            common = prev_ranks.index.intersection(curr_ranks.index)
            if len(common) < 2:
                turnover_data.append({"step": int(step), "turnover": float("nan")})
                continue

            prev_aligned = prev_ranks[common]
            curr_aligned = curr_ranks[common]

            # Count discordant pairs
            n = len(common)
            discordant = 0
            for j in range(n):
                for k in range(j + 1, n):
                    prev_order = prev_aligned.iloc[j] - prev_aligned.iloc[k]
                    curr_order = curr_aligned.iloc[j] - curr_aligned.iloc[k]
                    if prev_order * curr_order < 0:
                        discordant += 1

            max_discordant = n * (n - 1) / 2
            turnover = discordant / max_discordant if max_discordant > 0 else 0.0

            turnover_data.append({"step": int(step), "turnover": float(turnover)})

        return pd.DataFrame(turnover_data)
