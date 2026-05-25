"""
Memory reader for Procela variables.

Converts Procela variable hypothesis memory into structured DataFrames.
"""

from __future__ import annotations

import pandas as pd
from procela import Key, KeyAuthority, Mechanism, ResolutionPolicy, TimePoint, Variable

from .frames import (
    errors_frame,
    hypotheses_frame,
    resolutions_frame,
)


class MemoryReader:
    """
    Reads a Procela variable's memory and exposes records as DataFrames.

    The variable's memory is iterated once during construction.
    All subsequent property accesses return cached DataFrames.

    Parameters
    ----------
    variable : Variable
        A Procela variable with populated hypothesis memory.
    time_points : dict[int, TimePoint], optional
        Mapping from memory iteration timepoint/index to simulation step.
        If ``None``, the iteration index is used as the step number.
        This is typically obtained from the Executive's step mapping
        to ensure correct alignment across variables.
    """

    def __init__(
        self,
        variable: Variable,
        time_points: dict[int, TimePoint] | None = None,
    ) -> None:
        self._variable = variable
        self._time_points = (
            {t: i for i, t in time_points.items()} if time_points else None
        )

        self._hypotheses_records: list[dict[str, int | str | float]] = []
        self._resolutions_records: list[dict[str, int | str | float]] = []

        self._read_memory()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def hypotheses(self) -> pd.DataFrame:
        """
        Return all hypotheses as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns as defined in ``HYPOTHESES_SCHEMA``.
            One row per hypothesis ever proposed.

        Raises
        ------
        ValueError
            If no hypotheses were found in memory.
        """
        return hypotheses_frame(self._hypotheses_records)

    def resolutions(self) -> pd.DataFrame:
        """
        Return all resolutions as a DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns as defined in ``RESOLUTIONS_SCHEMA``.
            One row per simulation step.

        Raises
        ------
        ValueError
            If no resolutions were found in memory.
        """
        return resolutions_frame(self._resolutions_records)

    def errors(self) -> pd.DataFrame:
        """
        Compute per-hypothesis prediction errors.

        Joins hypotheses with their corresponding resolutions
        and computes absolute and squared error for each hypothesis.

        Returns
        -------
        pd.DataFrame
            Columns as defined in ``ERRORS_SCHEMA``.

        Raises
        ------
        ValueError
            If hypotheses or resolutions frames are empty.
        """
        return errors_frame(self.hypotheses(), self.resolutions())

    # ------------------------------------------------------------------
    # Internal: memory iteration
    # ------------------------------------------------------------------

    def _read_memory(self) -> None:
        """Iterate the variable's memory once and populate internal records."""
        if self._variable.memory is None:
            return

        for idx, (hypotheses, conclusion, _reasoning, time) in enumerate(
            self._variable.memory.records()
        ):
            step = (
                self._resolve_step(time)
                if self._time_points
                else self._resolve_step(idx)
            )

            if step < 0:
                continue

            # Collect hypotheses
            for hyp in hypotheses:
                if hyp.record is None:
                    continue

                self._hypotheses_records.append(
                    {
                        "step": step,
                        "variable": self._variable.name,
                        "mechanism": self._get_source_name(hyp.record.source),
                        "proposed": float(hyp.record.value),
                        "confidence": (
                            hyp.record.confidence
                            if hyp.record.confidence is not None
                            else 0.0
                        ),
                    }
                )

            # Collect resolution
            if conclusion is not None:
                self._resolutions_records.append(
                    {
                        "step": step,
                        "variable": self._variable.name,
                        "resolved": float(conclusion.value),
                        "policy": self._get_source_name(conclusion.source),
                        "num_hypotheses": len(hypotheses),
                    }
                )

    def _resolve_step(self, index: TimePoint | int | None) -> int:
        """
        Map a memory iteration timepoint or index to a simulation step.

        Parameters
        ----------
        index : TimePoint | int | None
            The iteration memory timepoint or index.

        Returns
        -------
        int
            The simulation step number.
            -1 is returned if could not be attached to a correct step.
        """
        step = -1  # skip this step

        if self._time_points is not None:
            if isinstance(index, TimePoint) and index in self._time_points:
                step = self._time_points[index]
        elif isinstance(index, int):
            step = index

        return step

    def _get_source_name(self, key: Key | None) -> str:
        """
        Return the source (mechanism, policy) name.

        Parameters
        ----------
        key : Key
            The source key to get the name of.

        Returns
        -------
        str
            The source name if the key is a mechanism's key
            or a policy's key, "unknown" otherwise.
        """
        if key is None:
            return "unknown"

        source = KeyAuthority.resolve(key)
        if isinstance(source, Mechanism) or isinstance(source, ResolutionPolicy):
            return str(source.name)

        return "unknown"
