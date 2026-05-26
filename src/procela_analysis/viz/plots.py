"""
Standard visualizations for Procela hypothesis memory analysis.

All functions return matplotlib Figure objects that can be saved,
embedded in reports, or displayed interactively. No Procela core
dependency — operates entirely on DataFrames.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (IBM Design Library)
PALETTE = [
    "#648FFF",  # blue
    "#785EF0",  # violet
    "#DC267F",  # magenta
    "#FE6100",  # orange
    "#FFB000",  # yellow
    "#009E73",  # green
    "#56B4E9",  # sky blue
    "#E69F00",  # gold
    "#F0E442",  # yellow-green
    "#0072B2",  # deep blue
]


def _get_color(mechanism: str, mechanism_order: list[str]) -> str:
    """Return a consistent color for a mechanism."""
    if mechanism in mechanism_order:
        idx = mechanism_order.index(mechanism)
        return PALETTE[idx % len(PALETTE)]
    return "#999999"


# ---------------------------------------------------------------------------
# Dominance plots
# ---------------------------------------------------------------------------


def dominance_timeline(
    dominance: pd.DataFrame,
    title: str = "Mechanism Dominance Over Time",
    xlabel: str = "Step",
    ylabel: str = "Confidence Share",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot confidence share over time as a stacked area chart.

    Parameters
    ----------
    dominance : pd.DataFrame
        As produced by ``MechanismEcology.dominance_curve()``.
        Columns: ``step``, ``mechanism``, ``confidence_share``.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mechanisms = sorted(dominance["mechanism"].unique())

    pivot = dominance.pivot_table(
        values="confidence_share",
        index="step",
        columns="mechanism",
        aggfunc="first",
    ).fillna(0.0)

    # Ensure consistent column order
    pivot = pivot[mechanisms]

    fig, ax = plt.subplots(figsize=figsize)
    ax.stackplot(
        pivot.index,
        *[pivot[m].values for m in mechanisms],
        labels=mechanisms,
        colors=[_get_color(m, mechanisms) for m in mechanisms],
        alpha=0.85,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_ylim(0, 1)
    ax.set_xlim(pivot.index.min(), pivot.index.max())

    fig.tight_layout()
    return fig


def dominance_heatmap(
    dominance: pd.DataFrame,
    title: str = "Mechanism Dominance Heatmap",
    label: str = "Confidence Share",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot confidence share as a heatmap with mechanisms on y-axis.

    Parameters
    ----------
    dominance : pd.DataFrame
        As produced by ``MechanismEcology.dominance_curve()``.
    title : str
        Plot title.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    pivot = dominance.pivot_table(
        index="mechanism",
        columns="step",
        values="confidence_share",
        aggfunc="first",
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
    )

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(label)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Error plots
# ---------------------------------------------------------------------------


def error_timeline(
    errors: pd.DataFrame,
    variable: str,
    window: int = 10,
    title: str = "Rolling Prediction Error",
    xlabel: str = "Step",
    ylabel: str = "Rolling MAE",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot rolling mean absolute error over time.

    Parameters
    ----------
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    variable : str
        Variable name to filter on.
    window : int
        Rolling window size.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ..memory.metrics import rolling_error

    err = rolling_error(errors, variable, window=window)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(err["step"], err["rolling_error"], color=PALETTE[0], linewidth=1.5)
    ax.fill_between(
        err["step"],
        0,
        err["rolling_error"],
        color=PALETTE[0],
        alpha=0.15,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(err["step"].min(), err["step"].max())

    fig.tight_layout()
    return fig


def per_mechanism_error(
    errors: pd.DataFrame,
    variable: str,
    window: int = 10,
    title: str = "Per-Mechanism Rolling Error",
    xlabel: str = "Step",
    ylabel: str = "Rolling MAE",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot rolling error for each mechanism as separate lines.

    Parameters
    ----------
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    variable : str
        Variable name to filter on.
    window : int
        Rolling window size.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    from ..mechanisms.profiler import MechanismProfiler

    profiler = MechanismProfiler(
        pd.DataFrame(
            columns=["step", "variable", "mechanism", "proposed", "confidence"]
        ),
        errors,
    )
    accuracy = profiler.rolling_mae(variable, window=window)

    mechanisms = sorted(accuracy["mechanism"].unique())

    fig, ax = plt.subplots(figsize=figsize)

    for mech in mechanisms:
        mech_data = accuracy[accuracy["mechanism"] == mech]
        ax.plot(
            mech_data["step"],
            mech_data["rolling_mae"],
            label=mech,
            color=_get_color(mech, mechanisms),
            linewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_xlim(accuracy["step"].min(), accuracy["step"].max())

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Epistemic signal plots
# ---------------------------------------------------------------------------


def fragility_timeline(
    fragility: pd.DataFrame,
    title: str = "Policy Fragility Over Time",
    xlabel: str = "Step",
    ylabel: str = "Fragility",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot raw and smoothed fragility.

    Parameters
    ----------
    fragility : pd.DataFrame
        As produced by ``metrics.fragility()``.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        fragility["step"],
        fragility["raw_fragility"],
        color="#CCCCCC",
        linewidth=1,
        alpha=0.7,
        label="Raw",
    )
    ax.plot(
        fragility["step"],
        fragility["smoothed_fragility"],
        color=PALETTE[3],
        linewidth=2,
        label="Smoothed",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(fragility["step"].min(), fragility["step"].max())

    fig.tight_layout()
    return fig


def coverage_timeline(
    coverage_data: dict[str, pd.DataFrame],
    title: str = "Mechanism Coverage Over Time",
    xlabel: str = "Step",
    ylabel: str = "Smoothed Coverage",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """
    Plot smoothed coverage for multiple mechanisms.

    Parameters
    ----------
    coverage_data : dict of str to pd.DataFrame
        Mapping of mechanism name to coverage DataFrame
        as produced by ``metrics.coverage()``.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    mechanisms = sorted(coverage_data.keys())

    fig, ax = plt.subplots(figsize=figsize)

    for mech in mechanisms:
        df = coverage_data[mech]
        ax.plot(
            df["step"],
            df["smoothed_coverage"],
            label=mech,
            color=_get_color(mech, mechanisms),
            linewidth=2,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Regime plots
# ---------------------------------------------------------------------------


def regime_bands(
    labels: pd.DataFrame,
    ax: plt.Axes | None = None,
    alpha: float = 0.12,
) -> plt.Axes:
    """
    Add shaded background bands for detected regimes.

    Parameters
    ----------
    labels : pd.DataFrame
        As produced by ``RegimeDetector.labels()``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. Creates new if None.
    alpha : float
        Transparency of regime bands.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    regimes = (
        labels.groupby("regime_label")["step"].agg(["min", "max"]).sort_values("min")
    )

    colors = ["#F5F5F5", "#EEEEEE"]
    for i, (_, row) in enumerate(regimes.iterrows()):
        ax.axvspan(
            row["min"] - 0.5,
            row["max"] + 0.5,
            alpha=alpha,
            color=colors[i % len(colors)],
            zorder=0,
        )

    return ax


def diversity_timeline(
    diversity: pd.DataFrame,
    title: str = "Mechanism Diversity Over Time",
    xlabel: str = "Step",
    ylabel: str = "Simpson Diversity",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot Simpson diversity index over time.

    Parameters
    ----------
    diversity : pd.DataFrame
        As produced by ``MechanismEcology.diversity_index()``.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        diversity["step"],
        diversity["diversity"],
        color=PALETTE[5],
        linewidth=2,
    )
    ax.fill_between(
        diversity["step"],
        0,
        diversity["diversity"],
        color=PALETTE[5],
        alpha=0.15,
    )

    max_d = (
        1.0 - 1.0 / diversity["num_active"].max()
        if diversity["num_active"].max() > 1
        else 0.5
    )
    ax.axhline(
        y=max_d,
        color="#999999",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
        label=f"Max diversity ({max_d:.2f})",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(diversity["step"].min(), diversity["step"].max())

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Turnover plot
# ---------------------------------------------------------------------------


def turnover_timeline(
    turnover: pd.DataFrame,
    title: str = "Rank Turnover Over Time",
    xlabel: str = "Step",
    ylabel: str = "Turnover",
    figsize: tuple[int, int] = (10, 4),
) -> plt.Figure:
    """
    Plot mechanism rank turnover over time.

    Parameters
    ----------
    turnover : pd.DataFrame
        As produced by ``MechanismEcology.turnover()``.
    title : str
        Plot title.
    xlabel : str
        X label.
    ylabel : str
        Y label.
    figsize : tuple of int
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    valid = turnover[~turnover["turnover"].isna()]
    ax.fill_between(
        valid["step"],
        0,
        valid["turnover"],
        color=PALETTE[4],
        alpha=0.2,
        step="mid",
    )
    ax.plot(
        valid["step"],
        valid["turnover"],
        color=PALETTE[4],
        linewidth=1.5,
        drawstyle="steps-mid",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(turnover["step"].min(), turnover["step"].max())

    fig.tight_layout()
    return fig
