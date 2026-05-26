"""
Self-contained HTML audit report for Procela simulations.

Generates a single HTML file with embedded CSS and Plotly
visualizations that documents the complete epistemic history
of a simulation: mechanism dominance, regime transitions,
governance actions, and per-mechanism falsifiability.

Designed as a supplementary material artifact for simulation-based
research papers. No network calls, no external dependencies at
render time.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ..mechanisms.ecology import MechanismEcology
from ..mechanisms.profiler import MechanismProfiler
from ..memory.metrics import (
    confidence_spread,
    disagreement_index,
    fragility,
    rolling_error,
)
from ..regimes.detector import RegimeDetector


class AuditReport:
    """
    Generates a self-contained HTML audit report from simulation memory.

    The report includes interactive Plotly charts and summary tables
    covering mechanism dominance, regime detection, governance actions,
    and per-mechanism falsifiability. All JavaScript is embedded inline
    — the output is a single file with no network dependencies.

    Parameters
    ----------
    hypotheses : pd.DataFrame
        As produced by ``MemoryReader.hypotheses()``.
    resolutions : pd.DataFrame
        As produced by ``MemoryReader.resolutions()``.
    errors : pd.DataFrame
        As produced by ``MemoryReader.errors()``.
    governance_log : pd.DataFrame, optional
        Governance action log with columns ``step``, ``action``,
        ``unit_name``, ``detail``, ``experiment_outcome``.
    title : str
        Report title displayed in the header.
    """

    def __init__(
        self,
        hypotheses: pd.DataFrame,
        resolutions: pd.DataFrame,
        errors: pd.DataFrame,
        governance_log: pd.DataFrame | None = None,
        title: str = "Procela Simulation Audit",
    ) -> None:
        """Self-contained HTML audit report for Procela simulations."""
        self._hypotheses = hypotheses
        self._resolutions = resolutions
        self._errors = errors
        self._governance = governance_log
        self._title = title

        self._variables = sorted(hypotheses["variable"].unique())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, filepath: str) -> None:
        """
        Generate a self-contained HTML audit report.

        Parameters
        ----------
        filepath : str
            Path to write the HTML file. Should end with ``.html``.
        """
        sections: list[str] = []

        sections.append(self._header())
        sections.append(self._executive_summary())

        for var in self._variables:
            sections.append(self._variable_section(var))

        if self._governance is not None and not self._governance.empty:
            sections.append(self._governance_section())

        sections.append(self._footer())

        html = "\n".join(sections)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html)

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _header(self) -> str:
        """HTML header with embedded Plotly JS."""
        # Extract Plotly JS bundle in a version-safe way.
        # We generate a tiny HTML fragment and pull out the JS script.
        plotly_html = pio.to_html(
            go.Figure(),
            include_plotlyjs=True,
            full_html=False,
        )

        script_start = plotly_html.find("<script>")
        script_end = plotly_html.find("</script>", script_start)

        plotly_js = ""
        if script_start != -1 and script_end != -1:
            plotly_js = plotly_html[script_start + len("<script>") : script_end]

        return f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{self._title}</title>
        <script>{plotly_js}</script>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont,
                'Segoe UI', Roboto, sans-serif;
                max-width: 1100px;
                margin: 0 auto;
                padding: 2rem 1.5rem;
                color: #222;
                background: #fafafa;
            }}
            h1 {{ font-size: 2rem; margin-bottom: 0.25rem; }}
            h2 {{ font-size: 1.4rem; margin: 2rem 0 1rem; border-bottom: 2px solid #ddd;
            padding-bottom: 0.25rem; }}
            h3 {{ font-size: 1.1rem; margin: 1.5rem 0 0.75rem; }}
            .timestamp {{ color: #888; font-size: 0.9rem; margin-bottom: 2rem; }}
            .summary-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }}
            .summary-card {{
                background: white;
                border: 1px solid #e0e0e0;
                border-radius: 6px;
                padding: 1rem;
                text-align: center;
            }}
            .summary-card .value {{
                font-size: 2rem;
                font-weight: bold;
                color: #648FFF;
            }}
            .summary-card .label {{
                font-size: 0.85rem;
                color: #666;
                margin-top: 0.25rem;
            }}
            .plot-container {{ margin: 1.5rem 0; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 1rem 0;
                font-size: 0.9rem;
            }}
            th, td {{
                padding: 0.5rem 0.75rem;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
            }}
            th {{ background: #f5f5f5; font-weight: 600; }}
            tr:hover {{ background: #fafafa; }}
            .success {{ color: #009E73; font-weight: 600; }}
            .failure {{ color: #DC267F; font-weight: 600; }}
            .footer {{
                margin-top: 3rem;
                padding-top: 1rem;
                border-top: 1px solid #ddd;
                color: #888;
                font-size: 0.85rem;
            }}
        </style>
    </head>
    <body>
    <h1>{self._title}</h1>
    <p class="timestamp">
        Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </p>
    """

    def _executive_summary(self) -> str:
        """Summary statistics cards."""
        total_steps = int(self._hypotheses["step"].max() + 1)
        total_mechanisms = int(self._hypotheses["mechanism"].nunique())
        total_variables = len(self._variables)
        mean_error = float(self._errors["absolute_error"].mean())
        gov_actions = len(self._governance) if self._governance is not None else 0

        return f"""<h2>Executive Summary</h2>
<div class="summary-grid">
    <div class="summary-card">
        <div class="value">{total_steps}</div>
        <div class="label">Simulation Steps</div>
    </div>
    <div class="summary-card">
        <div class="value">{total_mechanisms}</div>
        <div class="label">Mechanisms</div>
    </div>
    <div class="summary-card">
        <div class="value">{total_variables}</div>
        <div class="label">Variables</div>
    </div>
    <div class="summary-card">
        <div class="value">{mean_error:.3f}</div>
        <div class="label">Mean Absolute Error</div>
    </div>
    <div class="summary-card">
        <div class="value">{gov_actions}</div>
        <div class="label">Governance Actions</div>
    </div>
</div>
"""

    def _variable_section(self, variable: str) -> str:
        """Full analysis section for one variable."""
        parts: list[str] = []
        parts.append(f"<h2>Variable: {variable}</h2>")

        # Dominance timeline
        ecology = MechanismEcology(self._hypotheses, self._errors)
        dominance = ecology.dominance_curve(variable)
        parts.append("<h3>Mechanism Dominance</h3>")
        parts.append(self._fig_to_html(self._dominance_fig(dominance, variable)))

        # Regime detection
        parts.append("<h3>Regime Detection</h3>")
        detector = RegimeDetector(self._errors)
        labels = detector.detect(variable)
        regimes = detector.characterize(variable)

        parts.append(self._regimes_table(regimes))
        parts.append(self._fig_to_html(self._regime_error_fig(variable, labels)))

        # Per-mechanism falsifiability
        parts.append("<h3>Per-Mechanism Falsifiability</h3>")
        profiler = MechanismProfiler(self._hypotheses, self._errors)
        fals = profiler.falsifiability(variable)
        parts.append(self._falsifiability_table(fals))

        # Epistemic signals
        parts.append("<h3>Epistemic Signals</h3>")
        parts.append(self._fig_to_html(self._signals_fig(variable)))

        return "\n".join(parts)

    def _governance_section(self) -> str:
        """Governance action log section."""
        if self._governance is None or self._governance.empty:
            return ""

        parts: list[str] = []
        parts.append("<h2>Governance Actions</h2>")

        # Summary stats
        experiments = self._governance[
            self._governance["action"].isin(["experiment_start", "experiment_end"])
        ]
        if not experiments.empty:
            ends = experiments[experiments["action"] == "experiment_end"]
            successes = (ends["experiment_outcome"] == "success").sum()
            total_exp = len(ends)
            success_rate = successes / total_exp if total_exp > 0 else 0

            parts.append(f"""<div class="summary-grid">
    <div class="summary-card">
        <div class="value">{total_exp}</div>
        <div class="label">Experiments</div>
    </div>
    <div class="summary-card">
        <div class="value">{success_rate:.0%}</div>
        <div class="label">Success Rate</div>
    </div>
</div>""")

        # Action table
        parts.append("<table><thead><tr>")
        parts.append(
            "<th>Step</th><th>Unit</th><th>Action</th><th>Detail</th><th>Outcome</th>"
        )
        parts.append("</tr></thead><tbody>")

        for _, row in self._governance.iterrows():
            outcome = row.get("experiment_outcome", "")
            outcome_class = ""
            if outcome == "success":
                outcome_class = "success"
            elif outcome == "failure":
                outcome_class = "failure"

            parts.append(
                f"<tr>"
                f"<td>{int(row['step'])}</td>"
                f"<td>{row.get('unit_name', '')}</td>"
                f"<td>{row.get('action', '')}</td>"
                f"<td>{row.get('detail', '')}</td>"
                f'<td class="{outcome_class}">{outcome}</td>'
                f"</tr>"
            )

        parts.append("</tbody></table>")

        return "\n".join(parts)

    def _footer(self) -> str:
        """Close HTML."""
        return """<div class="footer">
    Generated by Procela Analysis. All data sourced from Procela variable
    memory with cryptographic audit trail.
</div>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Plot builders
    # ------------------------------------------------------------------

    def _dominance_fig(self, dominance: pd.DataFrame, variable: str) -> go.Figure:
        """Stacked area chart of mechanism dominance."""
        mechanisms = sorted(dominance["mechanism"].unique())
        pivot = dominance.pivot_table(
            index="step",
            columns="mechanism",
            values="confidence_share",
            aggfunc="first",
        ).fillna(0.0)
        pivot = pivot[mechanisms]

        fig = go.Figure()
        for mech in mechanisms:
            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=pivot[mech],
                    name=mech,
                    mode="lines",
                    stackgroup="one",
                    line=dict(width=0.5),
                )
            )

        fig.update_layout(
            title=f"Mechanism Dominance — {variable}",
            xaxis_title="Step",
            yaxis_title="Confidence Share",
            yaxis=dict(range=[0, 1]),
            height=400,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", y=-0.2),
        )

        return fig

    def _regime_error_fig(self, variable: str, labels: pd.DataFrame) -> go.Figure:
        """Roll error with regime bands."""
        err = rolling_error(self._errors, variable)

        fig = go.Figure()

        # Regime bands
        regimes = (
            labels.groupby("regime_label")["step"]
            .agg(["min", "max"])
            .sort_values("min")
        )
        colors = ["rgba(200,200,200,0.15)", "rgba(220,220,220,0.15)"]
        for i, (_, row) in enumerate(regimes.iterrows()):
            fig.add_vrect(
                x0=row["min"] - 0.5,
                x1=row["max"] + 0.5,
                fillcolor=colors[i % len(colors)],
                line_width=0,
                layer="below",
            )

        fig.add_trace(
            go.Scatter(
                x=err["step"],
                y=err["rolling_error"],
                mode="lines",
                name="Rolling Error",
                line=dict(color="#648FFF", width=2),
            )
        )

        fig.update_layout(
            title=f"Prediction Error with Detected Regimes — {variable}",
            xaxis_title="Step",
            yaxis_title="Rolling MAE",
            height=350,
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False,
        )

        return fig

    def _signals_fig(self, variable: str) -> go.Figure:
        """Epistemic signals dashboard."""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Policy Fragility",
                "Disagreement Index",
                "Confidence Spread",
                "Diversity Index",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.10,
        )

        # Fragility
        try:
            frag = fragility(self._hypotheses, variable)
            fig.add_trace(
                go.Scatter(
                    x=frag["step"],
                    y=frag["smoothed_fragility"],
                    mode="lines",
                    name="Fragility",
                    line=dict(color="#FE6100"),
                ),
                row=1,
                col=1,
            )
        except ValueError:
            pass

        # Disagreement
        try:
            disag = disagreement_index(self._hypotheses, variable)
            fig.add_trace(
                go.Scatter(
                    x=disag["step"],
                    y=disag["disagreement_index"],
                    mode="lines",
                    name="Disagreement",
                    line=dict(color="#DC267F"),
                ),
                row=1,
                col=2,
            )
        except ValueError:
            pass

        # Confidence spread
        try:
            cspread = confidence_spread(self._hypotheses, variable)
            fig.add_trace(
                go.Scatter(
                    x=cspread["step"],
                    y=cspread["confidence_spread"],
                    mode="lines",
                    name="Conf. Spread",
                    line=dict(color="#785EF0"),
                ),
                row=2,
                col=1,
            )
        except ValueError:
            pass

        # Diversity
        try:
            ecology = MechanismEcology(self._hypotheses, self._errors)
            div = ecology.diversity_index(variable)
            fig.add_trace(
                go.Scatter(
                    x=div["step"],
                    y=div["diversity"],
                    mode="lines",
                    name="Diversity",
                    line=dict(color="#009E73"),
                ),
                row=2,
                col=2,
            )
        except ValueError:
            pass

        fig.update_layout(
            height=600,
            margin=dict(l=40, r=20, t=60, b=40),
            showlegend=False,
        )

        return fig

    # ------------------------------------------------------------------
    # Table builders
    # ------------------------------------------------------------------

    def _regimes_table(self, regimes: pd.DataFrame) -> str:
        """HTML table for regime characterization."""
        if regimes.empty:
            return "<p>No regimes detected.</p>"

        rows = ["<table><thead><tr>"]
        rows.append("<th>Regime</th><th>Dominant Mechanism</th>")
        rows.append("<th>Mean Error</th><th>Steps</th><th>Duration</th>")
        rows.append("</tr></thead><tbody>")

        for _, r in regimes.iterrows():
            rows.append(
                f"<tr>"
                f"<td>{int(r['regime_label'])}</td>"
                f"<td><strong>{r['dominant_mechanism']}</strong></td>"
                f"<td>{r['mean_error']:.3f}</td>"
                f"<td>{int(r['start_step'])}–{int(r['end_step'])}</td>"
                f"<td>{int(r['duration'])}</td>"
                f"</tr>"
            )

        rows.append("</tbody></table>")
        return "\n".join(rows)

    def _falsifiability_table(self, fals: pd.DataFrame) -> str:
        """HTML table for per-mechanism falsifiability."""
        if fals.empty:
            return "<p>No falsifiability data available.</p>"

        rows = ["<table><thead><tr>"]
        rows.append("<th>Mechanism</th><th>Mean Error</th>")
        rows.append("<th>Error Volatility</th><th>Steps Active</th>")
        rows.append("<th>Falsifiability Score</th>")
        rows.append("</tr></thead><tbody>")

        for _, r in fals.iterrows():
            rows.append(
                f"<tr>"
                f"<td><strong>{r['mechanism']}</strong></td>"
                f"<td>{r['mean_error']:.3f}</td>"
                f"<td>{r['error_volatility']:.3f}</td>"
                f"<td>{int(r['steps_active'])}</td>"
                f"<td>{r['falsifiability_score']:.2f}</td>"
                f"</tr>"
            )

        rows.append("</tbody></table>")
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fig_to_html(fig: go.Figure) -> str:
        """Convert a Plotly figure to embedded HTML."""
        html = pio.to_html(
            fig,
            include_plotlyjs=True,
            full_html=False,
        )
        return f'<div class="plot-container">{html}</div>'
