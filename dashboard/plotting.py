"""Plotly figures for dashboard views."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .distribution_fitting import fitted_pmf_frame, geometric_pmf


def pmf_figure(distribution: pd.DataFrame, *, aggregation: str = "pooled_probability", title: str = "Empirical PMF"):
    y = aggregation if aggregation in distribution.columns else "pooled_probability"
    return px.bar(
        distribution,
        x="attempt_count",
        y=y,
        color="scenario_name" if "scenario_name" in distribution.columns else None,
        barmode="group",
        labels={"attempt_count": "Attempt count", y: "Probability / proportion"},
        title=title,
    )


def cdf_figure(distribution: pd.DataFrame, *, cdf_column: str = "pooled_cdf", title: str = "Empirical CDF"):
    return px.line(
        distribution,
        x="attempt_count",
        y=cdf_column,
        color="scenario_name" if "scenario_name" in distribution.columns else None,
        markers=True,
        labels={"attempt_count": "Attempt count", cdf_column: "P(N <= k)"},
        title=title,
    )


def survival_figure(distribution: pd.DataFrame, *, survival_column: str = "pooled_survival", title: str = "Survival Function"):
    return px.line(
        distribution,
        x="attempt_count",
        y=survival_column,
        color="scenario_name" if "scenario_name" in distribution.columns else None,
        markers=True,
        labels={"attempt_count": "Attempt count", survival_column: "P(N >= k)"},
        title=title,
    )


def queue_length_pmf(
    distribution: pd.DataFrame,
    *,
    probability_column: str = "pooled_probability",
):
    """Return a queue-length PMF chart using zero-filled integer support."""

    y = probability_column if probability_column in distribution.columns else "pooled_probability"
    fig = px.bar(
        distribution,
        x="queue_length",
        y=y,
        labels={"queue_length": "Queue length q", y: "Probability"},
        title="Queue-length PMF: P(Q = q)",
    )
    fig.update_layout(bargap=0.05)
    return fig


def queue_length_time_series(samples: pd.DataFrame):
    """Return fixed-time queue length samples over simulation time."""

    fig = go.Figure()
    if samples.empty:
        fig.update_layout(
            title="Queue length over time",
            xaxis_title="Simulation time (model days)",
            yaxis_title="Queue length q",
        )
        return fig

    clean = samples.sort_values(["replication", "sample_time"]).copy()
    clean["replication"] = clean["replication"].astype(str)
    return px.line(
        clean,
        x="sample_time",
        y="queue_length",
        color="replication",
        labels={
            "sample_time": "Simulation time (model days)",
            "queue_length": "Queue length q",
            "replication": "Replication",
        },
        title="Queue length over time",
    )


def grouped_fluid_simulation_comparison(frame: pd.DataFrame):
    """Return grouped bars comparing fluid estimates with simulation means."""

    if frame.empty:
        fig = go.Figure()
        fig.update_layout(
            title="Fluid estimate vs simulation mean unavailable",
            xaxis_title="Metric",
            yaxis_title="Value",
        )
        return fig
    fig = px.bar(
        frame,
        x="metric",
        y="value",
        color="estimate_source",
        facet_col="call_center",
        facet_col_wrap=2,
        barmode="group",
        labels={
            "metric": "Metric",
            "value": "Value",
            "estimate_source": "Estimate",
            "call_center": "Call center",
        },
        title="Fluid estimate vs simulation mean by call center",
    )
    fig.update_xaxes(tickangle=-20)
    fig.update_yaxes(matches=None)
    return fig


FITTING_LINE_OPTIONS = [
    "Selected model fit",
    "Fluid-flow p",
    "Stochastic-Q p",
]


def fitting_overlay(
    distribution: pd.DataFrame,
    fit_row: dict,
    *,
    title: str = "Fitting Overlay",
    visible_lines: list[str] | tuple[str, ...] | set[str] | None = None,
):
    """Return empirical PMF bars with user-selected fitting/reference lines."""

    visible = set(FITTING_LINE_OPTIONS if visible_lines is None else visible_lines)
    frame = fitted_pmf_frame(distribution, fit_row)
    fig = go.Figure()
    if frame.empty:
        fig.update_layout(title=title)
        return fig
    fig.add_bar(x=frame["attempt_count"], y=frame["empirical_probability"], name="Empirical")
    fit_name = "Geometric MLE" if fit_row["model"] == "geometric" else f"Fitted {fit_row['model']}"
    if "Selected model fit" in visible:
        fig.add_scatter(
            x=frame["attempt_count"],
            y=frame["fitted_probability"],
            name=fit_name,
            mode="lines+markers",
        )
    p_flow = fit_row.get("p_flow", np.nan)
    if (
        "Fluid-flow p" in visible
        and fit_row["model"] == "geometric"
        and pd.notna(p_flow)
        and np.isfinite(float(p_flow))
    ):
        k = frame["attempt_count"].to_numpy(dtype=int)
        fig.add_scatter(
            x=frame["attempt_count"],
            y=geometric_pmf(k, float(p_flow)),
            name="Fluid-flow p",
            mode="lines+markers",
            line={"dash": "dash"},
        )
    p_stochastic_mean_q = fit_row.get("p_stochastic_mean_q", np.nan)
    if (
        "Stochastic-Q p" in visible
        and fit_row["model"] == "geometric"
        and pd.notna(p_stochastic_mean_q)
        and np.isfinite(float(p_stochastic_mean_q))
    ):
        k = frame["attempt_count"].to_numpy(dtype=int)
        fig.add_scatter(
            x=frame["attempt_count"],
            y=geometric_pmf(k, float(p_stochastic_mean_q)),
            name="Stochastic-Q p",
            mode="lines+markers",
            line={"dash": "dot"},
        )
    fig.update_layout(title=title, xaxis_title="Attempt count", yaxis_title="Probability / proportion", barmode="overlay")
    return fig


def residual_figure(distribution: pd.DataFrame, fit_row: dict, *, title: str = "PMF Residuals"):
    frame = fitted_pmf_frame(distribution, fit_row)
    fig = px.bar(
        frame,
        x="attempt_count",
        y="residual",
        labels={"attempt_count": "Attempt count", "residual": "Empirical probability - fitted probability"},
        title=title,
    )
    fig.add_hline(y=0.0, line_width=1, line_dash="dash")
    return fig


def metric_comparison(summary: pd.DataFrame, metric: str):
    required = {"metric", "scenario_name", "mean"}
    if summary.empty or not required.issubset(summary.columns):
        fig = go.Figure()
        fig.update_layout(
            title=f"{metric} comparison unavailable",
            xaxis_title="Scenario",
            yaxis_title=metric,
        )
        return fig
    frame = summary[summary["metric"] == metric].copy()
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{metric} comparison unavailable",
            xaxis_title="Scenario",
            yaxis_title=metric,
        )
        return fig
    error_y = "standard_error" if "standard_error" in frame.columns else None
    return px.bar(
        frame,
        x="scenario_name",
        y="mean",
        error_y=error_y,
        labels={"scenario_name": "Scenario", "mean": metric},
        title=f"{metric} comparison",
    )


def sweep_line(frame: pd.DataFrame, y: str, *, color: str = "population"):
    clean = frame.dropna(subset=["sweep_value", y]).copy()
    return px.line(
        clean,
        x="sweep_value",
        y=y,
        color=color if color in clean.columns else None,
        markers=True,
        labels={"sweep_value": "Sweep value", y: y},
        title=f"Parameter sweep: {y}",
    )


def sweep_p_comparison(frame: pd.DataFrame, *, population: str):
    """Compare empirical, fitted, and fluid-flow p values across a sweep."""

    p_columns = {
        "geometric_p": "Geometric MLE p",
        "p_flow": "Fluid-flow p",
        "p_first_attempt": "First-attempt empirical probability",
    }
    available = [column for column in p_columns if column in frame.columns]
    clean = frame[frame["population"] == population].copy() if "population" in frame.columns else frame.copy()
    clean = clean.drop_duplicates(subset=["scenario_name", "sweep_value"])
    if not available or clean.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"{population} p comparison unavailable",
            xaxis_title="Sweep value",
            yaxis_title="p",
        )
        return fig
    long = clean.melt(
        id_vars=["scenario_name", "sweep_value"],
        value_vars=available,
        var_name="series",
        value_name="p_value",
    )
    long["series"] = long["series"].map(p_columns)
    long = long.replace([np.inf, -np.inf], np.nan).dropna(subset=["p_value"])
    return px.line(
        long,
        x="sweep_value",
        y="p_value",
        color="series",
        markers=True,
        labels={"sweep_value": "Sweep value", "p_value": "p", "series": "Series"},
        title=f"{population} geometric p comparison",
    )


def sweep_distribution_fit_overlay(
    distribution: pd.DataFrame,
    fitting_results: pd.DataFrame,
    *,
    population: str,
    aggregation: str = "pooled_probability",
    scenarios: list[str] | None = None,
):
    """Return PMF bars and geometric fitted lines for sweep scenarios."""

    y = aggregation if aggregation in distribution.columns else "pooled_probability"
    clean = distribution[distribution["population"] == population].copy()
    if scenarios is not None:
        clean = clean[clean["scenario_name"].isin(scenarios)]
    fig = go.Figure()
    if clean.empty:
        fig.update_layout(
            title=f"{population} attempt distribution unavailable",
            xaxis_title="Attempt count",
            yaxis_title="Proportion",
        )
        return fig

    if scenarios is None:
        scenario_names = list(dict.fromkeys(clean["scenario_name"].tolist()))
    else:
        available = set(clean["scenario_name"].dropna().tolist())
        scenario_names = [scenario_name for scenario_name in scenarios if scenario_name in available]
    colors = px.colors.qualitative.Plotly
    fit_rows = fitting_results[
        (fitting_results["population"] == population)
        & (fitting_results["model"] == "geometric")
        & (fitting_results["success"] == True)
    ]
    for index, scenario_name in enumerate(scenario_names):
        scenario_dist = clean[clean["scenario_name"] == scenario_name].sort_values("attempt_count")
        color = colors[index % len(colors)]
        fig.add_bar(
            x=scenario_dist["attempt_count"],
            y=scenario_dist[y],
            name=f"{scenario_name} empirical",
            marker_color=color,
            opacity=0.55,
            legendgroup=scenario_name,
            offsetgroup=scenario_name,
        )
        fit_subset = fit_rows[fit_rows["scenario_name"] == scenario_name]
        if fit_subset.empty:
            continue
        overlay_frame = fitted_pmf_frame(scenario_dist, fit_subset.iloc[0].to_dict())
        if overlay_frame.empty:
            continue
        fig.add_scatter(
            x=overlay_frame["attempt_count"],
            y=overlay_frame["fitted_probability"],
            name=f"{scenario_name} Geometric MLE",
            mode="lines+markers",
            line={"color": color, "width": 2},
            marker={"size": 6},
            legendgroup=scenario_name,
        )
        p_flow = fit_subset.iloc[0].get("p_flow", np.nan)
        if pd.notna(p_flow) and np.isfinite(float(p_flow)):
            k = overlay_frame["attempt_count"].to_numpy(dtype=int)
            fig.add_scatter(
                x=overlay_frame["attempt_count"],
                y=geometric_pmf(k, float(p_flow)),
                name=f"{scenario_name} Fluid-flow p",
                mode="lines",
                line={"color": color, "dash": "dash", "width": 2},
                legendgroup=scenario_name,
            )
    fig.update_layout(
        title=f"{population} attempt distribution with geometric fits",
        xaxis_title="Attempt count",
        yaxis_title="Proportion",
        barmode="group",
    )
    return fig


def waiting_policy_scatter(frame: pd.DataFrame, x: str, y: str):
    clean = frame.replace([np.inf, -np.inf], np.nan).copy()
    clean[x] = pd.to_numeric(clean[x], errors="coerce")
    clean[y] = pd.to_numeric(clean[y], errors="coerce")
    clean = clean.dropna(subset=[x, y])
    fig = px.scatter(
        clean,
        x=x,
        y=y,
        color="scenario_name",
        hover_data=[
            column
            for column in [
                "scenario_name",
                "completion_rate",
                "procedural_denial_rate",
                "left_without_enrollment_rate",
                "mean_attempt_count_all",
            ]
            if column in clean.columns
        ],
        labels={x: x, y: y},
        title="Policy and waiting-time analysis",
    )
    fig.update_traces(marker={"size": 10})
    if len(clean) >= 2 and clean[x].nunique() > 1:
        x_values = clean[x].to_numpy(dtype=float)
        y_values = clean[y].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x_values, y_values, 1)
        line_x = np.linspace(float(x_values.min()), float(x_values.max()), 100)
        line_y = slope * line_x + intercept
        fig.add_scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Fitted relation",
            line={"color": "black", "dash": "dash", "width": 2},
            hovertemplate=f"{x}: %{{x:.4g}}<br>{y}: %{{y:.4g}}<extra>Fitted relation</extra>",
        )
        if abs(slope) > 1e-12:
            x_intercept = -intercept / slope
            x_intercept_text = f"{x_intercept:.4g}"
        else:
            x_intercept_text = "undefined"
        fit_text = (
            f"y = {slope:.4g}x + {intercept:.4g}<br>"
            f"y-intercept = {intercept:.4g}<br>"
            f"x-intercept = {x_intercept_text}"
        )
        fig.add_annotation(
            text=fit_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1,
        )
    return fig
