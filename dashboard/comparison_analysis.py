"""Scenario comparison tables for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .distribution_analysis import population_sample
from .scenario_registry import PARAMETER_METADATA


def _mean_metric(replication_metrics: pd.DataFrame, scenario: str, metric: str) -> float:
    if replication_metrics.empty or metric not in replication_metrics.columns:
        return float("nan")
    subset = replication_metrics[replication_metrics["scenario_name"] == scenario]
    if subset.empty:
        return float("nan")
    values = subset[metric].dropna().astype(float)
    return float(values.mean()) if len(values) else float("nan")


def _fit_row(fitting_results: pd.DataFrame, scenario: str, population: str, model: str) -> pd.Series | None:
    if fitting_results.empty:
        return None
    subset = fitting_results[
        (fitting_results["scenario_name"] == scenario)
        & (fitting_results["population"] == population)
        & (fitting_results["model"] == model)
        & (fitting_results["success"] == True)
    ]
    return subset.iloc[0] if len(subset) else None


def build_comparison_table(run_result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one readable scenario-comparison row per scenario/population."""

    caller_records = run_result.get("caller_records", pd.DataFrame())
    fitting_results = run_result.get("fitting_results", pd.DataFrame())
    replication_metrics = run_result.get("replication_metrics", pd.DataFrame())
    parameters = run_result.get("parameters", pd.DataFrame())
    columns = [
        "scenario_name",
        "population",
        "sample_size",
        "mean_attempts",
        "p90_attempts",
        "p95_attempts",
        "geometric_p",
        "p_mle",
        "p_first_attempt",
        "p_flow",
        "flow_based_p",
        "p_mle_minus_p_flow",
        "p_mle_flow_absolute_error",
        "p_mle_flow_relative_error",
        "p_stochastic_mean_q",
        "stochastic_mean_q_flow_based_p",
        "p_mle_minus_p_stochastic_mean_q",
        "p_mle_stochastic_mean_q_absolute_error",
        "p_mle_stochastic_mean_q_relative_error",
        "stochastic_mean_q",
        "fluid_q_bar",
        "fluid_waiting_quantity",
        "fluid_service_quantity",
        "fluid_terminal_flow",
        "fluid_total_attempt_ending_flow",
        "fluid_flow_status",
        "geometric_rmse",
        "geometric_cdf_difference",
        "geometric_aic",
        "negative_binomial_aic",
        "completion_rate",
        "left_without_enrollment_rate",
        "abandonment_fraction",
        "staffing",
        "arrival_rate_lam",
        "mu_plus",
        "mu_minus",
        "thetaA",
        "thetaS",
        "thetaL",
    ]
    if caller_records.empty:
        return pd.DataFrame(columns=columns)

    parameter_lookup = {}
    if not parameters.empty and "scenario_name" in parameters.columns:
        parameter_lookup = {
            str(row.scenario_name): row._asdict()
            for row in parameters.itertuples(index=False)
        }

    rows = []
    for scenario_name, scenario_frame in caller_records.groupby("scenario_name"):
        param_row = parameter_lookup.get(str(scenario_name), {})
        for population in ["all", "completed", "abandoned"]:
            sample = population_sample(scenario_frame, population).sample
            geom = _fit_row(fitting_results, scenario_name, population, "geometric")
            nb = _fit_row(fitting_results, scenario_name, population, "negative_binomial")
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "population": population,
                    "sample_size": int(len(sample)),
                    "mean_attempts": float(np.mean(sample)) if len(sample) else np.nan,
                    "p90_attempts": float(np.percentile(sample, 90)) if len(sample) else np.nan,
                    "p95_attempts": float(np.percentile(sample, 95)) if len(sample) else np.nan,
                    "geometric_p": float(geom["p"]) if geom is not None else np.nan,
                    "p_mle": float(geom.get("p_mle", geom["p"])) if geom is not None else np.nan,
                    "p_first_attempt": float(geom.get("p_first_attempt", np.nan)) if geom is not None else np.nan,
                    "p_flow": float(geom.get("p_flow", np.nan)) if geom is not None else np.nan,
                    "flow_based_p": float(geom.get("flow_based_p", geom.get("p_flow", np.nan))) if geom is not None else np.nan,
                    "p_mle_minus_p_flow": float(geom.get("p_mle_minus_p_flow", np.nan)) if geom is not None else np.nan,
                    "p_mle_flow_absolute_error": float(geom.get("p_mle_flow_absolute_error", np.nan)) if geom is not None else np.nan,
                    "p_mle_flow_relative_error": float(geom.get("p_mle_flow_relative_error", np.nan)) if geom is not None else np.nan,
                    "p_stochastic_mean_q": float(geom.get("p_stochastic_mean_q", np.nan)) if geom is not None else np.nan,
                    "stochastic_mean_q_flow_based_p": float(geom.get("stochastic_mean_q_flow_based_p", np.nan)) if geom is not None else np.nan,
                    "p_mle_minus_p_stochastic_mean_q": float(geom.get("p_mle_minus_p_stochastic_mean_q", np.nan)) if geom is not None else np.nan,
                    "p_mle_stochastic_mean_q_absolute_error": float(geom.get("p_mle_stochastic_mean_q_absolute_error", np.nan)) if geom is not None else np.nan,
                    "p_mle_stochastic_mean_q_relative_error": float(geom.get("p_mle_stochastic_mean_q_relative_error", np.nan)) if geom is not None else np.nan,
                    "stochastic_mean_q": float(geom.get("stochastic_mean_q", np.nan)) if geom is not None else np.nan,
                    "fluid_q_bar": float(geom.get("fluid_q_bar", np.nan)) if geom is not None else np.nan,
                    "fluid_waiting_quantity": float(geom.get("fluid_waiting_quantity", np.nan)) if geom is not None else np.nan,
                    "fluid_service_quantity": float(geom.get("fluid_service_quantity", np.nan)) if geom is not None else np.nan,
                    "fluid_terminal_flow": float(geom.get("fluid_terminal_flow", np.nan)) if geom is not None else np.nan,
                    "fluid_total_attempt_ending_flow": float(geom.get("fluid_total_attempt_ending_flow", np.nan)) if geom is not None else np.nan,
                    "fluid_flow_status": str(geom.get("fluid_flow_status", "")) if geom is not None else "",
                    "geometric_rmse": float(geom["pmf_rmse"]) if geom is not None else np.nan,
                    "geometric_cdf_difference": float(geom["max_abs_cdf_difference"]) if geom is not None else np.nan,
                    "geometric_aic": float(geom["aic"]) if geom is not None else np.nan,
                    "negative_binomial_aic": float(nb["aic"]) if nb is not None else np.nan,
                    "completion_rate": _mean_metric(replication_metrics, scenario_name, "completion_rate"),
                    "left_without_enrollment_rate": _mean_metric(
                        replication_metrics, scenario_name, "left_without_enrollment_rate"
                    ),
                    "abandonment_fraction": _mean_metric(replication_metrics, scenario_name, "abandonment_fraction"),
                    "staffing": param_row.get("c", np.nan),
                    "arrival_rate_lam": param_row.get("lam", np.nan),
                    "mu_plus": param_row.get("mu_plus", np.nan),
                    "mu_minus": param_row.get("mu_minus", np.nan),
                    "thetaA": param_row.get("thetaA", np.nan),
                    "thetaS": param_row.get("thetaS", np.nan),
                    "thetaL": param_row.get("thetaL", np.nan),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _values_equal(left, right, *, tolerance: float) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return bool(np.isclose(float(left), float(right), rtol=tolerance, atol=tolerance))
    except (TypeError, ValueError):
        return str(left) == str(right)


def build_parameter_change_table(
    run_result: dict[str, pd.DataFrame],
    *,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """Return parameters whose values differ across compared scenarios."""

    parameters = run_result.get("parameters", pd.DataFrame())
    if parameters.empty or "scenario_name" not in parameters.columns:
        return pd.DataFrame(columns=["parameter", "meaning", "unit"])
    scenario_rows = parameters.drop_duplicates(subset=["scenario_name"], keep="first").copy()
    scenario_names = [str(name) for name in scenario_rows["scenario_name"]]
    ignored_columns = {"scenario_name"}
    rows = []
    for parameter in [column for column in scenario_rows.columns if column not in ignored_columns]:
        values = scenario_rows[parameter].tolist()
        first = values[0] if values else np.nan
        if all(_values_equal(first, value, tolerance=tolerance) for value in values[1:]):
            continue
        meta = PARAMETER_METADATA.get(parameter, {"label": parameter, "unit": ""})
        row = {
            "parameter": parameter,
            "meaning": meta["label"],
            "unit": meta["unit"],
        }
        for scenario_name, value in zip(scenario_names, values, strict=False):
            row[scenario_name] = value
        if len(scenario_names) == 2:
            left, right = values
            try:
                left_float = float(left)
                right_float = float(right)
                row["absolute_change"] = right_float - left_float
                row["percent_change"] = (
                    (right_float - left_float) / abs(left_float) * 100.0
                    if left_float != 0
                    else np.nan
                )
            except (TypeError, ValueError):
                row["absolute_change"] = np.nan
                row["percent_change"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)
