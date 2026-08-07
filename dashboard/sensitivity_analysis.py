"""One-dimensional parameter sweep support."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Callable

import numpy as np
import pandas as pd

from stochastic_simulation import SimulationParams

from .scenario_registry import EXPERIMENT_PRESETS, set_parameter_for_sweep
from .simulation_runner import (
    ScenarioRunOptions,
    add_distribution_and_fits,
    combine_run_results,
    run_scenarios,
)


def sweep_values(
    *,
    minimum: float,
    maximum: float,
    grid_points: int,
    scale: str,
) -> list[float]:
    """Create linear or log-spaced one-dimensional grid values."""

    if grid_points <= 0:
        raise ValueError("grid_points must be positive")
    if minimum > maximum:
        raise ValueError("minimum must be <= maximum")
    if scale == "log":
        if minimum <= 0 or maximum <= 0:
            raise ValueError("log-scale sweeps require positive bounds")
        return [float(value) for value in np.geomspace(minimum, maximum, grid_points)]
    return [float(value) for value in np.linspace(minimum, maximum, grid_points)]


def preset_values(
    preset_name: str,
    baseline_params: SimulationParams,
    *,
    grid_points: int,
) -> tuple[str, list[float]]:
    """Return parameter and default values for a named experiment preset."""

    preset = EXPERIMENT_PRESETS[preset_name]
    parameter = str(preset["parameter"])
    if "values" in preset:
        return parameter, [float(value) for value in preset["values"]]
    if "bounds" in preset:
        minimum, maximum = preset["bounds"]
        return parameter, sweep_values(minimum=float(minimum), maximum=float(maximum), grid_points=grid_points, scale="linear")
    low, high = preset["scale_bounds"]
    base_value = {
        "thetaA": baseline_params.thetaA,
        "thetaS": baseline_params.thetaS,
        "thetaL": baseline_params.thetaL,
        "c": baseline_params.c,
        "aht_minutes": 540.0 / (baseline_params.mu_plus + baseline_params.mu_minus),
        "lam": baseline_params.lam,
    }[parameter]
    return parameter, sweep_values(minimum=float(base_value * low), maximum=float(base_value * high), grid_points=grid_points, scale="linear")


def build_sweep_scenarios(
    baseline_params: SimulationParams,
    *,
    parameter: str,
    values: list[float],
) -> tuple[dict[str, SimulationParams], pd.DataFrame]:
    """Build sweep scenario parameters and the matching configuration table."""

    scenarios = {}
    config_rows = []
    for index, value in enumerate(values):
        params = set_parameter_for_sweep(baseline_params, parameter, value)
        scenario_name = f"{parameter}={value:.6g}"
        scenarios[scenario_name] = params
        row = asdict(params)
        row.update(
            {
                "scenario_name": scenario_name,
                "sweep_parameter": parameter,
                "sweep_value": value,
                "parameter_index": index,
            }
        )
        config_rows.append(row)
    return scenarios, pd.DataFrame(config_rows)


def finalize_parameter_sweep(
    combined: dict[str, pd.DataFrame],
    *,
    sweep_configuration: pd.DataFrame,
    parameter: str,
    populations: list[str],
    fit_negative_binomial_model: bool = False,
) -> dict[str, pd.DataFrame]:
    """Add sweep configuration, distributions, fits, and compact result table."""

    combined = dict(combined)
    combined["sweep_configuration"] = sweep_configuration
    enriched = add_distribution_and_fits(
        combined,
        populations=populations,
        fit_negative_binomial_model=fit_negative_binomial_model,
    )
    enriched["sweep_results"] = _sweep_result_table(enriched, parameter)
    return enriched


def run_parameter_sweep(
    baseline_params: SimulationParams,
    *,
    parameter: str,
    values: list[float],
    options: ScenarioRunOptions,
    populations: list[str],
    fit_negative_binomial_model: bool = False,
    common_random_numbers: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run a one-dimensional parameter sweep."""

    scenarios, sweep_configuration = build_sweep_scenarios(
        baseline_params,
        parameter=parameter,
        values=values,
    )
    run_options = options if common_random_numbers else replace(options, base_seed=options.base_seed + options.seed_stride)
    if not common_random_numbers:
        all_results = []
        for index, (scenario_name, params) in enumerate(scenarios.items()):
            scenario_options = replace(options, base_seed=options.base_seed + index * options.seed_stride)
            result = run_scenarios({scenario_name: params}, scenario_options, progress=progress)
            all_results.append(result)
        combined = combine_run_results(all_results)
    else:
        combined = run_scenarios(scenarios, run_options, progress=progress)
    return finalize_parameter_sweep(
        combined,
        sweep_configuration=sweep_configuration,
        parameter=parameter,
        populations=populations,
        fit_negative_binomial_model=fit_negative_binomial_model,
    )


def _sweep_result_table(result: dict[str, pd.DataFrame], parameter: str) -> pd.DataFrame:
    """Combine metrics and fits into a compact sweep table."""

    summary = result["scenario_summary"]
    fit = result["fitting_results"]
    config = result["sweep_configuration"][["scenario_name", "sweep_value"]]
    metric_pivot = (
        summary.pivot_table(index="scenario_name", columns="metric", values="mean", aggfunc="first")
        .reset_index()
        if not summary.empty
        else pd.DataFrame(columns=["scenario_name"])
    )
    merged = config.merge(metric_pivot, on="scenario_name", how="left")
    geom = fit[(fit["model"] == "geometric") & (fit["success"] == True)].copy()
    if not geom.empty:
        geom_columns = [
            "scenario_name",
            "population",
            "sample_size",
            "sample_mean",
            "p",
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
            "fluid_final_abandonment_flow",
            "fluid_short_abandonment_flow",
            "fluid_long_abandonment_flow",
            "fluid_service_success_flow",
            "fluid_service_failure_flow",
            "fluid_flow_status",
            "pmf_rmse",
            "max_abs_cdf_difference",
            "aic",
        ]
        geom = geom[[column for column in geom_columns if column in geom.columns]]
        geom = geom.rename(columns={"p": "geometric_p", "aic": "geometric_aic"})
        merged = merged.merge(geom, on="scenario_name", how="left")
    merged.insert(1, "sweep_parameter", parameter)
    return merged
