"""Shared helpers for weekly research scripts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from configs.paper_cc2_baseline import BASELINE, default_initial_state
from fluid_steady_state import FluidSteadyStateResult, solve_fluid_steady_state
from light_simulation import LightState, run_light_replications
from stochastic_simulation import SimulationParams


OUTPUT_DIR = Path("outputs/weekly_research")
CORE_METRICS = {
    "Q": ("mean_Q", "q_bar"),
    "B": ("mean_B", "b_bar"),
    "RS": ("mean_RS", "rS_bar"),
    "RL": ("mean_RL", "rL_bar"),
}


def ensure_output_dir() -> Path:
    """Create and return the weekly output directory."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def percent_difference(value: float, reference: float) -> float:
    """Return 100 * (value - reference) / reference, or nan for zero reference."""

    return float(np.nan) if reference == 0 else 100.0 * (value - reference) / reference


def summarize_light_vs_fluid(
    params: SimulationParams,
    n_replications: int | None = None,
    initial_state: LightState | None = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Compare light-simulator aggregate means against fluid steady state."""

    n_replications = BASELINE.n_replications if n_replications is None else n_replications
    initial_state = default_initial_state(params) if initial_state is None else initial_state
    light = run_light_replications(
        params,
        n_replications,
        horizon=params.T,
        initial_state=initial_state,
        validate=validate,
    )
    fluid = solve_fluid_steady_state(params).to_dict()

    rows = []
    for label, (light_metric, fluid_metric) in CORE_METRICS.items():
        sim_mean = float(light[light_metric].mean())
        sim_std = float(light[light_metric].std(ddof=1))
        fluid_value = float(fluid[fluid_metric])
        rows.append(
            {
                "metric": label,
                "simulation_mean": sim_mean,
                "simulation_std": sim_std,
                "fluid_value": fluid_value,
                "difference": sim_mean - fluid_value,
                "percent_difference": percent_difference(sim_mean, fluid_value),
            }
        )
    return pd.DataFrame(rows)


def scenario_core_row(
    scenario_name: str,
    changed_parameter: str,
    changed_value,
    params: SimulationParams,
    n_replications: int | None = None,
    initial_state: LightState | None = None,
) -> dict[str, object]:
    """Return one wide scenario row comparing Q, B, RS, and RL."""

    frame = summarize_light_vs_fluid(params, n_replications, initial_state)
    row: dict[str, object] = {
        "scenario_name": scenario_name,
        "changed_parameter": changed_parameter,
        "changed_value": changed_value,
    }
    for _, metric_row in frame.iterrows():
        metric = metric_row["metric"]
        row[f"fluid_{metric}"] = metric_row["fluid_value"]
        row[f"sim_{metric}_mean"] = metric_row["simulation_mean"]
        row[f"sim_{metric}_std"] = metric_row["simulation_std"]
        row[f"percent_error_{metric}"] = metric_row["percent_difference"]
    return row


def replace_aht(params: SimulationParams, aht_minutes: float) -> SimulationParams:
    """Return params with service rates implied by a new average handle time."""

    mu = BASELINE.model_day_minutes / aht_minutes
    return replace(
        params,
        mu_plus=BASELINE.p_plus * mu,
        mu_minus=(1.0 - BASELINE.p_plus) * mu,
    )


def aggregate_derived_metrics(params: SimulationParams, q: float, b: float, rS: float, rL: float):
    """Compute lightweight derived quantities from aggregate state values."""

    waiting = max(q - params.c, 0.0)
    lambda_hat = params.lam + params.deltaB * b + params.deltaS * rS + params.deltaL * rL
    theta = params.thetaA + params.thetaS + params.thetaL
    lambda_tilde = lambda_hat - theta * waiting
    procedural = params.thetaA * waiting
    average_wait = np.nan if lambda_hat <= 0 else waiting / lambda_hat * BASELINE.model_day_minutes
    asa = np.nan if lambda_tilde <= 0 else waiting / lambda_tilde * BASELINE.model_day_minutes
    return {
        "waiting_count": waiting,
        "procedural_denial_rate": procedural,
        "average_wait": average_wait,
        "ASA": asa,
        "endogenous_congestion_redials": params.deltaS * rS + params.deltaL * rL,
        "endogenous_congestion_recertification": params.deltaB * b,
    }


def metric_values_from_fluid(result: FluidSteadyStateResult) -> dict[str, float]:
    """Return fluid result values keyed by aggregate state label."""

    return {"Q": result.q_bar, "B": result.b_bar, "RS": result.rS_bar, "RL": result.rL_bar}


def with_seed(params: SimulationParams, seed: int) -> SimulationParams:
    """Return params with a new seed."""

    return replace(params, seed=seed)
