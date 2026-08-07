"""Thin dashboard runner around the existing stochastic simulation engine."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from experiments.cohort_policy_analysis import replication_metrics, scenario_summary
from stochastic_simulation import SimulationParams, simulate_one

from .distribution_analysis import attempt_distribution, population_sample
from .distribution_fitting import fit_geometric, fit_negative_binomial
from .fluid_analysis import compute_flow_based_geometric_p
from .scenario_registry import resolve_staffing_for_replication


ProgressCallback = Callable[[int, int, str], None]


FLOW_FITTING_COLUMNS = [
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
    "stochastic_mean_q_waiting_quantity",
    "stochastic_mean_q_service_quantity",
    "stochastic_mean_q_terminal_flow",
    "stochastic_mean_q_total_attempt_ending_flow",
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
    "fluid_overloaded",
    "fluid_finite",
    "fluid_flow_status",
    "fluid_flow_message",
]


@dataclass(frozen=True)
class ScenarioRunOptions:
    """Simulation options controlled by the dashboard."""

    replications: int = 5
    base_seed: int = 20260720
    dynamic_horizon: bool = True
    cohort_start: float = 260.0
    cohort_end: float = 420.0
    post_clearance_buffer: float = 20.0
    max_dynamic_horizon: float = 900.0
    max_events: int = 4_000_000
    validate: bool = False
    staffing_mode: str = "default"
    probability_33: float = 0.5
    seed_stride: int = 10_000
    queue_distribution_enabled: bool = False
    queue_warmup_time: float = 260.0
    queue_sample_interval: float = 1.0
    queue_observation_end: float | None = None


def replication_seeds(base_seed: int, replications: int) -> list[int]:
    """Return deterministic dashboard replication seeds."""

    if replications <= 0:
        raise ValueError("replications must be positive")
    if base_seed < 0:
        raise ValueError("base_seed must be nonnegative")
    return [int(base_seed + index) for index in range(replications)]


def run_scenarios(
    scenarios: dict[str, SimulationParams],
    options: ScenarioRunOptions,
    *,
    progress: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    """Run one or more scenarios and return DataFrames used by the dashboard."""

    if options.cohort_start >= options.cohort_end:
        raise ValueError("cohort_start must be less than cohort_end")
    total_runs = len(scenarios) * options.replications
    run_index = 0
    replication_rows = []
    termination_rows = []
    caller_frames = []
    parameter_rows = []
    queue_sample_frames = []

    if options.queue_distribution_enabled:
        if options.queue_warmup_time < 0:
            raise ValueError("queue_warmup_time must be nonnegative")
        if options.queue_sample_interval <= 0:
            raise ValueError("queue_sample_interval must be positive")
        if (
            options.queue_observation_end is not None
            and options.queue_observation_end < options.queue_warmup_time
        ):
            raise ValueError(
                "queue_observation_end must be greater than or equal to queue_warmup_time"
            )

    for scenario_name, base_params in scenarios.items():
        parameter_row = asdict(base_params)
        parameter_row.update({"scenario_name": scenario_name})
        parameter_rows.append(parameter_row)
        for replication, seed in enumerate(replication_seeds(options.base_seed, options.replications)):
            run_index += 1
            if progress is not None:
                progress(
                    run_index - 1,
                    total_runs,
                    f"Running {scenario_name} replication {replication + 1}/{options.replications}",
                )
            params = resolve_staffing_for_replication(
                base_params,
                mode=options.staffing_mode,
                probability_33=options.probability_33,
                seed=seed,
            )
            params = params.__class__(**{**asdict(params), "seed": seed})
            simulation_output = simulate_one(
                params,
                validate=options.validate,
                return_caller_records=True,
                return_end_state_distribution=True,
                return_cohort_summary=True,
                cohort_start=options.cohort_start,
                cohort_end=options.cohort_end,
                dynamic_horizon=options.dynamic_horizon,
                post_clearance_buffer=options.post_clearance_buffer,
                max_dynamic_horizon=options.max_dynamic_horizon,
                max_events=options.max_events,
                queue_distribution_enabled=options.queue_distribution_enabled,
                queue_warmup_time=options.queue_warmup_time,
                queue_sample_interval=options.queue_sample_interval,
                queue_observation_end=options.queue_observation_end,
            )
            if options.queue_distribution_enabled:
                result, path, extras = simulation_output
                queue_samples = path[["t", "waiting"]].copy()
                queue_samples = queue_samples.rename(
                    columns={"t": "sample_time", "waiting": "queue_length"}
                )
                queue_samples.insert(0, "seed", seed)
                queue_samples.insert(0, "replication", replication)
                queue_samples.insert(0, "scenario_name", scenario_name)
                queue_sample_frames.append(queue_samples)
            else:
                result, extras = simulation_output
            if options.dynamic_horizon:
                diagnostics = extras["dynamic_horizon_diagnostics"].iloc[0].copy()
            else:
                cohort_records_for_diag = extras["cohort_caller_state_records"]
                diagnostics = pd.Series(
                    {
                        "cohort_start": options.cohort_start,
                        "cohort_end": options.cohort_end,
                        "cohort_size": len(cohort_records_for_diag),
                        "cohort_completed": int((cohort_records_for_diag["terminal_outcome"] == "completed").sum()),
                        "cohort_left_without_enrollment": int((cohort_records_for_diag["terminal_outcome"] == "left_without_enrollment").sum()),
                        "cohort_unfinished": int((cohort_records_for_diag["terminal_outcome"] == "unfinished").sum()),
                        "cohort_clearance_time": np.nan,
                        "simulation_end_time": params.T,
                        "dynamic_horizon_success": True,
                        "events_processed": np.nan,
                        "termination_reason": "fixed_horizon",
                    }
                )
            cohort_records = extras["cohort_caller_state_records"].copy()
            cohort_records.insert(0, "seed", seed)
            cohort_records.insert(0, "replication", replication)
            cohort_records.insert(0, "scenario_name", scenario_name)
            caller_frames.append(cohort_records)
            replication_rows.append(
                replication_metrics(
                    scenario=scenario_name,
                    replication=replication,
                    seed=seed,
                    result=result,
                    diagnostics=diagnostics,
                    cohort_records=cohort_records,
                )
            )
            term = diagnostics.to_dict()
            term.update({"scenario_name": scenario_name, "replication": replication, "seed": seed})
            termination_rows.append(term)
            if progress is not None:
                progress(
                    run_index,
                    total_runs,
                    f"Finished {scenario_name} replication {replication + 1}/{options.replications}",
                )

    caller_records = pd.concat(caller_frames, ignore_index=True) if caller_frames else pd.DataFrame()
    queue_samples = pd.concat(queue_sample_frames, ignore_index=True) if queue_sample_frames else pd.DataFrame(
        columns=["scenario_name", "replication", "seed", "sample_time", "queue_length"]
    )
    replication_frame = pd.DataFrame(replication_rows)
    summary_frame = scenario_summary(replication_frame) if len(replication_frame) else pd.DataFrame()
    return {
        "parameters": pd.DataFrame(parameter_rows),
        "replication_metrics": replication_frame,
        "termination_diagnostics": pd.DataFrame(termination_rows),
        "caller_records": caller_records,
        "queue_samples": queue_samples,
        "scenario_summary": summary_frame,
    }


def combine_run_results(results: list[dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Combine scenario run outputs and recompute the cross-scenario summary."""

    if not results:
        return {
            "parameters": pd.DataFrame(),
            "replication_metrics": pd.DataFrame(),
            "termination_diagnostics": pd.DataFrame(),
            "caller_records": pd.DataFrame(),
            "queue_samples": pd.DataFrame(columns=["scenario_name", "replication", "seed", "sample_time", "queue_length"]),
            "scenario_summary": pd.DataFrame(),
        }
    combined = {
        key: pd.concat([result[key] for result in results], ignore_index=True)
        for key in [
            "parameters",
            "replication_metrics",
            "termination_diagnostics",
            "caller_records",
            "queue_samples",
        ]
    }
    replication_frame = combined["replication_metrics"]
    combined["scenario_summary"] = (
        scenario_summary(replication_frame)
        if len(replication_frame)
        else pd.DataFrame()
    )
    return combined


def relabel_run_result(
    result: dict[str, pd.DataFrame],
    scenario_name: str,
) -> dict[str, pd.DataFrame]:
    """Return a run result with all scenario_name columns relabeled."""

    relabeled = {}
    for key in [
        "parameters",
        "replication_metrics",
        "termination_diagnostics",
        "caller_records",
        "queue_samples",
    ]:
        frame = result.get(key, pd.DataFrame()).copy()
        if "scenario_name" in frame.columns:
            frame["scenario_name"] = scenario_name
        relabeled[key] = frame
    replication_frame = relabeled["replication_metrics"]
    relabeled["scenario_summary"] = (
        scenario_summary(replication_frame)
        if len(replication_frame)
        else pd.DataFrame()
    )
    return relabeled


def add_distribution_and_fits(
    run_result: dict[str, pd.DataFrame],
    *,
    populations: list[str],
    fit_negative_binomial_model: bool = False,
) -> dict[str, pd.DataFrame]:
    """Compute distribution and fitting tables for a run result."""

    caller_records = run_result["caller_records"]
    parameters = run_result.get("parameters", pd.DataFrame())
    parameter_lookup: dict[str, SimulationParams] = {}
    if not parameters.empty and "scenario_name" in parameters.columns:
        for row in parameters.to_dict(orient="records"):
            scenario_name = str(row["scenario_name"])
            values = {
                field: row[field]
                for field in SimulationParams.__dataclass_fields__
                if field in row
            }
            parameter_lookup[scenario_name] = SimulationParams(**values)
    replication_metrics = run_result.get("replication_metrics", pd.DataFrame())
    stochastic_mean_q_lookup: dict[str, float] = {}
    if not replication_metrics.empty and {"scenario_name", "mean_Q"}.issubset(
        replication_metrics.columns
    ):
        stochastic_mean_q_lookup = (
            replication_metrics.groupby("scenario_name")["mean_Q"]
            .mean()
            .astype(float)
            .to_dict()
        )
    distribution_frames = []
    fit_rows = []
    for scenario_name, scenario_frame in caller_records.groupby("scenario_name"):
        flow_result = None
        stochastic_mean_q_flow_result = None
        stochastic_mean_q = stochastic_mean_q_lookup.get(str(scenario_name), np.nan)
        if scenario_name in parameter_lookup:
            flow_result = compute_flow_based_geometric_p(parameter_lookup[scenario_name])
            if np.isfinite(stochastic_mean_q):
                stochastic_mean_q_flow_result = compute_flow_based_geometric_p(
                    parameter_lookup[scenario_name],
                    q_bar=stochastic_mean_q,
                )
        for population in populations:
            distribution = attempt_distribution(scenario_frame, population)
            if not distribution.empty:
                distribution.insert(0, "scenario_name", scenario_name)
            distribution_frames.append(distribution)
            sample = population_sample(scenario_frame, population).sample
            geometric_fit = fit_geometric(sample, population=population, scenario_name=scenario_name)
            if flow_result is not None:
                geometric_fit.update(
                    _flow_fit_fields(
                        geometric_fit,
                        flow_result,
                        stochastic_mean_q_flow_result=stochastic_mean_q_flow_result,
                        stochastic_mean_q=stochastic_mean_q,
                    )
                )
            fit_rows.append(geometric_fit)
            if fit_negative_binomial_model:
                fit_rows.append(fit_negative_binomial(sample, population=population, scenario_name=scenario_name))
    enriched = dict(run_result)
    enriched["attempt_distributions"] = pd.concat(distribution_frames, ignore_index=True) if distribution_frames else pd.DataFrame()
    enriched["fitting_results"] = pd.DataFrame(fit_rows)
    return enriched


def _p_error_fields(
    *,
    p_mle: float,
    p_value: float,
    prefix: str,
) -> dict[str, float]:
    """Return MLE-minus-reference p diagnostics for one flow-based p value."""

    if np.isfinite(p_mle) and np.isfinite(p_value):
        difference = p_mle - p_value
        absolute_error = abs(difference)
        relative_error = absolute_error / p_mle if p_mle > 0 else np.nan
    else:
        difference = np.nan
        absolute_error = np.nan
        relative_error = np.nan
    return {
        f"p_mle_minus_{prefix}": difference,
        f"p_mle_{prefix}_absolute_error": absolute_error,
        f"p_mle_{prefix}_relative_error": relative_error,
    }


def _flow_fit_fields(
    fit_row: dict,
    flow_result,
    *,
    stochastic_mean_q_flow_result=None,
    stochastic_mean_q: float = np.nan,
) -> dict[str, float | str | bool]:
    """Return flow-based p diagnostics aligned with a geometric fit row."""

    flow = flow_result.to_dict()
    p_mle = float(fit_row.get("p_mle", fit_row.get("p", np.nan)))
    p_flow = float(flow["p_flow"])
    flow_errors = _p_error_fields(
        p_mle=p_mle,
        p_value=p_flow if bool(flow["finite"]) else np.nan,
        prefix="p_flow",
    )

    stochastic_flow = (
        stochastic_mean_q_flow_result.to_dict()
        if stochastic_mean_q_flow_result is not None
        else {}
    )
    p_stochastic_mean_q = float(stochastic_flow.get("p_flow", np.nan))
    stochastic_errors = _p_error_fields(
        p_mle=p_mle,
        p_value=(
            p_stochastic_mean_q
            if bool(stochastic_flow.get("finite", False))
            else np.nan
        ),
        prefix="p_stochastic_mean_q",
    )
    fields = {
        "p_flow": p_flow,
        "flow_based_p": p_flow,
        "p_mle_minus_p_flow": flow_errors["p_mle_minus_p_flow"],
        "p_mle_flow_absolute_error": flow_errors["p_mle_p_flow_absolute_error"],
        "p_mle_flow_relative_error": flow_errors["p_mle_p_flow_relative_error"],
        "p_stochastic_mean_q": p_stochastic_mean_q,
        "stochastic_mean_q_flow_based_p": p_stochastic_mean_q,
        "p_mle_minus_p_stochastic_mean_q": stochastic_errors[
            "p_mle_minus_p_stochastic_mean_q"
        ],
        "p_mle_stochastic_mean_q_absolute_error": stochastic_errors[
            "p_mle_p_stochastic_mean_q_absolute_error"
        ],
        "p_mle_stochastic_mean_q_relative_error": stochastic_errors[
            "p_mle_p_stochastic_mean_q_relative_error"
        ],
        "stochastic_mean_q": float(stochastic_mean_q),
        "stochastic_mean_q_waiting_quantity": float(
            stochastic_flow.get("waiting_quantity", np.nan)
        ),
        "stochastic_mean_q_service_quantity": float(
            stochastic_flow.get("service_quantity", np.nan)
        ),
        "stochastic_mean_q_terminal_flow": float(
            stochastic_flow.get("terminal_flow", np.nan)
        ),
        "stochastic_mean_q_total_attempt_ending_flow": float(
            stochastic_flow.get("total_attempt_ending_flow", np.nan)
        ),
        "fluid_q_bar": float(flow["q_bar"]),
        "fluid_waiting_quantity": float(flow["waiting_quantity"]),
        "fluid_service_quantity": float(flow["service_quantity"]),
        "fluid_terminal_flow": float(flow["terminal_flow"]),
        "fluid_total_attempt_ending_flow": float(flow["total_attempt_ending_flow"]),
        "fluid_final_abandonment_flow": float(flow["final_abandonment_flow"]),
        "fluid_short_abandonment_flow": float(flow["short_abandonment_flow"]),
        "fluid_long_abandonment_flow": float(flow["long_abandonment_flow"]),
        "fluid_service_success_flow": float(flow["service_success_flow"]),
        "fluid_service_failure_flow": float(flow["service_failure_flow"]),
        "fluid_overloaded": bool(flow["overloaded"]),
        "fluid_finite": bool(flow["finite"]),
        "fluid_flow_status": str(flow["status"]),
        "fluid_flow_message": str(flow["message"]),
    }
    return fields
