"""Scenario and parameter construction for the Streamlit dashboard."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.cohort_policy_analysis import (
    FIXED_RATES,
    build_policy_scenarios,
    load_call_center_parameters,
    params_from_row,
    service_rates,
)
from stochastic_simulation import MINUTES_PER_MODEL_DAY, SimulationParams


PARAMETER_FILE = Path("data/call_center_parameters.csv")
CALL_CENTER_LABELS = {
    "Call Center 1": 1,
    "Call Center 2": 2,
    "Call Center 3": 3,
    "Call Center 4": 4,
}

PARAMETER_METADATA: dict[str, dict[str, str]] = {
    "lam": {"label": "Arrival rate", "unit": "callers/model day"},
    "c": {"label": "Staffing level", "unit": "agents"},
    "mu_plus": {"label": "Successful service completion rate", "unit": "per model day"},
    "mu_minus": {"label": "Failed service completion rate", "unit": "per model day"},
    "thetaA": {"label": "Final abandonment/loss rate", "unit": "per waiting caller per model day"},
    "thetaS": {"label": "Short-orbit abandonment rate", "unit": "per waiting caller per model day"},
    "thetaL": {"label": "Long-orbit abandonment rate", "unit": "per waiting caller per model day"},
    "deltaB": {"label": "Recertification call rate", "unit": "per B individual per model day"},
    "deltaS": {"label": "Short-redial return rate", "unit": "per short-orbit caller per model day"},
    "deltaL": {"label": "Long-redial return rate", "unit": "per long-orbit caller per model day"},
    "gamma": {"label": "Natural enrolled departure rate", "unit": "per B individual per model day"},
    "T": {"label": "Fixed simulation horizon", "unit": "model days"},
    "warmup": {"label": "Warmup/cohort start", "unit": "model days"},
    "q0": {"label": "Initial system callers", "unit": "callers"},
    "b0": {"label": "Initial enrolled pool B", "unit": "individuals"},
    "rs0": {"label": "Initial short orbit", "unit": "callers"},
    "rl0": {"label": "Initial long orbit", "unit": "callers"},
    "arrival_process": {"label": "Fresh-arrival process", "unit": ""},
    "lambda0": {"label": "Baseline sinusoidal arrival rate", "unit": "callers/model day"},
    "arrival_amplitude": {"label": "Sinusoidal relative arrival amplitude", "unit": "fraction"},
    "arrival_period": {"label": "Sinusoidal arrival period", "unit": "model days"},
    "arrival_phase": {"label": "Sinusoidal arrival phase", "unit": "radians"},
}

SWEEP_PARAMETERS = {
    "lam": "Arrival rate (lam)",
    "c": "Staffing level (c)",
    "aht_minutes": "Average handling time",
    "mu_plus": "Successful service rate (mu_plus)",
    "mu_minus": "Failed service rate (mu_minus)",
    "thetaA": "Final abandonment rate (thetaA)",
    "thetaS": "Short abandonment rate (thetaS)",
    "thetaL": "Long abandonment rate (thetaL)",
    "abandonment_scale": "All abandonment rates scale",
    "enroll_probability": "Service completion probability",
}

EXPERIMENT_PRESETS: dict[str, dict[str, Any]] = {
    "Abandonment absolute scale": {
        "parameter": "abandonment_scale",
        "values": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    },
    "Short-abandonment sensitivity": {"parameter": "thetaS", "scale_bounds": (0.25, 2.0)},
    "Long-abandonment sensitivity": {"parameter": "thetaL", "scale_bounds": (0.25, 2.0)},
    "Final-abandonment sensitivity": {"parameter": "thetaA", "scale_bounds": (0.25, 2.0)},
    "Service-completion sensitivity": {"parameter": "enroll_probability", "bounds": (0.1, 0.9)},
    "Staffing sensitivity": {"parameter": "c", "scale_bounds": (0.5, 2.0)},
    "Service-time sensitivity": {"parameter": "aht_minutes", "scale_bounds": (0.5, 2.0)},
    "Arrival-rate sensitivity": {"parameter": "lam", "scale_bounds": (0.5, 2.0)},
}


def load_parameter_table(parameter_file: str | Path = PARAMETER_FILE) -> pd.DataFrame:
    """Load the repository's call-center parameter CSV."""

    return pd.read_csv(parameter_file)


def call_center_row(label_or_id: str | int, parameter_file: str | Path = PARAMETER_FILE) -> dict[str, Any]:
    """Return one call-center parameter row as a dictionary."""

    call_center = CALL_CENTER_LABELS.get(str(label_or_id), label_or_id)
    return load_call_center_parameters(parameter_file, int(call_center))


def baseline_params(
    label_or_id: str | int,
    *,
    seed: int,
    horizon: float,
    warmup: float,
    parameter_file: str | Path = PARAMETER_FILE,
) -> SimulationParams:
    """Build baseline SimulationParams from the repository CSV."""

    row = call_center_row(label_or_id, parameter_file)
    return params_from_row(row, seed=seed, horizon=horizon, warmup=warmup)


def policy_scenarios_for_call_center(
    label_or_id: str | int,
    *,
    seed: int,
    horizon: float,
    warmup: float,
    parameter_file: str | Path = PARAMETER_FILE,
) -> dict[str, SimulationParams]:
    """Return the existing baseline/capacity policy scenarios."""

    row = call_center_row(label_or_id, parameter_file)
    return build_policy_scenarios(row, seed=seed, horizon=horizon, warmup=warmup)


def params_to_display_frame(params: SimulationParams) -> pd.DataFrame:
    """Return a human-readable parameter table."""

    rows = []
    inactive_constant_arrival_fields = {
        "lambda0",
        "arrival_amplitude",
        "arrival_period",
        "arrival_phase",
    }
    for name, value in asdict(params).items():
        if (
            params.arrival_process == "constant"
            and name in inactive_constant_arrival_fields
        ):
            continue
        meta = PARAMETER_METADATA.get(name, {"label": name, "unit": ""})
        rows.append(
            {
                "parameter": name,
                "meaning": meta["label"],
                "value": value,
                "unit": meta["unit"],
            }
        )
    return pd.DataFrame(rows)


def average_handling_time_minutes(params: SimulationParams) -> float:
    """Return AHT implied by service rates."""

    mu_total = params.mu_plus + params.mu_minus
    return float(MINUTES_PER_MODEL_DAY / mu_total) if mu_total > 0 else float("nan")


def enrollment_probability(params: SimulationParams) -> float:
    """Return service-success probability implied by service rates."""

    mu_total = params.mu_plus + params.mu_minus
    return float(params.mu_plus / mu_total) if mu_total > 0 else float("nan")


def apply_parameter_overrides(
    params: SimulationParams,
    overrides: dict[str, Any],
) -> SimulationParams:
    """Apply dashboard parameter overrides without mutating the base params."""

    overrides = dict(overrides)
    values = asdict(params)
    aht_minutes = overrides.pop("aht_minutes", None)
    enroll_prob_override = overrides.pop("enroll_probability", None)
    values.update({name: value for name, value in overrides.items() if value is not None})
    if aht_minutes is not None or enroll_prob_override is not None:
        current = SimulationParams(**values)
        chosen_aht = average_handling_time_minutes(current) if aht_minutes is None else float(aht_minutes)
        chosen_prob = (
            enrollment_probability(current)
            if enroll_prob_override is None
            else float(enroll_prob_override)
        )
        values["mu_plus"], values["mu_minus"] = service_rates(chosen_aht, chosen_prob)
    return SimulationParams(**values)


def apply_abandonment_transform(
    params: SimulationParams,
    *,
    abandonment_scale: float = 1.0,
    ratio_parameter: str | None = None,
    ratio_value: float | None = None,
) -> SimulationParams:
    """Apply abandonment scale and optional one-rate adjustment."""

    transformed = replace(
        params,
        thetaA=params.thetaA * float(abandonment_scale),
        thetaS=params.thetaS * float(abandonment_scale),
        thetaL=params.thetaL * float(abandonment_scale),
    )
    if ratio_parameter and ratio_value is not None:
        if ratio_parameter not in {"thetaA", "thetaS", "thetaL"}:
            raise ValueError("ratio_parameter must be thetaA, thetaS, thetaL, or None")
        transformed = replace(transformed, **{ratio_parameter: float(ratio_value)})
    return transformed


def resolve_staffing_for_replication(
    params: SimulationParams,
    *,
    mode: str,
    probability_33: float,
    seed: int,
) -> SimulationParams:
    """Resolve Call Center 1 fractional staffing for a single replication."""

    if mode == "fixed_32":
        return replace(params, c=32)
    if mode == "fixed_33":
        return replace(params, c=33)
    if mode == "stochastic_32_33":
        if probability_33 < 0 or probability_33 > 1:
            raise ValueError("probability_33 must lie in [0, 1]")
        agents = 33 if np.random.default_rng(seed).random() < probability_33 else 32
        return replace(params, c=agents)
    return params


def set_parameter_for_sweep(
    params: SimulationParams,
    parameter: str,
    value: float,
) -> SimulationParams:
    """Return params with one sweep parameter changed."""

    if parameter == "abandonment_scale":
        return apply_abandonment_transform(params, abandonment_scale=float(value))
    if parameter == "aht_minutes":
        return apply_parameter_overrides(params, {"aht_minutes": float(value)})
    if parameter == "enroll_probability":
        return apply_parameter_overrides(params, {"enroll_probability": float(value)})
    if parameter == "c":
        return replace(params, c=int(round(float(value))))
    if parameter in asdict(params):
        return replace(params, **{parameter: float(value)})
    raise ValueError(f"unsupported sweep parameter: {parameter}")


def validate_params_for_dashboard(params: SimulationParams) -> list[str]:
    """Return validation errors suitable for display."""

    errors: list[str] = []
    for name in ("lam", "mu_plus", "mu_minus", "thetaA", "thetaS", "thetaL", "deltaB", "deltaS", "deltaL", "gamma"):
        value = float(getattr(params, name))
        if not np.isfinite(value) or value < 0:
            errors.append(f"{name} must be finite and nonnegative")
    if params.arrival_process not in {"constant", "sinusoidal"}:
        errors.append("arrival_process must be constant or sinusoidal")
    if params.lambda0 is not None:
        if not np.isfinite(float(params.lambda0)) or float(params.lambda0) < 0:
            errors.append("lambda0 must be finite and nonnegative")
    if params.arrival_amplitude < 0 or params.arrival_amplitude > 1:
        errors.append("arrival_amplitude must lie in [0, 1]")
    if params.arrival_period <= 0:
        errors.append("arrival_period must be positive")
    if params.c < 0:
        errors.append("c must be nonnegative")
    if params.T <= 0:
        errors.append("T must be positive")
    if params.warmup < 0 or params.warmup >= params.T:
        errors.append("warmup must satisfy 0 <= warmup < T")
    prob = enrollment_probability(params)
    if not np.isnan(prob) and (prob < 0 or prob > 1):
        errors.append("implied service completion probability must lie in [0, 1]")
    return errors
