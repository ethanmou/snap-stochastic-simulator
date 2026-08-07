"""Streamlit sidebar controls for dashboard parameters."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st

from stochastic_simulation import SimulationParams

from .scenario_registry import (
    CALL_CENTER_LABELS,
    EXPERIMENT_PRESETS,
    PARAMETER_METADATA,
    SWEEP_PARAMETERS,
    apply_abandonment_transform,
    apply_parameter_overrides,
    average_handling_time_minutes,
    baseline_params,
    enrollment_probability,
    params_to_display_frame,
    validate_params_for_dashboard,
)
from .simulation_runner import ScenarioRunOptions


def _number(
    label: str,
    value: float,
    *,
    min_value: float | None = None,
    step: float = 1.0,
    key: str | None = None,
) -> float:
    return float(
        st.sidebar.number_input(
            label,
            value=float(value),
            min_value=min_value,
            step=step,
            format="%.6g",
            key=key,
        )
    )


def _preset_widget_key(preset: str) -> str:
    """Return a stable prefix for widgets whose defaults depend on a preset."""

    return str(preset).lower().replace(" ", "_")


def _reset_preset_parameter_widgets_on_change(preset: str) -> None:
    """Clear preset-bound widget state when the active preset changes."""

    active_key = "_active_call_center_preset"
    preset_key = _preset_widget_key(preset)
    if st.session_state.get(active_key) == preset:
        return
    for key in list(st.session_state):
        if str(key).startswith(f"{preset_key}:"):
            del st.session_state[key]
    st.session_state[active_key] = preset


def sidebar_configuration() -> tuple[SimulationParams, ScenarioRunOptions, dict[str, Any]]:
    """Render sidebar controls and return params, run options, and UI choices."""

    st.sidebar.header("Scenario")
    preset = st.sidebar.selectbox(
        "Call center preset",
        list(CALL_CENTER_LABELS) + ["Custom Parameters"],
        index=1,
        key="call_center_preset",
    )
    base_label = "Call Center 2" if preset == "Custom Parameters" else preset
    preset_key = _preset_widget_key(preset)
    _reset_preset_parameter_widgets_on_change(preset)

    st.sidebar.header("Run")
    dynamic_horizon = st.sidebar.toggle("Dynamic horizon", value=True)
    replications = int(st.sidebar.number_input("Replications", value=3, min_value=1, max_value=200, step=1))
    base_seed = int(st.sidebar.number_input("Random seed", value=20260720, min_value=0, step=1))
    cohort_start = _number("Cohort start (model days)", 260.0, min_value=0.0, step=10.0)
    cohort_end = _number("Cohort end (model days)", 420.0, min_value=0.0, step=10.0)
    max_dynamic_horizon = _number("Max dynamic horizon (model days)", 900.0, min_value=0.01, step=10.0)
    fixed_horizon = _number("Fixed horizon T (model days)", max_dynamic_horizon, min_value=0.01, step=10.0)
    post_clearance_buffer = _number("Post-clearance buffer (model days)", 20.0, min_value=0.0, step=1.0)
    max_events = int(st.sidebar.number_input("Max events", value=4_000_000, min_value=1, step=100_000))
    validate = st.sidebar.toggle("Validation mode", value=False)
    queue_distribution_enabled = st.sidebar.toggle("Record queue distribution", value=True)
    queue_warmup_time = cohort_start
    queue_sample_interval = 1.0
    queue_observation_end = None
    if queue_distribution_enabled:
        queue_warmup_time = _number(
            "Queue sampling warmup (model days)",
            cohort_start,
            min_value=0.0,
            step=10.0,
        )
        queue_sample_interval = _number(
            "Queue sample interval (model days)",
            1.0,
            min_value=0.000001,
            step=0.5,
        )
        queue_observation_end_value = _number(
            "Queue observation end (0 = simulation end)",
            0.0,
            min_value=0.0,
            step=10.0,
        )
        queue_observation_end = (
            None if queue_observation_end_value == 0.0 else queue_observation_end_value
        )

    params = baseline_params(
        base_label,
        seed=base_seed,
        horizon=max_dynamic_horizon if dynamic_horizon else fixed_horizon,
        warmup=cohort_start,
    )

    st.sidebar.header("Operational Parameters")
    lam = _number(
        "Arrival rate (lam, callers/model day)",
        params.lam,
        min_value=0.0,
        step=max(params.lam / 20, 0.1),
        key=f"{preset_key}:lam",
    )
    arrival_process = st.sidebar.selectbox(
        "Arrival process",
        ["constant", "sinusoidal"],
        index=0 if params.arrival_process == "constant" else 1,
        key=f"{preset_key}:arrival_process",
    )
    if arrival_process == "sinusoidal":
        lambda0 = _number(
            "Baseline sinusoidal arrival rate (lambda0, callers/model day)",
            params.lam if params.lambda0 is None else params.lambda0,
            min_value=0.0,
            step=max(params.lam / 20, 0.1),
            key=f"{preset_key}:lambda0",
        )
        arrival_amplitude = _number(
            "Arrival amplitude (fraction, 0-1)",
            params.arrival_amplitude,
            min_value=0.0,
            step=0.05,
            key=f"{preset_key}:arrival_amplitude",
        )
        arrival_period = _number(
            "Arrival period (model days)",
            params.arrival_period,
            min_value=0.000001,
            step=0.25,
            key=f"{preset_key}:arrival_period",
        )
        arrival_phase = _number(
            "Arrival phase (radians)",
            params.arrival_phase,
            step=0.25,
            key=f"{preset_key}:arrival_phase",
        )
    else:
        lambda0 = None
        arrival_amplitude = 0.0
        arrival_period = 1.0
        arrival_phase = 0.0
    staffing_value = _number(
        "Staffing level (c, agents)",
        params.c,
        min_value=0.0,
        step=1.0,
        key=f"{preset_key}:c",
    )
    aht = _number(
        "Average handling time (AHT, minutes)",
        average_handling_time_minutes(params),
        min_value=0.01,
        step=1.0,
        key=f"{preset_key}:aht_minutes",
    )
    enroll = _number(
        "Service completion probability (mu_plus/(mu_plus+mu_minus))",
        enrollment_probability(params),
        min_value=0.0,
        step=0.05,
        key=f"{preset_key}:enroll_probability",
    )
    q0 = int(
        st.sidebar.number_input(
            "Initial system callers (q0)",
            value=params.q0,
            min_value=0,
            step=1,
            key=f"{preset_key}:q0",
        )
    )
    b0 = int(
        st.sidebar.number_input(
            "Initial enrolled pool (b0)",
            value=params.b0,
            min_value=0,
            step=1,
            key=f"{preset_key}:b0",
        )
    )
    rs0 = int(
        st.sidebar.number_input(
            "Initial short orbit (rs0)",
            value=params.rs0,
            min_value=0,
            step=1,
            key=f"{preset_key}:rs0",
        )
    )
    rl0 = int(
        st.sidebar.number_input(
            "Initial long orbit (rl0)",
            value=params.rl0,
            min_value=0,
            step=1,
            key=f"{preset_key}:rl0",
        )
    )

    st.sidebar.header("Behavioral Parameters")
    thetaA = _number(
        "Final abandonment/loss rate (thetaA, per model day)",
        params.thetaA,
        min_value=0.0,
        step=max(params.thetaA / 20, 0.01),
        key=f"{preset_key}:thetaA",
    )
    thetaS = _number(
        "Short-orbit abandonment rate (thetaS, per model day)",
        params.thetaS,
        min_value=0.0,
        step=max(params.thetaS / 20, 0.01),
        key=f"{preset_key}:thetaS",
    )
    thetaL = _number(
        "Long-orbit abandonment rate (thetaL, per model day)",
        params.thetaL,
        min_value=0.0,
        step=max(params.thetaL / 20, 0.01),
        key=f"{preset_key}:thetaL",
    )
    deltaB = _number(
        "Recertification call rate (deltaB, per model day)",
        params.deltaB,
        min_value=0.0,
        step=0.001,
        key=f"{preset_key}:deltaB",
    )
    deltaS = _number(
        "Short-redial return rate (deltaS, per model day)",
        params.deltaS,
        min_value=0.0,
        step=0.1,
        key=f"{preset_key}:deltaS",
    )
    deltaL = _number(
        "Long-redial return rate (deltaL, per model day)",
        params.deltaL,
        min_value=0.0,
        step=0.01,
        key=f"{preset_key}:deltaL",
    )
    gamma = _number(
        "Natural enrolled departure rate (gamma, per model day)",
        params.gamma,
        min_value=0.0,
        step=0.001,
        key=f"{preset_key}:gamma",
    )

    st.sidebar.header("Abandonment Transforms")
    abandonment_scale = _number(
        "Abandonment scale",
        1.0,
        min_value=0.0,
        step=0.1,
        key=f"{preset_key}:abandonment_scale",
    )
    ratio_parameter = st.sidebar.selectbox(
        "Adjust one abandonment rate",
        ["None", "thetaA", "thetaS", "thetaL"],
        key=f"{preset_key}:ratio_parameter",
    )
    ratio_value = None
    if ratio_parameter != "None":
        baseline_value = {"thetaA": thetaA, "thetaS": thetaS, "thetaL": thetaL}[ratio_parameter]
        ratio_value = _number(
            f"Override {ratio_parameter}",
            baseline_value,
            min_value=0.0,
            step=max(baseline_value / 20, 0.01),
            key=f"{preset_key}:ratio_value:{ratio_parameter}",
        )

    staffing_mode = "default"
    probability_33 = 0.5
    if base_label == "Call Center 1":
        st.sidebar.header("Call Center 1 Staffing")
        staffing_mode = st.sidebar.radio(
            "32.5-agent handling",
            ["fixed_32", "fixed_33", "stochastic_32_33"],
            index=2,
            key=f"{preset_key}:staffing_mode",
        )
        if staffing_mode == "stochastic_32_33":
            probability_33 = _number(
                "Probability of 33 agents",
                0.5,
                min_value=0.0,
                step=0.05,
                key=f"{preset_key}:probability_33",
            )

    params = apply_parameter_overrides(
        params,
        {
            "lam": lam,
            "arrival_process": arrival_process,
            "lambda0": lambda0,
            "arrival_amplitude": arrival_amplitude,
            "arrival_period": arrival_period,
            "arrival_phase": arrival_phase,
            "c": int(round(staffing_value)),
            "aht_minutes": aht,
            "enroll_probability": enroll,
            "q0": q0,
            "b0": b0,
            "rs0": rs0,
            "rl0": rl0,
            "thetaA": thetaA,
            "thetaS": thetaS,
            "thetaL": thetaL,
            "deltaB": deltaB,
            "deltaS": deltaS,
            "deltaL": deltaL,
            "gamma": gamma,
            "T": max_dynamic_horizon if dynamic_horizon else fixed_horizon,
            "warmup": cohort_start,
        },
    )
    params = apply_abandonment_transform(
        params,
        abandonment_scale=abandonment_scale,
        ratio_parameter=None if ratio_parameter == "None" else ratio_parameter,
        ratio_value=ratio_value,
    )

    options = ScenarioRunOptions(
        replications=replications,
        base_seed=base_seed,
        dynamic_horizon=dynamic_horizon,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        post_clearance_buffer=post_clearance_buffer,
        max_dynamic_horizon=max_dynamic_horizon,
        max_events=max_events,
        validate=validate,
        staffing_mode=staffing_mode,
        probability_33=probability_33,
        queue_distribution_enabled=queue_distribution_enabled,
        queue_warmup_time=queue_warmup_time,
        queue_sample_interval=queue_sample_interval,
        queue_observation_end=queue_observation_end,
    )
    choices = {
        "preset": preset,
        "base_label": base_label,
        "params_table": params_to_display_frame(params),
        "validation_errors": validate_params_for_dashboard(params),
    }
    return params, options, choices


def sweep_controls(params: SimulationParams) -> dict[str, Any]:
    """Render parameter sweep controls."""

    preset = st.selectbox("Experiment preset", ["Custom sweep"] + list(EXPERIMENT_PRESETS))
    if preset == "Custom sweep":
        parameter = st.selectbox(
            "Sweep parameter",
            list(SWEEP_PARAMETERS),
            format_func=lambda key: SWEEP_PARAMETERS[key],
        )
    else:
        parameter = str(EXPERIMENT_PRESETS[preset]["parameter"])
        parameter_options = list(SWEEP_PARAMETERS)
        st.selectbox(
            "Sweep parameter",
            parameter_options,
            index=parameter_options.index(parameter),
            format_func=lambda key: SWEEP_PARAMETERS[key],
            disabled=True,
            key=f"sweep_parameter_for_{preset}",
        )
        st.caption("Preset loaded the target parameter and default grid; edit the range below before running.")
    base_value = asdict(params).get(parameter, 1.0)
    if parameter == "aht_minutes":
        base_value = average_handling_time_minutes(params)
    if parameter == "enroll_probability":
        base_value = enrollment_probability(params)
    if parameter == "abandonment_scale":
        base_value = 1.0
    default_minimum = float(max(float(base_value) * 0.5, 0.000001))
    default_maximum = float(max(float(base_value) * 1.5, default_minimum))
    default_grid_points = 5
    if preset != "Custom sweep":
        preset_config = EXPERIMENT_PRESETS[preset]
        if "values" in preset_config:
            preset_values = [float(value) for value in preset_config["values"]]
            default_minimum = min(preset_values)
            default_maximum = max(preset_values)
            default_grid_points = len(preset_values)
        elif "bounds" in preset_config:
            default_minimum, default_maximum = (float(value) for value in preset_config["bounds"])
        elif "scale_bounds" in preset_config:
            low, high = preset_config["scale_bounds"]
            default_minimum = float(base_value) * float(low)
            default_maximum = float(base_value) * float(high)
    key_prefix = f"{preset}:{parameter}"
    minimum = st.number_input(
        "Minimum",
        value=default_minimum,
        min_value=0.0,
        format="%.6g",
        key=f"{key_prefix}:minimum",
    )
    maximum = st.number_input(
        "Maximum",
        value=float(max(default_maximum, minimum)),
        min_value=0.0,
        format="%.6g",
        key=f"{key_prefix}:maximum",
    )
    grid_points = int(
        st.number_input(
            "Grid points",
            value=default_grid_points,
            min_value=1,
            max_value=50,
            step=1,
            key=f"{key_prefix}:grid_points",
        )
    )
    scale = st.radio("Grid scale", ["linear", "log"], horizontal=True, key=f"{key_prefix}:scale")
    common_random_numbers = st.toggle("Common random numbers", value=True)
    populations = st.multiselect("Populations", ["all", "completed", "abandoned"], default=["all", "completed", "abandoned"])
    return {
        "preset": preset,
        "parameter": parameter,
        "minimum": float(minimum),
        "maximum": float(maximum),
        "grid_points": grid_points,
        "scale": scale,
        "common_random_numbers": common_random_numbers,
        "populations": populations,
    }
