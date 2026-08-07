"""Fluid-flow approximation for geometric attempt-count termination probability."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import pandas as pd

from fluid_steady_state import solve_fluid_steady_state
from stochastic_simulation import MINUTES_PER_MODEL_DAY, SimulationParams


ANALYSIS_CACHE_VERSION = "flow_based_p_v4"
_FLOAT_TOLERANCE = 1e-10
FLUID_STATE_METRIC_MAP = {
    "Q": ("mean_Q", "q_bar"),
    "B": ("mean_B", "b_bar"),
    "RS": ("mean_RS", "rS_bar"),
    "RL": ("mean_RL", "rL_bar"),
}
DASHBOARD_GROUPED_METRICS = {
    "Mean queue length": ("mean_Q", "q_bar"),
    "Mean short-orbit size": ("mean_RS", "rS_bar"),
    "Average waiting time (min)": (
        "average_wait_including_abandonments_minutes",
        "average_waiting_time_minutes",
    ),
}


@dataclass(frozen=True)
class FluidSteadyStateResult:
    """Structured fluid steady-state result for dashboard diagnostics."""

    q_bar: float
    waiting_quantity: float
    service_quantity: float
    overloaded: bool
    finite: bool
    status: str
    message: str

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return a flat mapping suitable for DataFrame rows."""

        return asdict(self)


@dataclass(frozen=True)
class FlowBasedGeometricResult:
    """Attempt-ending flow components for the system-level geometric p."""

    p_flow: float
    q_bar: float
    waiting_quantity: float
    service_quantity: float
    terminal_flow: float
    total_attempt_ending_flow: float
    final_abandonment_flow: float
    short_abandonment_flow: float
    long_abandonment_flow: float
    service_success_flow: float
    service_failure_flow: float
    overloaded: bool
    finite: bool
    status: str
    message: str

    def to_dict(self) -> dict[str, float | bool | str]:
        """Return a flat mapping suitable for DataFrame rows."""

        return asdict(self)


def _unavailable_fluid(status: str, message: str) -> FluidSteadyStateResult:
    return FluidSteadyStateResult(
        q_bar=float("nan"),
        waiting_quantity=float("nan"),
        service_quantity=float("nan"),
        overloaded=False,
        finite=False,
        status=status,
        message=message,
    )


def _unavailable_flow(
    status: str,
    message: str,
    *,
    q_bar: float = float("nan"),
    waiting_quantity: float = float("nan"),
    service_quantity: float = float("nan"),
    overloaded: bool = False,
) -> FlowBasedGeometricResult:
    return FlowBasedGeometricResult(
        p_flow=float("nan"),
        q_bar=q_bar,
        waiting_quantity=waiting_quantity,
        service_quantity=service_quantity,
        terminal_flow=float("nan"),
        total_attempt_ending_flow=float("nan"),
        final_abandonment_flow=float("nan"),
        short_abandonment_flow=float("nan"),
        long_abandonment_flow=float("nan"),
        service_success_flow=float("nan"),
        service_failure_flow=float("nan"),
        overloaded=overloaded,
        finite=False,
        status=status,
        message=message,
    )


def compute_fluid_steady_state_q(params: SimulationParams) -> FluidSteadyStateResult:
    """Compute the paper steady-state q_bar used by the flow p approximation.

    The formula is valid for constant-arrival settings with positive
    denominators. It is intentionally returned as a structured result instead
    of raising for expected dashboard edge cases such as thetaA == 0.
    """

    if params.arrival_process != "constant":
        return _unavailable_fluid(
            "undefined_nonconstant_arrival",
            "Fluid steady-state q_bar is only defined for constant arrivals.",
        )
    if params.gamma + params.deltaB == 0:
        return _unavailable_fluid(
            "undefined_gamma_plus_deltaB_zero",
            "gamma + deltaB must be nonzero for the fluid steady-state formula.",
        )
    if params.mu_plus <= 0:
        return _unavailable_fluid(
            "undefined_mu_plus_nonpositive",
            "mu_plus must be positive for the fluid steady-state formula.",
        )
    if params.gamma <= 0:
        return _unavailable_fluid(
            "undefined_gamma_nonpositive",
            "gamma must be positive for the fluid steady-state formula.",
        )
    threshold = params.gamma * params.c * params.mu_plus / (
        params.gamma + params.deltaB
    )
    excess_load = params.lam - threshold
    overloaded = bool(excess_load > _FLOAT_TOLERANCE)
    overload_gap = max(excess_load, 0.0)
    base_quantity = min(
        float(params.c),
        (1.0 + params.deltaB / params.gamma) * params.lam / params.mu_plus,
    )

    if params.thetaA == 0 and overloaded:
        return _unavailable_fluid(
            "undefined_no_finite_fluid_steady_state",
            "thetaA is zero with positive excess load, so finite q_bar is unavailable.",
        )
    if params.thetaA < 0:
        return _unavailable_fluid(
            "undefined_thetaA_negative",
            "thetaA must be nonnegative for the fluid steady-state formula.",
        )

    q_bar = base_quantity
    if overload_gap > 0:
        q_bar += overload_gap / params.thetaA
    waiting_quantity = max(q_bar - float(params.c), 0.0)
    service_quantity = min(q_bar, float(params.c))
    return FluidSteadyStateResult(
        q_bar=float(q_bar),
        waiting_quantity=float(waiting_quantity),
        service_quantity=float(service_quantity),
        overloaded=overloaded,
        finite=True,
        status="ok",
        message="Fluid steady-state q_bar computed from the paper formula.",
    )


def build_fluid_state_comparison(
    params: SimulationParams, replication_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Compare stochastic aggregate means with fluid steady-state bars.

    This helper intentionally returns only the four aggregate state variables
    used for the dashboard sanity check. It is cheap and deterministic, so the
    Streamlit app computes it from the current parameters instead of storing it
    in the simulation cache.
    """

    if replication_metrics.empty:
        return pd.DataFrame(), "No replication metrics are available."
    missing = [
        stochastic_metric
        for stochastic_metric, _fluid_metric in FLUID_STATE_METRIC_MAP.values()
        if stochastic_metric not in replication_metrics.columns
    ]
    if "average_wait_including_abandonments_minutes" not in replication_metrics.columns:
        missing.append("average_wait_including_abandonments_minutes")
    if missing:
        return (
            pd.DataFrame(),
            f"Replication metrics are missing columns: {', '.join(missing)}.",
        )
    try:
        fluid_result = solve_fluid_steady_state(params)
    except ValueError as exc:
        return pd.DataFrame(), str(exc)
    fluid = fluid_result.to_dict()
    fluid["average_waiting_time_minutes"] = fluid_average_waiting_time_minutes(
        params,
        fluid,
    )

    rows = []
    for metric, (stochastic_metric, fluid_metric) in {
        **FLUID_STATE_METRIC_MAP,
        "Average wait (min)": (
            "average_wait_including_abandonments_minutes",
            "average_waiting_time_minutes",
        ),
    }.items():
        stochastic_mean = float(replication_metrics[stochastic_metric].mean())
        stochastic_std = float(replication_metrics[stochastic_metric].std(ddof=1))
        fluid_value = float(fluid[fluid_metric])
        difference = stochastic_mean - fluid_value
        percent_difference = (
            float("nan")
            if fluid_value == 0.0
            else 100.0 * difference / fluid_value
        )
        rows.append(
            {
                "metric": metric,
                "stochastic_mean": stochastic_mean,
                "stochastic_std": stochastic_std,
                "fluid_value": fluid_value,
                "difference": difference,
                "percent_difference": percent_difference,
            }
        )
    return pd.DataFrame(rows), "ok"


def fluid_average_waiting_time_minutes(
    params: SimulationParams,
    fluid_values: dict[str, float],
) -> float:
    """Return fluid average wait, including abandonments, in minutes.

    The approximation is waiting_bar / lambda_hat, where waiting_bar is
    max(q_bar - c, 0) and lambda_hat includes fresh, recertification, short
    redial, and long redial arrivals.
    """

    waiting_bar = max(float(fluid_values["q_bar"]) - float(params.c), 0.0)
    lambda_hat = (
        params.lam
        + params.deltaB * float(fluid_values["b_bar"])
        + params.deltaS * float(fluid_values["rS_bar"])
        + params.deltaL * float(fluid_values["rL_bar"])
    )
    if lambda_hat <= 0:
        return float("nan")
    return waiting_bar / lambda_hat * MINUTES_PER_MODEL_DAY


def build_call_center_grouped_comparison(
    params_by_call_center: dict[str, SimulationParams],
    replication_metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """Return fluid-vs-simulation rows for the dashboard grouped chart."""

    if replication_metrics.empty:
        return pd.DataFrame(), "No replication metrics are available."
    rows = []
    for call_center, params in params_by_call_center.items():
        scenario_frame = replication_metrics[
            replication_metrics["scenario_name"] == call_center
        ]
        if scenario_frame.empty:
            continue
        try:
            fluid = solve_fluid_steady_state(params).to_dict()
        except ValueError as exc:
            return pd.DataFrame(), f"{call_center}: {exc}"
        fluid["average_waiting_time_minutes"] = fluid_average_waiting_time_minutes(
            params,
            fluid,
        )
        for metric, (stochastic_metric, fluid_metric) in DASHBOARD_GROUPED_METRICS.items():
            if stochastic_metric not in scenario_frame.columns:
                return (
                    pd.DataFrame(),
                    f"{call_center} metrics are missing column: {stochastic_metric}.",
                )
            rows.append(
                {
                    "call_center": call_center,
                    "metric": metric,
                    "estimate_source": "Simulation mean",
                    "value": float(scenario_frame[stochastic_metric].mean()),
                }
            )
            rows.append(
                {
                    "call_center": call_center,
                    "metric": metric,
                    "estimate_source": "Fluid estimate",
                    "value": float(fluid[fluid_metric]),
                }
            )
    if not rows:
        return pd.DataFrame(), "No matching call-center scenario metrics are available."
    return pd.DataFrame(rows), "ok"


def compute_flow_based_geometric_p(
    params: SimulationParams,
    q_bar: float | None = None,
) -> FlowBasedGeometricResult:
    """Compute system-level fluid-flow approximation to geometric p.

    p_flow is terminal attempt-ending flow divided by all attempt-ending flow.
    It is not fitted from caller attempt records and is intentionally shared
    across completed, abandoned, and overall population panels.
    """

    fluid = compute_fluid_steady_state_q(params)
    if q_bar is not None and math.isfinite(float(q_bar)):
        q_value = float(q_bar)
        fluid = FluidSteadyStateResult(
            q_bar=q_value,
            waiting_quantity=max(q_value - float(params.c), 0.0),
            service_quantity=min(q_value, float(params.c)),
            overloaded=bool(q_value > params.c + _FLOAT_TOLERANCE),
            finite=True,
            status="ok_supplied_q_bar",
            message="Using caller-supplied q_bar for the flow approximation.",
        )
    elif not fluid.finite:
        return _unavailable_flow(
            fluid.status,
            fluid.message,
            q_bar=fluid.q_bar,
            waiting_quantity=fluid.waiting_quantity,
            service_quantity=fluid.service_quantity,
            overloaded=fluid.overloaded,
        )

    theta_total = params.thetaA + params.thetaS + params.thetaL
    mu_total = params.mu_plus + params.mu_minus
    waiting = fluid.waiting_quantity
    service = fluid.service_quantity

    final_abandonment_flow = params.thetaA * waiting
    short_abandonment_flow = params.thetaS * waiting
    long_abandonment_flow = params.thetaL * waiting
    service_success_flow = params.mu_plus * service
    service_failure_flow = params.mu_minus * service
    terminal_flow = final_abandonment_flow + service_success_flow
    total_attempt_ending_flow = theta_total * waiting + mu_total * service

    if total_attempt_ending_flow == 0:
        return _unavailable_flow(
            "undefined_zero_attempt_ending_flow",
            "Total attempt-ending flow is zero, so p_flow is unavailable.",
            q_bar=fluid.q_bar,
            waiting_quantity=waiting,
            service_quantity=service,
            overloaded=fluid.overloaded,
        )

    p_flow = terminal_flow / total_attempt_ending_flow
    if p_flow < -_FLOAT_TOLERANCE or p_flow > 1.0 + _FLOAT_TOLERANCE:
        return _unavailable_flow(
            "undefined_invalid_probability",
            "Computed p_flow is outside [0, 1]. Check model rates.",
            q_bar=fluid.q_bar,
            waiting_quantity=waiting,
            service_quantity=service,
            overloaded=fluid.overloaded,
        )
    p_flow = min(max(p_flow, 0.0), 1.0)

    return FlowBasedGeometricResult(
        p_flow=float(p_flow),
        q_bar=fluid.q_bar,
        waiting_quantity=waiting,
        service_quantity=service,
        terminal_flow=float(terminal_flow),
        total_attempt_ending_flow=float(total_attempt_ending_flow),
        final_abandonment_flow=float(final_abandonment_flow),
        short_abandonment_flow=float(short_abandonment_flow),
        long_abandonment_flow=float(long_abandonment_flow),
        service_success_flow=float(service_success_flow),
        service_failure_flow=float(service_failure_flow),
        overloaded=fluid.overloaded,
        finite=True,
        status=fluid.status,
        message=fluid.message,
    )
