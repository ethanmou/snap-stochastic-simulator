"""Fluid steady-state benchmark for the SNAP call-center model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from stochastic_simulation import (
    SimulationParams,
    _validate_params,
)


@dataclass(frozen=True)
class FluidSteadyStateResult:
    """Deterministic fluid steady-state values for the aggregate state."""

    q_bar: float
    b_bar: float
    rS_bar: float
    rL_bar: float

    def to_dict(self) -> dict[str, float]:
        """Return the benchmark as a flat dictionary."""

        return asdict(self)


def _validate_fluid_requirements(params: SimulationParams) -> None:
    """Validate denominator assumptions required by the closed-form formulas."""

    required_positive = {
        "thetaA": params.thetaA,
        "gamma": params.gamma,
        "deltaS": params.deltaS,
        "deltaL": params.deltaL,
        "mu_plus": params.mu_plus,
    }
    for name, value in required_positive.items():
        if value <= 0:
            raise ValueError(
                f"{name} must be positive for the fluid steady-state formulas"
            )


def solve_fluid_steady_state(params: SimulationParams) -> FluidSteadyStateResult:
    """Compute the Section 4.1 fluid steady-state benchmark.

    The formulas here intentionally follow the task specification exactly. The
    result is a deterministic first-order benchmark, not an exact stochastic
    mean for a finite-horizon simulation.
    """

    _validate_params(params)
    _validate_fluid_requirements(params)

    threshold = params.gamma * params.c * params.mu_plus / (
        params.gamma + params.deltaB
    )
    overload_gap = max(params.lam - threshold, 0.0)

    q_bar = (
        min(
            float(params.c),
            (1.0 + params.deltaB / params.gamma) * params.lam / params.mu_plus,
        )
        + overload_gap / params.thetaA
    )
    b_bar = min(
        params.lam / params.gamma,
        params.c * params.mu_plus / (params.gamma + params.deltaB),
    )
    rS_bar = params.thetaS / (params.deltaS * params.thetaA) * overload_gap
    rL_bar = min(
        params.c * params.mu_minus / params.deltaL,
        (1.0 + params.deltaB / params.gamma)
        * params.lam
        * params.mu_minus
        / (params.deltaL * params.mu_plus),
    ) + params.thetaL / (params.deltaL * params.thetaA) * overload_gap

    return FluidSteadyStateResult(
        q_bar=q_bar,
        b_bar=b_bar,
        rS_bar=rS_bar,
        rL_bar=rL_bar,
    )


def fluid_steady_state(params: SimulationParams) -> FluidSteadyStateResult:
    """Alias for solve_fluid_steady_state with the paper-facing name."""

    return solve_fluid_steady_state(params)


def fluid_ode_rhs(t: float, y, params: SimulationParams) -> np.ndarray:
    """Return the Section 4.1 fluid ODE right-hand side for y = [q, b, rS, rL]."""

    del t
    _validate_params(params)
    q, b, rS, rL = np.asarray(y, dtype=float)
    mu = params.mu_plus + params.mu_minus
    theta = params.thetaA + params.thetaS + params.thetaL
    waiting = max(q - params.c, 0.0)
    busy = min(q, float(params.c))

    return np.array(
        [
            params.lam
            + params.deltaB * b
            + params.deltaS * rS
            + params.deltaL * rL
            - theta * waiting
            - mu * busy,
            params.mu_plus * busy - (params.deltaB + params.gamma) * b,
            params.thetaS * waiting - params.deltaS * rS,
            params.mu_minus * busy + params.thetaL * waiting - params.deltaL * rL,
        ],
        dtype=float,
    )


def solve_fluid_ode(
    params: SimulationParams,
    horizon: float,
    initial_state,
    t_eval,
    max_step: float | None = None,
) -> "pd.DataFrame":
    """Solve the fluid ODE on requested evaluation times using explicit Euler steps.

    ``max_step`` controls the internal integration step size, not the output
    frequency; output rows are still determined by ``t_eval``. This avoids a
    scipy dependency and is intended for deterministic validation trajectories
    and sample-path comparison plots. The returned DataFrame has columns
    ``t, q, b, rS, rL``.
    """

    _validate_params(params)
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_step is None:
        max_step = min(horizon / 5000.0, 0.01)
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError("max_step must be positive")
    times = np.asarray(t_eval, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("t_eval must be a nonempty one-dimensional sequence")
    if times[0] < 0 or times[-1] > horizon:
        raise ValueError("t_eval must lie within [0, horizon]")
    if np.any(np.diff(times) < 0):
        raise ValueError("t_eval must be sorted")

    if isinstance(initial_state, dict):
        y = np.array(
            [
                initial_state["Q"],
                initial_state["B"],
                initial_state["RS"],
                initial_state["RL"],
            ],
            dtype=float,
        )
    else:
        y = np.asarray(initial_state, dtype=float)
    if y.shape != (4,):
        raise ValueError("initial_state must contain Q, B, RS, and RL")
    if (y < 0).any() or not np.isfinite(y).all():
        raise ValueError("initial_state must be finite and nonnegative")

    rows = []
    current_t = 0.0
    for target_t in times:
        while current_t < target_t:
            step = min(max_step, target_t - current_t)
            dy = fluid_ode_rhs(current_t, y, params)
            y = y + step * dy
            current_t += step
        rows.append({"t": target_t, "q": y[0], "b": y[1], "rS": y[2], "rL": y[3]})

    return pd.DataFrame(rows, columns=["t", "q", "b", "rS", "rL"])
