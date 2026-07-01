"""Barebones aggregate Gillespie simulator for the SNAP call-center model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from stochastic_simulation import (
    SimulationParams,
    _validate_params,
    sample_event,
)


class LightState(NamedTuple):
    """Aggregate Markov-chain state for the light simulator."""

    Q: int
    B: int
    RS: int
    RL: int


@dataclass(frozen=True)
class LightSimulationResult:
    """Core aggregate outputs from the simplified simulator."""

    mean_Q: float
    mean_B: float
    mean_RS: float
    mean_RL: float
    final_Q: int
    final_B: int
    final_RS: int
    final_RL: int

    def to_dict(self) -> dict[str, int | float]:
        """Return a flat dictionary suitable for DataFrame construction."""

        return asdict(self)


def _state_counts(Q: int, capacity: int) -> tuple[int, int]:
    """Return aggregate waiting and service counts under work conservation."""

    in_service = min(Q, capacity)
    waiting = max(Q - capacity, 0)
    return waiting, in_service


def _event_rates(params: SimulationParams, Q: int, B: int, RS: int, RL: int) -> np.ndarray:
    """Compute the same event rates used by the full simulator."""

    waiting, in_service = _state_counts(Q, params.c)
    return np.array(
        [
            params.lam,
            params.deltaB * B,
            params.deltaS * RS,
            params.deltaL * RL,
            params.thetaA * waiting,
            params.thetaS * waiting,
            params.thetaL * waiting,
            params.mu_plus * in_service,
            params.mu_minus * in_service,
            params.gamma * B,
        ],
        dtype=float,
    )


def _validate_light_state(
    *,
    t: float,
    event: int | None,
    rates: np.ndarray | None,
    total_rate: float | None,
    Q: int,
    B: int,
    RS: int,
    RL: int,
) -> None:
    """Validate aggregate state invariants for the light simulator."""

    state = {"Q": Q, "B": B, "RS": RS, "RL": RL}
    prefix = f"Invalid light simulation state at t={t}, event={event}"

    for name, value in state.items():
        if not math.isfinite(float(value)):
            raise ValueError(f"{prefix}: {name} is not finite; state={state}")
    if Q < 0 or B < 0 or RS < 0 or RL < 0:
        raise ValueError(f"{prefix}: negative state value; state={state}")
    if rates is not None:
        if not np.isfinite(rates).all():
            raise ValueError(f"{prefix}: non-finite event rate; rates={rates}")
        if (rates < 0).any():
            raise ValueError(f"{prefix}: negative event rate; rates={rates}")
    if total_rate is not None:
        if not math.isfinite(total_rate):
            raise ValueError(f"{prefix}: total_rate is not finite")
        if total_rate < 0.0:
            raise ValueError(f"{prefix}: total_rate cannot be negative")


def simulate_light(
    params: SimulationParams,
    horizon: float | None = None,
    initial_state: LightState | None = None,
    seed: int | None = None,
    record_path: bool = False,
    record_dt: float | None = None,
    validate: bool = False,
) -> LightSimulationResult | pd.DataFrame:
    """Run a simplified aggregate Gillespie simulation.

    This simulator intentionally avoids caller-level records. It tracks only
    aggregate state variables while preserving the same CTMC rates and
    transitions as the full simulator.
    """

    if initial_state is None:
        light_state = LightState(params.q0, params.b0, params.rs0, params.rl0)
    elif isinstance(initial_state, LightState):
        light_state = initial_state
    else:
        raise TypeError("initial_state must be a LightState or None")
    horizon = params.T if horizon is None else float(horizon)
    _validate_params(params)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and finite")
    rng = np.random.default_rng(params.seed if seed is None else seed)

    Q = light_state.Q
    B = light_state.B
    RS = light_state.RS
    RL = light_state.RL
    area_Q = 0.0
    area_B = 0.0
    area_RS = 0.0
    area_RL = 0.0
    t = 0.0
    path_rows: list[dict[str, float | int]] = []
    if record_path:
        if record_dt is None:
            record_dt = horizon / 500.0
        if record_dt <= 0 or not math.isfinite(record_dt):
            raise ValueError("record_dt must be positive and finite")
        next_record_t = 0.0

    while t < horizon:
        rates = _event_rates(params, Q, B, RS, RL)
        total_rate = float(rates.sum())
        if validate:
            _validate_light_state(
                t=t,
                event=None,
                rates=rates,
                total_rate=total_rate,
                Q=Q,
                B=B,
                RS=RS,
                RL=RL,
            )

        event_time = (
            horizon
            if total_rate == 0.0
            else t + rng.exponential(1.0 / total_rate)
        )
        interval_end = min(event_time, horizon)
        if record_path:
            while next_record_t <= interval_end:
                path_rows.append(
                    {"t": next_record_t, "Q": Q, "B": B, "RS": RS, "RL": RL}
                )
                next_record_t += float(record_dt)
        observed_start = max(t, params.warmup)
        observed_dt = max(0.0, interval_end - observed_start)
        if observed_dt > 0:
            area_Q += Q * observed_dt
            area_B += B * observed_dt
            area_RS += RS * observed_dt
            area_RL += RL * observed_dt

        if event_time > horizon or total_rate == 0.0:
            t = horizon
            break

        t = event_time
        event = sample_event(rates, total_rate, rng)

        if event == 0:
            Q += 1
        elif event == 1:
            B -= 1
            Q += 1
        elif event == 2:
            RS -= 1
            Q += 1
        elif event == 3:
            RL -= 1
            Q += 1
        elif event in (4, 5, 6):
            Q -= 1
            if event == 5:
                RS += 1
            elif event == 6:
                RL += 1
        elif event in (7, 8):
            Q -= 1
            if event == 7:
                B += 1
            else:
                RL += 1
        else:
            B -= 1

        if validate:
            _validate_light_state(
                t=t,
                event=event,
                rates=None,
                total_rate=None,
                Q=Q,
                B=B,
                RS=RS,
                RL=RL,
            )

    if record_path:
        return pd.DataFrame(path_rows, columns=["t", "Q", "B", "RS", "RL"])

    observation_time = horizon - params.warmup
    if observation_time <= 0:
        raise ValueError("warmup must be shorter than horizon")

    return LightSimulationResult(
        mean_Q=area_Q / observation_time,
        mean_B=area_B / observation_time,
        mean_RS=area_RS / observation_time,
        mean_RL=area_RL / observation_time,
        final_Q=Q,
        final_B=B,
        final_RS=RS,
        final_RL=RL,
    )


def run_light_replications(
    params: SimulationParams,
    n_replications: int,
    horizon: float | None = None,
    initial_state: LightState | None = None,
    validate: bool = False,
) -> pd.DataFrame:
    """Run independent light-simulator replications."""

    if not isinstance(n_replications, (int, np.integer)) or isinstance(
        n_replications, bool
    ):
        raise ValueError("n_replications must be an integer")
    if n_replications <= 0:
        raise ValueError("n_replications must be positive")

    seed_sequence = np.random.SeedSequence(params.seed)
    child_sequences = seed_sequence.spawn(n_replications)
    rows = []
    for replication, child_sequence in enumerate(child_sequences):
        replication_seed = int(child_sequence.generate_state(1, dtype=np.uint64)[0])
        result = simulate_light(
            replace(params, seed=replication_seed),
            horizon=horizon,
            initial_state=initial_state,
            validate=validate,
        )
        if not isinstance(result, LightSimulationResult):
            raise TypeError("run_light_replications requires record_path=False")
        row = {"replication": replication, "seed": replication_seed}
        row.update(result.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)
