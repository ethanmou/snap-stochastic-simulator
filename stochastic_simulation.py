"""Event-driven stochastic simulator for the SNAP call center model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Optional

import numpy as np
import pandas as pd


MINUTES_PER_MODEL_DAY = 540.0


@dataclass(frozen=True)
class SimulationParams:
    """Inputs for one continuous-time simulation replication."""

    T: float
    warmup: float
    c: int
    lam: float
    mu_plus: float
    mu_minus: float
    thetaA: float
    thetaS: float
    thetaL: float
    deltaB: float
    deltaS: float
    deltaL: float
    gamma: float
    q0: int = 0
    b0: int = 0
    rs0: int = 0
    rl0: int = 0
    seed: Optional[int] = None


@dataclass(frozen=True)
class SimulationResult:
    """Summary statistics and invariant diagnostics for one replication."""

    total_arrivals: int
    fresh_arrivals: int
    recertification_arrivals: int
    short_redial_arrivals: int
    long_redial_arrivals: int
    service_successes: int
    service_failures: int
    abandon_lost: int
    abandon_short: int
    abandon_long: int
    total_abandonments: int
    mean_Q: float
    mean_waiting: float
    mean_B: float
    mean_RS: float
    mean_RL: float
    utilization: float
    average_wait_including_abandonments_minutes: float
    average_speed_to_answer_minutes: float
    average_time_to_abandonment_minutes: float
    abandonment_fraction: float
    procedural_denial_rate_per_model_day: float
    total_arrival_rate_per_model_day: float
    effective_arrival_rate_per_model_day: float
    returning_call_rate_per_model_day: float
    final_Q: int
    final_waiting: int
    final_in_service: int
    final_B: int
    final_RS: int
    final_RL: int
    min_Q: int
    min_B: int
    min_RS: int
    min_RL: int
    max_in_service: int

    def to_dict(self) -> dict[str, int | float]:
        """Return the result as a flat dictionary suitable for a DataFrame row."""

        return asdict(self)


def _validate_params(params: SimulationParams) -> None:
    real_fields = (
        "T",
        "warmup",
        "lam",
        "mu_plus",
        "mu_minus",
        "thetaA",
        "thetaS",
        "thetaL",
        "deltaB",
        "deltaS",
        "deltaL",
        "gamma",
    )
    for name in real_fields:
        value = getattr(params, name)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")

    if params.T <= 0:
        raise ValueError("T must be positive")
    if params.warmup < 0 or params.warmup >= params.T:
        raise ValueError("warmup must satisfy 0 <= warmup < T")

    for name in real_fields[2:]:
        if getattr(params, name) < 0:
            raise ValueError(f"{name} must be nonnegative")

    integer_fields = ("c", "q0", "b0", "rs0", "rl0")
    for name in integer_fields:
        value = getattr(params, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
        if value < 0:
            raise ValueError(f"{name} must be nonnegative")

    if params.seed is not None:
        if not isinstance(params.seed, (int, np.integer)) or isinstance(params.seed, bool):
            raise ValueError("seed must be an integer or None")
        if params.seed < 0:
            raise ValueError("seed must be nonnegative")


def simulate_one(params: SimulationParams) -> SimulationResult:
    """Run one Gillespie simulation of the SNAP call center model.

    Time averages and event statistics are collected after ``params.warmup``.
    Queue service is FCFS, while abandonment selects a uniformly random waiting
    caller, as implied by identical exponential abandonment clocks.
    """

    _validate_params(params)
    rng = np.random.default_rng(params.seed)

    B = params.b0
    RS = params.rs0
    RL = params.rl0
    initial_in_service = min(params.q0, params.c)
    in_service = [0.0] * initial_in_service
    waiting_queue = [0.0] * (params.q0 - initial_in_service)

    counts = {
        "fresh_arrivals": 0,
        "recertification_arrivals": 0,
        "short_redial_arrivals": 0,
        "long_redial_arrivals": 0,
        "service_successes": 0,
        "service_failures": 0,
        "abandon_lost": 0,
        "abandon_short": 0,
        "abandon_long": 0,
    }
    answered_waits: list[float] = []
    abandoned_waits: list[float] = []
    areas = {
        "Q": 0.0,
        "waiting": 0.0,
        "B": 0.0,
        "RS": 0.0,
        "RL": 0.0,
        "busy": 0.0,
    }

    min_Q = params.q0
    min_B = B
    min_RS = RS
    min_RL = RL
    max_in_service = len(in_service)
    t = 0.0

    def admit_caller(entry_time: float, event_time: float, collect: bool) -> None:
        if len(in_service) < params.c:
            in_service.append(event_time)
            if collect:
                answered_waits.append(event_time - entry_time)
        else:
            waiting_queue.append(entry_time)

    def start_next_waiting(event_time: float, collect: bool) -> None:
        if waiting_queue and len(in_service) < params.c:
            entry_time = waiting_queue.pop(0)
            in_service.append(event_time)
            if collect:
                answered_waits.append(event_time - entry_time)

    while t < params.T:
        n_waiting = len(waiting_queue)
        n_service = len(in_service)
        rates = np.array(
            [
                params.lam,
                params.deltaB * B,
                params.deltaS * RS,
                params.deltaL * RL,
                params.thetaA * n_waiting,
                params.thetaS * n_waiting,
                params.thetaL * n_waiting,
                params.mu_plus * n_service,
                params.mu_minus * n_service,
                params.gamma * B,
            ],
            dtype=float,
        )
        total_rate = float(rates.sum())
        event_time = (
            params.T
            if total_rate == 0.0
            else t + rng.exponential(1.0 / total_rate)
        )
        interval_end = min(event_time, params.T)

        observed_start = max(t, params.warmup)
        observed_dt = max(0.0, interval_end - observed_start)
        if observed_dt:
            areas["Q"] += (n_waiting + n_service) * observed_dt
            areas["waiting"] += n_waiting * observed_dt
            areas["B"] += B * observed_dt
            areas["RS"] += RS * observed_dt
            areas["RL"] += RL * observed_dt
            areas["busy"] += n_service * observed_dt

        if event_time > params.T or total_rate == 0.0:
            t = params.T
            break

        t = event_time
        collect = t >= params.warmup
        event = min(
            int(
                np.searchsorted(
                    np.cumsum(rates), rng.random() * total_rate, side="right"
                )
            ),
            len(rates) - 1,
        )

        if event == 0:  # Fresh arrival
            if collect:
                counts["fresh_arrivals"] += 1
            admit_caller(t, t, collect)
        elif event == 1:  # Recertification arrival
            B -= 1
            if collect:
                counts["recertification_arrivals"] += 1
            admit_caller(t, t, collect)
        elif event == 2:  # Short-redial arrival
            RS -= 1
            if collect:
                counts["short_redial_arrivals"] += 1
            admit_caller(t, t, collect)
        elif event == 3:  # Long-redial arrival
            RL -= 1
            if collect:
                counts["long_redial_arrivals"] += 1
            admit_caller(t, t, collect)
        elif event in (4, 5, 6):  # Waiting-caller abandonment
            abandoned_index = int(rng.integers(len(waiting_queue)))
            entry_time = waiting_queue.pop(abandoned_index)
            if collect:
                abandoned_waits.append(t - entry_time)
            if event == 4:
                if collect:
                    counts["abandon_lost"] += 1
            elif event == 5:
                RS += 1
                if collect:
                    counts["abandon_short"] += 1
            else:
                RL += 1
                if collect:
                    counts["abandon_long"] += 1
        elif event in (7, 8):  # Service completion
            completed_index = int(rng.integers(len(in_service)))
            in_service.pop(completed_index)
            if event == 7:
                B += 1
                if collect:
                    counts["service_successes"] += 1
            else:
                RL += 1
                if collect:
                    counts["service_failures"] += 1
            start_next_waiting(t, collect)
        else:  # Enrolled departure
            B -= 1

        Q = len(waiting_queue) + len(in_service)
        min_Q = min(min_Q, Q)
        min_B = min(min_B, B)
        min_RS = min(min_RS, RS)
        min_RL = min(min_RL, RL)
        max_in_service = max(max_in_service, len(in_service))

    observation_time = params.T - params.warmup
    total_arrivals = sum(
        counts[name]
        for name in (
            "fresh_arrivals",
            "recertification_arrivals",
            "short_redial_arrivals",
            "long_redial_arrivals",
        )
    )
    total_abandonments = sum(
        counts[name] for name in ("abandon_lost", "abandon_short", "abandon_long")
    )
    all_waits = answered_waits + abandoned_waits

    def mean_minutes(values: list[float]) -> float:
        return (float(np.mean(values)) * MINUTES_PER_MODEL_DAY) if values else 0.0

    return SimulationResult(
        total_arrivals=total_arrivals,
        fresh_arrivals=counts["fresh_arrivals"],
        recertification_arrivals=counts["recertification_arrivals"],
        short_redial_arrivals=counts["short_redial_arrivals"],
        long_redial_arrivals=counts["long_redial_arrivals"],
        service_successes=counts["service_successes"],
        service_failures=counts["service_failures"],
        abandon_lost=counts["abandon_lost"],
        abandon_short=counts["abandon_short"],
        abandon_long=counts["abandon_long"],
        total_abandonments=total_abandonments,
        mean_Q=areas["Q"] / observation_time,
        mean_waiting=areas["waiting"] / observation_time,
        mean_B=areas["B"] / observation_time,
        mean_RS=areas["RS"] / observation_time,
        mean_RL=areas["RL"] / observation_time,
        utilization=areas["busy"] / (params.c * observation_time) if params.c else 0.0,
        average_wait_including_abandonments_minutes=mean_minutes(all_waits),
        average_speed_to_answer_minutes=mean_minutes(answered_waits),
        average_time_to_abandonment_minutes=mean_minutes(abandoned_waits),
        abandonment_fraction=total_abandonments / total_arrivals if total_arrivals else 0.0,
        procedural_denial_rate_per_model_day=counts["abandon_lost"] / observation_time,
        total_arrival_rate_per_model_day=total_arrivals / observation_time,
        effective_arrival_rate_per_model_day=(total_arrivals - total_abandonments)
        / observation_time,
        returning_call_rate_per_model_day=(
            counts["recertification_arrivals"]
            + counts["short_redial_arrivals"]
            + counts["long_redial_arrivals"]
        )
        / observation_time,
        final_Q=len(waiting_queue) + len(in_service),
        final_waiting=len(waiting_queue),
        final_in_service=len(in_service),
        final_B=B,
        final_RS=RS,
        final_RL=RL,
        min_Q=min_Q,
        min_B=min_B,
        min_RS=min_RS,
        min_RL=min_RL,
        max_in_service=max_in_service,
    )


def run_replications(params: SimulationParams, n_replications: int) -> pd.DataFrame:
    """Run independent replications and return one summary row per run."""

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
        result = simulate_one(replace(params, seed=replication_seed))
        row = {"replication": replication, "seed": replication_seed}
        row.update(result.to_dict())
        rows.append(row)

    return pd.DataFrame(rows)
