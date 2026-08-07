"""Compare many short replications with one long replication."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from light_simulation import run_light_replications, simulate_light
from stochastic_simulation import SimulationParams, run_replications, simulate_one


OUTPUT_PATH = Path("outputs/long_vs_many_replications_comparison.csv")
METRICS = (
    "mean_Q",
    "mean_B",
    "mean_RS",
    "mean_RL",
)


def percent_difference(reference: float, value: float) -> float:
    """Return percent difference relative to the many-replication mean."""

    if reference == 0:
        return float("nan")
    return 100.0 * (value - reference) / reference


def _comparison_rows(label: str, many: pd.DataFrame, one_long: dict[str, float]):
    """Yield comparison rows for one simulator family."""

    for metric in METRICS:
        many_mean = float(many[metric].mean())
        many_std = float(many[metric].std(ddof=1))
        long_value = float(one_long[metric])
        yield {
            "simulator": label,
            "metric": metric,
            "many_replications_mean": many_mean,
            "many_replications_std": many_std,
            "one_long_replication_value": long_value,
            "difference": long_value - many_mean,
            "percent_difference": percent_difference(many_mean, long_value),
        }


def validation_params(seed: int = 2026) -> SimulationParams:
    """Return a stable congested parameter set for long-run validation."""

    return SimulationParams(
        T=160.0,
        warmup=80.0,
        c=4,
        lam=30.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=3.0,
        thetaS=2.0,
        thetaL=1.0,
        deltaB=0.2,
        deltaS=2.0,
        deltaL=0.5,
        gamma=0.7,
        q0=0,
        b0=20,
        rs0=2,
        rl0=2,
        seed=seed,
    )


def build_comparison(
    n_light_replications: int = 100, n_full_replications: int = 30
) -> pd.DataFrame:
    """Build the many-short vs one-long comparison table."""

    base = validation_params()
    light_many_params = base
    light_long_params = replace(base, T=700.0, warmup=350.0, seed=9090)
    full_many_params = replace(base, seed=3030)
    full_long_params = replace(base, T=700.0, warmup=350.0, seed=9191)

    light_many = run_light_replications(
        light_many_params, n_light_replications, validate=True
    )
    light_long = simulate_light(light_long_params, validate=True).to_dict()

    full_many = run_replications(full_many_params, n_full_replications, validate=True)
    full_long = simulate_one(full_long_params, validate=True).to_dict()

    rows = []
    rows.extend(_comparison_rows("light", light_many, light_long))
    rows.extend(_comparison_rows("full", full_many, full_long))
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_comparison()
    frame.to_csv(OUTPUT_PATH, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved comparison rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
