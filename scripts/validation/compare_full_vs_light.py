"""Compare the full simulator with the aggregate light simulator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from light_simulation import simulate_light
from stochastic_simulation import SimulationParams, simulate_one


OUTPUT_PATH = Path("outputs/full_vs_light_comparison.csv")
CORE_METRICS = (
    "mean_Q",
    "mean_B",
    "mean_RS",
    "mean_RL",
    "final_Q",
    "final_B",
    "final_RS",
    "final_RL",
)


def default_params(seed: int = 2026) -> SimulationParams:
    """Return a reproducible parameter set for validation scripts."""

    return SimulationParams(
        T=30.0,
        warmup=5.0,
        c=20,
        lam=80.0,
        mu_plus=10.0,
        mu_minus=2.0,
        thetaA=1.5,
        thetaS=1.0,
        thetaL=0.5,
        deltaB=0.04,
        deltaS=1.2,
        deltaL=0.15,
        gamma=0.02,
        q0=0,
        b0=50,
        rs0=4,
        rl0=6,
        seed=seed,
    )


def percent_difference(full_value: float, light_value: float) -> float:
    """Return percent difference relative to the full simulator value."""

    if full_value == 0:
        return 0.0 if light_value == 0 else float("inf")
    return ((light_value - full_value) / abs(full_value)) * 100.0


def build_comparison(
    params: SimulationParams | None = None, n_replications: int = 25
) -> pd.DataFrame:
    """Run matched full/light replications and return a tidy comparison table."""

    params = default_params() if params is None else params
    rows = []
    for replication in range(n_replications):
        seed = int(params.seed) + replication
        run_params = replace(params, seed=seed)
        full = simulate_one(run_params, validate=True).to_dict()
        light = simulate_light(run_params, validate=True).to_dict()
        for metric in CORE_METRICS:
            full_value = float(full[metric])
            light_value = float(light[metric])
            rows.append(
                {
                    "replication": replication,
                    "seed": seed,
                    "metric": metric,
                    "full": full_value,
                    "light": light_value,
                    "difference": light_value - full_value,
                    "percent_difference": percent_difference(
                        full_value, light_value
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_comparison()
    frame.to_csv(OUTPUT_PATH, index=False)
    summary = (
        frame.groupby("metric")[["full", "light", "difference", "percent_difference"]]
        .mean()
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(f"\nSaved comparison rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
