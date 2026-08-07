"""Compare aggregate light-simulator time averages with fluid steady state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from configs.paper_cc2_baseline import BASELINE, default_initial_state
from configs.paper_cc2_baseline import make_baseline_params
from fluid_steady_state import solve_fluid_steady_state
from light_simulation import run_light_replications


OUTPUT_PATH = Path("outputs/light_vs_fluid_comparison.csv")
METRIC_MAP = {
    "Q": ("mean_Q", "q_bar"),
    "B": ("mean_B", "b_bar"),
    "RS": ("mean_RS", "rS_bar"),
    "RL": ("mean_RL", "rL_bar"),
}


def default_params(seed: int = 2026):
    """Return a clearly labeled example parameter set for aggregate validation."""

    return make_baseline_params(seed=seed)


def percent_difference(light_mean: float, fluid_value: float) -> float:
    """Return 100 * (light_mean - fluid_value) / fluid_value."""

    if fluid_value == 0:
        return float(np.nan)
    return 100.0 * (light_mean - fluid_value) / fluid_value


def build_comparison(n_replications: int = 100) -> pd.DataFrame:
    """Run light replications and compare aggregate means with fluid bars."""

    params = default_params()
    light = run_light_replications(
        params,
        n_replications,
        horizon=params.T,
        initial_state=default_initial_state(params),
        validate=True,
    )
    fluid = solve_fluid_steady_state(params).to_dict()

    rows = []
    for label, (light_metric, fluid_metric) in METRIC_MAP.items():
        light_mean = float(light[light_metric].mean())
        light_std = float(light[light_metric].std(ddof=1))
        fluid_value = float(fluid[fluid_metric])
        rows.append(
            {
                "metric": label,
                "light_mean": light_mean,
                "light_std": light_std,
                "fluid_value": fluid_value,
                "difference": light_mean - fluid_value,
                "percent_difference": percent_difference(light_mean, fluid_value),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = build_comparison(BASELINE.n_replications)
    frame.to_csv(OUTPUT_PATH, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved comparison rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
