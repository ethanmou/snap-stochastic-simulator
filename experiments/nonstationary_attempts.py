"""Run non-stationary arrival and enrollment-attempt experiments."""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_cc2_baseline import BASELINE, make_baseline_params, zero_initial_state
from fluid_steady_state import solve_fluid_ode
from stochastic_simulation import (
    SimulationParams,
    attempt_distribution_table,
    simulate_one,
    summarize_attempt_records,
)


OUTPUT_DIR = Path("outputs/nonstationary_attempts")
SCENARIOS = (
    ("sinusoidal_amp_025", 0.25),
    ("sinusoidal_amp_050", 0.50),
)


def _scenario_params(amplitude: float) -> SimulationParams:
    """Return CC2 baseline params with weekly sinusoidal fresh arrivals."""

    base = make_baseline_params()
    return replace(
        base,
        arrival_process="sinusoidal",
        lambda0=base.lam,
        arrival_amplitude=amplitude,
        arrival_period=7.0,
        arrival_phase=0.0,
    )


def _mean_path(paths: list[pd.DataFrame]) -> pd.DataFrame:
    """Average replicated stochastic state paths at each recorded time."""

    return (
        pd.concat(paths, ignore_index=True)
        .groupby("t", as_index=False)[["Q", "B", "RS", "RL"]]
        .mean()
    )


def _path_comparison(stochastic: pd.DataFrame, fluid: pd.DataFrame) -> pd.DataFrame:
    """Return a time-indexed stochastic-vs-fluid path comparison."""

    frame = stochastic.merge(fluid, on="t", how="inner")
    for stochastic_col, fluid_col in (("Q", "q"), ("B", "b"), ("RS", "rS"), ("RL", "rL")):
        frame[f"{stochastic_col}_difference"] = frame[stochastic_col] - frame[fluid_col]
    return frame


def _plot_paths(scenario: str, stochastic: pd.DataFrame, fluid: pd.DataFrame) -> None:
    """Write Q/B/RS/RL stochastic mean vs fluid path plots."""

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = {
        "Q": ("Q", "q"),
        "B": ("B", "b"),
        "RS": ("RS", "rS"),
        "RL": ("RL", "rL"),
    }
    for label, (stochastic_col, fluid_col) in specs.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(stochastic["t"], stochastic[stochastic_col], label="stochastic mean")
        ax.plot(fluid["t"], fluid[fluid_col], label="fluid ODE", color="black")
        ax.set_title(f"{scenario}: {label}(t)")
        ax.set_xlabel("model day")
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"{scenario}_{label}_path.png", dpi=150)
        plt.close(fig)


def _plot_attempt_distribution(scenario: str, distribution: pd.DataFrame) -> None:
    """Write a bar plot of completed enrollment attempt counts."""

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(distribution) > 0:
        ax.bar(distribution["attempt_count"], distribution["num_callers"])
    ax.set_title(f"{scenario}: completed attempt-count distribution")
    ax.set_xlabel("attempt count before enrollment")
    ax.set_ylabel("completed enrollments")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{scenario}_attempt_distribution.png", dpi=150)
    plt.close(fig)


def _attempt_mean_over_time(records_by_replication: list[list[int]], scenario: str) -> None:
    """Plot cumulative mean attempt count over completed enrollment records."""

    os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = [value for replication in records_by_replication for value in replication]
    if len(records) == 0:
        frame = pd.DataFrame(columns=["completed_enrollment_index", "mean_attempt_count"])
    else:
        keep_indices = set(
            np.unique(
                np.linspace(1, len(records), min(len(records), 1000), dtype=int)
            )
        )
        cumulative = []
        running_sum = 0.0
        for index, value in enumerate(records, start=1):
            running_sum += value
            if index in keep_indices:
                cumulative.append(
                    {
                        "completed_enrollment_index": index,
                        "mean_attempt_count": running_sum / index,
                    }
                )
        frame = pd.DataFrame(cumulative)
    frame.to_csv(OUTPUT_DIR / f"{scenario}_mean_attempts_over_completed_enrollments.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(frame) > 0:
        ax.plot(frame["completed_enrollment_index"], frame["mean_attempt_count"])
    ax.set_title(f"{scenario}: cumulative mean attempts")
    ax.set_xlabel("completed enrollment index")
    ax.set_ylabel("mean attempt count")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"{scenario}_mean_attempts_over_time.png", dpi=150)
    plt.close(fig)


def run_scenario(scenario: str, amplitude: float) -> None:
    """Run one non-stationary arrival scenario and write outputs."""

    params = _scenario_params(amplitude)
    initial_state = zero_initial_state()
    t_eval = np.arange(0.0, params.T, 0.1)
    fluid = solve_fluid_ode(params, params.T, initial_state, t_eval)
    fluid.to_csv(OUTPUT_DIR / f"{scenario}_fluid_path.csv", index=False)

    paths: list[pd.DataFrame] = []
    records_by_replication: list[list[int]] = []
    metric_rows = []
    for replication in range(BASELINE.n_replications):
        run_params = replace(params, seed=BASELINE.seed + replication)
        result, records, path = simulate_one(
            run_params,
            return_attempt_records=True,
            record_path=True,
            record_dt=0.1,
        )
        paths.append(path)
        records_by_replication.append(records)
        metric_rows.append(
            {
                "scenario": scenario,
                "replication": replication,
                "seed": run_params.seed,
                **result.to_dict(),
            }
        )

    stochastic_mean = _mean_path(paths)
    stochastic_mean.to_csv(OUTPUT_DIR / f"{scenario}_stochastic_mean_path.csv", index=False)
    _path_comparison(stochastic_mean, fluid).to_csv(
        OUTPUT_DIR / f"{scenario}_path_comparison.csv", index=False
    )
    pd.DataFrame(metric_rows).to_csv(OUTPUT_DIR / f"{scenario}_replication_metrics.csv", index=False)

    all_records = [value for records in records_by_replication for value in records]
    attempt_metrics = {
        "scenario": scenario,
        "arrival_amplitude": amplitude,
        "arrival_period_model_days": params.arrival_period,
        **summarize_attempt_records(all_records),
    }
    pd.DataFrame([attempt_metrics]).to_csv(
        OUTPUT_DIR / f"{scenario}_attempt_metrics.csv", index=False
    )
    distribution = attempt_distribution_table(all_records)
    distribution.to_csv(OUTPUT_DIR / f"{scenario}_attempt_distribution.csv", index=False)

    _plot_paths(scenario, stochastic_mean, fluid)
    _plot_attempt_distribution(scenario, distribution)
    _attempt_mean_over_time(records_by_replication, scenario)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for scenario, amplitude in SCENARIOS:
        run_scenario(scenario, amplitude)
    print(f"Saved non-stationary attempt outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
