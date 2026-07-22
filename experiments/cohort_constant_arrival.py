"""Run constant-arrival cohort analyses for person-level call burden."""

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

from configs.paper_cc2_baseline import make_baseline_params, zero_initial_state
from fluid_steady_state import solve_fluid_ode, solve_fluid_steady_state
from stochastic_simulation import (
    caller_attempt_distribution_table,
    simulate_one,
    time_to_enrollment_summary_table,
)


OUTPUT_DIR = Path("outputs/cohort_caller")
FIGURE_DIR = Path("outputs/figures")
SCENARIOS = (
    {"horizon": 450.0, "cohort_start": 260.0, "cohort_end": 320.0},
    {"horizon": 500.0, "cohort_start": 260.0, "cohort_end": 320.0},
)


def constant_arrival_params(horizon: float, cohort_start: float, seed: int):
    """Return baseline parameters forced to constant fresh arrivals."""

    base = make_baseline_params(seed=seed)
    return replace(
        base,
        T=horizon,
        warmup=cohort_start,
        arrival_process="constant",
        lambda0=None,
        arrival_amplitude=0.0,
    )


def fluid_warmup_diagnostics(
    horizon: float = 250.0, dt: float = 0.5, tolerance: float = 0.05
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Solve the constant-arrival fluid path and flag the first near-steady time."""

    params = constant_arrival_params(horizon, 0.0, seed=20260628)
    t_eval = np.arange(0.0, horizon + dt, dt)
    fluid = solve_fluid_ode(params, horizon, zero_initial_state(), t_eval)
    steady = solve_fluid_steady_state(params).to_dict()

    diagnostics = fluid.copy()
    for path_col, steady_col in (("q", "q_bar"), ("b", "b_bar"), ("rS", "rS_bar"), ("rL", "rL_bar")):
        scale = max(abs(steady[steady_col]), 1.0)
        diagnostics[f"{path_col}_relative_gap"] = (
            diagnostics[path_col] - steady[steady_col]
        ).abs() / scale
    gap_cols = ["q_relative_gap", "b_relative_gap", "rS_relative_gap", "rL_relative_gap"]
    diagnostics["max_relative_gap"] = diagnostics[gap_cols].max(axis=1)
    diagnostics["within_tolerance"] = diagnostics["max_relative_gap"] <= tolerance
    recommended = diagnostics.loc[diagnostics["within_tolerance"], "t"]
    summary = pd.DataFrame(
        [
            {
                "horizon": horizon,
                "tolerance": tolerance,
                "recommended_warmup": (
                    float(recommended.iloc[0]) if len(recommended) > 0 else float("nan")
                ),
                **steady,
            }
        ]
    )
    return diagnostics, summary


def _plot_outputs(
    cohort_records: pd.DataFrame,
    unfinished_cohort_callers: pd.DataFrame,
    end_state: pd.DataFrame,
    cohort_end_state: pd.DataFrame,
) -> None:
    """Write simple person-level burden plots."""

    os.environ.setdefault("MPLCONFIGDIR", str(FIGURE_DIR / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(cohort_records) > 0:
        counts = cohort_records["attempt_count"].value_counts().sort_index()
        ax.bar(counts.index, counts.values)
    ax.set_title("Cohort attempt-count distribution")
    ax.set_xlabel("attempt count before enrollment")
    ax.set_ylabel("completed cohort callers")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cohort_attempt_count_constant_arrival.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if len(cohort_records) > 0:
        ax.hist(cohort_records["time_to_enrollment"], bins=30)
    ax.set_title("Cohort time to enrollment")
    ax.set_xlabel("model days from first call to enrollment")
    ax.set_ylabel("completed cohort callers")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "cohort_time_to_enrollment_constant_arrival.png", dpi=150)
    plt.close(fig)

    _plot_end_state_distributions(
        end_state,
        title="End-of-horizon attempt counts by location",
        filename_stem="end_state_attempt_distribution",
    )
    _plot_end_state_distributions(
        cohort_end_state,
        title="Cohort end-of-horizon attempt counts by location",
        filename_stem="cohort_end_state_attempt_distribution",
        ylabel="cohort callers",
    )
    _plot_left_without_enrollment_distribution(unfinished_cohort_callers)


def left_without_enrollment_distribution_table(
    unfinished_cohort_callers: pd.DataFrame,
) -> pd.DataFrame:
    """Return attempt-count distribution for cohort callers lost before enrollment."""

    columns = ["scenario", "attempt_count", "count", "proportion"]
    if len(unfinished_cohort_callers) == 0:
        return pd.DataFrame(columns=columns)

    lost = unfinished_cohort_callers[
        unfinished_cohort_callers["current_location"] == "left_without_enrollment"
    ].copy()
    if len(lost) == 0:
        return pd.DataFrame(columns=columns)

    grouped = (
        lost.groupby(["scenario", "current_attempt_count"])
        .size()
        .reset_index(name="count")
        .rename(columns={"current_attempt_count": "attempt_count"})
    )
    totals = grouped.groupby("scenario")["count"].transform("sum")
    grouped["proportion"] = grouped["count"] / totals
    return grouped[columns].sort_values(["scenario", "attempt_count"]).reset_index(
        drop=True
    )


def _plot_left_without_enrollment_distribution(
    unfinished_cohort_callers: pd.DataFrame,
) -> None:
    """Plot attempt-count distribution for cohort callers lost before enrollment."""

    import matplotlib.pyplot as plt

    distribution = left_without_enrollment_distribution_table(
        unfinished_cohort_callers
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_title("Cohort left-without-enrollment attempt counts")
    ax.set_xlabel("attempt count")
    ax.set_ylabel("cohort callers")
    if len(distribution) > 0:
        scenarios = sorted(distribution["scenario"].unique())
        attempt_counts = sorted(
            int(value) for value in distribution["attempt_count"].unique()
        )
        x = np.asarray(attempt_counts, dtype=float)
        width = 0.8 / max(len(scenarios), 1)
        for index, scenario in enumerate(scenarios):
            scenario_frame = distribution[distribution["scenario"] == scenario]
            counts_by_attempt = {
                int(row.attempt_count): int(row.count)
                for row in scenario_frame.itertuples(index=False)
            }
            offset = (index - (len(scenarios) - 1) / 2.0) * width
            ax.bar(
                x + offset,
                [
                    counts_by_attempt.get(attempt_count, 0)
                    for attempt_count in attempt_counts
                ],
                width=width,
                label=str(scenario),
            )
        ax.set_xticks(x)
        if len(scenarios) > 1:
            ax.legend(fontsize=8)
    else:
        ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR
        / "cohort_left_without_enrollment_attempt_distribution_constant_arrival.png",
        dpi=150,
    )
    plt.close(fig)


def _plot_end_state_distributions(
    end_state: pd.DataFrame,
    *,
    title: str,
    filename_stem: str,
    ylabel: str = "callers",
) -> None:
    """Write faceted and per-location end-state attempt-count plots."""

    import matplotlib.pyplot as plt

    locations = ["queue", "service", "short_orbit", "long_orbit"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)
    for ax, location in zip(axes.ravel(), locations):
        _plot_end_state_location(ax, end_state, location, ylabel=ylabel)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / f"{filename_stem}_constant_arrival.png", dpi=150)
    plt.close(fig)

    for location in locations:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_end_state_location(ax, end_state, location, ylabel=ylabel)
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR
            / f"{filename_stem}_{location}_constant_arrival.png",
            dpi=150,
        )
        plt.close(fig)


def _plot_end_state_location(
    ax, end_state: pd.DataFrame, location: str, *, ylabel: str = "callers"
) -> None:
    """Plot one end-of-horizon attempt-count distribution for a single location."""

    frame = end_state[end_state["location"] == location]
    ax.set_title(location.replace("_", " "))
    ax.set_xlabel("attempt count")
    ax.set_ylabel(ylabel)
    if len(frame) == 0:
        ax.set_xticks([])
        return

    scenarios = sorted(frame["scenario"].unique()) if "scenario" in frame else ["run"]
    attempt_counts = sorted(int(value) for value in frame["attempt_count"].unique())
    x = np.asarray(attempt_counts, dtype=float)
    width = 0.8 / max(len(scenarios), 1)
    for index, scenario in enumerate(scenarios):
        scenario_frame = frame[frame["scenario"] == scenario]
        counts_by_attempt = {
            int(row.attempt_count): int(row.count)
            for row in scenario_frame.itertuples(index=False)
        }
        offset = (index - (len(scenarios) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            [counts_by_attempt.get(attempt_count, 0) for attempt_count in attempt_counts],
            width=width,
            label=str(scenario),
        )
    ax.set_xticks(x)
    if len(scenarios) > 1:
        ax.legend(fontsize=8)


def run() -> None:
    """Run all constant-arrival cohort scenarios and write outputs."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    caller_frames = []
    unfinished_frames = []
    end_state_frames = []
    cohort_end_state_frames = []

    for scenario_id, scenario in enumerate(SCENARIOS):
        params = constant_arrival_params(
            scenario["horizon"],
            scenario["cohort_start"],
            seed=20260628 + scenario_id,
        )
        result, extras = simulate_one(
            params,
            validate=True,
            return_caller_records=True,
            return_end_state_distribution=True,
            return_cohort_summary=True,
            cohort_start=scenario["cohort_start"],
            cohort_end=scenario["cohort_end"],
        )
        del result

        scenario_label = (
            f"h{int(scenario['horizon'])}_"
            f"c{int(scenario['cohort_start'])}_{int(scenario['cohort_end'])}"
        )
        summary = extras["cohort_summary"].copy()
        summary.insert(0, "scenario", scenario_label)
        summary_rows.append(summary)

        caller_records = extras["caller_records"]
        cohort_records = caller_records[caller_records["cohort_member"]].copy()
        cohort_records.insert(0, "scenario", scenario_label)
        caller_frames.append(cohort_records)

        unfinished = extras["unfinished_cohort_callers"].copy()
        unfinished.insert(0, "scenario", scenario_label)
        unfinished_frames.append(unfinished)

        end_state = extras["end_state_attempt_distribution"].copy()
        end_state.insert(0, "scenario", scenario_label)
        end_state_frames.append(end_state)

        cohort_end_state = extras["cohort_end_state_attempt_distribution"].copy()
        cohort_end_state.insert(0, "scenario", scenario_label)
        cohort_end_state_frames.append(cohort_end_state)

    robustness = pd.concat(summary_rows, ignore_index=True)
    cohort_records_all = pd.concat(caller_frames, ignore_index=True)
    unfinished_all = pd.concat(unfinished_frames, ignore_index=True)
    end_state_all = pd.concat(end_state_frames, ignore_index=True)
    cohort_end_state_all = pd.concat(cohort_end_state_frames, ignore_index=True)

    robustness.to_csv(
        OUTPUT_DIR / "cohort_robustness_constant_arrival.csv", index=False
    )
    cohort_records_all.to_csv(
        OUTPUT_DIR / "cohort_caller_records_constant_arrival.csv", index=False
    )
    unfinished_all.to_csv(
        OUTPUT_DIR / "unfinished_cohort_callers_constant_arrival.csv", index=False
    )
    end_state_all.to_csv(
        OUTPUT_DIR / "end_state_attempt_distribution_constant_arrival.csv",
        index=False,
    )
    cohort_end_state_all.to_csv(
        OUTPUT_DIR / "cohort_end_state_attempt_distribution_constant_arrival.csv",
        index=False,
    )

    caller_attempt_distribution_table(cohort_records_all).to_csv(
        OUTPUT_DIR / "cohort_attempt_distribution_constant_arrival.csv",
        index=False,
    )
    time_to_enrollment_summary_table(cohort_records_all).to_csv(
        OUTPUT_DIR / "cohort_time_to_enrollment_summary_constant_arrival.csv",
        index=False,
    )
    left_without_enrollment_distribution_table(unfinished_all).to_csv(
        OUTPUT_DIR
        / "cohort_left_without_enrollment_attempt_distribution_constant_arrival.csv",
        index=False,
    )

    fluid_path, fluid_summary = fluid_warmup_diagnostics()
    fluid_path.to_csv(OUTPUT_DIR / "fluid_trajectory_constant_arrival.csv", index=False)
    fluid_summary.to_csv(
        OUTPUT_DIR / "fluid_warmup_diagnostic_constant_arrival.csv", index=False
    )

    _plot_outputs(
        cohort_records_all,
        unfinished_all,
        end_state_all,
        cohort_end_state_all,
    )
    print(f"Saved constant-arrival cohort outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
