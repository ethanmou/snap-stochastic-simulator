"""Dynamic-horizon cohort policy analysis for caller-level outcomes."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stochastic_simulation import MINUTES_PER_MODEL_DAY, SimulationParams, simulate_one


FIXED_RATES = {
    "deltaB": 1.0 / 130.0,
    "deltaS": 3.0,
    "deltaL": 1.0 / 9.0,
    "gamma": 1.0 / 260.0,
}
DEFAULT_OUTPUT_DIR = Path("outputs/cohort_policy_analysis")
SCENARIOS = (
    "baseline",
    "double_staffing",
    "half_service_time",
    "combined_capacity",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the policy analysis."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-file", default="call center parameters.csv")
    parser.add_argument("--call-center", type=int, default=2)
    parser.add_argument("--cohort-start", type=float, default=260.0)
    parser.add_argument("--cohort-end", type=float, default=420.0)
    parser.add_argument("--post-clearance-buffer", type=float, default=20.0)
    parser.add_argument("--replications", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dynamic-horizon", type=float, default=900.0)
    parser.add_argument("--max-events", type=int, default=2_000_000)
    parser.add_argument(
        "--plot-attempt-cutoff",
        type=int,
        default=None,
        help="Optional visual cutoff; CSV outputs always retain the full tail.",
    )
    return parser.parse_args()


def prepare_output_dirs(output_dir: Path) -> dict[str, Path]:
    """Create and return the output directory structure."""

    paths = {
        "root": output_dir,
        "logs": output_dir / "logs",
        "tables": output_dir / "tables",
        "plots": output_dir / "plots",
        "validation": output_dir / "tests_or_validation",
        "matplotlib": output_dir / ".matplotlib",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    Path("outputs/policy_analysis").mkdir(parents=True, exist_ok=True)
    return paths


def configure_logging(log_path: Path) -> logging.Logger:
    """Configure file and console logging."""

    logger = logging.getLogger("cohort_policy_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_call_center_parameters(
    parameter_file: str | Path, call_center: int
) -> dict[str, Any]:
    """Load one call-center row from the supplied parameter CSV."""

    parameter_file = Path(parameter_file)
    if not parameter_file.exists():
        raise FileNotFoundError(f"parameter file not found: {parameter_file}")
    frame = pd.read_csv(parameter_file)
    required = {
        "CC",
        "lambda",
        "c",
        "AHT_used_min",
        "enroll_pct",
        "mu_plus",
        "mu_minus",
        "theta_A",
        "theta_S",
        "theta_L",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"parameter file is missing columns: {sorted(missing)}")
    matches = frame[frame["CC"] == call_center]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for CC={call_center}")
    return matches.iloc[0].to_dict()


def service_rates(aht_minutes: float, enroll_probability: float) -> tuple[float, float]:
    """Return successful and unsuccessful service rates per model day."""

    if aht_minutes <= 0:
        raise ValueError("aht_minutes must be positive")
    if enroll_probability < 0 or enroll_probability > 1:
        raise ValueError("enroll_probability must lie in [0, 1]")
    mu_total = MINUTES_PER_MODEL_DAY / aht_minutes
    return enroll_probability * mu_total, (1.0 - enroll_probability) * mu_total


def params_from_row(
    row: dict[str, Any],
    *,
    seed: int,
    horizon: float,
    warmup: float,
    staffing: int | None = None,
    aht_minutes: float | None = None,
) -> SimulationParams:
    """Build SimulationParams from the CSV row and optional policy overrides."""

    enroll_probability = float(row["enroll_pct"]) / 100.0
    chosen_aht = float(row["AHT_used_min"] if aht_minutes is None else aht_minutes)
    mu_plus, mu_minus = service_rates(chosen_aht, enroll_probability)
    return SimulationParams(
        T=horizon,
        warmup=warmup,
        c=int(round(float(row["c"] if staffing is None else staffing))),
        lam=float(row["lambda"]),
        mu_plus=mu_plus,
        mu_minus=mu_minus,
        thetaA=float(row["theta_A"]),
        thetaS=float(row["theta_S"]),
        thetaL=float(row["theta_L"]),
        deltaB=FIXED_RATES["deltaB"],
        deltaS=FIXED_RATES["deltaS"],
        deltaL=FIXED_RATES["deltaL"],
        gamma=FIXED_RATES["gamma"],
        q0=0,
        b0=0,
        rs0=0,
        rl0=0,
        seed=seed,
        arrival_process="constant",
        lambda0=None,
        arrival_amplitude=0.0,
    )


def build_policy_scenarios(
    row: dict[str, Any],
    *,
    seed: int,
    horizon: float,
    warmup: float,
) -> dict[str, SimulationParams]:
    """Return baseline and capacity policy scenarios for one call center."""

    base_staffing = int(round(float(row["c"])))
    base_aht = float(row["AHT_used_min"])
    return {
        "baseline": params_from_row(row, seed=seed, horizon=horizon, warmup=warmup),
        "double_staffing": params_from_row(
            row,
            seed=seed,
            horizon=horizon,
            warmup=warmup,
            staffing=base_staffing * 2,
            aht_minutes=base_aht,
        ),
        "half_service_time": params_from_row(
            row,
            seed=seed,
            horizon=horizon,
            warmup=warmup,
            staffing=base_staffing,
            aht_minutes=base_aht / 2.0,
        ),
        "combined_capacity": params_from_row(
            row,
            seed=seed,
            horizon=horizon,
            warmup=warmup,
            staffing=base_staffing * 2,
            aht_minutes=base_aht / 2.0,
        ),
    }


def scenario_parameter_table(
    scenarios: dict[str, SimulationParams], row: dict[str, Any]
) -> pd.DataFrame:
    """Return a compact table describing scenario parameters."""

    enroll_probability = float(row["enroll_pct"]) / 100.0
    rows = []
    for scenario, params in scenarios.items():
        mu_total = params.mu_plus + params.mu_minus
        rows.append(
            {
                "scenario_name": scenario,
                "call_center": int(row["CC"]),
                "lambda": params.lam,
                "staffing": params.c,
                "aht_used_min": MINUTES_PER_MODEL_DAY / mu_total,
                "enrollment_probability": enroll_probability,
                "mu_plus": params.mu_plus,
                "mu_minus": params.mu_minus,
                "theta_A": params.thetaA,
                "theta_S": params.thetaS,
                "theta_L": params.thetaL,
                "delta_B": params.deltaB,
                "delta_S": params.deltaS,
                "delta_L": params.deltaL,
                "gamma": params.gamma,
                "arrival_process": params.arrival_process,
            }
        )
    return pd.DataFrame(rows)


def percentile(values: pd.Series, q: float) -> float:
    """Return percentile or NaN for empty values."""

    clean = values.dropna()
    return float(np.percentile(clean, q)) if len(clean) > 0 else float("nan")


def replication_metrics(
    *,
    scenario: str,
    replication: int,
    seed: int,
    result,
    diagnostics: pd.Series,
    cohort_records: pd.DataFrame,
) -> dict[str, Any]:
    """Compute one replication-level policy metric row."""

    completed = cohort_records[cohort_records["terminal_outcome"] == "completed"]
    left = cohort_records[
        cohort_records["terminal_outcome"] == "left_without_enrollment"
    ]
    cohort_size = int(diagnostics["cohort_size"])
    completion_count = int(diagnostics["cohort_completed"])
    left_count = int(diagnostics["cohort_left_without_enrollment"])
    unfinished_count = int(diagnostics["cohort_unfinished"])
    attempts = cohort_records["attempt_count"] if cohort_size > 0 else pd.Series(dtype=float)
    return {
        "scenario_name": scenario,
        "replication": replication,
        "seed": seed,
        "cohort_start": diagnostics["cohort_start"],
        "cohort_end": diagnostics["cohort_end"],
        "cohort_size": cohort_size,
        "completion_count": completion_count,
        "completion_rate": completion_count / cohort_size if cohort_size > 0 else np.nan,
        "left_without_enrollment_count": left_count,
        "left_without_enrollment_rate": left_count / cohort_size
        if cohort_size > 0
        else np.nan,
        "unfinished_count": unfinished_count,
        "mean_attempt_count_all": float(attempts.mean()) if cohort_size > 0 else np.nan,
        "median_attempt_count_all": float(attempts.median()) if cohort_size > 0 else np.nan,
        "mean_attempt_count_completed": float(completed["attempt_count"].mean())
        if len(completed) > 0
        else np.nan,
        "mean_attempt_count_left": float(left["attempt_count"].mean())
        if len(left) > 0
        else np.nan,
        "p90_attempt_count_all": percentile(attempts, 90) if cohort_size > 0 else np.nan,
        "p95_attempt_count_all": percentile(attempts, 95) if cohort_size > 0 else np.nan,
        "maximum_attempt_count": int(attempts.max()) if cohort_size > 0 else 0,
        "mean_time_in_system_completed": float(completed["time_in_system"].mean())
        if len(completed) > 0
        else np.nan,
        "median_time_in_system_completed": float(completed["time_in_system"].median())
        if len(completed) > 0
        else np.nan,
        "mean_time_to_exit_left": float(left["time_in_system"].mean())
        if len(left) > 0
        else np.nan,
        "cohort_clearance_time": diagnostics["cohort_clearance_time"],
        "simulation_end_time": diagnostics["simulation_end_time"],
        "dynamic_horizon_success": bool(diagnostics["dynamic_horizon_success"]),
        "events_processed": int(diagnostics["events_processed"]),
        "termination_reason": diagnostics["termination_reason"],
        "mean_Q": result.mean_Q,
        "mean_B": result.mean_B,
        "mean_RS": result.mean_RS,
        "mean_RL": result.mean_RL,
        "average_wait_including_abandonments_minutes": result.average_wait_including_abandonments_minutes,
        "average_speed_to_answer_minutes": result.average_speed_to_answer_minutes,
        "average_time_to_abandonment_minutes": result.average_time_to_abandonment_minutes,
        "abandonment_fraction": result.abandonment_fraction,
    }


def attempt_distribution_rows(
    *,
    scenario: str,
    replication: int,
    cohort_records: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return per-replication attempt-count distributions by outcome group."""

    groups = {
        "all": cohort_records,
        "completed": cohort_records[
            cohort_records["terminal_outcome"] == "completed"
        ],
        "left_without_enrollment": cohort_records[
            cohort_records["terminal_outcome"] == "left_without_enrollment"
        ],
    }
    rows: list[dict[str, Any]] = []
    for outcome_group, frame in groups.items():
        denominator = int(len(frame))
        counts = frame["attempt_count"].value_counts().to_dict()
        for attempt_count in sorted(int(value) for value in counts):
            caller_count = int(counts[attempt_count])
            rows.append(
                {
                    "scenario_name": scenario,
                    "replication": replication,
                    "outcome_group": outcome_group,
                    "attempt_count": attempt_count,
                    "caller_count": caller_count,
                    "cohort_denominator": denominator,
                    "proportion": caller_count / denominator
                    if denominator > 0
                    else np.nan,
                }
            )
        if denominator == 0:
            rows.append(
                {
                    "scenario_name": scenario,
                    "replication": replication,
                    "outcome_group": outcome_group,
                    "attempt_count": 0,
                    "caller_count": 0,
                    "cohort_denominator": 0,
                    "proportion": np.nan,
                }
            )
    return rows


def summarize_attempt_distributions(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate attempt distributions across replications with zero-filled gaps."""

    rows = []
    for (scenario, group), subset in frame.groupby(["scenario_name", "outcome_group"]):
        valid_reps = sorted(
            subset.loc[subset["cohort_denominator"] > 0, "replication"].unique()
        )
        attempts = sorted(int(value) for value in subset["attempt_count"].unique())
        for attempt_count in attempts:
            counts = []
            proportions = []
            for replication in valid_reps:
                rep_rows = subset[
                    (subset["replication"] == replication)
                    & (subset["attempt_count"] == attempt_count)
                ]
                denominator_rows = subset[
                    (subset["replication"] == replication)
                    & (subset["cohort_denominator"] > 0)
                ]
                if len(denominator_rows) == 0:
                    continue
                if len(rep_rows) == 0:
                    counts.append(0.0)
                    proportions.append(0.0)
                else:
                    counts.append(float(rep_rows["caller_count"].sum()))
                    proportions.append(float(rep_rows["proportion"].fillna(0.0).sum()))
            rows.append(
                {
                    "scenario_name": scenario,
                    "outcome_group": group,
                    "attempt_count": attempt_count,
                    **summary_stats_array(counts, prefix="count"),
                    **summary_stats_array(proportions, prefix="proportion"),
                    "number_of_valid_replications": len(counts),
                }
            )
    return pd.DataFrame(rows)


def summary_stats_array(values: list[float], *, prefix: str) -> dict[str, float]:
    """Return mean, sample SD, and SE for a list."""

    if len(values) == 0:
        return {
            f"mean_{prefix}": np.nan,
            f"standard_deviation_{prefix}": np.nan,
            f"standard_error_{prefix}": np.nan,
        }
    array = np.asarray(values, dtype=float)
    sd = float(np.std(array, ddof=1)) if len(values) > 1 else 0.0
    return {
        f"mean_{prefix}": float(np.mean(array)),
        f"standard_deviation_{prefix}": sd,
        f"standard_error_{prefix}": sd / math.sqrt(len(values))
        if len(values) > 0
        else np.nan,
    }


def scenario_summary(replication_metrics_frame: pd.DataFrame) -> pd.DataFrame:
    """Return scenario-level metric means and confidence intervals."""

    metrics = [
        "completion_rate",
        "left_without_enrollment_rate",
        "mean_attempt_count_all",
        "median_attempt_count_all",
        "mean_attempt_count_completed",
        "mean_attempt_count_left",
        "p90_attempt_count_all",
        "p95_attempt_count_all",
        "maximum_attempt_count",
        "mean_time_in_system_completed",
        "median_time_in_system_completed",
        "mean_time_to_exit_left",
        "cohort_clearance_time",
        "simulation_end_time",
        "cohort_size",
        "completion_count",
        "left_without_enrollment_count",
        "unfinished_count",
        "mean_Q",
        "mean_B",
        "mean_RS",
        "mean_RL",
        "average_wait_including_abandonments_minutes",
        "average_speed_to_answer_minutes",
        "average_time_to_abandonment_minutes",
        "abandonment_fraction",
    ]
    rows = []
    valid = replication_metrics_frame[
        replication_metrics_frame["dynamic_horizon_success"]
    ]
    for scenario, subset in valid.groupby("scenario_name"):
        for metric in metrics:
            values = subset[metric].dropna().astype(float)
            stats = summary_stats_array(values.tolist(), prefix="value")
            se = stats["standard_error_value"]
            rows.append(
                {
                    "scenario_name": scenario,
                    "metric": metric,
                    "mean": stats["mean_value"],
                    "standard_deviation": stats["standard_deviation_value"],
                    "standard_error": se,
                    "ci95_lower": stats["mean_value"] - 1.96 * se
                    if not math.isnan(se)
                    else np.nan,
                    "ci95_upper": stats["mean_value"] + 1.96 * se
                    if not math.isnan(se)
                    else np.nan,
                    "number_of_valid_replications": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def baseline_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each policy scenario against baseline for key metrics."""

    columns = [
        "scenario_name",
        "metric",
        "scenario_mean",
        "baseline_mean",
        "absolute_difference_from_baseline",
        "percent_difference_from_baseline",
        "method",
    ]
    if "metric" not in summary.columns or len(summary) == 0:
        return pd.DataFrame(columns=columns)
    key_metrics = {
        "completion_rate",
        "left_without_enrollment_rate",
        "mean_attempt_count_all",
        "p90_attempt_count_all",
        "cohort_clearance_time",
    }
    subset = summary[summary["metric"].isin(key_metrics)]
    baseline = subset[subset["scenario_name"] == "baseline"].set_index("metric")
    if len(baseline) == 0:
        return pd.DataFrame(columns=columns)
    rows = []
    for row in subset.itertuples(index=False):
        if row.scenario_name == "baseline":
            continue
        baseline_mean = float(baseline.loc[row.metric, "mean"])
        diff = float(row.mean) - baseline_mean
        rows.append(
            {
                "scenario_name": row.scenario_name,
                "metric": row.metric,
                "scenario_mean": row.mean,
                "baseline_mean": baseline_mean,
                "absolute_difference_from_baseline": diff,
                "percent_difference_from_baseline": 100.0 * diff / baseline_mean
                if baseline_mean != 0
                else np.nan,
                "method": "normal approximation CI uses mean +/- 1.96 SE",
            }
        )
    return pd.DataFrame(rows, columns=columns)


def plot_attempt_distribution(
    summary: pd.DataFrame,
    *,
    outcome_group: str,
    output_path: Path,
    cutoff: int | None,
    scenario_order: tuple[str, ...] | None = None,
    log_scale: bool = False,
) -> None:
    """Plot mean attempt-count proportions by scenario for one outcome group."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = summary[summary["outcome_group"] == outcome_group].copy()
    if cutoff is not None:
        frame = frame[frame["attempt_count"] <= cutoff]
    fig, ax = plt.subplots(figsize=(14, 7))
    order = SCENARIOS if scenario_order is None else scenario_order
    scenarios = [name for name in order if name in set(frame["scenario_name"])]
    if not scenarios:
        scenarios = sorted(frame["scenario_name"].unique())
    attempts = sorted(int(value) for value in frame["attempt_count"].unique())
    x = np.asarray(attempts, dtype=float)
    width = 0.8 / max(len(scenarios), 1)
    for index, scenario in enumerate(scenarios):
        scenario_frame = frame[frame["scenario_name"] == scenario]
        means = {
            int(row.attempt_count): float(row.mean_proportion)
            for row in scenario_frame.itertuples(index=False)
        }
        ses = {
            int(row.attempt_count): float(row.standard_error_proportion)
            for row in scenario_frame.itertuples(index=False)
        }
        offset = (index - (len(scenarios) - 1) / 2.0) * width
        ax.bar(
            x + offset,
            [means.get(attempt, 0.0) for attempt in attempts],
            yerr=[ses.get(attempt, 0.0) for attempt in attempts],
            width=width,
            label=scenario,
            capsize=2,
        )
        for attempt in attempts:
            value = means.get(attempt, 0.0)
            if value <= 0:
                continue
            if value >= 0.005 or log_scale:
                ax.text(
                    attempt + offset,
                    value,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
    ax.set_title(f"Mean attempt-count proportion: {outcome_group}")
    ax.set_xlabel("attempt count")
    ax.set_ylabel("mean proportion")
    ax.set_xticks(x)
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("mean proportion (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_metric(
    summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    scenario_order: tuple[str, ...] | None = None,
) -> None:
    """Plot scenario means with 95% confidence intervals for one metric."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = summary[summary["metric"] == metric].copy()
    order = SCENARIOS if scenario_order is None else scenario_order
    frame["scenario_name"] = pd.Categorical(
        frame["scenario_name"], categories=order, ordered=True
    )
    frame = frame.sort_values("scenario_name")
    errors = [
        frame["mean"] - frame["ci95_lower"],
        frame["ci95_upper"] - frame["mean"],
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame["scenario_name"].astype(str), frame["mean"], yerr=errors, capsize=4)
    ax.set_title(title)
    ax.set_xlabel("scenario")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    """Run all policy scenarios and write analysis artifacts."""

    paths = prepare_output_dirs(args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(paths["matplotlib"]))
    logger = configure_logging(paths["logs"] / "analysis.log")
    logger.info("Starting cohort policy analysis")

    row = load_call_center_parameters(Path(args.parameter_file), args.call_center)
    scenarios = build_policy_scenarios(
        row,
        seed=args.seed,
        horizon=args.max_dynamic_horizon,
        warmup=args.cohort_start,
    )
    scenario_parameter_table(scenarios, row).to_csv(
        paths["tables"] / "parameter_scenarios.csv", index=False
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    replication_seeds = [
        int(seed.generate_state(1, dtype=np.uint64)[0])
        for seed in seed_sequence.spawn(args.replications)
    ]
    config = {
        "parameter_file": args.parameter_file,
        "call_center": args.call_center,
        "cohort_start": args.cohort_start,
        "cohort_end": args.cohort_end,
        "post_clearance_buffer": args.post_clearance_buffer,
        "replications": args.replications,
        "seed": args.seed,
        "seed_strategy": "matched replication seeds reused across scenarios",
        "replication_seeds": replication_seeds,
        "max_dynamic_horizon": args.max_dynamic_horizon,
        "max_events": args.max_events,
        "scenarios": list(scenarios),
    }
    (paths["root"] / "run_config.json").write_text(json.dumps(config, indent=2))

    replication_rows = []
    termination_rows = []
    caller_frames = []
    attempt_rows = []
    for scenario_name, base_params in scenarios.items():
        logger.info("Running scenario %s", scenario_name)
        for replication, seed in enumerate(replication_seeds):
            params = replace(base_params, seed=seed)
            result, extras = simulate_one(
                params,
                validate=False,
                return_caller_records=True,
                return_end_state_distribution=True,
                return_cohort_summary=True,
                cohort_start=args.cohort_start,
                cohort_end=args.cohort_end,
                dynamic_horizon=True,
                post_clearance_buffer=args.post_clearance_buffer,
                max_dynamic_horizon=args.max_dynamic_horizon,
                max_events=args.max_events,
            )
            diagnostics = extras["dynamic_horizon_diagnostics"].iloc[0].copy()
            cohort_records = extras["cohort_caller_state_records"].copy()
            cohort_records.insert(0, "seed", seed)
            cohort_records.insert(0, "replication", replication)
            cohort_records.insert(0, "scenario_name", scenario_name)
            caller_frames.append(cohort_records)
            replication_rows.append(
                replication_metrics(
                    scenario=scenario_name,
                    replication=replication,
                    seed=seed,
                    result=result,
                    diagnostics=diagnostics,
                    cohort_records=cohort_records,
                )
            )
            term = diagnostics.to_dict()
            term.update(
                {
                    "scenario_name": scenario_name,
                    "replication": replication,
                    "seed": seed,
                }
            )
            termination_rows.append(term)
            attempt_rows.extend(
                attempt_distribution_rows(
                    scenario=scenario_name,
                    replication=replication,
                    cohort_records=cohort_records,
                )
            )
            logger.info(
                "%s rep %s seed %s ended at %.3f: %s",
                scenario_name,
                replication,
                seed,
                diagnostics["simulation_end_time"],
                diagnostics["termination_reason"],
            )

    replication_frame = pd.DataFrame(replication_rows)
    termination_frame = pd.DataFrame(termination_rows)
    caller_frame = pd.concat(caller_frames, ignore_index=True)
    attempt_frame = pd.DataFrame(attempt_rows)
    attempt_summary = summarize_attempt_distributions(attempt_frame)
    scenario_summary_frame = scenario_summary(replication_frame)
    comparison_frame = baseline_comparison(scenario_summary_frame)

    replication_frame.to_csv(paths["tables"] / "replication_metrics.csv", index=False)
    termination_frame.to_csv(paths["tables"] / "termination_diagnostics.csv", index=False)
    caller_frame.to_csv(paths["tables"] / "cohort_caller_records.csv", index=False)
    attempt_frame.to_csv(
        paths["tables"] / "attempt_distribution_by_replication.csv", index=False
    )
    attempt_summary.to_csv(
        paths["tables"] / "attempt_distribution_summary.csv", index=False
    )
    scenario_summary_frame.to_csv(paths["tables"] / "scenario_summary.csv", index=False)
    comparison_frame.to_csv(
        paths["tables"] / "baseline_policy_comparison.csv", index=False
    )

    cutoff = args.plot_attempt_cutoff
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="all",
        output_path=paths["plots"] / "attempt_distribution_all.png",
        cutoff=cutoff,
    )
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="all",
        output_path=paths["plots"] / "attempt_distribution_all_logy.png",
        cutoff=cutoff,
        log_scale=True,
    )
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="completed",
        output_path=paths["plots"] / "attempt_distribution_completed.png",
        cutoff=cutoff,
    )
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="completed",
        output_path=paths["plots"] / "attempt_distribution_completed_logy.png",
        cutoff=cutoff,
        log_scale=True,
    )
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="left_without_enrollment",
        output_path=paths["plots"] / "attempt_distribution_left.png",
        cutoff=cutoff,
    )
    plot_attempt_distribution(
        attempt_summary,
        outcome_group="left_without_enrollment",
        output_path=paths["plots"] / "attempt_distribution_left_logy.png",
        cutoff=cutoff,
        log_scale=True,
    )
    plot_metric(
        scenario_summary_frame,
        "completion_rate",
        paths["plots"] / "completion_rate_comparison.png",
        "Completion rate by scenario",
    )
    plot_metric(
        scenario_summary_frame,
        "mean_attempt_count_all",
        paths["plots"] / "mean_attempt_count_comparison.png",
        "Mean attempt count by scenario",
    )
    plot_metric(
        scenario_summary_frame,
        "cohort_clearance_time",
        paths["plots"] / "cohort_clearance_time_comparison.png",
        "Cohort clearance time by scenario",
    )

    write_validation_summary(paths["validation"] / "validation_summary.md", termination_frame)
    write_readme(
        paths["root"] / "README.md",
        args=args,
        row=row,
        scenario_summary_frame=scenario_summary_frame,
        comparison_frame=comparison_frame,
        termination_frame=termination_frame,
    )
    logger.info("Analysis complete: %s", paths["root"])
    return {
        "replication_metrics": replication_frame,
        "termination_diagnostics": termination_frame,
        "scenario_summary": scenario_summary_frame,
        "comparison": comparison_frame,
    }


def write_validation_summary(path: Path, termination_frame: pd.DataFrame) -> None:
    """Write validation notes for termination behavior."""

    success = termination_frame.groupby("scenario_name")["dynamic_horizon_success"].agg(
        ["sum", "count"]
    )
    lines = [
        "# Validation Summary",
        "",
        "Dynamic horizon success counts:",
        "",
        "```text",
        success.to_string(),
        "```",
        "",
        "Successful replications have zero unfinished cohort callers and stop after the post-clearance buffer.",
        "Safety-limited or zero-cohort runs are retained in diagnostics and excluded from scenario summaries.",
    ]
    path.write_text("\n".join(lines) + "\n")


def metric_mean(summary: pd.DataFrame, scenario: str, metric: str) -> float:
    """Return one scenario metric mean from the summary table."""

    rows = summary[
        (summary["scenario_name"] == scenario) & (summary["metric"] == metric)
    ]
    return float(rows.iloc[0]["mean"]) if len(rows) else float("nan")


def write_readme(
    path: Path,
    *,
    args: argparse.Namespace,
    row: dict[str, Any],
    scenario_summary_frame: pd.DataFrame,
    comparison_frame: pd.DataFrame,
    termination_frame: pd.DataFrame,
) -> None:
    """Write a Markdown research summary."""

    success_counts = termination_frame.groupby("scenario_name")[
        "dynamic_horizon_success"
    ].agg(["sum", "count"])
    baseline_completion = metric_mean(
        scenario_summary_frame, "baseline", "completion_rate"
    )
    double_completion = metric_mean(
        scenario_summary_frame, "double_staffing", "completion_rate"
    )
    half_completion = metric_mean(
        scenario_summary_frame, "half_service_time", "completion_rate"
    )
    baseline_attempts = metric_mean(
        scenario_summary_frame, "baseline", "mean_attempt_count_all"
    )
    double_attempts = metric_mean(
        scenario_summary_frame, "double_staffing", "mean_attempt_count_all"
    )
    half_attempts = metric_mean(
        scenario_summary_frame, "half_service_time", "mean_attempt_count_all"
    )
    command = (
        "python -m experiments.cohort_policy_analysis "
        f"--parameter-file \"{args.parameter_file}\" "
        f"--call-center {args.call_center} "
        f"--cohort-start {args.cohort_start:g} "
        f"--cohort-end {args.cohort_end:g} "
        f"--post-clearance-buffer {args.post_clearance_buffer:g} "
        f"--replications {args.replications} "
        f"--seed {args.seed} "
        f"--output-dir {args.output_dir}"
    )
    lines = [
        "# Cohort Policy Analysis",
        "",
        "## Objective",
        "Evaluate Call Center 2 policy scenarios using a fixed first-arrival cohort and dynamic simulation horizon.",
        "",
        "## Repository Modules Changed",
        "- `stochastic_simulation.py`: optional dynamic-horizon cohort stopping and richer caller-state outputs.",
        "- `experiments/cohort_policy_analysis.py`: reproducible policy-analysis runner.",
        "- `tests/test_stochastic_simulation.py` and `tests/test_cohort_policy_analysis.py`: validation coverage.",
        "",
        "## Parameter Source",
        f"Machine-readable source: `{args.parameter_file}`.",
        "",
        "## Baseline Call Center 2 Parameters",
        f"- lambda: {row['lambda']}",
        f"- staffing: {row['c']}",
        f"- observed AHT: {row.get('preset_AHT_min', np.nan)} minutes",
        f"- AHT used: {row['AHT_used_min']} minutes",
        f"- enrollment probability: {float(row['enroll_pct']) / 100.0:.2f}",
        f"- theta_A/theta_S/theta_L: {row['theta_A']}, {row['theta_S']}, {row['theta_L']}",
        f"- fixed delta_B/delta_S/delta_L/gamma: {FIXED_RATES['deltaB']:.8f}, {FIXED_RATES['deltaS']}, {FIXED_RATES['deltaL']:.8f}, {FIXED_RATES['gamma']:.8f}",
        "",
        "## Cohort Window",
        f"`{args.cohort_start:g} <= first_arrival_time < {args.cohort_end:g}`. Membership is fixed by first arrival and redials do not change it.",
        "",
        "## Dynamic Stopping Rule",
        "The simulation continues after the cohort window closes until every cohort caller either completes enrollment or permanently leaves without enrollment.",
        f"After the last cohort terminal outcome, it runs an additional {args.post_clearance_buffer:g} model days.",
        f"Safety horizon: {args.max_dynamic_horizon:g}; max events: {args.max_events}.",
        "",
        "## Replications And Seeds",
        f"Replications requested: {args.replications}. Matched replication seeds are reused across scenarios for common random numbers.",
        "",
        "## Scenario Definitions",
        "- baseline: CSV Call Center 2 staffing and AHT used.",
        "- double_staffing: staffing doubled, AHT unchanged.",
        "- half_service_time: staffing unchanged, AHT halved and service rates recomputed.",
        "- combined_capacity: staffing doubled and AHT halved.",
        "",
        "## Output Files",
        "- `tables/parameter_scenarios.csv`: scenario parameters.",
        "- `tables/replication_metrics.csv`: replication-level outcomes.",
        "- `tables/scenario_summary.csv`: means, standard errors, and 95% normal CIs.",
        "- `tables/baseline_policy_comparison.csv`: policy differences from baseline.",
        "- `tables/attempt_distribution_by_replication.csv`: raw per-replication distributions.",
        "- `tables/attempt_distribution_summary.csv`: distribution summaries with zero-filled missing categories.",
        "- `tables/termination_diagnostics.csv`: dynamic stopping diagnostics.",
        "- `tables/cohort_caller_records.csv`: individual-level cohort caller states.",
        "- `plots/*.png`: required policy comparison plots.",
        "",
        "## Main Numerical Findings",
        f"- Baseline completion rate mean: {baseline_completion:.4f}; mean attempts all: {baseline_attempts:.4f}.",
        f"- Double staffing completion rate mean: {double_completion:.4f}; mean attempts all: {double_attempts:.4f}.",
        f"- Half service time completion rate mean: {half_completion:.4f}; mean attempts all: {half_attempts:.4f}.",
        "",
        "## Interpretation",
        "Policies with more service capacity reduce repeated attempts and left-without-enrollment rates in these generated results. Compare `baseline_policy_comparison.csv` for exact absolute and percent changes.",
        "",
        "## Termination Diagnostics",
        "```text",
        success_counts.to_string(),
        "```",
        "",
        "## Limitations",
        "- Normal-approximation confidence intervals are first-pass summaries.",
        "- The B recertification pool remains aggregate, so recertification calls start new caller episodes.",
        "- If fewer than 30 replications are run, the reason should be runtime management for this initial pass.",
        "",
        "## Reproduce",
        "```bash",
        command,
        "```",
    ]
    if args.replications < 30:
        lines.extend(
            [
                "",
                "## Replication Count Note",
                f"This initial run used {args.replications} replications rather than 30 to keep runtime practical while preserving at least 10 independent replications.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    """Command-line entry point."""

    run_analysis(parse_args())


if __name__ == "__main__":
    main()
