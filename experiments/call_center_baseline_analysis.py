"""Run baseline-only dynamic cohort analysis for selected call centers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.cohort_policy_analysis import (
    DEFAULT_OUTPUT_DIR,
    attempt_distribution_rows,
    load_call_center_parameters,
    params_from_row,
    plot_attempt_distribution,
    plot_metric,
    prepare_output_dirs,
    replication_metrics,
    scenario_parameter_table,
    scenario_summary,
    summarize_attempt_distributions,
)
from stochastic_simulation import simulate_one


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameter-file", default="call center parameters.csv")
    parser.add_argument("--call-centers", default="2,3,4")
    parser.add_argument("--cohort-start", type=float, default=260.0)
    parser.add_argument("--cohort-end", type=float, default=420.0)
    parser.add_argument("--post-clearance-buffer", type=float, default=20.0)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-dynamic-horizon", type=float, default=900.0)
    parser.add_argument("--max-events", type=int, default=5_000_000)
    parser.add_argument("--plot-attempt-cutoff", type=int, default=None)
    return parser.parse_args()


def configure_append_logger(log_path: Path) -> logging.Logger:
    """Configure an append-mode logger for the baseline call-center run."""

    logger = logging.getLogger("call_center_baseline_analysis")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_call_centers(raw: str) -> list[int]:
    """Parse a comma-separated call-center list."""

    centers = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not centers:
        raise ValueError("at least one call center must be provided")
    return centers


def run_baseline_call_center_analysis(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Run baseline-only scenarios for selected call centers."""

    paths = prepare_output_dirs(args.output_dir)
    os.environ.setdefault("MPLCONFIGDIR", str(paths["matplotlib"]))
    logger = configure_append_logger(paths["logs"] / "analysis.log")
    call_centers = parse_call_centers(args.call_centers)
    logger.info("Starting baseline-only call-center analysis for %s", call_centers)

    seed_sequence = np.random.SeedSequence(args.seed)
    replication_seeds = [
        int(seed.generate_state(1, dtype=np.uint64)[0])
        for seed in seed_sequence.spawn(args.replications)
    ]

    scenario_params = {}
    source_rows: dict[str, dict[str, Any]] = {}
    for call_center in call_centers:
        row = load_call_center_parameters(args.parameter_file, call_center)
        scenario_name = f"call_center_{call_center}"
        source_rows[scenario_name] = row
        scenario_params[scenario_name] = params_from_row(
            row,
            seed=args.seed,
            horizon=args.max_dynamic_horizon,
            warmup=args.cohort_start,
        )

    parameter_tables = []
    for scenario_name, params in scenario_params.items():
        table = scenario_parameter_table(
            {scenario_name: params}, source_rows[scenario_name]
        )
        parameter_tables.append(table)
    parameter_frame = pd.concat(parameter_tables, ignore_index=True)
    parameter_frame.to_csv(
        paths["tables"] / "call_center_baseline_parameter_scenarios.csv",
        index=False,
    )

    config = {
        "parameter_file": args.parameter_file,
        "call_centers": call_centers,
        "cohort_start": args.cohort_start,
        "cohort_end": args.cohort_end,
        "post_clearance_buffer": args.post_clearance_buffer,
        "replications": args.replications,
        "seed": args.seed,
        "seed_strategy": "matched replication seeds reused across call centers",
        "replication_seeds": replication_seeds,
        "max_dynamic_horizon": args.max_dynamic_horizon,
        "max_events": args.max_events,
    }
    (paths["root"] / "call_center_baseline_run_config.json").write_text(
        json.dumps(config, indent=2)
    )

    replication_rows = []
    termination_rows = []
    attempt_rows = []
    for scenario_name, base_params in scenario_params.items():
        logger.info("Running baseline scenario %s", scenario_name)
        for replication, seed in enumerate(replication_seeds):
            params = replace(base_params, seed=seed)
            result, extras = simulate_one(
                params,
                validate=False,
                return_caller_records=True,
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
            termination = diagnostics.to_dict()
            termination.update(
                {
                    "scenario_name": scenario_name,
                    "replication": replication,
                    "seed": seed,
                }
            )
            termination_rows.append(termination)
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
    attempt_frame = pd.DataFrame(attempt_rows)
    attempt_summary = summarize_attempt_distributions(attempt_frame)
    summary_frame = scenario_summary(replication_frame)

    replication_frame.to_csv(
        paths["tables"] / "call_center_baseline_replication_metrics.csv",
        index=False,
    )
    termination_frame.to_csv(
        paths["tables"] / "call_center_baseline_termination_diagnostics.csv",
        index=False,
    )
    attempt_frame.to_csv(
        paths["tables"] / "call_center_baseline_attempt_distribution_by_replication.csv",
        index=False,
    )
    attempt_summary.to_csv(
        paths["tables"] / "call_center_baseline_attempt_distribution_summary.csv",
        index=False,
    )
    summary_frame.to_csv(
        paths["tables"] / "call_center_baseline_scenario_summary.csv",
        index=False,
    )

    scenario_order = tuple(scenario_params)
    for outcome_group, filename in (
        ("all", "call_center_baseline_attempt_distribution_all.png"),
        ("completed", "call_center_baseline_attempt_distribution_completed.png"),
        ("left_without_enrollment", "call_center_baseline_attempt_distribution_left.png"),
    ):
        plot_attempt_distribution(
            attempt_summary,
            outcome_group=outcome_group,
            output_path=paths["plots"] / filename,
            cutoff=args.plot_attempt_cutoff,
            scenario_order=scenario_order,
        )
        plot_attempt_distribution(
            attempt_summary,
            outcome_group=outcome_group,
            output_path=paths["plots"] / filename.replace(".png", "_logy.png"),
            cutoff=args.plot_attempt_cutoff,
            scenario_order=scenario_order,
            log_scale=True,
        )

    plot_metric(
        summary_frame,
        "completion_rate",
        paths["plots"] / "call_center_baseline_completion_rate.png",
        "Baseline completion rate by call center",
        scenario_order=scenario_order,
    )
    plot_metric(
        summary_frame,
        "mean_attempt_count_all",
        paths["plots"] / "call_center_baseline_mean_attempt_count.png",
        "Baseline mean attempt count by call center",
        scenario_order=scenario_order,
    )
    plot_metric(
        summary_frame,
        "cohort_clearance_time",
        paths["plots"] / "call_center_baseline_cohort_clearance_time.png",
        "Baseline cohort clearance time by call center",
        scenario_order=scenario_order,
    )
    if "call_center_2" in scenario_params:
        plot_replication_attempt_distribution_heatmaps(
            attempt_frame,
            scenario_name="call_center_2",
            output_dir=paths["plots"],
            cutoff=args.plot_attempt_cutoff,
        )

    append_readme_section(paths["root"] / "README.md", summary_frame, termination_frame)
    logger.info("Baseline-only call-center analysis complete")
    return {
        "replication_metrics": replication_frame,
        "termination_diagnostics": termination_frame,
        "scenario_summary": summary_frame,
        "attempt_distribution_summary": attempt_summary,
    }


def metric_mean(summary: pd.DataFrame, scenario: str, metric: str) -> float:
    """Return a scenario metric mean from a summary table."""

    rows = summary[
        (summary["scenario_name"] == scenario) & (summary["metric"] == metric)
    ]
    return float(rows.iloc[0]["mean"]) if len(rows) else float("nan")


def plot_replication_attempt_distribution_heatmaps(
    attempt_frame: pd.DataFrame,
    *,
    scenario_name: str,
    output_dir: Path,
    outcome_group: str = "all",
    cutoff: int | None = None,
) -> None:
    """Plot per-replication attempt-count distributions for one baseline scenario."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = attempt_frame[
        (attempt_frame["scenario_name"] == scenario_name)
        & (attempt_frame["outcome_group"] == outcome_group)
    ].copy()
    if cutoff is not None:
        frame = frame[frame["attempt_count"] <= cutoff]
    if frame.empty:
        return

    replications = sorted(int(value) for value in frame["replication"].unique())
    attempts = sorted(int(value) for value in frame["attempt_count"].unique())
    count_matrix = pd.DataFrame(0.0, index=replications, columns=attempts)
    proportion_matrix = pd.DataFrame(0.0, index=replications, columns=attempts)
    for row in frame.itertuples(index=False):
        count_matrix.loc[int(row.replication), int(row.attempt_count)] = float(
            row.caller_count
        )
        proportion_matrix.loc[int(row.replication), int(row.attempt_count)] = float(
            row.proportion
        )

    _plot_attempt_heatmap(
        proportion_matrix,
        output_dir / "call_center_2_baseline_attempt_distribution_by_replication.png",
        title="Call Center 2 baseline attempt distribution by replication",
        colorbar_label="proportion within replication",
        value_format="{:.3f}",
        annotate_threshold=0.001,
    )
    _plot_attempt_heatmap(
        np.log10(count_matrix + 1.0),
        output_dir
        / "call_center_2_baseline_attempt_distribution_by_replication_log_count.png",
        title="Call Center 2 baseline attempt counts by replication",
        colorbar_label="log10(caller_count + 1)",
        value_format="{:.0f}",
        annotate_values=count_matrix,
        annotate_threshold=1.0,
    )


def _plot_attempt_heatmap(
    values: pd.DataFrame,
    output_path: Path,
    *,
    title: str,
    colorbar_label: str,
    value_format: str,
    annotate_threshold: float,
    annotate_values: pd.DataFrame | None = None,
) -> None:
    """Render a readable replication-by-attempt heatmap."""

    import matplotlib.pyplot as plt

    fig_width = max(14.0, 0.45 * len(values.columns))
    fig_height = max(6.0, 0.45 * len(values.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    ax.set_title(title)
    ax.set_xlabel("attempt count")
    ax.set_ylabel("replication")
    ax.set_xticks(np.arange(len(values.columns)))
    ax.set_xticklabels([str(value) for value in values.columns])
    ax.set_yticks(np.arange(len(values.index)))
    ax.set_yticklabels([str(value) for value in values.index])

    labels = annotate_values if annotate_values is not None else values
    for row_index, replication in enumerate(values.index):
        for column_index, attempt_count in enumerate(values.columns):
            label_value = float(labels.loc[replication, attempt_count])
            if label_value >= annotate_threshold:
                ax.text(
                    column_index,
                    row_index,
                    value_format.format(label_value),
                    ha="center",
                    va="center",
                    color="white" if values.iloc[row_index, column_index] > values.max().max() / 2 else "black",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def append_readme_section(
    readme_path: Path, summary_frame: pd.DataFrame, termination_frame: pd.DataFrame
) -> None:
    """Append a concise selected-call-center baseline summary to README.md."""

    start_marker = "<!-- call-center-baseline-start -->"
    end_marker = "<!-- call-center-baseline-end -->"
    scenario_names = sorted(summary_frame["scenario_name"].unique())
    call_center_label = ", ".join(name.replace("call_center_", "") for name in scenario_names)
    success = termination_frame.groupby("scenario_name")["dynamic_horizon_success"].agg(
        ["sum", "count"]
    )
    lines = [
        start_marker,
        "",
        "## Additional Baseline Call Center Results",
        "",
        f"Call Centers {call_center_label} were run with their CSV baseline parameters only; no policy changes were applied.",
        "",
        "```text",
        success.to_string(),
        "```",
        "",
        "Key means:",
    ]
    for scenario in scenario_names:
        lines.append(
            f"- {scenario}: completion_rate={metric_mean(summary_frame, scenario, 'completion_rate'):.4f}, "
            f"left_without_enrollment_rate={metric_mean(summary_frame, scenario, 'left_without_enrollment_rate'):.4f}, "
            f"mean_attempt_count_all={metric_mean(summary_frame, scenario, 'mean_attempt_count_all'):.4f}"
        )
    lines.append("")
    lines.append(
        "The call-center baseline attempt-distribution plots include larger annotated linear-scale versions and log-y versions so small tail probabilities remain visible."
    )
    lines.append(end_marker)
    section = "\n".join(lines) + "\n"
    existing = readme_path.read_text() if readme_path.exists() else ""
    if start_marker in existing and end_marker in existing:
        before = existing.split(start_marker, 1)[0].rstrip()
        after = existing.split(end_marker, 1)[1].lstrip()
        readme_path.write_text(f"{before}\n\n{section}\n{after}".rstrip() + "\n")
    else:
        with readme_path.open("a") as handle:
            handle.write("\n" + section)


def main() -> None:
    """Command-line entry point."""

    run_baseline_call_center_analysis(parse_args())


if __name__ == "__main__":
    main()
