"""Run first-pass staffing and AHT policy evaluations."""

from __future__ import annotations

from dataclasses import replace
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_cc2_baseline import BASELINE, default_initial_state, make_baseline_params
from fluid_steady_state import solve_fluid_steady_state
from light_simulation import run_light_replications
from scripts.research_utils import (
    aggregate_derived_metrics,
    ensure_output_dir,
    replace_aht,
)


CSV_PATH = "policy_evaluation.csv"


def _policy_scenarios():
    base = make_baseline_params()
    staffing_aht = (
        ("baseline", 52, 21.9),
        # One representative combined intervention for the weekly policy pass.
        ("increase_agent", 102, 21.9),
        ("decrease_aht", 52, 11.5),
        ("bundled", 102, 11.5),
    )
    for name, c, aht in staffing_aht:
        yield "staffing_aht", name, replace_aht(replace(base, c=c), aht), aht


def _scenario_row(policy_family: str, scenario_name: str, params, aht_minutes):
    initial_state = default_initial_state(params)
    light = run_light_replications(
        params,
        BASELINE.n_replications,
        horizon=params.T,
        initial_state=initial_state,
        validate=False,
    )
    fluid = solve_fluid_steady_state(params)
    sim = {
        "Q": float(light["mean_Q"].mean()),
        "B": float(light["mean_B"].mean()),
        "RS": float(light["mean_RS"].mean()),
        "RL": float(light["mean_RL"].mean()),
    }
    sim_std = {
        "Q": float(light["mean_Q"].std(ddof=1)),
        "B": float(light["mean_B"].std(ddof=1)),
        "RS": float(light["mean_RS"].std(ddof=1)),
        "RL": float(light["mean_RL"].std(ddof=1)),
    }
    fluid_values = {
        "Q": fluid.q_bar,
        "B": fluid.b_bar,
        "RS": fluid.rS_bar,
        "RL": fluid.rL_bar,
    }
    fluid_derived = aggregate_derived_metrics(
        params, fluid.q_bar, fluid.b_bar, fluid.rS_bar, fluid.rL_bar
    )
    sim_derived = aggregate_derived_metrics(
        params, sim["Q"], sim["B"], sim["RS"], sim["RL"]
    )
    row = {
        "policy_family": policy_family,
        "scenario_name": scenario_name,
        "c": params.c,
        "aht_minutes": aht_minutes,
    }
    for metric in ("Q", "B", "RS", "RL"):
        row[f"fluid_{metric}"] = fluid_values[metric]
        row[f"sim_{metric}_mean"] = sim[metric]
        row[f"sim_{metric}_std"] = sim_std[metric]
    for name, value in fluid_derived.items():
        row[f"fluid_{name}"] = value
    for name, value in sim_derived.items():
        row[f"sim_{name}"] = value
    return row


def _plot_policy_family(frame: pd.DataFrame, policy_family: str, metric: str, filename, output_dir):
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = frame[frame["policy_family"] == policy_family]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(subset["scenario_name"], subset[f"sim_{metric}"], marker="o", label="simulation")
    ax.plot(subset["scenario_name"], subset[f"fluid_{metric}"], marker="s", label="fluid")
    ax.set_title(f"{policy_family}: {metric}")
    ax.set_xlabel("scenario")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=150)
    plt.close(fig)


def main():
    output_dir = ensure_output_dir()
    rows = [
        _scenario_row(policy_family, scenario_name, params, aht)
        for policy_family, scenario_name, params, aht in _policy_scenarios()
    ]
    frame = pd.DataFrame(rows)
    csv_path = output_dir / CSV_PATH
    frame.to_csv(csv_path, index=False)

    _plot_policy_family(
        frame,
        "staffing_aht",
        "average_wait",
        "staffing_aht_vs_average_wait.png",
        output_dir,
    )
    _plot_policy_family(
        frame,
        "staffing_aht",
        "procedural_denial_rate",
        "staffing_aht_vs_procedural_denial_rate.png",
        output_dir,
    )
    print(frame.to_string(index=False))
    print(f"\nSaved policy evaluation to {csv_path}")
    return frame


if __name__ == "__main__":
    main()
