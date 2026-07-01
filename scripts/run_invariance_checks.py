"""Run structural invariance checks for the fluid equations and light simulator."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_cc2_baseline import BASELINE, make_baseline_params
from scripts.research_utils import ensure_output_dir, scenario_core_row


CSV_PATH = "invariance_checks.csv"


def _scenario_rows():
    base = make_baseline_params()
    return [
        ("gamma_deltaB_scale", "scale", 1.0, base),
        (
            "gamma_deltaB_scale",
            "scale",
            2.0,
            replace(base, gamma=base.gamma * 2.0, deltaB=base.deltaB * 2.0),
        ),
    ]


def _add_ratios(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    baseline = frame.iloc[0]
    for metric in ("Q", "B", "RS", "RL"):
        frame[f"fluid_{metric}_ratio_to_baseline"] = (
            frame[f"fluid_{metric}"] / baseline[f"fluid_{metric}"]
        )
        frame[f"sim_{metric}_ratio_to_baseline"] = (
            frame[f"sim_{metric}_mean"] / baseline[f"sim_{metric}_mean"]
        )
    return frame


def main():
    output_dir = ensure_output_dir()
    rows = [
        scenario_core_row(name, parameter, value, params, BASELINE.n_replications)
        for name, parameter, value, params in _scenario_rows()
    ]
    frame = _add_ratios(pd.DataFrame(rows))
    csv_path = output_dir / CSV_PATH
    frame.to_csv(csv_path, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved invariance checks to {csv_path}")
    return frame


if __name__ == "__main__":
    main()
