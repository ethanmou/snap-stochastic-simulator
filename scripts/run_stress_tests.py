"""Run first-round stress tests around the CC2 baseline."""

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


OUTPUT_PATH = "stress_tests.csv"


def build_scenarios():
    """Yield a compact set of named stress-test scenarios.

    The full prompt listed a broad grid. For the weekly pass we keep one or two
    representative conditions per stress family so the experiment remains fast
    enough to rerun while still checking each structural direction.
    """

    base = make_baseline_params()
    for c in (1, 100):
        yield f"capacity_c_{c}", "c", c, replace(base, c=c)
    for multiplier in (0.1, 10):
        yield (
            f"lambda_x_{multiplier}",
            "lambda_multiplier",
            multiplier,
            replace(base, lam=base.lam * multiplier),
        )
    theta_scenarios = (
        (4, 3, 3),
        (1, 1, 9),
    )
    for thetaA, thetaS, thetaL in theta_scenarios:
        yield (
            f"theta_{thetaA}_{thetaS}_{thetaL}",
            "theta_tuple",
            f"{thetaA},{thetaS},{thetaL}",
            replace(base, thetaA=thetaA, thetaS=thetaS, thetaL=thetaL),
        )


def main():
    output_dir = ensure_output_dir()
    rows = [
        scenario_core_row(
            scenario_name,
            changed_parameter,
            changed_value,
            params,
            BASELINE.n_replications,
        )
        for scenario_name, changed_parameter, changed_value, params in build_scenarios()
    ]
    frame = pd.DataFrame(rows)
    output_path = output_dir / OUTPUT_PATH
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved stress tests to {output_path}")
    return frame


if __name__ == "__main__":
    main()
