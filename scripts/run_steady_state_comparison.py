"""Run CC2 light-simulation vs fluid steady-state comparison."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.paper_cc2_baseline import BASELINE, make_baseline_params
from scripts.research_utils import ensure_output_dir, summarize_light_vs_fluid


OUTPUT_PATH = "steady_state_comparison.csv"


def main():
    output_dir = ensure_output_dir()
    params = make_baseline_params()
    frame = summarize_light_vs_fluid(params, BASELINE.n_replications)
    output_path = output_dir / OUTPUT_PATH
    frame.to_csv(output_path, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved steady-state comparison to {output_path}")
    return frame


if __name__ == "__main__":
    main()
