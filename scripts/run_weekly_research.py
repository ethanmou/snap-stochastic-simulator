"""Run all weekly research validation experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_invariance_checks
from scripts import run_policy_evaluation
from scripts import run_sample_path_plots
from scripts import run_steady_state_comparison
from scripts import run_stress_tests
from scripts.research_utils import ensure_output_dir


def main():
    ensure_output_dir()
    run_steady_state_comparison.main()
    run_sample_path_plots.main()
    run_stress_tests.main()
    run_invariance_checks.main()
    run_policy_evaluation.main()
    print("Finished weekly research scripts.")


if __name__ == "__main__":
    main()
