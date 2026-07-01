"""Generate stochastic sample-path plots against the fluid ODE trajectory."""

from __future__ import annotations

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
from light_simulation import simulate_light
from scripts.research_utils import ensure_output_dir


PLOT_SPECS = {
    "Q": ("Q", "q", "sample_path_Q.png"),
    "B": ("B", "b", "sample_path_B.png"),
    "RS": ("RS", "rS", "sample_path_RS.png"),
    "RL": ("RL", "rL", "sample_path_RL.png"),
}


def main():
    output_dir = ensure_output_dir()
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = make_baseline_params()
    initial_state = zero_initial_state()
    t_eval = np.arange(0.0, params.T , 0.05)
    fluid = solve_fluid_ode(params, params.T, initial_state, t_eval)
    paths = [
        simulate_light(
            params,
            horizon=params.T,
            initial_state=initial_state,
            seed=BASELINE.seed + replication,
            record_path=True,
            record_dt=0.05,
            validate=False,
        )
        for replication in range(BASELINE.n_replications)
    ]
    mean_path = (
        pd.concat(paths, ignore_index=True)
        .groupby("t", as_index=False)[["Q", "B", "RS", "RL"]]
        .mean()
    )

    for label, (path_col, fluid_col, filename) in PLOT_SPECS.items():
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(
            mean_path["t"],
            mean_path[path_col],
            color="tab:blue",
            linewidth=2.0,
            label="stochastic mean",
        )
        ax.plot(fluid["t"], fluid[fluid_col], color="black", linewidth=2.0, label="fluid ODE")
        ax.set_title(f"{label}(t): stochastic mean vs fluid trajectory")
        ax.set_xlabel("model day")
        ax.set_ylabel(label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)
    print(f"Saved sample path plots to {output_dir}")


if __name__ == "__main__":
    main()
