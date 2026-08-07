# SNAP Stochastic Simulator

Event-driven stochastic simulator for a SNAP call-center model. The simulator
runs continuous-time replications, tracks queue and redial dynamics, and returns
summary metrics as Python dataclasses or pandas DataFrames.

## Features

- Gillespie-style continuous-time event simulation
- Reproducible random seeds for single runs and independent replications
- Queue, service, abandonment, redial, and recertification state tracking
- Summary metrics for arrivals, waits, utilization, abandonment, and diagnostics
- Pytest coverage for invariants, reproducibility, and parameter validation

## Requirements

- Python 3.10 or newer
- numpy
- pandas
- matplotlib
- plotly
- scipy
- streamlit
- pytest

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run The Demo

```bash
python run_simulation_demo.py
```

## Run The Interactive Dashboard

Recommended command from the repository root:

```bash
streamlit run apps/streamlit_app.py
```

The historical root-level wrapper still works:

```bash
streamlit run streamlit_app.py
```

The dashboard keeps the stochastic engine in `stochastic_simulation.simulate_one`
as the single source of simulation behavior. It adds Streamlit controls for call
center presets, operational and behavioral parameters, single-scenario runs,
scenario comparison, one-dimensional parameter sweeps, policy/waiting-time
analysis, and CSV/ZIP exports.

Dashboard sections:

- `Simulation Overview`: replication-level metrics, cohort outcomes, final
  states, wait summaries, and actual dynamic horizon diagnostics.
- `Attempt Distribution`: empirical PMF, CDF, survival function, and the
  distribution table for all, completed, or abandoned callers.
- `Distribution Fitting`: geometric fitting with optional shifted negative
  binomial comparison and residual diagnostics.
- `Scenario Comparison`: overlays and tables for call-center/policy scenarios.
- `Parameter Sweep`: one-dimensional sensitivity experiments only; no 2D
  heatmaps are generated.
- `Policy and Waiting-Time Analysis`: scatter plots relating existing wait
  metrics to completion, abandonment, attempts, and fitted abandoned geometric
  parameters.
- `Data Export`: parameters, replication metrics, caller records,
  distributions, fitting results, and diagnostics.

Distribution aggregation is explicit. `pooled_probability[k]` pools callers
across replications before normalizing. `replication_mean_probability[k]`
normalizes inside each replication first, zero-fills missing attempt counts, and
then averages probabilities. `replication_mean_count[k]` is a mean count, not a
probability.

The simulator's caller-level `attempt_count` is a retry/redial count with
support `0, 1, ...`. Dashboard fitting uses `N = simulation_attempt_count + 1`
so the geometric model is `P(N = k) = p(1-p)^(k-1)` for `k = 1, 2, ...`; its
MLE is `p_hat = 1 / sample_mean`. The negative-binomial comparison is shifted
the same way, with `N = 1 + failures before r successes`. Discrete CDF
differences are reported as diagnostics, not continuous KS p-values.

Parameter sweeps support `lam`, `c`, `aht_minutes`, `mu_plus`, `mu_minus`,
`thetaA`, `thetaS`, `thetaL`, `abandonment_scale`, and
`enroll_probability`. Built-in presets cover abandonment absolute scale,
short/long/final abandonment sensitivity, service-completion sensitivity,
staffing sensitivity, service-time sensitivity, and arrival-rate sensitivity.
Seeds are deterministic by default: replication `r` uses `base_seed + r`, and
common random numbers reuse that sequence across scenarios or sweep values.

## Project Layout

Core simulator modules stay at the repository root because scripts and tests
import them directly:

- `stochastic_simulation.py`: full caller-level stochastic simulator
- `light_simulation.py`: minimal aggregate Gillespie simulator
- `fluid_steady_state.py`: fluid steady-state and ODE helpers

Supporting code is grouped by purpose:

- `apps/`: Streamlit dashboard entry point
- `dashboard/`: dashboard controls, plotting, exports, and analysis helpers
- `configs/`: reusable parameter presets
- `data/`: tracked input data and parameter samples
- `docs/`: research and validation notes
- `experiments/`: research experiment scripts
- `scripts/`: weekly research runners and validation scripts
- `tests/`: pytest test suite
- `outputs/`: generated tables, plots, caches, and reports

Run validation scripts directly with module paths, for example:

```bash
python -m scripts.validation.compare_light_vs_fluid
python -m scripts.validation.compare_full_vs_light
python -m scripts.validation.compare_long_vs_many_replications
```

## Run Tests

```bash
python -m pytest
```

## Basic Usage

```python
from stochastic_simulation import SimulationParams, run_replications, simulate_one

params = SimulationParams(
    T=100.0,
    warmup=20.0,
    c=52,
    lam=623.3,
    mu_plus=13.5,
    mu_minus=13.5,
    thetaA=4.0,
    thetaS=3.0,
    thetaL=3.0,
    deltaB=1 / 130,
    deltaS=3.0,
    deltaL=1 / 9,
    gamma=1 / 260,
    seed=123,
)

result = simulate_one(params)
replications = run_replications(params, 10)
```

Time is represented in model days. Wait metrics are reported in minutes using
540 minutes per model day.
