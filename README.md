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
