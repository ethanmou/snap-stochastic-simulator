# Fluid Benchmark Notes

## Source And Scope

The implemented benchmark comes from Section 4.1 of "Due Process on Hold: A
Queueing Framework for Improving Access in SNAP." The fluid model is a
deterministic approximation to the stochastic queueing network. It is useful as
a first-order steady-state benchmark, not as an exact finite-horizon stochastic
prediction.

The ODE state variables are:

- `q`: total callers in the call-center system, waiting plus in service
- `b`: enrolled / beneficiary pool
- `rS`: short-redial orbit
- `rL`: long-redial orbit

## Implementation

`fluid_steady_state.py` implements:

- `solve_fluid_steady_state(params)`, which computes the closed-form
  steady-state formulas for `q_bar`, `b_bar`, `rS_bar`, and `rL_bar`
- `fluid_ode_rhs(t, y, params)`, which returns the Section 4.1 ODE right-hand
  side for future ODE simulation or validation

The comparison does not depend on a dashboard or plotting framework.

## Metric Interpretation

The fluid steady-state result is intentionally minimal. It answers only:

- What is the deterministic steady-state value of `q`?
- What is the deterministic steady-state value of `b`?
- What is the deterministic steady-state value of `rS`?
- What is the deterministic steady-state value of `rL`?

The light-vs-fluid comparison focuses only on aggregate state variables. The
fluid model provides deterministic steady-state values for `q`, `b`, `rS`, and
`rL`. The light simulator provides stochastic time averages for `Q`, `B`, `RS`,
and `RL`. These are the directly comparable quantities.

The fluid result does not expose detailed performance metrics such as
abandonment fraction, procedural denial rate, average wait time, or utilization.
Those metrics remain the responsibility of the full stochastic simulator.

## Run The Comparison

```bash
.venv/bin/python compare_light_vs_fluid.py
```

The script compares 100 light Gillespie replications against the deterministic
fluid steady state and writes:

```text
outputs/light_vs_fluid_comparison.csv
```

## Run Tests

```bash
.venv/bin/python -m pytest
```
