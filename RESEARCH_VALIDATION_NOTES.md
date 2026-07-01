# Research Validation Notes

## Full Stochastic Simulator

`stochastic_simulation.py` implements the detailed Gillespie simulator for the
SNAP call-center model. It tracks caller-level queue entry times for wait
metrics, service occupancy, enrolled beneficiaries `B`, short-redial population
`RS`, long-redial population `RL`, event counts, utilization, abandonment, and
procedural-denial flow metrics.

The core state is:

- `Q`: total callers in queue or service
- `B`: enrolled or recertification-eligible population
- `RS`: short-redial population
- `RL`: long-redial population
- `in_service`: callers currently being served in the full simulator

The full simulator now supports `validate=True`, which checks nonnegative
states, finite values, capacity constraints, and event-rate validity during the
event loop.

## Barebones Light Simulator

`light_simulation.py` implements the same continuous-time Markov chain using
only aggregate state variables. It avoids individual waiting-time records,
detailed event counts, and performance metrics. It uses:

- `waiting = max(Q - c, 0)`
- `in_service = min(Q, c)`

This makes the model transitions shorter and easier to inspect manually. It is
intended to validate aggregate CTMC behavior in the full simulator, not to
replace detailed waiting-time metrics.

The light simulator is intentionally minimal. It tracks only aggregate CTMC
state variables and diagnostics for invariant checks. It does not compute
caller-level waiting-time metrics, abandonment fraction, procedural denial rate,
average wait time, utilization, or other detailed performance metrics. Those
detailed metrics remain the responsibility of the full stochastic simulator.

## Full vs Light Comparison

Run:

```bash
.venv/bin/python compare_full_vs_light.py
```

This writes:

```text
outputs/full_vs_light_comparison.csv
```

Directly comparable metrics are:

- `mean_Q`
- `mean_B`
- `mean_RS`
- `mean_RL`
- `final_Q`
- `final_B`
- `final_RS`
- `final_RL`

Metrics that depend on caller-level wait histories, such as average speed to
answer or time to abandonment, are not directly comparable because the light
simulator intentionally does not retain per-caller queue entry records.

## Long Run vs Many Replications

Run:

```bash
.venv/bin/python compare_long_vs_many_replications.py
```

This writes:

```text
outputs/long_vs_many_replications_comparison.csv
```

The script compares many shorter independent replications with one longer
trajectory. It reports mean, standard deviation, one-long-run value, difference,
and percent difference for aggregate state time averages: `mean_Q`, `mean_B`,
`mean_RS`, and `mean_RL`.

The goal is not exact equality. The comparison is meant to show whether the
single long trajectory and independent-replication estimates are broadly
consistent up to simulation noise.

The long-vs-many replication comparison is a sanity check for aggregate
time-average state variables. It compares many short replications against one
long replication only on `mean_Q`, `mean_B`, `mean_RS`, and `mean_RL`. It
intentionally does not compare performance metrics such as utilization,
abandonment rates, service completion rates, or procedural denial rates.

## Light vs Fluid Comparison

Run:

```bash
.venv/bin/python compare_light_vs_fluid.py
```

This writes:

```text
outputs/light_vs_fluid_comparison.csv
```

The light-vs-fluid comparison focuses only on aggregate state variables. The
fluid model provides deterministic steady-state values for `q`, `b`, `rS`, and
`rL`. The light simulator provides stochastic time averages for `Q`, `B`, `RS`,
and `RL`. These are the directly comparable quantities.

## Repeat-Attempt Proxy Metrics

`compute_repeat_attempt_proxies(...)` computes aggregate approximations using
available event counts:

- `abandonments_per_fresh_arrival`
- `abandonments_per_external_arrival`
- `redial_arrivals_per_fresh_arrival`
- `redial_arrivals_per_external_arrival`
- `total_attempts_per_fresh_arrival`
- `total_attempts_per_external_arrival`
- `repeat_attempt_share`

These are proxy metrics, not exact caller-level histories. They do not identify
how many attempts a specific caller made, and they should not be interpreted as
the exact expected number of attempts per caller unless additional model
assumptions justify that interpretation.

## Fluid Benchmark Status

Fluid and steady-state benchmarking is now implemented in `fluid_steady_state.py`
using the Section 4.1 formulas from the paper. See `FLUID_BENCHMARK_NOTES.md`
for the equations, metric interpretation, and light-vs-fluid comparison command.

## Tests

Run:

```bash
.venv/bin/python -m pytest
```

The tests cover full-simulator invariants, light-simulator invariants, event
sampling bounds, full-vs-light aggregate consistency, repeat-attempt proxy edge
cases, and the fluid steady-state benchmark.
