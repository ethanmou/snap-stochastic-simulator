import pandas as pd
import pytest

from experiments.cohort_policy_analysis import (
    attempt_distribution_rows,
    build_policy_scenarios,
    load_call_center_parameters,
    service_rates,
    summarize_attempt_distributions,
)
from stochastic_simulation import Caller, SimulationParams, simulate_one


def tiny_params(**overrides):
    values = {
        "T": 20.0,
        "warmup": 1.0,
        "c": 5,
        "lam": 2.0,
        "mu_plus": 20.0,
        "mu_minus": 0.0,
        "thetaA": 0.0,
        "thetaS": 0.0,
        "thetaL": 0.0,
        "deltaB": 0.0,
        "deltaS": 0.0,
        "deltaL": 0.0,
        "gamma": 0.0,
        "q0": 0,
        "b0": 0,
        "rs0": 0,
        "rl0": 0,
        "seed": 123,
    }
    values.update(overrides)
    return SimulationParams(**values)


def test_cohort_membership_boundary_convention():
    before = Caller(queue_entry_time=0.9, first_arrival_time=0.9)
    inside = Caller(queue_entry_time=1.0, first_arrival_time=1.0)
    at_end = Caller(queue_entry_time=2.0, first_arrival_time=2.0)

    assert not before.is_cohort_member(1.0, 2.0)
    assert inside.is_cohort_member(1.0, 2.0)
    assert not at_end.is_cohort_member(1.0, 2.0)


def test_redial_does_not_change_cohort_membership():
    caller = Caller(
        queue_entry_time=5.0,
        first_arrival_time=1.2,
        attempt_count=3,
        source_type="long_orbit_return",
    )

    assert caller.is_cohort_member(1.0, 2.0)


def test_dynamic_horizon_waits_until_cohort_window_closes_and_buffer_elapses():
    result, extras = simulate_one(
        tiny_params(seed=5),
        dynamic_horizon=True,
        cohort_start=1.0,
        cohort_end=2.0,
        post_clearance_buffer=0.5,
        max_dynamic_horizon=20.0,
        return_caller_records=True,
    )
    del result
    diag = extras["dynamic_horizon_diagnostics"].iloc[0]

    assert diag["simulation_end_time"] >= 2.0
    assert diag["dynamic_horizon_success"]
    assert diag["simulation_end_time"] == pytest.approx(
        diag["cohort_clearance_time"] + 0.5
    )


def test_dynamic_horizon_safety_limit_reports_unsuccessful_replication():
    result, extras = simulate_one(
        tiny_params(c=0, mu_plus=0.0, thetaA=0.0, seed=7),
        dynamic_horizon=True,
        cohort_start=0.5,
        cohort_end=1.0,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=2.0,
        max_events=1000,
        return_caller_records=True,
    )
    del result
    diag = extras["dynamic_horizon_diagnostics"].iloc[0]

    assert not diag["dynamic_horizon_success"]
    assert diag["termination_reason"] == "max_dynamic_horizon"
    assert diag["cohort_unfinished"] > 0


def test_dynamic_horizon_zero_cohort_is_explicit():
    result, extras = simulate_one(
        tiny_params(lam=0.0, seed=11),
        dynamic_horizon=True,
        cohort_start=1.0,
        cohort_end=2.0,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=3.0,
        return_caller_records=True,
    )
    del result
    diag = extras["dynamic_horizon_diagnostics"].iloc[0]

    assert diag["cohort_size"] == 0
    assert not diag["dynamic_horizon_success"]
    assert diag["termination_reason"] == "zero_cohort"


def test_attempt_distribution_proportions_sum_for_valid_groups():
    frame = pd.DataFrame(
        [
            {"attempt_count": 0, "terminal_outcome": "completed"},
            {"attempt_count": 1, "terminal_outcome": "completed"},
            {"attempt_count": 1, "terminal_outcome": "left_without_enrollment"},
        ]
    )
    rows = pd.DataFrame(
        attempt_distribution_rows(
            scenario="toy", replication=0, cohort_records=frame
        )
    )

    for _, subset in rows.groupby("outcome_group"):
        if subset["cohort_denominator"].iloc[0] > 0:
            assert subset["proportion"].sum() == pytest.approx(1.0)


def test_attempt_distribution_summary_zero_fills_missing_categories():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "toy",
                "replication": 0,
                "outcome_group": "all",
                "attempt_count": 0,
                "caller_count": 2,
                "cohort_denominator": 2,
                "proportion": 1.0,
            },
            {
                "scenario_name": "toy",
                "replication": 1,
                "outcome_group": "all",
                "attempt_count": 1,
                "caller_count": 2,
                "cohort_denominator": 2,
                "proportion": 1.0,
            },
        ]
    )
    summary = summarize_attempt_distributions(frame)
    attempt_0 = summary[summary["attempt_count"] == 0].iloc[0]
    attempt_1 = summary[summary["attempt_count"] == 1].iloc[0]

    assert attempt_0["mean_count"] == pytest.approx(1.0)
    assert attempt_1["mean_count"] == pytest.approx(1.0)


def test_policy_construction_changes_only_intended_capacity_parameters():
    row = load_call_center_parameters("call center parameters.csv", 2)
    scenarios = build_policy_scenarios(row, seed=1, horizon=10.0, warmup=1.0)

    assert scenarios["baseline"].c == 52
    assert scenarios["double_staffing"].c == 104
    assert scenarios["half_service_time"].c == 52
    assert scenarios["baseline"].lam == scenarios["double_staffing"].lam
    assert scenarios["baseline"].thetaA == scenarios["half_service_time"].thetaA


def test_half_service_time_recomputes_rates_from_aht_and_enrollment_probability():
    mu_plus, mu_minus = service_rates(10.0, 0.5)

    assert mu_plus == pytest.approx(27.0)
    assert mu_minus == pytest.approx(27.0)
