from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from stochastic_simulation import SimulationParams, run_replications, simulate_one


def make_params(**overrides):
    values = {
        "T": 5.0,
        "warmup": 1.0,
        "c": 8,
        "lam": 20.0,
        "mu_plus": 8.0,
        "mu_minus": 2.0,
        "thetaA": 2.0,
        "thetaS": 1.5,
        "thetaL": 0.5,
        "deltaB": 0.1,
        "deltaS": 2.0,
        "deltaL": 0.2,
        "gamma": 0.05,
        "q0": 0,
        "b0": 10,
        "rs0": 2,
        "rl0": 3,
        "seed": 1234,
    }
    values.update(overrides)
    return SimulationParams(**values)


def test_no_arrival_case():
    result = simulate_one(
        make_params(
            lam=0.0,
            deltaB=0.0,
            deltaS=0.0,
            deltaL=0.0,
            gamma=0.0,
            q0=0,
            b0=0,
            rs0=0,
            rl0=0,
        )
    )

    assert result.total_arrivals == 0
    assert result.final_Q == 0
    assert result.mean_Q == 0.0
    assert result.utilization == 0.0
    assert result.abandonment_fraction == 0.0


def test_large_staffing_has_no_wait_or_abandonment():
    result = simulate_one(
        make_params(
            T=2.0,
            warmup=0.0,
            c=100,
            lam=10.0,
            thetaA=0.0,
            thetaS=0.0,
            thetaL=0.0,
            deltaB=0.0,
            deltaS=0.0,
            deltaL=0.0,
            gamma=0.0,
            b0=0,
            rs0=0,
            rl0=0,
        )
    )

    assert result.total_arrivals > 0
    assert result.total_abandonments == 0
    assert result.mean_waiting == 0.0
    assert result.average_speed_to_answer_minutes == 0.0
    assert result.max_in_service <= 100


def test_same_seed_is_reproducible():
    first = simulate_one(make_params(seed=777))
    second = simulate_one(make_params(seed=777))

    assert first == second


def test_states_remain_nonnegative_and_capacity_is_respected():
    params = make_params(c=3, lam=80.0, q0=20, b0=30, rs0=20, rl0=20, seed=9)
    result = simulate_one(params)

    assert result.min_Q >= 0
    assert result.min_B >= 0
    assert result.min_RS >= 0
    assert result.min_RL >= 0
    assert result.final_waiting >= 0
    assert result.final_in_service >= 0
    assert result.max_in_service <= params.c


def test_output_metrics_are_finite_numbers():
    result = simulate_one(make_params())

    for name, value in asdict(result).items():
        assert isinstance(value, (int, float)), name
        assert np.isfinite(value), name


def test_run_replications_returns_independent_summary_rows():
    frame = run_replications(make_params(seed=42), 3)

    assert isinstance(frame, pd.DataFrame)
    assert len(frame) == 3
    assert frame["replication"].tolist() == [0, 1, 2]
    assert frame["seed"].nunique() == 3
    assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()

    repeated = run_replications(make_params(seed=42), 3)
    pd.testing.assert_frame_equal(frame, repeated)


def test_invalid_parameters_are_rejected():
    with pytest.raises(ValueError):
        simulate_one(make_params(warmup=5.0))
    with pytest.raises(ValueError):
        simulate_one(make_params(c=-1))
