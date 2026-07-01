from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from compare_full_vs_light import CORE_METRICS
from compare_light_vs_fluid import build_comparison as build_light_vs_fluid
from compare_long_vs_many_replications import METRICS as LONG_VS_MANY_METRICS
from configs.paper_cc2_baseline import BASELINE, default_initial_state
from configs.paper_cc2_baseline import make_baseline_params
from configs.paper_cc2_baseline import PaperCC2Baseline
from fluid_steady_state import FluidSteadyStateResult, fluid_ode_rhs
from fluid_steady_state import solve_fluid_ode, solve_fluid_steady_state
from light_simulation import LightState, run_light_replications, simulate_light
from stochastic_simulation import SimulationParams, run_replications, simulate_one
from stochastic_simulation import compute_repeat_attempt_proxies, sample_event


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
    result = simulate_one(params, validate=True)

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


def test_light_simulator_runs_and_returns_finite_values():
    result = simulate_light(make_params(seed=55), validate=True)

    assert set(result.to_dict()) == {
        "mean_Q",
        "mean_B",
        "mean_RS",
        "mean_RL",
        "final_Q",
        "final_B",
        "final_RS",
        "final_RL",
    }
    for name in result.to_dict():
        value = getattr(result, name)
        assert isinstance(value, (int, float)), name
        assert np.isfinite(value), name


def test_light_simulator_preserves_nonnegative_states():
    params = make_params(c=3, lam=80.0, q0=20, b0=30, rs0=20, rl0=20, seed=99)
    path = simulate_light(params, record_path=True, record_dt=0.25, validate=True)

    assert path[["Q", "B", "RS", "RL"]].ge(0).all().all()
    waiting = (path["Q"] - params.c).clip(lower=0)
    in_service = path["Q"].clip(upper=params.c)
    assert waiting.ge(0).all()
    assert in_service.le(params.c).all()


def test_run_light_replications_returns_core_columns_only():
    frame = run_light_replications(make_params(seed=42), 3, validate=True)

    assert list(frame.columns) == [
        "replication",
        "seed",
        "mean_Q",
        "mean_B",
        "mean_RS",
        "mean_RL",
        "final_Q",
        "final_B",
        "final_RS",
        "final_RL",
    ]
    assert len(frame) == 3
    assert np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all()


def test_light_simulator_records_aggregate_path():
    params = make_params(T=2.0, warmup=0.5, seed=101)
    path = simulate_light(
        params,
        horizon=2.0,
        initial_state=LightState(0, 10, 2, 3),
        record_path=True,
        record_dt=0.5,
        validate=True,
    )

    assert list(path.columns) == ["t", "Q", "B", "RS", "RL"]
    assert path[["Q", "B", "RS", "RL"]].ge(0).all().all()


def test_event_sampling_stays_in_range():
    rng = np.random.default_rng(2026)
    rates = np.array([0.0, 2.0, 0.0, 3.0, 5.0])
    total_rate = float(rates.sum())

    for _ in range(1000):
        event = sample_event(rates, total_rate, rng)
        assert 0 <= event < len(rates)
        assert rates[event] > 0


def test_full_and_light_core_outputs_match_for_same_seed():
    params = make_params(
        T=8.0,
        warmup=2.0,
        c=4,
        lam=30.0,
        mu_plus=0.0,
        mu_minus=0.0,
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
        seed=2026,
    )

    full = simulate_one(params, validate=True)
    light = simulate_light(params, validate=True)

    for metric in ("mean_Q", "mean_B", "mean_RS", "mean_RL"):
        assert getattr(light, metric) == pytest.approx(getattr(full, metric))
    for metric in ("final_Q", "final_B", "final_RS", "final_RL"):
        assert getattr(light, metric) == getattr(full, metric)


def test_repeat_attempt_proxies_are_finite_and_handle_zero_denominators():
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
    proxies = compute_repeat_attempt_proxies(result)

    assert result.total_attempts_per_fresh_arrival == 0.0
    assert result.repeat_attempt_share == 0.0
    assert set(proxies).issuperset(
        {"total_attempts_per_fresh_arrival", "repeat_attempt_share"}
    )
    assert all(np.isfinite(value) for value in proxies.values())


def test_fluid_steady_state_returns_finite_nonnegative_values():
    params = make_params(
        T=160.0,
        warmup=80.0,
        c=4,
        lam=30.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=3.0,
        thetaS=2.0,
        thetaL=1.0,
        deltaB=0.2,
        deltaS=2.0,
        deltaL=0.5,
        gamma=0.7,
    )
    result = solve_fluid_steady_state(params)

    assert set(result.to_dict()) == {"q_bar", "b_bar", "rS_bar", "rL_bar"}
    assert isinstance(result, FluidSteadyStateResult)
    for name in ("q_bar", "b_bar", "rS_bar", "rL_bar"):
        value = getattr(result, name)
        assert np.isfinite(value), name
        assert value >= 0, name


def test_fluid_steady_state_returns_finite_values_for_cc2_baseline():
    result = solve_fluid_steady_state(make_baseline_params())

    for value in result.to_dict().values():
        assert np.isfinite(value)
        assert value >= 0


@pytest.mark.parametrize("field", ["thetaA", "gamma", "deltaS", "deltaL", "mu_plus"])
def test_fluid_steady_state_rejects_required_positive_zero_fields(field):
    params = make_params(**{field: 0.0})

    with pytest.raises(ValueError, match=field):
        solve_fluid_steady_state(params)


def test_fluid_underloaded_case_has_no_waiting_or_short_redial_overload():
    params = make_params(
        c=10,
        lam=5.0,
        mu_plus=6.0,
        thetaA=3.0,
        thetaS=2.0,
        thetaL=1.0,
        deltaB=0.2,
        deltaS=2.0,
        deltaL=0.5,
        gamma=0.7,
    )
    result = solve_fluid_steady_state(params)

    assert max(result.q_bar - params.c, 0.0) == pytest.approx(0.0)
    assert result.rS_bar == pytest.approx(0.0)


def test_fluid_ode_rhs_returns_finite_four_vector():
    params = make_params(thetaA=3.0, gamma=0.7, deltaS=2.0, deltaL=0.5, mu_plus=6.0)
    rhs = fluid_ode_rhs(0.0, [3.0, 10.0, 2.0, 4.0], params)

    assert rhs.shape == (4,)
    assert np.isfinite(rhs).all()


def test_solve_fluid_ode_returns_expected_columns():
    params = make_baseline_params()
    frame = solve_fluid_ode(params, 1.0, default_initial_state(params), [0.0, 0.5, 1.0])

    assert list(frame.columns) == ["t", "q", "b", "rS", "rL"]
    assert np.isfinite(frame.to_numpy()).all()
    assert frame[["q", "b", "rS", "rL"]].ge(0).all().all()


def test_solve_fluid_ode_rejects_invalid_max_step():
    params = make_baseline_params()

    with pytest.raises(ValueError, match="max_step must be positive"):
        solve_fluid_ode(
            params,
            1.0,
            default_initial_state(params),
            [0.0, 0.5, 1.0],
            max_step=0.0,
        )


def test_solve_fluid_ode_rejects_unsorted_t_eval():
    params = make_baseline_params()

    with pytest.raises(ValueError, match="t_eval must be sorted"):
        solve_fluid_ode(params, 1.0, default_initial_state(params), [0.0, 0.8, 0.5])


def test_solve_fluid_ode_baseline_short_horizon_is_finite_and_nonnegative():
    params = make_baseline_params(T=0.5, warmup=0.1)
    frame = solve_fluid_ode(
        params,
        0.5,
        default_initial_state(params),
        np.linspace(0.0, 0.5, 6),
        max_step=0.005,
    )

    assert np.isfinite(frame.to_numpy()).all()
    assert frame[["q", "b", "rS", "rL"]].ge(0).all().all()


def test_gamma_deltaB_scaling_preserves_ratio():
    base = make_baseline_params()
    scaled = make_baseline_params(gamma=base.gamma * 2, deltaB=base.deltaB * 2)

    base_ratio = base.gamma / (base.gamma + base.deltaB)
    scaled_ratio = scaled.gamma / (scaled.gamma + scaled.deltaB)
    assert scaled_ratio == pytest.approx(base_ratio)


def test_aht_conversion_sets_mu():
    config = PaperCC2Baseline(aht_minutes=21.9)

    assert config.mu == pytest.approx(BASELINE.model_day_minutes / 21.9)
    assert config.mu_plus + config.mu_minus == pytest.approx(config.mu)


def test_light_vs_fluid_comparison_only_uses_core_aggregate_rows():
    frame = build_light_vs_fluid(n_replications=3)

    assert frame["metric"].tolist() == ["Q", "B", "RS", "RL"]
    assert list(frame.columns) == [
        "metric",
        "light_mean",
        "light_std",
        "fluid_value",
        "difference",
        "percent_difference",
    ]


def test_long_vs_many_only_compares_core_time_average_metrics():
    assert LONG_VS_MANY_METRICS == ("mean_Q", "mean_B", "mean_RS", "mean_RL")


def test_full_vs_light_only_compares_core_shared_state_metrics():
    assert CORE_METRICS == (
        "mean_Q",
        "mean_B",
        "mean_RS",
        "mean_RL",
        "final_Q",
        "final_B",
        "final_RS",
        "final_RL",
    )
