import json

import numpy as np
import pandas as pd
import pytest

import dashboard.policy_analysis as policy_analysis
from dashboard.distribution_analysis import (
    attempt_distribution,
    population_frame,
    population_sample,
    population_summary,
)
from dashboard.comparison_analysis import build_comparison_table, build_parameter_change_table
from dashboard.distribution_fitting import (
    fit_geometric,
    fit_negative_binomial,
    geometric_pmf,
)
from dashboard.export_utils import safe_filename, zip_dataframes
from dashboard.fluid_analysis import build_call_center_grouped_comparison
from dashboard.fluid_analysis import build_fluid_state_comparison
from dashboard.fluid_analysis import compute_flow_based_geometric_p, compute_fluid_steady_state_q
from dashboard.plotting import (
    fitting_overlay,
    grouped_fluid_simulation_comparison,
    metric_comparison,
    queue_length_time_series,
    sweep_distribution_fit_overlay,
    sweep_p_comparison,
    waiting_policy_scatter,
)
from dashboard.policy_analysis import build_policy_grid, build_policy_wait_table
from dashboard.queue_analysis import (
    load_cached_queue_outputs,
    queue_length_distribution,
    save_cached_queue_outputs,
)
from dashboard.scenario_registry import (
    apply_parameter_overrides,
    baseline_params,
    params_to_display_frame,
    resolve_staffing_for_replication,
    set_parameter_for_sweep,
    validate_params_for_dashboard,
)
from dashboard.sensitivity_analysis import run_parameter_sweep, sweep_values
from dashboard.simulation_runner import ScenarioRunOptions, replication_seeds, run_scenarios
from dashboard.simulation_runner import add_distribution_and_fits
from dashboard.simulation_runner import combine_run_results
from dashboard.simulation_runner import relabel_run_result
from stochastic_simulation import SimulationParams


def caller_frame():
    return pd.DataFrame(
        [
            {"scenario_name": "toy", "replication": 0, "attempt_count": 0, "terminal_outcome": "completed"},
            {"scenario_name": "toy", "replication": 0, "attempt_count": 0, "terminal_outcome": "completed"},
            {"scenario_name": "toy", "replication": 0, "attempt_count": 1, "terminal_outcome": "left_without_enrollment"},
            {"scenario_name": "toy", "replication": 1, "attempt_count": 1, "terminal_outcome": "completed"},
            {"scenario_name": "toy", "replication": 1, "attempt_count": 2, "terminal_outcome": "left_without_enrollment"},
        ]
    )


def tiny_params(**overrides):
    values = {
        "T": 4.0,
        "warmup": 0.0,
        "c": 3,
        "lam": 5.0,
        "mu_plus": 30.0,
        "mu_minus": 0.0,
        "thetaA": 0.3,
        "thetaS": 0.2,
        "thetaL": 0.1,
        "deltaB": 0.0,
        "deltaS": 2.0,
        "deltaL": 1.0,
        "gamma": 0.0,
        "q0": 0,
        "b0": 0,
        "rs0": 0,
        "rl0": 0,
        "seed": 123,
    }
    values.update(overrides)
    return SimulationParams(**values)


def test_attempt_count_aggregation_pooled_and_mean_replication_pmf():
    distribution = attempt_distribution(caller_frame(), "all")
    attempt_1 = distribution[distribution["attempt_count"] == 1].iloc[0]
    attempt_2 = distribution[distribution["attempt_count"] == 2].iloc[0]

    assert attempt_1["simulation_attempt_count"] == 0
    assert attempt_1["pooled_count"] == 2
    assert attempt_1["pooled_probability"] == pytest.approx(2 / 5)
    assert attempt_1["replication_mean_probability"] == pytest.approx((2 / 3 + 0 / 2) / 2)
    assert attempt_2["replication_mean_probability"] == pytest.approx((1 / 3 + 1 / 2) / 2)


def test_attempt_distribution_zero_fills_missing_attempt_counts():
    distribution = attempt_distribution(caller_frame(), "all")
    attempt_3 = distribution[distribution["attempt_count"] == 3].iloc[0]

    assert attempt_3["pooled_count"] == 1
    assert attempt_3["number_of_replications_with_nonzero_count"] == 1
    assert attempt_3["number_of_valid_replications"] == 2


def test_completed_and_abandoned_population_filtering():
    frame = caller_frame()

    assert len(population_frame(frame, "completed")) == 3
    assert len(population_frame(frame, "abandoned")) == 2
    assert population_sample(frame, "completed").sample.tolist() == [1, 1, 2]
    assert population_summary(frame, "abandoned")["sample_size"] == 2


def test_geometric_mle_and_pmf_for_positive_support_sample():
    sample = [1, 1, 2, 2, 3, 3]
    fit = fit_geometric(sample, population="all", scenario_name="toy")

    assert fit["success"]
    assert fit["p"] == pytest.approx(1 / np.mean(sample))
    assert fit["p_mle"] == pytest.approx(fit["p"])
    assert fit["p_first_attempt"] == pytest.approx(2 / 6)
    assert geometric_pmf(np.array([1, 2]), fit["p"]).sum() < 1.0
    assert fit["pmf_rmse"] >= 0.0
    assert fit["max_abs_cdf_difference"] >= 0.0


def test_geometric_generated_sample_estimates_true_p():
    rng = np.random.default_rng(20260725)
    sample = rng.geometric(p=0.35, size=5000)
    fit = fit_geometric(sample)

    assert fit["success"]
    assert fit["p"] == pytest.approx(0.35, abs=0.02)


def test_flow_based_p_known_numeric_example():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )

    fluid = compute_fluid_steady_state_q(params)
    flow = compute_flow_based_geometric_p(params)

    assert fluid.finite
    assert fluid.q_bar == pytest.approx(40.0)
    assert fluid.waiting_quantity == pytest.approx(30.0)
    assert fluid.service_quantity == pytest.approx(10.0)
    assert flow.final_abandonment_flow == pytest.approx(60.0)
    assert flow.short_abandonment_flow == pytest.approx(150.0)
    assert flow.long_abandonment_flow == pytest.approx(150.0)
    assert flow.service_success_flow == pytest.approx(60.0)
    assert flow.service_failure_flow == pytest.approx(20.0)
    assert flow.terminal_flow == pytest.approx(120.0)
    assert flow.total_attempt_ending_flow == pytest.approx(440.0)
    assert flow.p_flow == pytest.approx(120.0 / 440.0)


def test_fluid_state_comparison_uses_only_core_aggregate_states():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )
    replication_metrics = pd.DataFrame(
        [
            {
                "mean_Q": 41.0,
                "mean_B": 100.0,
                "mean_RS": 10.0,
                "mean_RL": 90.0,
                "average_wait_including_abandonments_minutes": 5.0,
            },
            {
                "mean_Q": 39.0,
                "mean_B": 110.0,
                "mean_RS": 14.0,
                "mean_RL": 100.0,
                "average_wait_including_abandonments_minutes": 7.0,
            },
        ]
    )

    comparison, message = build_fluid_state_comparison(params, replication_metrics)

    assert message == "ok"
    assert comparison["metric"].tolist() == ["Q", "B", "RS", "RL", "Average wait (min)"]
    assert list(comparison.columns) == [
        "metric",
        "stochastic_mean",
        "stochastic_std",
        "fluid_value",
        "difference",
        "percent_difference",
    ]
    q_row = comparison[comparison["metric"] == "Q"].iloc[0]
    assert q_row["stochastic_mean"] == pytest.approx(40.0)
    assert q_row["fluid_value"] == pytest.approx(40.0)
    wait_row = comparison[comparison["metric"] == "Average wait (min)"].iloc[0]
    assert wait_row["stochastic_mean"] == pytest.approx(6.0)
    assert wait_row["fluid_value"] == pytest.approx(30.0 / 440.0 * 540.0)


def test_call_center_grouped_comparison_contains_chart_metrics():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )
    replication_metrics = pd.DataFrame(
        [
            {
                "scenario_name": "Call Center X",
                "mean_Q": 41.0,
                "mean_RS": 10.0,
                "mean_RL": 90.0,
                "average_wait_including_abandonments_minutes": 5.0,
            },
            {
                "scenario_name": "Call Center X",
                "mean_Q": 39.0,
                "mean_RS": 14.0,
                "mean_RL": 100.0,
                "average_wait_including_abandonments_minutes": 7.0,
            },
        ]
    )

    grouped, message = build_call_center_grouped_comparison(
        {"Call Center X": params},
        replication_metrics,
    )

    assert message == "ok"
    assert set(grouped["metric"]) == {
        "Mean queue length",
        "Mean short-orbit size",
        "Average waiting time (min)",
    }
    assert set(grouped["estimate_source"]) == {"Simulation mean", "Fluid estimate"}
    q_sim = grouped[
        (grouped["metric"] == "Mean queue length")
        & (grouped["estimate_source"] == "Simulation mean")
    ].iloc[0]
    wait_fluid = grouped[
        (grouped["metric"] == "Average waiting time (min)")
        & (grouped["estimate_source"] == "Fluid estimate")
    ].iloc[0]
    assert q_sim["value"] == pytest.approx(40.0)
    assert wait_fluid["value"] == pytest.approx(30.0 / 440.0 * 540.0)


def test_flow_based_p_underloaded_reduces_to_service_success_fraction():
    params = tiny_params(
        c=100,
        lam=10.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )

    flow = compute_flow_based_geometric_p(params)

    assert flow.finite
    assert flow.waiting_quantity == pytest.approx(0.0)
    assert flow.service_quantity > 0
    assert flow.p_flow == pytest.approx(6.0 / 8.0)


def test_flow_based_p_thetaA_zero_with_positive_excess_is_unavailable():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=0.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )

    flow = compute_flow_based_geometric_p(params)

    assert not flow.finite
    assert np.isnan(flow.p_flow)
    assert flow.status == "undefined_no_finite_fluid_steady_state"


def test_flow_based_p_thetaL_zero_keeps_service_failure_in_denominator():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=0.0,
        deltaB=0.1,
        gamma=0.2,
    )

    flow = compute_flow_based_geometric_p(params)

    assert flow.long_abandonment_flow == pytest.approx(0.0)
    assert flow.service_failure_flow == pytest.approx(20.0)
    assert flow.total_attempt_ending_flow == pytest.approx(290.0)
    assert flow.p_flow == pytest.approx(120.0 / 290.0)


def test_flow_based_p_recomputes_for_relevant_parameter_changes():
    base = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )
    base_flow = compute_flow_based_geometric_p(base)

    assert compute_flow_based_geometric_p(tiny_params(**{**base.__dict__, "thetaS": 6.0})).p_flow != pytest.approx(base_flow.p_flow)
    assert compute_flow_based_geometric_p(tiny_params(**{**base.__dict__, "deltaB": 0.2})).q_bar != pytest.approx(base_flow.q_bar)
    assert compute_flow_based_geometric_p(tiny_params(**{**base.__dict__, "mu_minus": 3.0})).p_flow != pytest.approx(base_flow.p_flow)
    assert compute_flow_based_geometric_p(tiny_params(**{**base.__dict__, "c": 12})).q_bar != pytest.approx(base_flow.q_bar)


def test_negative_binomial_failure_handling_for_empty_sample():
    fit = fit_negative_binomial([])

    assert not fit["success"]
    assert fit["error"] == "empty sample"


def test_replication_seeds_are_deterministic():
    assert replication_seeds(10, 4) == [10, 11, 12, 13]


def test_call_center_1_stochastic_staffing_is_seed_deterministic():
    params = baseline_params("Call Center 1", seed=1, horizon=5.0, warmup=0.0)
    resolved_once = [
        resolve_staffing_for_replication(params, mode="stochastic_32_33", probability_33=0.5, seed=seed).c
        for seed in range(10, 20)
    ]
    resolved_twice = [
        resolve_staffing_for_replication(params, mode="stochastic_32_33", probability_33=0.5, seed=seed).c
        for seed in range(10, 20)
    ]

    assert resolved_once == resolved_twice
    assert set(resolved_once).issubset({32, 33})


def test_dashboard_runner_forwards_dynamic_horizon_diagnostics():
    options = ScenarioRunOptions(
        replications=1,
        base_seed=222,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=4.0,
        max_events=100_000,
    )
    result = run_scenarios({"tiny": tiny_params(T=4.0)}, options)

    assert "termination_diagnostics" in result
    assert result["termination_diagnostics"]["simulation_end_time"].notna().all()
    assert "caller_records" in result


def test_dashboard_runner_returns_fixed_time_queue_samples_when_enabled():
    options = ScenarioRunOptions(
        replications=2,
        base_seed=222,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        max_dynamic_horizon=4.0,
        max_events=100_000,
        queue_distribution_enabled=True,
        queue_warmup_time=1.0,
        queue_sample_interval=1.0,
        queue_observation_end=3.0,
    )
    result = run_scenarios({"tiny": tiny_params(T=4.0, warmup=0.0)}, options)

    samples = result["queue_samples"]
    assert {"scenario_name", "replication", "seed", "sample_time", "queue_length"}.issubset(samples.columns)
    assert samples["sample_time"].min() == pytest.approx(1.0)
    assert samples["sample_time"].max() == pytest.approx(3.0)
    assert set(samples["replication"]) == {0, 1}


def test_queue_length_distribution_zero_fills_support_and_uses_inclusive_survival():
    samples = pd.DataFrame(
        [
            {"replication": 0, "queue_length": 0},
            {"replication": 0, "queue_length": 2},
            {"replication": 0, "queue_length": 2},
            {"replication": 1, "queue_length": 0},
            {"replication": 1, "queue_length": 0},
            {"replication": 1, "queue_length": 3},
        ]
    )

    distribution = queue_length_distribution(samples)

    assert distribution["queue_length"].tolist() == [0, 1, 2, 3]
    q1 = distribution[distribution["queue_length"] == 1].iloc[0]
    q2 = distribution[distribution["queue_length"] == 2].iloc[0]
    assert q1["pooled_count"] == 0
    assert q2["pooled_probability"] == pytest.approx(2 / 6)
    assert q2["replication_mean_probability"] == pytest.approx((2 / 3 + 0 / 3) / 2)
    assert q2["number_of_replications_with_nonzero_count"] == 1
    assert q2["number_of_replications"] == 2
    assert q2["pooled_survival_probability"] == pytest.approx(3 / 6)


def test_queue_cache_round_trips_exact_sampling_settings(tmp_path, monkeypatch):
    import dashboard.queue_analysis as queue_analysis

    monkeypatch.setattr(queue_analysis, "QUEUE_CACHE_DIR", tmp_path)
    params = tiny_params()
    options = ScenarioRunOptions(
        replications=1,
        queue_distribution_enabled=True,
        queue_warmup_time=1.0,
        queue_sample_interval=0.5,
        queue_observation_end=2.0,
    )
    samples = pd.DataFrame(
        [{"scenario_name": "tiny", "replication": 0, "seed": 1, "sample_time": 1.0, "queue_length": 0}]
    )
    distribution = queue_length_distribution(samples)

    path = save_cached_queue_outputs(params, options, samples, distribution)
    loaded, loaded_path = load_cached_queue_outputs(params, options)
    changed, _ = load_cached_queue_outputs(
        params,
        ScenarioRunOptions(
            replications=1,
            queue_distribution_enabled=True,
            queue_warmup_time=1.0,
            queue_sample_interval=1.0,
            queue_observation_end=2.0,
        ),
    )

    assert loaded_path == path
    assert loaded is not None
    assert loaded["queue_distribution"].loc[0, "pooled_probability"] == pytest.approx(1.0)
    assert changed is None


def test_combine_run_results_recomputes_multi_scenario_summary():
    options = ScenarioRunOptions(
        replications=1,
        base_seed=222,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=4.0,
        max_events=100_000,
    )
    first = run_scenarios({"first": tiny_params(T=4.0, lam=2.0)}, options)
    second = run_scenarios({"second": tiny_params(T=4.0, lam=4.0)}, options)

    combined = combine_run_results([first, second])

    assert set(combined["parameters"]["scenario_name"]) == {"first", "second"}
    assert set(combined["replication_metrics"]["scenario_name"]) == {"first", "second"}
    assert set(combined["caller_records"]["scenario_name"]).issubset({"first", "second"})
    assert "metric" in combined["scenario_summary"].columns


def test_relabel_run_result_reuses_single_scenario_for_comparison_name():
    options = ScenarioRunOptions(
        replications=1,
        base_seed=222,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=4.0,
        max_events=100_000,
    )
    single = run_scenarios({"single_scenario": tiny_params(T=4.0)}, options)

    relabeled = relabel_run_result(single, "call_center_2_custom")

    assert set(relabeled["parameters"]["scenario_name"]) == {"call_center_2_custom"}
    assert set(relabeled["replication_metrics"]["scenario_name"]) == {"call_center_2_custom"}
    assert set(relabeled["caller_records"]["scenario_name"]).issubset({"call_center_2_custom"})
    assert set(relabeled["scenario_summary"]["scenario_name"]) == {"call_center_2_custom"}


def test_sweep_values_and_parameter_sweep_shape():
    options = ScenarioRunOptions(
        replications=1,
        base_seed=333,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=4.0,
        max_events=100_000,
    )
    values = sweep_values(minimum=2.0, maximum=4.0, grid_points=2, scale="linear")
    result = run_parameter_sweep(
        tiny_params(T=4.0),
        parameter="lam",
        values=values,
        options=options,
        populations=["all", "completed", "abandoned"],
    )

    assert set(result["sweep_configuration"]["sweep_value"]) == {2.0, 4.0}
    assert result["fitting_results"]["population"].isin(["all", "completed", "abandoned"]).all()
    assert "sweep_results" in result


def test_build_comparison_table_contains_concrete_scenario_population_metrics():
    options = ScenarioRunOptions(
        replications=1,
        base_seed=444,
        dynamic_horizon=True,
        cohort_start=0.1,
        cohort_end=0.5,
        post_clearance_buffer=0.1,
        max_dynamic_horizon=4.0,
        max_events=100_000,
    )
    run_result = run_scenarios({"tiny": tiny_params(T=4.0)}, options)
    enriched = add_distribution_and_fits(
        run_result,
        populations=["all", "completed", "abandoned"],
        fit_negative_binomial_model=True,
    )
    comparison = build_comparison_table(enriched)

    assert {"all", "completed", "abandoned"}.issubset(set(comparison["population"]))
    assert {
        "sample_size",
        "mean_attempts",
        "p90_attempts",
        "p95_attempts",
        "geometric_p",
        "p_mle",
        "p_first_attempt",
        "p_flow",
        "p_mle_flow_absolute_error",
        "fluid_q_bar",
        "fluid_terminal_flow",
        "fluid_total_attempt_ending_flow",
        "fluid_flow_status",
        "geometric_rmse",
        "geometric_cdf_difference",
        "geometric_aic",
        "negative_binomial_aic",
        "completion_rate",
        "left_without_enrollment_rate",
        "staffing",
        "arrival_rate_lam",
    }.issubset(comparison.columns)


def test_distribution_enrichment_adds_system_level_flow_p_to_geometric_rows():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )
    run_result = {
        "caller_records": pd.DataFrame(
            [
                {"scenario_name": "toy", "replication": 0, "attempt_count": 1, "terminal_outcome": "completed"},
                {"scenario_name": "toy", "replication": 0, "attempt_count": 2, "terminal_outcome": "completed"},
                {"scenario_name": "toy", "replication": 0, "attempt_count": 3, "terminal_outcome": "left_without_enrollment"},
            ]
        ),
        "parameters": pd.DataFrame([{**params.__dict__, "scenario_name": "toy"}]),
    }

    enriched = add_distribution_and_fits(run_result, populations=["all", "completed", "abandoned"])
    fits = enriched["fitting_results"]
    geometric = fits[fits["model"] == "geometric"]

    assert geometric["p_flow"].notna().all()
    assert geometric["p_flow"].nunique() == 1
    assert geometric.iloc[0]["p_flow"] == pytest.approx(120.0 / 440.0)


def test_distribution_enrichment_adds_stochastic_mean_q_flow_p():
    params = tiny_params(
        c=10,
        lam=100.0,
        mu_plus=6.0,
        mu_minus=2.0,
        thetaA=2.0,
        thetaS=5.0,
        thetaL=5.0,
        deltaB=0.1,
        gamma=0.2,
    )
    run_result = {
        "caller_records": pd.DataFrame(
            [
                {"scenario_name": "toy", "replication": 0, "attempt_count": 1, "terminal_outcome": "completed"},
                {"scenario_name": "toy", "replication": 0, "attempt_count": 2, "terminal_outcome": "completed"},
            ]
        ),
        "parameters": pd.DataFrame([{**params.__dict__, "scenario_name": "toy"}]),
        "replication_metrics": pd.DataFrame(
            [
                {"scenario_name": "toy", "replication": 0, "mean_Q": 48.0},
                {"scenario_name": "toy", "replication": 1, "mean_Q": 52.0},
            ]
        ),
    }

    enriched = add_distribution_and_fits(run_result, populations=["all"])
    fit = enriched["fitting_results"].iloc[0]

    assert fit["stochastic_mean_q"] == pytest.approx(50.0)
    assert fit["p_flow"] == pytest.approx(120.0 / 440.0)
    assert fit["p_stochastic_mean_q"] == pytest.approx(140.0 / 560.0)
    assert fit["p_mle_stochastic_mean_q_absolute_error"] == pytest.approx(
        abs(fit["p_mle"] - 140.0 / 560.0)
    )


def test_parameter_change_table_reports_changed_scenario_parameters():
    run_result = {
        "parameters": pd.DataFrame(
            [
                {
                    "scenario_name": "call_center_2",
                    "lam": 623.3,
                    "c": 52,
                    "thetaA": 4.0,
                },
                {
                    "scenario_name": "call_center_2_custom",
                    "lam": 925.0,
                    "c": 70,
                    "thetaA": 4.0,
                },
            ]
        )
    }

    changes = build_parameter_change_table(run_result)

    assert set(changes["parameter"]) == {"lam", "c"}
    assert "call_center_2" in changes.columns
    assert "call_center_2_custom" in changes.columns
    lam_change = changes[changes["parameter"] == "lam"].iloc[0]
    assert lam_change["absolute_change"] == pytest.approx(925.0 - 623.3)


def test_policy_grid_builds_twenty_fixed_capacity_scenarios():
    scenarios, config = build_policy_grid(tiny_params(c=10))

    assert len(scenarios) == 20
    assert len(config) == 20
    assert {"staffing_multiplier", "aht_multiplier", "staffing", "aht_minutes"}.issubset(config.columns)


def test_policy_wait_table_contains_relation_metrics_and_policy_parameters():
    run_result = {
        "replication_metrics": pd.DataFrame(
            [
                {
                    "scenario_name": "policy_a",
                    "average_wait_including_abandonments_minutes": 2.0,
                    "average_speed_to_answer_minutes": 1.0,
                    "average_time_to_abandonment_minutes": 3.0,
                    "left_without_enrollment_rate": 0.2,
                    "completion_rate": 0.8,
                    "mean_attempt_count_all": 1.5,
                    "abandonment_fraction": 0.2,
                    "replication": 1,
                }
            ]
        ),
        "fitting_results": pd.DataFrame(
            [
                {
                    "scenario_name": "policy_a",
                    "population": "abandoned",
                    "model": "geometric",
                    "success": True,
                    "p": 0.7,
                }
            ]
        ),
    }
    policy_parameters = pd.DataFrame(
        [
            {
                "scenario_name": "policy_a",
                "staffing": 12,
                "aht_minutes": 20.0,
            }
        ]
    )

    wait_table = build_policy_wait_table(run_result, policy_parameters)

    assert wait_table.loc[0, "geometric_p"] == pytest.approx(0.7)
    assert wait_table.loc[0, "staffing"] == 12
    assert wait_table.loc[0, "procedural_denial_rate"] == pytest.approx(0.2)
    assert "replication" not in wait_table.columns


def test_policy_dataset_loader_falls_back_to_compatible_persistent_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(policy_analysis, "POLICY_CACHE_DIR", tmp_path)
    cache_path = tmp_path / "previous_policy_grid"
    cache_path.mkdir()
    pd.DataFrame(
        [
            {
                "scenario_name": "staff_1.00_aht_1.00",
                "average_wait_including_abandonments_minutes": 2.0,
                "left_without_enrollment_rate": 0.2,
                "replication": 1,
            }
        ]
    ).to_csv(cache_path / "policy_wait_table.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario_name": "staff_1.00_aht_1.00",
                "staffing": 52,
                "aht_minutes": 20.0,
            }
        ]
    ).to_csv(cache_path / "policy_parameters.csv", index=False)
    metadata = {
        "version": policy_analysis.POLICY_CACHE_VERSION,
        "base_label": "Call Center 2",
        "options": {"replications": 99},
        "staffing_multipliers": policy_analysis.STAFFING_MULTIPLIERS,
        "aht_multipliers": policy_analysis.AHT_MULTIPLIERS,
    }
    (cache_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    dataset, loaded_path = policy_analysis.load_policy_dataset_if_available(
        "Call Center 2",
        tiny_params(seed=999),
        ScenarioRunOptions(replications=2, base_seed=111),
    )

    assert loaded_path == cache_path
    assert dataset is not None
    assert dataset["policy_wait_table"].loc[0, "left_without_enrollment_rate"] == pytest.approx(0.2)
    assert dataset["policy_wait_table"].loc[0, "procedural_denial_rate"] == pytest.approx(0.2)
    assert "replication" not in dataset["policy_wait_table"].columns


def test_set_parameter_for_sweep_abandonment_scale_and_aht():
    params = tiny_params(thetaA=1.0, thetaS=2.0, thetaL=3.0)
    scaled = set_parameter_for_sweep(params, "abandonment_scale", 2.0)
    faster = set_parameter_for_sweep(params, "aht_minutes", 10.0)

    assert scaled.thetaA == 2.0
    assert scaled.thetaS == 4.0
    assert scaled.thetaL == 6.0
    assert faster.mu_plus + faster.mu_minus == pytest.approx(54.0)


def test_export_schema_helpers():
    payload = zip_dataframes({"caller records": caller_frame()})

    assert safe_filename("bad / name.csv") == "bad_name.csv"
    assert payload.startswith(b"PK")


def test_metric_comparison_handles_empty_summary_without_metric_column():
    fig = metric_comparison(pd.DataFrame(), "completion_rate")

    assert "unavailable" in fig.layout.title.text


def test_queue_length_time_series_draws_replication_lines():
    samples = pd.DataFrame(
        [
            {"scenario_name": "tiny", "replication": 0, "seed": 1, "sample_time": 1.0, "queue_length": 2},
            {"scenario_name": "tiny", "replication": 0, "seed": 1, "sample_time": 2.0, "queue_length": 3},
            {"scenario_name": "tiny", "replication": 1, "seed": 2, "sample_time": 1.0, "queue_length": 1},
        ]
    )

    fig = queue_length_time_series(samples)

    assert fig.layout.title.text == "Queue length over time"
    assert {trace.name for trace in fig.data} == {"0", "1"}
    assert all(trace.mode == "lines" for trace in fig.data)


def test_grouped_fluid_simulation_comparison_draws_grouped_bars():
    frame = pd.DataFrame(
        [
            {
                "call_center": "Call Center 1",
                "metric": "Mean queue length",
                "estimate_source": "Simulation mean",
                "value": 10.0,
            },
            {
                "call_center": "Call Center 1",
                "metric": "Mean queue length",
                "estimate_source": "Fluid estimate",
                "value": 8.0,
            },
        ]
    )

    fig = grouped_fluid_simulation_comparison(frame)

    assert {trace.name for trace in fig.data} == {"Simulation mean", "Fluid estimate"}
    assert fig.layout.barmode == "group"


def test_waiting_policy_scatter_adds_fitted_relation_line():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "fast",
                "average_wait_including_abandonments_minutes": 1.0,
                "left_without_enrollment_rate": 0.1,
            },
            {
                "scenario_name": "medium",
                "average_wait_including_abandonments_minutes": 3.0,
                "left_without_enrollment_rate": 0.2,
            },
            {
                "scenario_name": "slow",
                "average_wait_including_abandonments_minutes": 5.0,
                "left_without_enrollment_rate": 0.4,
            },
        ]
    )

    fig = waiting_policy_scatter(
        frame,
        "average_wait_including_abandonments_minutes",
        "left_without_enrollment_rate",
    )

    assert any(trace.name == "Fitted relation" and trace.mode == "lines" for trace in fig.data)
    assert any("y-intercept" in annotation.text for annotation in fig.layout.annotations)
    assert all(getattr(trace, "text", None) is None for trace in fig.data if trace.mode == "markers")


def test_sweep_distribution_fit_overlay_draws_empirical_and_geometric_traces():
    distribution = pd.DataFrame(
        [
            {
                "scenario_name": "lam=1",
                "population": "all",
                "attempt_count": 1,
                "pooled_probability": 0.6,
            },
            {
                "scenario_name": "lam=1",
                "population": "all",
                "attempt_count": 2,
                "pooled_probability": 0.4,
            },
        ]
    )
    fitting = pd.DataFrame(
        [
            {
                "scenario_name": "lam=1",
                "population": "all",
                "model": "geometric",
                "success": True,
                "p": 0.6,
            }
        ]
    )

    fig = sweep_distribution_fit_overlay(distribution, fitting, population="all")

    assert any(trace.type == "bar" and trace.name == "lam=1 empirical" for trace in fig.data)
    assert any(trace.type == "scatter" and trace.name == "lam=1 Geometric MLE" for trace in fig.data)


def test_fitting_overlay_draws_flow_based_geometric_curve_when_available():
    distribution = pd.DataFrame(
        [
            {"attempt_count": 1, "pooled_probability": 0.6},
            {"attempt_count": 2, "pooled_probability": 0.4},
        ]
    )
    fit_row = {
        "model": "geometric",
        "success": True,
        "p": 0.6,
        "p_flow": 0.5,
    }

    fig = fitting_overlay(distribution, fit_row)

    assert {trace.name for trace in fig.data} == {"Empirical", "Geometric MLE", "Fluid-flow p"}


def test_fitting_overlay_draws_stochastic_mean_q_curve_when_available():
    distribution = pd.DataFrame(
        [
            {"attempt_count": 1, "pooled_probability": 0.6},
            {"attempt_count": 2, "pooled_probability": 0.4},
        ]
    )
    fit_row = {
        "model": "geometric",
        "success": True,
        "p": 0.6,
        "p_flow": 0.5,
        "p_stochastic_mean_q": 0.45,
    }

    fig = fitting_overlay(distribution, fit_row)

    assert {trace.name for trace in fig.data} == {
        "Empirical",
        "Geometric MLE",
        "Fluid-flow p",
        "Stochastic-Q p",
    }


def test_fitting_overlay_respects_selected_fitting_lines():
    distribution = pd.DataFrame(
        [
            {"attempt_count": 1, "pooled_probability": 0.6},
            {"attempt_count": 2, "pooled_probability": 0.4},
        ]
    )
    fit_row = {
        "model": "geometric",
        "success": True,
        "p": 0.6,
        "p_flow": 0.5,
        "p_stochastic_mean_q": 0.45,
    }

    fig = fitting_overlay(
        distribution,
        fit_row,
        visible_lines=["Stochastic-Q p"],
    )

    assert {trace.name for trace in fig.data} == {"Empirical", "Stochastic-Q p"}


def test_sweep_p_comparison_draws_mle_flow_and_first_attempt_series():
    frame = pd.DataFrame(
        [
            {
                "scenario_name": "lam=1",
                "population": "all",
                "sweep_value": 1.0,
                "geometric_p": 0.6,
                "p_flow": 0.5,
                "p_first_attempt": 0.62,
            },
            {
                "scenario_name": "lam=2",
                "population": "all",
                "sweep_value": 2.0,
                "geometric_p": 0.4,
                "p_flow": 0.45,
                "p_first_attempt": 0.42,
            },
        ]
    )

    fig = sweep_p_comparison(frame, population="all")

    assert {trace.name for trace in fig.data} == {
        "Geometric MLE p",
        "Fluid-flow p",
        "First-attempt empirical probability",
    }


def test_sweep_distribution_fit_overlay_respects_selected_scenario_order():
    distribution = pd.DataFrame(
        [
            {
                "scenario_name": "lam=2",
                "population": "all",
                "attempt_count": 1,
                "pooled_probability": 0.4,
            },
            {
                "scenario_name": "lam=1",
                "population": "all",
                "attempt_count": 1,
                "pooled_probability": 0.6,
            },
        ]
    )
    fitting = pd.DataFrame(
        [
            {
                "scenario_name": "lam=2",
                "population": "all",
                "model": "geometric",
                "success": True,
                "p": 0.4,
            },
            {
                "scenario_name": "lam=1",
                "population": "all",
                "model": "geometric",
                "success": True,
                "p": 0.6,
            },
        ]
    )

    fig = sweep_distribution_fit_overlay(
        distribution,
        fitting,
        population="all",
        scenarios=["lam=1", "lam=2"],
    )

    assert fig.data[0].name == "lam=1 empirical"
    assert fig.data[1].name == "lam=1 Geometric MLE"
    assert fig.data[2].name == "lam=2 empirical"


def test_sinusoidal_arrival_overrides_and_display_visibility():
    params = apply_parameter_overrides(
        tiny_params(),
        {
            "arrival_process": "sinusoidal",
            "lambda0": 6.0,
            "arrival_amplitude": 0.25,
            "arrival_period": 2.0,
            "arrival_phase": 0.5,
        },
    )

    assert params.arrival_process == "sinusoidal"
    assert params.lambda0 == 6.0
    assert params.arrival_amplitude == 0.25
    assert validate_params_for_dashboard(params) == []
    assert "lambda0" in set(params_to_display_frame(params)["parameter"])


def test_constant_arrival_hides_inactive_sinusoidal_display_fields():
    frame = params_to_display_frame(tiny_params(arrival_process="constant"))

    assert "arrival_process" in set(frame["parameter"])
    assert "lambda0" not in set(frame["parameter"])
    assert "arrival_amplitude" not in set(frame["parameter"])
