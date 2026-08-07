"""Lightweight result model documentation for dashboard tables."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardSchemas:
    """Column groups returned by dashboard service functions."""

    replication_metrics: tuple[str, ...] = (
        "scenario_name",
        "replication",
        "seed",
        "cohort_size",
        "completion_count",
        "left_without_enrollment_count",
        "unfinished_count",
        "completion_rate",
        "mean_attempt_count_all",
        "simulation_end_time",
    )
    caller_records: tuple[str, ...] = (
        "scenario_name",
        "replication",
        "seed",
        "caller_id",
        "first_arrival_time",
        "attempt_count",
        "terminal_outcome",
        "time_in_system",
    )
    attempt_distribution: tuple[str, ...] = (
        "scenario_name",
        "population",
        "attempt_count",
        "simulation_attempt_count",
        "pooled_count",
        "pooled_probability",
        "replication_mean_probability",
    )
    fitting_results: tuple[str, ...] = (
        "scenario_name",
        "population",
        "model",
        "sample_size",
        "p",
        "p_mle",
        "p_first_attempt",
        "p_flow",
        "flow_based_p",
        "p_mle_flow_absolute_error",
        "p_stochastic_mean_q",
        "stochastic_mean_q_flow_based_p",
        "p_mle_stochastic_mean_q_absolute_error",
        "stochastic_mean_q",
        "fluid_q_bar",
        "fluid_waiting_quantity",
        "fluid_service_quantity",
        "fluid_terminal_flow",
        "fluid_total_attempt_ending_flow",
        "fluid_flow_status",
        "pmf_rmse",
        "max_abs_cdf_difference",
        "aic",
    )
