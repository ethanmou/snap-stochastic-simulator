"""Streamlit dashboard for the SNAP call-center stochastic simulator."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.controls import sidebar_configuration, sweep_controls
from dashboard.comparison_analysis import build_comparison_table, build_parameter_change_table
from dashboard.export_utils import safe_filename, timestamp_token, zip_dataframes
from dashboard.fluid_analysis import (
    ANALYSIS_CACHE_VERSION,
    build_call_center_grouped_comparison,
    build_fluid_state_comparison,
)
from dashboard.plotting import (
    FITTING_LINE_OPTIONS,
    cdf_figure,
    fitting_overlay,
    grouped_fluid_simulation_comparison,
    metric_comparison,
    pmf_figure,
    queue_length_pmf,
    queue_length_time_series,
    residual_figure,
    survival_figure,
    sweep_distribution_fit_overlay,
    sweep_line,
    sweep_p_comparison,
    waiting_policy_scatter,
)
from dashboard.policy_analysis import (
    load_or_generate_policy_dataset,
    load_policy_dataset_if_available,
)
from dashboard.queue_analysis import (
    load_cached_queue_outputs,
    queue_length_distribution,
    save_cached_queue_outputs,
)
from dashboard.scenario_registry import (
    baseline_params,
    policy_scenarios_for_call_center,
)
from dashboard.sensitivity_analysis import (
    build_sweep_scenarios,
    finalize_parameter_sweep,
    sweep_values,
)
from dashboard.simulation_runner import (
    ScenarioRunOptions,
    add_distribution_and_fits,
    combine_run_results,
    relabel_run_result,
    run_scenarios,
)
from stochastic_simulation import SimulationParams


st.set_page_config(page_title="SNAP Call-Center Simulation Dashboard", layout="wide")
st.title("SNAP Call-Center Stochastic Simulation Dashboard")


def cache_key_payload(*payloads) -> str:
    """Return a stable JSON key for detecting stale interactive results."""

    return json.dumps(payloads, sort_keys=True, default=str)


def format_metric_value(value, *, precision: int = 4) -> str:
    """Format nullable scalar values for Streamlit metric cards."""

    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{precision}g}"
    except (TypeError, ValueError):
        return "n/a"


def streamlit_progress(label: str, target=st):
    """Return a progress callback created inside a cached function."""

    bar = target.progress(0.0, text=f"{label}: 0 complete")

    def update(done: int, total: int, message: str) -> None:
        remaining = max(total - done, 0)
        bar.progress(
            done / max(total, 1),
            text=f"{label}: {done}/{total} complete, {remaining} remaining - {message}",
        )

    return update


def session_cached(cache_name: str, key: str, factory):
    """Return a cached session result without replaying old Streamlit elements."""

    cache = st.session_state.setdefault(cache_name, {})
    if key in cache:
        return cache[key], True
    result = factory()
    cache[key] = result
    return result, False


def run_result_succeeded(result: dict[str, pd.DataFrame]) -> bool:
    """Return whether all replications in a run reached the intended horizon."""

    diagnostics = result.get("termination_diagnostics", pd.DataFrame())
    if diagnostics.empty or "dynamic_horizon_success" not in diagnostics.columns:
        return False
    return bool(diagnostics["dynamic_horizon_success"].fillna(False).all())


def max_event_reusable_options(options_dict: dict) -> dict:
    """Return options where max_events is ignored for successful-run reuse."""

    reusable = dict(options_dict)
    reusable.pop("max_events", None)
    return reusable


def run_single_dashboard(params_dict: dict, options_dict: dict, progress_target=st):
    params = SimulationParams(**params_dict)
    options = ScenarioRunOptions(**options_dict)
    return run_scenarios(
        {"current_custom": params},
        options,
        progress=streamlit_progress("Updating current custom scenario", target=progress_target),
    )


def enrich_single_dashboard(result: dict[str, pd.DataFrame], include_nb: bool) -> dict[str, pd.DataFrame]:
    return add_distribution_and_fits(
        result,
        populations=["all", "completed", "abandoned"],
        fit_negative_binomial_model=include_nb,
    )


def scenario_slug(label: str) -> str:
    """Return the dashboard's stable scenario identifier."""

    return label.lower().replace(" ", "_")


def run_compare_dashboard(
    params_dict: dict,
    options_dict: dict,
    base_label: str,
    selected: list[str],
    reusable_results: dict[str, dict[str, pd.DataFrame]] | None = None,
    progress_target=st,
):
    base = SimulationParams(**params_dict)
    options = ScenarioRunOptions(**options_dict)
    reusable_results = reusable_results or {}
    scenarios = {}
    if "Current Custom" in selected:
        scenarios[f"{scenario_slug(base_label)}_custom"] = base
    for label in ["Call Center 1", "Call Center 2", "Call Center 3", "Call Center 4"]:
        if label in selected:
            scenarios[scenario_slug(label)] = baseline_params(
                label,
                seed=options.base_seed,
                horizon=base.T,
                warmup=base.warmup,
            )
    policy_names = {"baseline", "double_staffing", "half_service_time", "combined_capacity"}
    requested_policies = policy_names.intersection(selected)
    if requested_policies:
        policies = policy_scenarios_for_call_center(base_label, seed=options.base_seed, horizon=base.T, warmup=base.warmup)
        for name in requested_policies:
            scenarios[f"{scenario_slug(base_label)}_{name}"] = policies[name]
    total_runs = max(len(scenarios) * options.replications, 1)
    progress_bar = progress_target.progress(
        0.0,
        text=f"Scenario comparison: 0/{total_runs} replication(s) processed",
    )

    def update_progress(done: int, message: str) -> None:
        remaining = max(total_runs - done, 0)
        progress_bar.progress(
            done / total_runs,
            text=f"Scenario comparison: {done}/{total_runs} processed, {remaining} remaining - {message}",
        )

    scenario_results = []
    for scenario_index, (scenario_name, scenario_params) in enumerate(scenarios.items()):
        scenario_start = scenario_index * options.replications
        scenario_key = cache_key_payload(
            scenario_name,
            asdict(scenario_params),
            asdict(options),
        )
        if scenario_name in reusable_results:
            result = reusable_results[scenario_name]
            st.session_state.setdefault("comparison_scenario_cache", {}).setdefault(
                scenario_key,
                result,
            )
            scenario_results.append(result)
            update_progress(
                scenario_start + options.replications,
                f"Reused existing result for {scenario_name}",
            )
            continue
        result, _from_cache = session_cached(
            "comparison_scenario_cache",
            scenario_key,
            lambda scenario_name=scenario_name, scenario_params=scenario_params: run_scenarios(
                {scenario_name: scenario_params},
                options,
                progress=lambda done, _total, message, scenario_start=scenario_start: update_progress(
                    scenario_start + done,
                    message,
                ),
            ),
        )
        scenario_results.append(result)
        if _from_cache:
            update_progress(
                scenario_start + options.replications,
                f"Loaded cached result for {scenario_name}",
            )
    return combine_run_results(scenario_results)


def enrich_compare_dashboard(result: dict[str, pd.DataFrame], include_nb: bool) -> dict[str, pd.DataFrame]:
    enriched = add_distribution_and_fits(
        result,
        populations=["all", "completed", "abandoned"],
        fit_negative_binomial_model=include_nb,
    )
    enriched["comparison_table"] = build_comparison_table(enriched)
    enriched["parameter_changes"] = build_parameter_change_table(enriched)
    return enriched


def run_sweep_dashboard(params_dict: dict, options_dict: dict, sweep_config: dict, progress_target=st):
    params = SimulationParams(**params_dict)
    options = ScenarioRunOptions(**options_dict)
    parameter = sweep_config["parameter"]
    values = sweep_values(
        minimum=sweep_config["minimum"],
        maximum=sweep_config["maximum"],
        grid_points=sweep_config["grid_points"],
        scale=sweep_config["scale"],
    )
    scenarios, sweep_configuration = build_sweep_scenarios(
        params,
        parameter=parameter,
        values=values,
    )
    total_runs = max(len(scenarios) * options.replications, 1)
    progress_bar = progress_target.progress(
        0.0,
        text=f"Parameter sweep: 0/{total_runs} replication(s) processed",
    )

    def update_progress(done: int, message: str) -> None:
        remaining = max(total_runs - done, 0)
        progress_bar.progress(
            done / total_runs,
            text=f"Parameter sweep: {done}/{total_runs} processed, {remaining} remaining - {message}",
        )

    scenario_results = []
    success_cache = st.session_state.setdefault("sweep_success_cache", {})
    for index, (scenario_name, scenario_params) in enumerate(scenarios.items()):
        scenario_start = index * options.replications
        scenario_options = (
            options
            if sweep_config["common_random_numbers"]
            else replace(options, base_seed=options.base_seed + index * options.seed_stride)
        )
        scenario_options_dict = asdict(scenario_options)
        reusable_key = cache_key_payload(
            scenario_name,
            asdict(scenario_params),
            max_event_reusable_options(scenario_options_dict),
        )
        success_entry = success_cache.get(reusable_key)
        if (
            success_entry is not None
            and int(success_entry["max_events"]) <= int(scenario_options.max_events)
        ):
            scenario_results.append(success_entry["result"])
            update_progress(
                scenario_start + options.replications,
                f"Reused successful result for {scenario_name}",
            )
            continue
        exact_key = cache_key_payload(
            scenario_name,
            asdict(scenario_params),
            scenario_options_dict,
        )
        result, _from_cache = session_cached(
            "sweep_scenario_cache",
            exact_key,
            lambda scenario_name=scenario_name, scenario_params=scenario_params, scenario_options=scenario_options: run_scenarios(
                {scenario_name: scenario_params},
                scenario_options,
                progress=lambda done, _total, message, scenario_start=scenario_start: update_progress(
                    scenario_start + done,
                    message,
                ),
            ),
        )
        if _from_cache:
            update_progress(
                scenario_start + options.replications,
                f"Loaded cached result for {scenario_name}",
            )
        if run_result_succeeded(result):
            success_cache[reusable_key] = {
                "max_events": int(scenario_options.max_events),
                "result": result,
            }
        scenario_results.append(result)
    combined = combine_run_results(scenario_results)
    combined["sweep_configuration"] = sweep_configuration
    return combined


def enrich_sweep_dashboard(
    result: dict[str, pd.DataFrame],
    sweep_config: dict,
    include_nb: bool,
) -> dict[str, pd.DataFrame]:
    return finalize_parameter_sweep(
        result,
        sweep_configuration=result["sweep_configuration"],
        parameter=sweep_config["parameter"],
        populations=sweep_config["populations"],
        fit_negative_binomial_model=include_nb,
    )


params, options, choices = sidebar_configuration()
include_nb = st.sidebar.toggle("Fit negative binomial", value=True)
if st.sidebar.button("Clear dashboard cache"):
    st.cache_data.clear()
    for key in (
        "single_cache",
        "single_enriched_cache",
        "comparison_cache",
        "comparison_enriched_cache",
        "comparison_scenario_cache",
        "sweep_cache",
        "sweep_enriched_cache",
        "sweep_scenario_cache",
        "sweep_success_cache",
        "comparison_result",
        "comparison_key",
        "sweep_result",
        "sweep_key",
    ):
        st.session_state.pop(key, None)
    st.sidebar.success("Dashboard cache cleared.")

if choices["validation_errors"]:
    st.error("\n".join(choices["validation_errors"]))
    st.stop()

st.sidebar.subheader("Current Setup")
st.sidebar.caption(
    f"{choices['base_label']} | c={params.c} | lam={params.lam:.4g} | "
    f"theta=({params.thetaA:.4g}, {params.thetaS:.4g}, {params.thetaL:.4g})"
)

tabs = st.tabs(
    [
        "Simulation Overview",
        "Queue Distribution",
        "Attempt Distribution",
        "Distribution Fitting",
        "Scenario Comparison",
        "Parameter Sweep",
        "Policy and Waiting-Time Analysis",
        "Data Export",
    ]
)

current_single_key = cache_key_payload(asdict(params), asdict(options))
with tabs[0]:
    current_custom_progress = st.empty()
single_raw_result, single_from_cache = session_cached(
    "single_cache",
    current_single_key,
    lambda: run_single_dashboard(
        asdict(params),
        asdict(options),
        current_custom_progress,
    ),
)
single_result, _single_enriched_from_cache = session_cached(
    "single_enriched_cache",
    # The simulation cache key above already includes every simulation option.
    # This enrichment key adds the analysis schema so new derived columns such
    # as p_flow cannot be silently omitted from older cached fitting tables.
    cache_key_payload(current_single_key, include_nb, ANALYSIS_CACHE_VERSION),
    lambda: enrich_single_dashboard(single_raw_result, include_nb),
)

caller_records = single_result["caller_records"]
replication_metrics = single_result["replication_metrics"]
scenario_summary = single_result["scenario_summary"]
distributions = single_result["attempt_distributions"]
fits = single_result["fitting_results"]
queue_samples_current = single_result.get("queue_samples", pd.DataFrame())
queue_distribution_current = (
    queue_length_distribution(queue_samples_current)
    if options.queue_distribution_enabled and not queue_samples_current.empty
    else pd.DataFrame()
)

with tabs[0]:
    st.subheader("Simulation Overview")
    with st.expander("Final parameters passed to simulation"):
        st.dataframe(choices["params_table"], use_container_width=True, hide_index=True)
    if replication_metrics.empty:
        st.warning("No replication metrics returned.")
    else:
        mean_row = replication_metrics.mean(numeric_only=True)
        cols = st.columns(4)
        metrics = [
            ("Cohort size", "cohort_size"),
            ("Completed", "completion_count"),
            ("Abandoned", "left_without_enrollment_count"),
            ("Unfinished", "unfinished_count"),
            ("Completion rate", "completion_rate"),
            ("Mean attempts", "mean_attempt_count_all"),
            ("P95 attempts", "p95_attempt_count_all"),
            ("Actual horizon", "simulation_end_time"),
        ]
        for index, (label, column) in enumerate(metrics):
            value = mean_row.get(column, float("nan"))
            cols[index % 4].metric(label, f"{value:.4g}" if pd.notna(value) else "n/a")
        st.dataframe(replication_metrics, use_container_width=True, hide_index=True)
        if scenario_summary.empty or "metric" not in scenario_summary.columns:
            st.warning(
                "Scenario-level summary is unavailable for the current run. "
                "This usually means no replication satisfied the dynamic-horizon success filter; "
                "check the termination diagnostics in Data Export or increase max_dynamic_horizon/max_events."
            )
        st.subheader("Fluid State Sanity Check")
        fluid_comparison, fluid_message = build_fluid_state_comparison(
            params, replication_metrics
        )
        if fluid_comparison.empty:
            st.info(f"Fluid state comparison unavailable: {fluid_message}")
        else:
            st.dataframe(
                fluid_comparison,
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Grouped Comparison Chart")
        grouped_label = (
            "Custom Parameters"
            if choices["preset"] == "Custom Parameters"
            else choices["base_label"]
        )
        grouped_metrics = replication_metrics.copy()
        grouped_metrics["scenario_name"] = grouped_label
        grouped_frame, grouped_message = build_call_center_grouped_comparison(
            {grouped_label: params},
            grouped_metrics,
        )
        if grouped_frame.empty:
            st.info(f"Grouped comparison unavailable: {grouped_message}")
        else:
            st.plotly_chart(
                grouped_fluid_simulation_comparison(grouped_frame),
                use_container_width=True,
            )
            with st.expander("Grouped comparison data"):
                st.dataframe(grouped_frame, use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Queue Distribution")
    if not options.queue_distribution_enabled:
        st.info(
            "Turn on `Record queue distribution` in the sidebar to collect fixed-time steady-state queue samples."
        )
    else:
        cached_queue, queue_cache_path = load_cached_queue_outputs(params, options)
        if cached_queue is not None:
            queue_samples = cached_queue["queue_samples"]
            queue_distribution = cached_queue["queue_distribution"]
            st.caption(f"Loaded saved queue-distribution data from `{queue_cache_path}`.")
        else:
            queue_samples = queue_samples_current
            queue_distribution = queue_distribution_current
            queue_cache_path = save_cached_queue_outputs(
                params,
                options,
                queue_samples,
                queue_distribution,
            )
            st.caption(f"Newly simulated queue-distribution data and saved it to `{queue_cache_path}`.")
        if queue_distribution.empty:
            st.warning("No queue samples are available for the current sampling settings.")
        else:
            queue_probability = st.radio(
                "Probability estimate",
                ["pooled_probability", "replication_mean_probability"],
                horizontal=True,
                key="queue_probability_estimate",
            )
            st.plotly_chart(
                queue_length_pmf(queue_distribution, probability_column=queue_probability),
                use_container_width=True,
            )
            queue_replications = sorted(queue_samples["replication"].dropna().unique().tolist())
            selected_queue_replications = st.multiselect(
                "Queue time-series replications",
                queue_replications,
                default=queue_replications,
                key="queue_time_series_replications",
            )
            queue_time_samples = queue_samples[queue_samples["replication"].isin(selected_queue_replications)]
            st.plotly_chart(queue_length_time_series(queue_time_samples), use_container_width=True)
            st.caption("`pooled_survival_probability` uses the inclusive convention P(Q >= q).")
            st.dataframe(queue_distribution, use_container_width=True, hide_index=True)
            st.download_button(
                "Download queue distribution CSV",
                data=queue_distribution.to_csv(index=False),
                file_name=f"{safe_filename(choices['base_label'])}_queue_distribution.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download raw queue samples CSV",
                data=queue_samples.to_csv(index=False),
                file_name=f"{safe_filename(choices['base_label'])}_queue_samples.csv",
                mime="text/csv",
            )
            with st.expander("Raw queue sample preview"):
                preview_rows = min(len(queue_samples), 1000)
                st.caption(f"Showing {preview_rows} of {len(queue_samples)} fixed-time queue samples.")
                st.dataframe(queue_samples.head(1000), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("Attempt Distribution")
    population = st.selectbox("Population", ["all", "completed", "abandoned"], key="dist_population")
    aggregation = st.radio("PMF aggregation", ["pooled_probability", "replication_mean_probability"], horizontal=True)
    frame = distributions[distributions["population"] == population].copy()
    st.plotly_chart(pmf_figure(frame, aggregation=aggregation, title=f"{population} empirical PMF"), use_container_width=True)
    st.plotly_chart(cdf_figure(frame, title=f"{population} empirical CDF"), use_container_width=True)
    st.plotly_chart(survival_figure(frame, title=f"{population} survival function"), use_container_width=True)
    st.dataframe(frame, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Distribution Fitting")
    population_fit = st.selectbox("Population", ["all", "completed", "abandoned"], key="fit_population")
    model_options = ["geometric", "negative_binomial"] if include_nb else ["geometric"]
    fit_model = st.selectbox("Model overlay", model_options, key="fit_model")
    geometric_subset = fits[(fits["population"] == population_fit) & (fits["model"] == "geometric")]
    geometric_row = (
        geometric_subset.iloc[0].to_dict()
        if not geometric_subset.empty and bool(geometric_subset.iloc[0].get("success", False))
        else {}
    )
    if geometric_row:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Geometric MLE p", format_metric_value(geometric_row.get("p_mle", geometric_row.get("p"))))
        metric_cols[1].metric("Empirical first-attempt probability", format_metric_value(geometric_row.get("p_first_attempt")))
        metric_cols[2].metric("Flow-based p", format_metric_value(geometric_row.get("p_flow")))
        metric_cols[3].metric("Stochastic-Q p", format_metric_value(geometric_row.get("p_stochastic_mean_q")))
        metric_cols[4].metric("MLE-stochastic-Q absolute error", format_metric_value(geometric_row.get("p_mle_stochastic_mean_q_absolute_error")))
        if pd.isna(geometric_row.get("p_flow", pd.NA)):
            st.caption(
                "Fluid-flow p unavailable: "
                f"{geometric_row.get('fluid_flow_status', 'unknown')} - "
                f"{geometric_row.get('fluid_flow_message', '')}"
            )
    with st.expander("Flow-based p details"):
        st.latex(
            r"""
            p_{\mathrm{flow}} =
            \frac{\theta_A (q_{\mathrm{bar}} - c)^+ + \mu_+ (q_{\mathrm{bar}} \wedge c)}
            {(\theta_A + \theta_S + \theta_L)(q_{\mathrm{bar}} - c)^+
             + (\mu_+ + \mu_-)(q_{\mathrm{bar}} \wedge c)}
            """
        )
        st.caption(
            "The fluid-flow value uses the paper steady-state q_bar. The stochastic-Q value uses the "
            "same flow formula after replacing q_bar with the current scenario's stochastic mean_Q. "
            "Neither value is fitted from the observed attempt-count records."
        )
        if geometric_row:
            flow_detail_fields = [
                ("q_bar", "fluid_q_bar"),
                ("waiting_quantity", "fluid_waiting_quantity"),
                ("service_quantity", "fluid_service_quantity"),
                ("final_abandonment_flow", "fluid_final_abandonment_flow"),
                ("short_abandonment_flow", "fluid_short_abandonment_flow"),
                ("long_abandonment_flow", "fluid_long_abandonment_flow"),
                ("service_success_flow", "fluid_service_success_flow"),
                ("service_failure_flow", "fluid_service_failure_flow"),
                ("terminal_flow", "fluid_terminal_flow"),
                ("total_attempt_ending_flow", "fluid_total_attempt_ending_flow"),
                ("p_flow", "p_flow"),
                ("stochastic_mean_Q", "stochastic_mean_q"),
                ("stochastic_Q_waiting_quantity", "stochastic_mean_q_waiting_quantity"),
                ("stochastic_Q_service_quantity", "stochastic_mean_q_service_quantity"),
                ("stochastic_Q_terminal_flow", "stochastic_mean_q_terminal_flow"),
                ("stochastic_Q_total_attempt_ending_flow", "stochastic_mean_q_total_attempt_ending_flow"),
                ("p_stochastic_mean_q", "p_stochastic_mean_q"),
                ("overloaded", "fluid_overloaded"),
                ("status", "fluid_flow_status"),
                ("message", "fluid_flow_message"),
            ]
            detail_frame = pd.DataFrame(
                [
                    {"quantity": label, "value": geometric_row.get(column, pd.NA)}
                    for label, column in flow_detail_fields
                ]
            )
            st.dataframe(detail_frame, use_container_width=True, hide_index=True)
        else:
            st.info("No geometric fit is available for this population.")
    fit_subset = fits[(fits["population"] == population_fit) & (fits["model"] == fit_model)]
    dist_subset = distributions[distributions["population"] == population_fit]
    selected_fitting_lines = st.multiselect(
        "Fitting lines",
        FITTING_LINE_OPTIONS,
        default=FITTING_LINE_OPTIONS,
        help="Select which fitted/reference lines to show. Empirical bars are always displayed.",
    )
    if fit_subset.empty or not bool(fit_subset.iloc[0].get("success", False)):
        st.warning("Fitting unavailable for this population/model.")
    else:
        fit_row = fit_subset.iloc[0].to_dict()
        st.plotly_chart(
            fitting_overlay(
                dist_subset,
                fit_row,
                title=f"{population_fit} {fit_model} fit",
                visible_lines=selected_fitting_lines,
            ),
            use_container_width=True,
        )
        st.plotly_chart(residual_figure(dist_subset, fit_row, title=f"{population_fit} {fit_model} residuals"), use_container_width=True)
    st.dataframe(fits, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Scenario Comparison")
    st.write("Compare call centers and existing capacity policy scenarios.")
    selected_scenarios = st.multiselect(
        "Scenarios",
        [
            "Current Custom",
            "Call Center 1",
            "Call Center 2",
            "Call Center 3",
            "Call Center 4",
            "baseline",
            "double_staffing",
            "half_service_time",
            "combined_capacity",
        ],
        default=["Current Custom", "Call Center 2", "Call Center 3", "Call Center 4"],
    )
    current_comparison_key = cache_key_payload(
        asdict(params),
        asdict(options),
        choices["base_label"],
        selected_scenarios,
    )
    if st.button("Run Scenario Comparison"):
        if not selected_scenarios:
            st.warning("Select at least one scenario.")
        else:
            comparison_progress = st.empty()
            reusable_results = {}
            if "Current Custom" in selected_scenarios:
                custom_scenario_name = f"{scenario_slug(choices['base_label'])}_custom"
                reusable_results[custom_scenario_name] = relabel_run_result(
                    single_raw_result,
                    custom_scenario_name,
                )
            st.session_state["comparison_result"], comparison_from_cache = session_cached(
                "comparison_cache",
                current_comparison_key,
                lambda: run_compare_dashboard(
                    asdict(params),
                    asdict(options),
                    choices["base_label"],
                    selected_scenarios,
                    reusable_results,
                    comparison_progress,
                ),
            )
            st.session_state["comparison_key"] = current_comparison_key
            if comparison_from_cache:
                comparison_progress.progress(
                    1.0,
                    text="Scenario comparison: loaded matching cached result.",
                )
                st.info("Loaded matching cached scenario-comparison result.")
    if (
        "comparison_result" in st.session_state
        and st.session_state.get("comparison_key") != current_comparison_key
    ):
        st.info(
            "Scenario comparison settings changed after the last run. "
            "Click `Run Scenario Comparison` again to recompute `Current Custom` with the current sidebar settings."
        )
    elif "comparison_result" in st.session_state:
        compare_result, _comparison_enriched_from_cache = session_cached(
            "comparison_enriched_cache",
            cache_key_payload(st.session_state["comparison_key"], include_nb, ANALYSIS_CACHE_VERSION),
            lambda: enrich_compare_dashboard(st.session_state["comparison_result"], include_nb),
        )
        st.session_state["comparison_enriched_result"] = compare_result
        comparison_dist = compare_result["attempt_distributions"]
        comparison_fit = compare_result["fitting_results"]
        comparison_metrics = compare_result["scenario_summary"]
        comparison_table = compare_result.get("comparison_table", build_comparison_table(compare_result))
        parameter_changes = compare_result.get("parameter_changes", build_parameter_change_table(compare_result))
        selected_population = st.selectbox("Comparison population", ["all", "completed", "abandoned"], key="compare_pop")
        subset = comparison_dist[comparison_dist["population"] == selected_population]
        table_subset = comparison_table[comparison_table["population"] == selected_population]
        scenario_count = subset["scenario_name"].nunique()
        if scenario_count > 5:
            st.warning("More than 5 scenarios are selected; charts may be crowded.")
        st.subheader("Changed parameters")
        if parameter_changes.empty:
            st.info("No parameter differences detected across the selected scenarios.")
        else:
            st.dataframe(parameter_changes, use_container_width=True, hide_index=True)
        st.plotly_chart(pmf_figure(subset, title=f"{selected_population} PMF by scenario"), use_container_width=True)
        st.plotly_chart(cdf_figure(subset, title=f"{selected_population} CDF by scenario"), use_container_width=True)
        st.subheader("Concrete comparison table")
        st.dataframe(table_subset, use_container_width=True, hide_index=True)
        with st.expander("Raw fitting results"):
            st.dataframe(comparison_fit, use_container_width=True, hide_index=True)
        with st.expander("Raw scenario metric summary"):
            st.dataframe(comparison_metrics, use_container_width=True, hide_index=True)

with tabs[5]:
    st.subheader("Parameter Sweep")
    sweep_config = sweep_controls(params)
    current_sweep_key = cache_key_payload(
        asdict(params),
        asdict(options),
        sweep_config,
    )
    runs = sweep_config["grid_points"] * options.replications
    st.info(f"Planned simulation runs: {runs}.")
    run_sweep_clicked = st.button("Run Parameter Sweep")
    sweep_is_stale = st.session_state.get("sweep_key") != current_sweep_key
    if run_sweep_clicked:
        sweep_progress = st.empty()
        st.session_state["sweep_result"], sweep_from_cache = session_cached(
            "sweep_cache",
            current_sweep_key,
            lambda: run_sweep_dashboard(
                asdict(params),
                asdict(options),
                sweep_config,
                sweep_progress,
            ),
        )
        st.session_state["sweep_key"] = current_sweep_key
        st.session_state["sweep_finalize_config"] = dict(sweep_config)
        if sweep_from_cache:
            sweep_progress.progress(
                1.0,
                text="Parameter sweep: loaded matching cached result.",
            )
            st.info("Loaded matching cached parameter-sweep result.")
    if (
        "sweep_result" in st.session_state
        and st.session_state.get("sweep_key") != current_sweep_key
    ):
        st.info(
            "Sweep settings changed after the last run. Showing the previous sweep result until "
            "you click `Run Parameter Sweep` again with the current sidebar and sweep settings."
        )
    if "sweep_result" in st.session_state:
        sweep_result, _sweep_enriched_from_cache = session_cached(
            "sweep_enriched_cache",
            cache_key_payload(st.session_state["sweep_key"], include_nb, ANALYSIS_CACHE_VERSION),
            lambda: enrich_sweep_dashboard(
                st.session_state["sweep_result"],
                st.session_state.get("sweep_finalize_config", sweep_config),
                include_nb,
            ),
        )
        st.session_state["sweep_enriched_result"] = sweep_result
        sweep_table = sweep_result["sweep_results"]
        population_level_metrics = [
            "geometric_p",
            "p_first_attempt",
            "p_flow",
            "p_mle_flow_absolute_error",
            "pmf_rmse",
            "max_abs_cdf_difference",
        ]
        scenario_level_metrics = [
            "completion_rate",
            "left_without_enrollment_rate",
            "mean_attempt_count_all",
        ]
        y_options = [
            column
            for column in population_level_metrics + scenario_level_metrics
            if column in sweep_table.columns
        ]
        y_metric = st.selectbox("Sweep chart metric", y_options)
        if y_metric in scenario_level_metrics:
            chart_frame = sweep_table.drop_duplicates(
                subset=["scenario_name", "sweep_value"]
            )
            expected_points = (
                sweep_result["sweep_configuration"]["sweep_value"].nunique()
                if "sweep_configuration" in sweep_result
                else chart_frame["sweep_value"].nunique()
            )
            valid_points = chart_frame.dropna(subset=[y_metric])["sweep_value"].nunique()
            st.caption(
                f"`{y_metric}` is scenario-level, so it is plotted once per sweep value rather than once per population."
            )
            if valid_points < expected_points:
                st.warning(
                    f"Only {valid_points} of {expected_points} sweep values have a non-missing `{y_metric}`. "
                    "Check dynamic-horizon success and termination diagnostics for the missing points."
                )
            st.plotly_chart(sweep_line(chart_frame, y_metric, color=None), use_container_width=True)
        else:
            st.plotly_chart(sweep_line(sweep_table, y_metric), use_container_width=True)
        p_comparison_populations = (
            [value for value in ["all", "completed", "abandoned"] if value in set(sweep_table.get("population", pd.Series(dtype=str)).dropna())]
            if "population" in sweep_table.columns
            else []
        )
        if p_comparison_populations:
            p_comparison_population = st.selectbox(
                "p comparison population",
                p_comparison_populations,
                key="sweep_p_comparison_population",
            )
            st.plotly_chart(
                sweep_p_comparison(sweep_table, population=p_comparison_population),
                use_container_width=True,
            )
        st.subheader("Attempt distribution by sweep scenario")
        distribution_population = st.selectbox(
            "Distribution population",
            ["all", "completed", "abandoned"],
            key="sweep_distribution_population",
        )
        sweep_distribution = sweep_result["attempt_distributions"]
        sweep_fits = sweep_result["fitting_results"]
        available_distribution_scenarios = (
            sorted(sweep_distribution["scenario_name"].dropna().unique().tolist())
            if "scenario_name" in sweep_distribution.columns
            else []
        )
        selected_distribution_scenarios = st.multiselect(
            "Distribution scenarios",
            available_distribution_scenarios,
            default=available_distribution_scenarios,
            key="sweep_distribution_scenarios",
        )
        if not selected_distribution_scenarios:
            st.warning("Select at least one sweep scenario to draw the attempt distribution.")
        else:
            st.plotly_chart(
                sweep_distribution_fit_overlay(
                    sweep_distribution,
                    sweep_fits,
                    population=distribution_population,
                    scenarios=selected_distribution_scenarios,
                ),
                use_container_width=True,
            )
        st.dataframe(sweep_table, use_container_width=True, hide_index=True)
        if "termination_diagnostics" in sweep_result:
            with st.expander("Sweep termination diagnostics"):
                diagnostics = sweep_result["termination_diagnostics"]
                if not diagnostics.empty and "termination_reason" in diagnostics.columns:
                    reason_counts = (
                        diagnostics.groupby(["scenario_name", "termination_reason"])
                        .size()
                        .reset_index(name="replications")
                    )
                    st.dataframe(reason_counts, use_container_width=True, hide_index=True)
                st.dataframe(diagnostics, use_container_width=True, hide_index=True)

with tabs[6]:
    st.subheader("Policy and Waiting-Time Analysis")
    policy_base_params = baseline_params(
        choices["base_label"],
        seed=options.base_seed,
        horizon=params.T,
        warmup=params.warmup,
    )
    policy_dataset, policy_cache_path = load_policy_dataset_if_available(
        choices["base_label"],
        policy_base_params,
        options,
    )
    if policy_dataset is None:
        st.info(
            "Persistent policy-grid data has not been generated for the current call center and run settings."
        )
        policy_progress = st.empty()
        if st.button("Generate Persistent Policy Dataset"):
            policy_dataset, from_policy_cache, policy_cache_path = load_or_generate_policy_dataset(
                choices["base_label"],
                policy_base_params,
                options,
                progress=streamlit_progress("Policy grid", target=policy_progress),
            )
            if from_policy_cache:
                policy_progress.progress(1.0, text="Policy grid: loaded persistent cached dataset.")
            else:
                policy_progress.progress(1.0, text="Policy grid: generated and saved persistent dataset.")
    else:
        st.caption(f"Loaded persistent policy dataset from `{policy_cache_path}`.")

    if policy_dataset is not None:
        wait_table = policy_dataset["policy_wait_table"]
        wait_x_options = [
            "average_wait_including_abandonments_minutes",
            "average_speed_to_answer_minutes",
            "average_time_to_abandonment_minutes",
        ]
        wait_y_options = [
            "procedural_denial_rate",
            "completion_rate",
            "mean_attempt_count_all",
            "abandonment_fraction",
        ]
        wait_x_options = [column for column in wait_x_options if column in wait_table.columns]
        wait_y_options = [column for column in wait_y_options if column in wait_table.columns]
        if not wait_x_options or not wait_y_options:
            st.warning("Persistent policy-grid data is missing plottable metrics.")
        else:
            x_metric = st.selectbox("Waiting-time x metric", wait_x_options)
            y_metric = st.selectbox("Outcome y metric", wait_y_options)
            st.plotly_chart(waiting_policy_scatter(wait_table, x_metric, y_metric), use_container_width=True)
            st.dataframe(wait_table, use_container_width=True, hide_index=True)

with tabs[7]:
    st.subheader("Data Export")
    token = timestamp_token()
    export_name = safe_filename(f"{token}_{choices['base_label']}_single")
    frames = {
        "parameters": single_result["parameters"],
        "replication_metrics": replication_metrics,
        "caller_records": caller_records,
        "attempt_distributions": distributions,
        "fitting_results": fits,
        "termination_diagnostics": single_result["termination_diagnostics"],
        "queue_samples": queue_samples_current,
        "queue_distribution": queue_distribution_current,
    }
    st.download_button(
        "Download single-scenario results ZIP",
        data=zip_dataframes(
            frames,
            metadata={
                "options": asdict(options),
                "analysis_cache_version": ANALYSIS_CACHE_VERSION,
                "attempt_count_support": "N = simulation_attempt_count + 1",
            },
        ),
        file_name=f"{export_name}.zip",
        mime="application/zip",
    )
    for name, frame in frames.items():
        st.download_button(
            f"Download {name} CSV",
            data=frame.to_csv(index=False),
            file_name=f"{export_name}_{safe_filename(name)}.csv",
            mime="text/csv",
        )
    if (
        "comparison_enriched_result" in st.session_state
        and st.session_state.get("comparison_key") == current_comparison_key
    ):
        comparison_export_result = st.session_state["comparison_enriched_result"]
        compare_frames = {
            "parameter_changes": comparison_export_result.get(
                "parameter_changes",
                build_parameter_change_table(comparison_export_result),
            ),
            "comparison_table": comparison_export_result.get(
                "comparison_table",
                build_comparison_table(comparison_export_result),
            ),
            "comparison_metrics": comparison_export_result["scenario_summary"],
            "comparison_fitting": comparison_export_result["fitting_results"],
            "combined_distribution": comparison_export_result["attempt_distributions"],
            "queue_samples": comparison_export_result.get("queue_samples", pd.DataFrame()),
        }
        st.download_button(
            "Download scenario-comparison results ZIP",
            data=zip_dataframes(
                compare_frames,
                metadata={"options": asdict(options), "analysis_cache_version": ANALYSIS_CACHE_VERSION},
            ),
            file_name=f"{timestamp_token()}_scenario_comparison.zip",
            mime="application/zip",
        )
    if "sweep_enriched_result" in st.session_state:
        sweep_export_result = st.session_state["sweep_enriched_result"]
        sweep_frames = {
            "sweep_configuration": sweep_export_result["sweep_configuration"],
            "sweep_metrics": sweep_export_result["sweep_results"],
            "sweep_fitting": sweep_export_result["fitting_results"],
            "sweep_distributions": sweep_export_result["attempt_distributions"],
            "queue_samples": sweep_export_result.get("queue_samples", pd.DataFrame()),
        }
        st.download_button(
            "Download parameter-sweep results ZIP",
            data=zip_dataframes(
                sweep_frames,
                metadata={"options": asdict(options), "analysis_cache_version": ANALYSIS_CACHE_VERSION},
            ),
            file_name=f"{timestamp_token()}_parameter_sweep.zip",
            mime="application/zip",
        )
