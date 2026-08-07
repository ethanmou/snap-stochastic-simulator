"""Persistent policy-grid data for waiting-time relationship analysis."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from stochastic_simulation import MINUTES_PER_MODEL_DAY, SimulationParams

from .scenario_registry import (
    apply_parameter_overrides,
    average_handling_time_minutes,
    enrollment_probability,
)
from .simulation_runner import ScenarioRunOptions, add_distribution_and_fits, run_scenarios


POLICY_CACHE_VERSION = 1
POLICY_CACHE_DIR = Path("outputs/dashboard_policy_cache")
STAFFING_MULTIPLIERS = [0.75, 0.9, 1.0, 1.1, 1.25]
AHT_MULTIPLIERS = [1.25, 1.0, 0.85, 0.7]


def _policy_cache_payload(base_label: str, base_params: SimulationParams, options: ScenarioRunOptions) -> dict:
    return {
        "version": POLICY_CACHE_VERSION,
        "base_label": base_label,
        "base_params": asdict(base_params),
        "options": asdict(options),
        "staffing_multipliers": STAFFING_MULTIPLIERS,
        "aht_multipliers": AHT_MULTIPLIERS,
    }


def policy_cache_key(base_label: str, base_params: SimulationParams, options: ScenarioRunOptions) -> str:
    """Return a stable key for one persistent policy dataset."""

    payload = _policy_cache_payload(base_label, base_params, options)
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def policy_cache_path(cache_key: str) -> Path:
    """Return the directory for one cached policy dataset."""

    return POLICY_CACHE_DIR / cache_key


def build_policy_grid(base_params: SimulationParams) -> tuple[dict[str, SimulationParams], pd.DataFrame]:
    """Build the fixed 20-scenario policy grid."""

    base_aht = average_handling_time_minutes(base_params)
    base_enrollment_probability = enrollment_probability(base_params)
    rows = []
    scenarios = {}
    for staffing_multiplier in STAFFING_MULTIPLIERS:
        for aht_multiplier in AHT_MULTIPLIERS:
            staffing = max(1, int(round(base_params.c * staffing_multiplier)))
            aht_minutes = base_aht * aht_multiplier
            scenario_name = f"staff_{staffing_multiplier:.2f}_aht_{aht_multiplier:.2f}"
            params = apply_parameter_overrides(
                base_params,
                {
                    "c": staffing,
                    "aht_minutes": aht_minutes,
                    "enroll_probability": base_enrollment_probability,
                },
            )
            scenarios[scenario_name] = params
            mu_total = params.mu_plus + params.mu_minus
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "staffing_multiplier": staffing_multiplier,
                    "aht_multiplier": aht_multiplier,
                    "staffing": staffing,
                    "aht_minutes": aht_minutes,
                    "service_rate_per_agent": mu_total,
                    "total_service_capacity_per_model_day": staffing * mu_total,
                    "total_service_capacity_per_day_minutes": staffing * MINUTES_PER_MODEL_DAY / aht_minutes,
                }
            )
    return scenarios, pd.DataFrame(rows)


def build_policy_wait_table(run_result: dict[str, pd.DataFrame], policy_parameters: pd.DataFrame) -> pd.DataFrame:
    """Return compact policy metrics used for relation plots."""

    replication_metrics = run_result["replication_metrics"]
    fitting_results = run_result["fitting_results"]
    wait_table = replication_metrics.groupby("scenario_name", as_index=False).mean(numeric_only=True)
    geom = fitting_results[
        (fitting_results["model"] == "geometric")
        & (fitting_results["population"] == "abandoned")
        & (fitting_results["success"] == True)
    ][["scenario_name", "p"]].rename(columns={"p": "geometric_p"})
    wait_table = wait_table.merge(geom, on="scenario_name", how="left")
    wait_table = wait_table.merge(policy_parameters, on="scenario_name", how="left")
    return _with_policy_metric_aliases(wait_table)


def _with_policy_metric_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dashboard-facing policy metric names while preserving cached schemas."""

    result = frame.drop(columns=["replication"], errors="ignore").copy()
    if "procedural_denial_rate" not in result.columns and "left_without_enrollment_rate" in result.columns:
        result["procedural_denial_rate"] = result["left_without_enrollment_rate"]
    return result


def _read_cached_policy_dataset(path: Path) -> dict[str, pd.DataFrame] | None:
    wait_table_path = path / "policy_wait_table.csv"
    parameter_path = path / "policy_parameters.csv"
    metadata_path = path / "metadata.json"
    if not wait_table_path.exists() or not parameter_path.exists() or not metadata_path.exists():
        return None
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    if metadata.get("version") != POLICY_CACHE_VERSION:
        return None
    return {
        "policy_wait_table": _with_policy_metric_aliases(pd.read_csv(wait_table_path)),
        "policy_parameters": pd.read_csv(parameter_path),
        "metadata": pd.DataFrame([metadata]),
    }


def _read_policy_metadata(path: Path) -> dict | None:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with metadata_path.open() as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def _policy_cache_is_compatible(metadata: dict | None, base_label: str) -> bool:
    if metadata is None:
        return False
    return (
        metadata.get("version") == POLICY_CACHE_VERSION
        and metadata.get("base_label") == base_label
        and metadata.get("staffing_multipliers") == STAFFING_MULTIPLIERS
        and metadata.get("aht_multipliers") == AHT_MULTIPLIERS
    )


def _compatible_policy_cache_paths(base_label: str) -> list[Path]:
    if not POLICY_CACHE_DIR.exists():
        return []
    candidates = []
    for path in POLICY_CACHE_DIR.iterdir():
        if not path.is_dir():
            continue
        metadata = _read_policy_metadata(path)
        if _policy_cache_is_compatible(metadata, base_label):
            candidates.append(path)
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)


def _find_compatible_cached_policy_dataset(base_label: str) -> tuple[dict[str, pd.DataFrame] | None, Path | None]:
    for path in _compatible_policy_cache_paths(base_label):
        cached = _read_cached_policy_dataset(path)
        if cached is not None:
            return cached, path
    return None, None


def _write_cached_policy_dataset(path: Path, dataset: dict[str, pd.DataFrame], metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    dataset["policy_wait_table"].to_csv(path / "policy_wait_table.csv", index=False)
    dataset["policy_parameters"].to_csv(path / "policy_parameters.csv", index=False)
    with (path / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)


def load_or_generate_policy_dataset(
    base_label: str,
    base_params: SimulationParams,
    options: ScenarioRunOptions,
    *,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[dict[str, pd.DataFrame], bool, Path]:
    """Load a persistent policy dataset or generate and cache it once."""

    key = policy_cache_key(base_label, base_params, options)
    path = policy_cache_path(key)
    cached = _read_cached_policy_dataset(path)
    if cached is not None:
        return cached, True, path
    cached, compatible_path = _find_compatible_cached_policy_dataset(base_label)
    if cached is not None and compatible_path is not None:
        return cached, True, compatible_path

    scenarios, policy_parameters = build_policy_grid(base_params)
    run_result = run_scenarios(scenarios, options, progress=progress)
    enriched = add_distribution_and_fits(
        run_result,
        populations=["abandoned"],
        fit_negative_binomial_model=False,
    )
    dataset = {
        "policy_wait_table": build_policy_wait_table(enriched, policy_parameters),
        "policy_parameters": policy_parameters,
    }
    metadata = _policy_cache_payload(base_label, base_params, options)
    metadata["cache_key"] = key
    _write_cached_policy_dataset(path, dataset, metadata)
    dataset["metadata"] = pd.DataFrame([metadata])
    return dataset, False, path


def load_policy_dataset_if_available(
    base_label: str,
    base_params: SimulationParams,
    options: ScenarioRunOptions,
) -> tuple[dict[str, pd.DataFrame] | None, Path]:
    """Load the persistent policy dataset if it already exists."""

    key = policy_cache_key(base_label, base_params, options)
    path = policy_cache_path(key)
    cached = _read_cached_policy_dataset(path)
    if cached is not None:
        return cached, path
    compatible, compatible_path = _find_compatible_cached_policy_dataset(base_label)
    if compatible is not None and compatible_path is not None:
        return compatible, compatible_path
    return None, path
