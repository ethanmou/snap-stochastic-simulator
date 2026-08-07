"""Attempt-count aggregation for caller-level dashboard records."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


POPULATION_OUTCOMES = {
    "all": None,
    "completed": "completed",
    "abandoned": "left_without_enrollment",
    "left_without_enrollment": "left_without_enrollment",
    "unfinished": "unfinished",
}


@dataclass(frozen=True)
class PopulationSample:
    """Caller-level sample for one population."""

    population: str
    sample: np.ndarray
    frame: pd.DataFrame
    support_shift: int = 1


def validate_caller_records(frame: pd.DataFrame) -> list[str]:
    """Return caller-record schema and consistency issues."""

    errors: list[str] = []
    required = {"scenario_name", "replication", "attempt_count", "terminal_outcome"}
    missing = required.difference(frame.columns)
    if missing:
        return [f"caller records missing columns: {sorted(missing)}"]
    attempts = pd.to_numeric(frame["attempt_count"].dropna(), errors="coerce")
    if attempts.isna().any() or not np.equal(np.mod(attempts.to_numpy(dtype=float), 1.0), 0.0).all():
        errors.append("attempt_count must contain integer values")
    if (attempts < 0).any():
        errors.append("simulation attempt_count must be nonnegative")
    overlap = frame["terminal_outcome"].isin(["completed", "left_without_enrollment", "unfinished"])
    if not overlap.all():
        errors.append("terminal_outcome contains unsupported values")
    return errors


def population_frame(frame: pd.DataFrame, population: str) -> pd.DataFrame:
    """Filter caller records to a supported population."""

    if population not in POPULATION_OUTCOMES:
        raise ValueError(f"unsupported population: {population}")
    outcome = POPULATION_OUTCOMES[population]
    if outcome is None:
        return frame.copy()
    return frame[frame["terminal_outcome"] == outcome].copy()


def population_sample(
    frame: pd.DataFrame,
    population: str,
    *,
    support_shift: int = 1,
) -> PopulationSample:
    """Return positive-support attempt counts for fitting.

    The simulator records redial/retry count with support 0, 1, ...
    The dashboard fits a geometric distribution on call attempt number
    N = simulation_attempt_count + 1, with support 1, 2, ...
    """

    subset = population_frame(frame, population)
    values = subset["attempt_count"].astype(int).to_numpy() + support_shift
    return PopulationSample(population=population, sample=values, frame=subset, support_shift=support_shift)


def attempt_distribution(
    frame: pd.DataFrame,
    population: str,
    *,
    support_shift: int = 1,
) -> pd.DataFrame:
    """Aggregate pooled and mean-replication attempt distributions.

    Missing attempt counts within a replication are treated as zero counts.
    """

    errors = validate_caller_records(frame)
    if errors:
        raise ValueError("; ".join(errors))
    subset = population_frame(frame, population)
    columns = [
        "population",
        "attempt_count",
        "simulation_attempt_count",
        "pooled_count",
        "pooled_probability",
        "replication_mean_count",
        "replication_mean_probability",
        "replication_std_probability",
        "number_of_replications_with_nonzero_count",
        "number_of_valid_replications",
        "pooled_cdf",
        "pooled_survival",
        "mean_replication_cdf",
        "mean_replication_survival",
    ]
    if subset.empty:
        return pd.DataFrame(columns=columns)

    subset = subset.copy()
    subset["attempt_count"] = subset["attempt_count"].astype(int) + support_shift
    subset["simulation_attempt_count"] = subset["attempt_count"] - support_shift
    replications = sorted(subset["replication"].unique())
    min_attempt = int(subset["attempt_count"].min())
    max_attempt = int(subset["attempt_count"].max())
    attempts = list(range(min_attempt, max_attempt + 1))
    pooled_total = int(len(subset))

    rows = []
    for attempt in attempts:
        rep_counts = []
        rep_probs = []
        nonzero = 0
        for replication in replications:
            rep_frame = subset[subset["replication"] == replication]
            denominator = len(rep_frame)
            count = int((rep_frame["attempt_count"] == attempt).sum())
            rep_counts.append(float(count))
            rep_probs.append(float(count / denominator) if denominator else 0.0)
            if count > 0:
                nonzero += 1
        pooled_count = int(sum(rep_counts))
        rows.append(
            {
                "population": population,
                "attempt_count": attempt,
                "simulation_attempt_count": attempt - support_shift,
                "pooled_count": pooled_count,
                "pooled_probability": pooled_count / pooled_total if pooled_total else np.nan,
                "replication_mean_count": float(np.mean(rep_counts)) if rep_counts else np.nan,
                "replication_mean_probability": float(np.mean(rep_probs)) if rep_probs else np.nan,
                "replication_std_probability": float(np.std(rep_probs, ddof=1)) if len(rep_probs) > 1 else 0.0,
                "number_of_replications_with_nonzero_count": nonzero,
                "number_of_valid_replications": len(replications),
            }
        )
    result = pd.DataFrame(rows, columns=columns[:-4])
    result["pooled_cdf"] = result["pooled_probability"].cumsum()
    result["pooled_survival"] = result["pooled_probability"][::-1].cumsum()[::-1]
    result["mean_replication_cdf"] = result["replication_mean_probability"].cumsum()
    result["mean_replication_survival"] = result["replication_mean_probability"][::-1].cumsum()[::-1]
    return result[columns]


def population_summary(frame: pd.DataFrame, population: str) -> dict[str, float | int | str]:
    """Return caller-level summary metrics for one population."""

    sample = population_sample(frame, population).sample
    if len(sample) == 0:
        return {
            "population": population,
            "sample_size": 0,
            "mean_attempts": np.nan,
            "median_attempts": np.nan,
            "attempt_std": np.nan,
            "p90_attempts": np.nan,
            "p95_attempts": np.nan,
            "max_attempts": 0,
        }
    return {
        "population": population,
        "sample_size": int(len(sample)),
        "mean_attempts": float(np.mean(sample)),
        "median_attempts": float(np.median(sample)),
        "attempt_std": float(np.std(sample, ddof=1)) if len(sample) > 1 else 0.0,
        "p90_attempts": float(np.percentile(sample, 90)),
        "p95_attempts": float(np.percentile(sample, 95)),
        "max_attempts": int(np.max(sample)),
    }
