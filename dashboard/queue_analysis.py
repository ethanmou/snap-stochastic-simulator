"""Steady-state queue-length aggregation and persistent dashboard cache."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_simulation import SimulationParams

from .simulation_runner import ScenarioRunOptions


QUEUE_SAMPLE_COLUMNS = [
    "scenario_name",
    "replication",
    "seed",
    "sample_time",
    "queue_length",
]

QUEUE_DISTRIBUTION_COLUMNS = [
    "queue_length",
    "pooled_count",
    "pooled_probability",
    "replication_mean_probability",
    "replication_std_probability",
    "number_of_replications_with_nonzero_count",
    "number_of_replications",
    "pooled_cdf",
    "pooled_survival_probability",
]

QUEUE_CACHE_VERSION = 1
QUEUE_CACHE_DIR = Path("outputs/dashboard_queue_cache")


def queue_length_distribution(samples: pd.DataFrame) -> pd.DataFrame:
    """Return the empirical steady-state queue-length distribution.

    The survival column uses the inclusive convention P(Q >= q).
    Missing integer queue lengths between zero and the maximum observed queue
    length are included with zero probability.
    """

    if samples.empty:
        return pd.DataFrame(columns=QUEUE_DISTRIBUTION_COLUMNS)
    missing = {"replication", "queue_length"}.difference(samples.columns)
    if missing:
        raise ValueError(f"queue samples missing columns: {sorted(missing)}")

    frame = samples.copy()
    frame["queue_length"] = pd.to_numeric(frame["queue_length"], errors="coerce")
    if frame["queue_length"].isna().any():
        raise ValueError("queue_length must be numeric")
    values = frame["queue_length"].to_numpy(dtype=float)
    if (values < 0).any() or not np.equal(np.mod(values, 1.0), 0.0).all():
        raise ValueError("queue_length must contain nonnegative integers")
    frame["queue_length"] = frame["queue_length"].astype(int)

    replications = sorted(frame["replication"].dropna().unique().tolist())
    max_queue_length = int(frame["queue_length"].max())
    pooled_total = int(len(frame))
    rows = []
    for queue_length in range(max_queue_length + 1):
        replication_probabilities = []
        nonzero_replications = 0
        pooled_count = int((frame["queue_length"] == queue_length).sum())
        for replication in replications:
            replication_frame = frame[frame["replication"] == replication]
            denominator = len(replication_frame)
            count = int((replication_frame["queue_length"] == queue_length).sum())
            if count > 0:
                nonzero_replications += 1
            replication_probabilities.append(count / denominator if denominator else 0.0)
        rows.append(
            {
                "queue_length": queue_length,
                "pooled_count": pooled_count,
                "pooled_probability": pooled_count / pooled_total if pooled_total else np.nan,
                "replication_mean_probability": (
                    float(np.mean(replication_probabilities))
                    if replication_probabilities
                    else np.nan
                ),
                "replication_std_probability": (
                    float(np.std(replication_probabilities, ddof=1))
                    if len(replication_probabilities) > 1
                    else 0.0
                ),
                "number_of_replications_with_nonzero_count": nonzero_replications,
                "number_of_replications": len(replications),
            }
        )

    result = pd.DataFrame(rows, columns=QUEUE_DISTRIBUTION_COLUMNS[:-2])
    result["pooled_cdf"] = result["pooled_probability"].cumsum()
    result["pooled_survival_probability"] = (
        result["pooled_probability"][::-1].cumsum()[::-1]
    )
    return result[QUEUE_DISTRIBUTION_COLUMNS]


def queue_cache_key(params: SimulationParams, options: ScenarioRunOptions) -> str:
    """Return a persistent cache key for queue-distribution outputs."""

    payload = {
        "version": QUEUE_CACHE_VERSION,
        "params": asdict(params),
        "options": asdict(options),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def queue_cache_path(cache_key: str) -> Path:
    """Return the directory used for one cached queue-distribution result."""

    return QUEUE_CACHE_DIR / cache_key


def load_cached_queue_outputs(
    params: SimulationParams,
    options: ScenarioRunOptions,
) -> tuple[dict[str, pd.DataFrame] | None, Path]:
    """Load queue samples and distribution if the exact cache exists."""

    key = queue_cache_key(params, options)
    path = queue_cache_path(key)
    metadata_path = path / "metadata.json"
    samples_path = path / "queue_samples.csv"
    distribution_path = path / "queue_distribution.csv"
    if not (
        metadata_path.exists()
        and samples_path.exists()
        and distribution_path.exists()
    ):
        return None, path
    try:
        with metadata_path.open() as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError:
        return None, path
    if metadata.get("version") != QUEUE_CACHE_VERSION:
        return None, path
    return (
        {
            "queue_samples": pd.read_csv(samples_path),
            "queue_distribution": pd.read_csv(distribution_path),
            "metadata": pd.DataFrame([metadata]),
        },
        path,
    )


def save_cached_queue_outputs(
    params: SimulationParams,
    options: ScenarioRunOptions,
    queue_samples: pd.DataFrame,
    queue_distribution: pd.DataFrame,
) -> Path:
    """Persist queue samples and distribution for exact future reuse."""

    key = queue_cache_key(params, options)
    path = queue_cache_path(key)
    path.mkdir(parents=True, exist_ok=True)
    queue_samples.to_csv(path / "queue_samples.csv", index=False)
    queue_distribution.to_csv(path / "queue_distribution.csv", index=False)
    metadata = {
        "version": QUEUE_CACHE_VERSION,
        "cache_key": key,
        "params": asdict(params),
        "options": asdict(options),
        "survival_convention": "pooled_survival_probability = P(Q >= q)",
    }
    with (path / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2, default=str)
    return path
