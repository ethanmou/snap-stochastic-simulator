"""Discrete distribution fitting diagnostics for attempt counts."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def geometric_pmf(k: np.ndarray, p: float) -> np.ndarray:
    """Geometric PMF P(N=k)=p(1-p)^(k-1), k=1,2,..."""

    return p * np.power(1.0 - p, k - 1)


def _empirical_distribution(sample: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(sample).value_counts().sort_index()
    frame = counts.rename_axis("attempt_count").reset_index(name="count")
    frame["probability"] = frame["count"] / frame["count"].sum()
    frame["cdf"] = frame["probability"].cumsum()
    return frame


def fit_geometric(sample: Iterable[int], *, population: str = "", scenario_name: str = "") -> dict[str, float | int | str | bool]:
    """Fit a positive-support geometric distribution by MLE."""

    values = np.asarray(list(sample), dtype=int)
    if len(values) == 0:
        return {"scenario_name": scenario_name, "population": population, "model": "geometric", "success": False, "error": "empty sample", "sample_size": 0}
    if (values < 1).any():
        return {"scenario_name": scenario_name, "population": population, "model": "geometric", "success": False, "error": "attempt counts must be >= 1", "sample_size": int(len(values))}

    sample_mean = float(np.mean(values))
    sample_variance = float(np.var(values, ddof=1)) if len(values) > 1 else 0.0
    p_hat = min(max(1.0 / sample_mean, 1e-12), 1.0)
    p_first_attempt = float(np.mean(values == 1))
    log_likelihood = float(np.sum(np.log(p_hat) + (values - 1) * np.log1p(-p_hat))) if p_hat < 1.0 else (0.0 if np.all(values == 1) else -math.inf)
    empirical = _empirical_distribution(values)
    k = empirical["attempt_count"].to_numpy(dtype=int)
    fitted = geometric_pmf(k, p_hat)
    empirical_cdf = empirical["cdf"].to_numpy(dtype=float)
    fitted_cdf = 1.0 - np.power(1.0 - p_hat, k)
    residual = empirical["probability"].to_numpy(dtype=float) - fitted
    chi_square = float(np.sum(np.square(empirical["count"] - len(values) * fitted) / np.maximum(len(values) * fitted, 1e-12)))
    return {
        "scenario_name": scenario_name,
        "population": population,
        "model": "geometric",
        "success": True,
        "error": "",
        "support": "N = simulation_attempt_count + 1; N starts at 1",
        "sample_size": int(len(values)),
        "sample_mean": sample_mean,
        "sample_variance": sample_variance,
        "p": float(p_hat),
        "p_mle": float(p_hat),
        "p_first_attempt": p_first_attempt,
        "r": np.nan,
        "implied_mean": float(1.0 / p_hat),
        "implied_variance": float((1.0 - p_hat) / (p_hat * p_hat)),
        "log_likelihood": log_likelihood,
        "aic": float(2 - 2 * log_likelihood),
        "bic": float(math.log(len(values)) - 2 * log_likelihood),
        "pmf_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pmf_mae": float(np.mean(np.abs(residual))),
        "max_abs_cdf_difference": float(np.max(np.abs(empirical_cdf - fitted_cdf))),
        "chi_square_statistic": chi_square,
        "ks_style_statistic": float(np.max(np.abs(empirical_cdf - fitted_cdf))),
    }


def negative_binomial_pmf(k: np.ndarray, r: float, p: float) -> np.ndarray:
    """Shifted negative-binomial PMF for N=1+failures before r successes."""

    x = k - 1
    coeff = np.exp(
        np.array([math.lgamma(xi + r) - math.lgamma(r) - math.lgamma(xi + 1.0) for xi in x])
    )
    return coeff * np.power(p, r) * np.power(1.0 - p, x)


def fit_negative_binomial(sample: Iterable[int], *, population: str = "", scenario_name: str = "") -> dict[str, float | int | str | bool]:
    """Fit a shifted negative binomial as an optional diagnostic model."""

    values = np.asarray(list(sample), dtype=int)
    base = {"scenario_name": scenario_name, "population": population, "model": "negative_binomial", "sample_size": int(len(values))}
    if len(values) == 0:
        return {**base, "success": False, "error": "empty sample"}
    if (values < 1).any():
        return {**base, "success": False, "error": "attempt counts must be >= 1"}
    try:
        from scipy.optimize import minimize
    except Exception as exc:  # pragma: no cover - depends on optional dependency state
        return {**base, "success": False, "error": f"scipy unavailable: {exc}"}

    x = values - 1

    def nll(raw: np.ndarray) -> float:
        log_r, logit_p = raw
        r = max(math.exp(float(log_r)), 1e-12)
        p = 1.0 / (1.0 + math.exp(-float(logit_p)))
        p = min(max(p, 1e-12), 1.0 - 1e-12)
        log_probs = [
            math.lgamma(int(xi) + r)
            - math.lgamma(r)
            - math.lgamma(int(xi) + 1.0)
            + r * math.log(p)
            + int(xi) * math.log1p(-p)
            for xi in x
        ]
        return -float(np.sum(log_probs))

    mean_x = float(np.mean(x))
    var_x = float(np.var(x, ddof=1)) if len(x) > 1 else mean_x
    r0 = max(mean_x * mean_x / max(var_x - mean_x, 1e-6), 0.25) if mean_x > 0 else 10.0
    p0 = min(max(r0 / (r0 + max(mean_x, 1e-9)), 1e-6), 1.0 - 1e-6)
    result = minimize(nll, np.array([math.log(r0), math.log(p0 / (1.0 - p0))]), method="Nelder-Mead")
    if not result.success:
        return {**base, "success": False, "error": result.message}
    r_hat = max(math.exp(float(result.x[0])), 1e-12)
    p_hat = 1.0 / (1.0 + math.exp(-float(result.x[1])))
    p_hat = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    log_likelihood = -float(result.fun)
    empirical = _empirical_distribution(values)
    k = empirical["attempt_count"].to_numpy(dtype=int)
    fitted = negative_binomial_pmf(k, r_hat, p_hat)
    fitted_cdf = np.cumsum(fitted)
    residual = empirical["probability"].to_numpy(dtype=float) - fitted
    return {
        **base,
        "success": True,
        "error": "",
        "support": "N = 1 + failures before r successes",
        "sample_mean": float(np.mean(values)),
        "sample_variance": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        "p": float(p_hat),
        "r": float(r_hat),
        "implied_mean": float(1.0 + r_hat * (1.0 - p_hat) / p_hat),
        "implied_variance": float(r_hat * (1.0 - p_hat) / (p_hat * p_hat)),
        "log_likelihood": log_likelihood,
        "aic": float(4 - 2 * log_likelihood),
        "bic": float(2 * math.log(len(values)) - 2 * log_likelihood),
        "pmf_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "pmf_mae": float(np.mean(np.abs(residual))),
        "max_abs_cdf_difference": float(np.max(np.abs(empirical["cdf"].to_numpy(dtype=float) - fitted_cdf))),
        "chi_square_statistic": float(np.sum(np.square(empirical["count"] - len(values) * fitted) / np.maximum(len(values) * fitted, 1e-12))),
        "ks_style_statistic": float(np.max(np.abs(empirical["cdf"].to_numpy(dtype=float) - fitted_cdf))),
    }


def fitted_pmf_frame(
    distribution: pd.DataFrame,
    fit: dict[str, float | int | str | bool],
) -> pd.DataFrame:
    """Return empirical/fitted/residual rows for plotting."""

    if distribution.empty or not fit.get("success"):
        return pd.DataFrame(columns=["attempt_count", "empirical_probability", "fitted_probability", "residual"])
    k = distribution["attempt_count"].to_numpy(dtype=int)
    if fit["model"] == "geometric":
        fitted = geometric_pmf(k, float(fit["p"]))
    else:
        fitted = negative_binomial_pmf(k, float(fit["r"]), float(fit["p"]))
    empirical = distribution["pooled_probability"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "attempt_count": k,
            "empirical_probability": empirical,
            "fitted_probability": fitted,
            "residual": empirical - fitted,
        }
    )
