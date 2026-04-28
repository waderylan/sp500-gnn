"""Statistical tests for forecast and portfolio comparisons.

The functions in this module operate on saved evaluation artifacts. They do
not train models, choose hyperparameters, or inspect test results for model
selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


TRADING_WEEKS_PER_YEAR = 52


@dataclass(frozen=True)
class DMResult:
    """Diebold-Mariano test result for one model-vs-baseline comparison."""

    dm_stat: float
    p_value: float
    n_weeks: int
    mean_loss_diff: float


def _as_clean_1d(values: np.ndarray | pd.Series) -> np.ndarray:
    """Return finite one-dimensional float values."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def annualized_sharpe(net_returns: np.ndarray | pd.Series) -> float:
    """Compute annualized Sharpe from weekly net returns with zero risk-free rate."""
    returns = _as_clean_1d(net_returns)
    if returns.size < 2:
        return float("nan")
    vol = float(np.std(returns, ddof=1))
    if vol <= 0.0:
        return float("nan")
    return float(np.mean(returns) / vol * np.sqrt(TRADING_WEEKS_PER_YEAR))


def diebold_mariano_test(
    errors_model: np.ndarray,
    errors_baseline: np.ndarray,
    *,
    horizon: int = 1,
) -> dict[str, float]:
    """
    One-sided Diebold-Mariano test with Harvey-Leybourne-Newbold correction.

    The input arrays are per-week losses, normally weekly MSE averaged across
    stocks. The tested alternative is that the model has lower expected loss
    than the baseline. A positive mean loss differential therefore favors the
    model:

    ``loss_diff[t] = errors_baseline[t] - errors_model[t]``.
    """
    model = np.asarray(errors_model, dtype=float).reshape(-1)
    baseline = np.asarray(errors_baseline, dtype=float).reshape(-1)
    if model.shape != baseline.shape:
        raise ValueError(f"Shape mismatch: {model.shape} vs {baseline.shape}")

    valid = np.isfinite(model) & np.isfinite(baseline)
    diff = baseline[valid] - model[valid]
    n_weeks = int(diff.size)
    if n_weeks <= max(2, horizon):
        return {
            "dm_stat": float("nan"),
            "p_value": float("nan"),
            "n_weeks": n_weeks,
            "mean_loss_diff": float("nan"),
        }

    mean_diff = float(np.mean(diff))
    centered = diff - mean_diff

    # Newey-West long-run variance for h-step-ahead forecast errors. For the
    # current one-week-ahead setup, horizon=1 and this reduces to var(diff).
    gamma0 = float(np.dot(centered, centered) / n_weeks)
    long_run_var = gamma0
    for lag in range(1, horizon):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n_weeks)
        weight = 1.0 - lag / horizon
        long_run_var += 2.0 * weight * cov

    if long_run_var <= 0.0:
        dm_stat = float("inf") if mean_diff > 0 else float("-inf")
    else:
        dm_stat = mean_diff / np.sqrt(long_run_var / n_weeks)

    hln_scale = np.sqrt((n_weeks + 1 - 2 * horizon + horizon * (horizon - 1) / n_weeks) / n_weeks)
    dm_hln = float(dm_stat * hln_scale)
    p_value = float(student_t.sf(dm_hln, df=n_weeks - 1))

    return {
        "dm_stat": dm_hln,
        "p_value": p_value,
        "n_weeks": n_weeks,
        "mean_loss_diff": mean_diff,
    }


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction.

    Returns a boolean array where ``True`` means the null is rejected at the
    requested FDR level. NaN p-values are never rejected.
    """
    p_values = np.asarray(p_values, dtype=float).reshape(-1)
    rejected = np.zeros(p_values.shape, dtype=bool)
    finite = np.isfinite(p_values)
    if not finite.any():
        return rejected

    finite_indices = np.flatnonzero(finite)
    finite_p = p_values[finite]
    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    thresholds = alpha * np.arange(1, sorted_p.size + 1) / sorted_p.size
    passed = sorted_p <= thresholds
    if passed.any():
        max_rank = int(np.flatnonzero(passed).max())
        rejected[finite_indices[order[: max_rank + 1]]] = True
    return rejected


def benjamini_hochberg_adjusted_p(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float).reshape(-1)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted

    finite_indices = np.flatnonzero(finite)
    finite_p = p_values[finite]
    order = np.argsort(finite_p)
    sorted_p = finite_p[order]
    m = sorted_p.size
    sorted_adj = sorted_p * m / np.arange(1, m + 1)
    sorted_adj = np.minimum.accumulate(sorted_adj[::-1])[::-1]
    sorted_adj = np.clip(sorted_adj, 0.0, 1.0)
    adjusted[finite_indices[order]] = sorted_adj
    return adjusted


def run_all_dm_tests(
    model_errors: dict[str, np.ndarray],
    baseline_errors: dict[str, np.ndarray],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Run model-vs-baseline DM tests and FDR-correct within each baseline group."""
    rows: list[dict[str, object]] = []
    for baseline_name, baseline_values in baseline_errors.items():
        for model_name, model_values in model_errors.items():
            if model_name == baseline_name:
                continue
            result = diebold_mariano_test(model_values, baseline_values)
            rows.append(
                {
                    "model": model_name,
                    "baseline": baseline_name,
                    **result,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "baseline",
                "dm_stat",
                "p_value",
                "p_value_bh",
                "rejected_bh",
                "n_weeks",
                "mean_loss_diff",
            ]
        )

    df["p_value_bh"] = np.nan
    df["rejected_bh"] = False
    for baseline_name, idx in df.groupby("baseline").groups.items():
        p_values = df.loc[idx, "p_value"].to_numpy(dtype=float)
        df.loc[idx, "p_value_bh"] = benjamini_hochberg_adjusted_p(p_values)
        df.loc[idx, "rejected_bh"] = benjamini_hochberg(p_values, alpha=alpha)

    return df.sort_values(["baseline", "p_value", "model"]).reset_index(drop=True)


def _circular_block_indices(
    n_obs: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample circular block bootstrap indices with length exactly n_obs."""
    if n_obs <= 0:
        raise ValueError("n_obs must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    starts = rng.integers(0, n_obs, size=int(np.ceil(n_obs / block_size)))
    blocks = [(start + np.arange(block_size)) % n_obs for start in starts]
    return np.concatenate(blocks)[:n_obs]


def block_bootstrap_sharpe(
    net_returns_a: np.ndarray,
    net_returns_b: np.ndarray | None,
    block_size: int = 8,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """
    Circular block bootstrap CI for Sharpe or Sharpe difference.

    If ``net_returns_b`` is provided, the reported point estimate and interval
    are for ``Sharpe(A) - Sharpe(B)``.
    """
    a = np.asarray(net_returns_a, dtype=float).reshape(-1)
    if net_returns_b is None:
        valid = np.isfinite(a)
        a = a[valid]
        b = None
    else:
        b = np.asarray(net_returns_b, dtype=float).reshape(-1)
        if a.shape != b.shape:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        valid = np.isfinite(a) & np.isfinite(b)
        a = a[valid]
        b = b[valid]

    n_obs = int(a.size)
    if n_obs < 2:
        return {
            "point_estimate": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_weeks": n_obs,
            "block_size": block_size,
            "n_bootstrap": n_bootstrap,
        }

    point = annualized_sharpe(a)
    if b is not None:
        point -= annualized_sharpe(b)

    rng = np.random.default_rng(seed)
    boot_values = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = _circular_block_indices(n_obs, block_size, rng)
        sample_value = annualized_sharpe(a[idx])
        if b is not None:
            sample_value -= annualized_sharpe(b[idx])
        boot_values[i] = sample_value

    boot_values = boot_values[np.isfinite(boot_values)]
    if boot_values.size == 0:
        ci_lower = ci_upper = float("nan")
    else:
        ci_lower, ci_upper = np.percentile(boot_values, [2.5, 97.5])

    return {
        "point_estimate": float(point),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_weeks": n_obs,
        "block_size": block_size,
        "n_bootstrap": n_bootstrap,
    }
