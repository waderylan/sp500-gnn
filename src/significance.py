"""
Statistical significance tests: Diebold-Mariano with HLN correction,
FDR correction via Benjamini-Hochberg, and block bootstrap for Sharpe ratio.

Honesty rule: if no improvement survives FDR correction, report it as-is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def diebold_mariano_test(errors_model: np.ndarray,
                          errors_baseline: np.ndarray) -> dict[str, float]:
    """
    One-sided Diebold-Mariano test with HLN small-sample correction.

    H₀: model is not better than baseline (i.e., model MSE >= baseline MSE).
    H₁: model is better.

    Args:
        errors_model: Per-week mean squared errors for the GNN model,
                      averaged across all stocks. Shape (num_test_weeks,).
        errors_baseline: Per-week mean squared errors for the baseline. Same shape.

    Returns:
        Dict with keys: 'dm_stat', 'p_value' (one-sided), 'n_weeks'.

    Shape assertion: errors_model.shape == errors_baseline.shape.
    """
    raise NotImplementedError


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a set of p-values.

    Args:
        p_values: ndarray of shape (num_tests,).
        alpha: Target FDR level.

    Returns:
        Boolean ndarray of shape (num_tests,) — True where null is rejected.
    """
    raise NotImplementedError


def run_all_dm_tests(model_errors: dict[str, np.ndarray],
                     baseline_errors: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Run DM tests for all (GNN variant, baseline) pairs and apply BH correction within
    each baseline group.

    Args:
        model_errors: Dict mapping model_name -> per-week MSE array, shape (num_test_weeks,).
        baseline_errors: Dict mapping baseline_name -> per-week MSE array.

    Returns:
        DataFrame with columns: ['model', 'baseline', 'dm_stat', 'p_value',
                                  'p_value_bh', 'rejected_bh'].
        Covers 9 primary tests (3 GNN variants × 3 baselines) + 99 sector tests.
    """
    raise NotImplementedError


def block_bootstrap_sharpe(net_returns_a: np.ndarray,
                            net_returns_b: np.ndarray | None,
                            block_size: int = 8,
                            n_bootstrap: int = 5000,
                            seed: int = 42) -> dict[str, float]:
    """
    Block bootstrap confidence interval for Sharpe ratio (or difference in Sharpe).

    If net_returns_b is provided, bootstraps the difference in Sharpe (A - B).
    If None, bootstraps the Sharpe of A alone.

    Args:
        net_returns_a: Weekly net returns for model A, shape (num_weeks,).
        net_returns_b: Weekly net returns for model B (e.g., equal-weight), or None.
        block_size: Length of each bootstrap block (default: 8 weeks).
        n_bootstrap: Number of bootstrap replicates (default: 5000).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: 'point_estimate', 'ci_lower' (2.5th pct), 'ci_upper' (97.5th pct).
    """
    raise NotImplementedError
