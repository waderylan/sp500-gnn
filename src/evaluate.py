"""
Evaluation metrics: MSE, MAE, R², and sector-level breakdowns.

All evaluation runs on val predictions during development.
Test-set evaluation is gated behind Task 5.1 (see CLAUDE.md Rule 4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute MSE, MAE, and R² between true and predicted RV.

    Args:
        y_true: ndarray of shape (num_weeks, num_stocks) or (N,).
        y_pred: ndarray of same shape as y_true.

    Returns:
        Dict with keys 'mse', 'mae', 'r2'.

    Shape assertion: y_true.shape == y_pred.shape.
    """
    raise NotImplementedError


def compute_sector_metrics(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            tickers: list[str],
                            sector_map: dict[str, str]) -> pd.DataFrame:
    """
    Compute MSE, MAE, R² broken down by GICS sector.

    Args:
        y_true: ndarray of shape (num_weeks, num_stocks).
        y_pred: ndarray of same shape.
        tickers: Ordered list of ticker symbols (column order of y_true/y_pred).
        sector_map: Dict mapping ticker -> sector name.

    Returns:
        DataFrame with columns ['sector', 'mse', 'mae', 'r2'], one row per sector.
    """
    raise NotImplementedError


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Aggregate metrics from all models into a single comparison table.

    Args:
        results: Dict mapping model_name -> metrics dict (from compute_metrics).

    Returns:
        DataFrame with model names as index and metric columns.
    """
    raise NotImplementedError
