"""
Inverse-volatility portfolio construction, weekly rebalancing, and performance metrics.

Weights = 1/predicted_RV, clipped at MAX_WEIGHT, normalized to sum to 1.
Transaction costs deducted at TRANSACTION_COST_BPS per unit of turnover.
Sharpe ratio uses FRED DTB3 T-bill rates as risk-free rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def compute_weights(predicted_rv: np.ndarray) -> np.ndarray:
    """
    Compute inverse-volatility portfolio weights from predicted RV.

    Pipeline:
        1. raw_weights = 1.0 / predicted_rv
        2. weights = raw_weights / raw_weights.sum()
        3. weights = weights.clip(max=config.MAX_WEIGHT)
        4. weights = weights / weights.sum()  # re-normalize after clip

    Args:
        predicted_rv: ndarray of shape (num_stocks,) — positive values.

    Returns:
        ndarray of shape (num_stocks,) summing to 1.0.

    Shape assertion: result.shape == predicted_rv.shape, abs(result.sum() - 1.0) < 1e-6.
    """
    raise NotImplementedError


def compute_portfolio_returns(weights: np.ndarray,
                               actual_returns: np.ndarray,
                               prev_weights: np.ndarray | None) -> dict[str, float]:
    """
    Compute gross and net portfolio return for one week, deducting transaction costs.

    Args:
        weights: Current week's portfolio weights, shape (num_stocks,).
        actual_returns: Realized returns this week, shape (num_stocks,).
        prev_weights: Prior week's weights for turnover calculation, or None for week 1.

    Returns:
        Dict with keys: 'gross_return', 'net_return', 'turnover', 'max_weight'.
    """
    raise NotImplementedError


def compute_sharpe(net_returns: np.ndarray,
                   tbill_rates: pd.Series,
                   weeks: pd.DatetimeIndex) -> float:
    """
    Compute annualized Sharpe ratio net of transaction costs.

    Sharpe = (annualized_return - mean_annualized_tbill) / annualized_volatility.

    Args:
        net_returns: Weekly net returns, shape (num_weeks,).
        tbill_rates: Daily FRED DTB3 rates (decimal), indexed by date.
        weeks: DatetimeIndex of portfolio weeks (used to align T-bill rates).

    Returns:
        Annualized Sharpe ratio (float).
    """
    raise NotImplementedError


def run_backtest(predicted_rv_by_week: np.ndarray,
                 actual_returns_by_week: np.ndarray,
                 tbill_rates: pd.Series,
                 weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Run the full portfolio backtest over the evaluation period.

    Args:
        predicted_rv_by_week: ndarray of shape (num_weeks, num_stocks).
        actual_returns_by_week: ndarray of shape (num_weeks, num_stocks).
        tbill_rates: Daily FRED DTB3 rates.
        weeks: DatetimeIndex of portfolio weeks.

    Returns:
        DataFrame with columns: ['week', 'gross_return', 'net_return', 'turnover', 'max_weight'].

    Logs max single-stock weight, turnover, gross return, net return each week.
    """
    raise NotImplementedError
