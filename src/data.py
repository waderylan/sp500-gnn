"""
Price download, log-return computation, weekly realized-volatility calculation,
and train/val/test split generation.

All functions are lookahead-safe: no window extends past the prediction week start.
Outputs are saved to data/raw/ and data/features/ as parquet files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config


def download_prices() -> pd.DataFrame:
    """
    Download daily OHLCV data for S&P 500 constituents via yfinance and apply
    universe filters (MIN_COVERAGE, constituent history). Saves prices.parquet
    and tickers.json to DATA_RAW_DIR.

    Returns:
        DataFrame of shape (num_trading_days, num_stocks) containing adjusted close prices.

    Lookahead safety: no rolling windows; point-in-time constituent list used.
    """
    raise NotImplementedError


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from adjusted close prices.

    Args:
        prices: DataFrame of shape (num_trading_days, num_stocks).

    Returns:
        DataFrame of shape (num_trading_days, num_stocks). First row is NaN and dropped.

    Lookahead safety: each day's return uses only that day and the prior day's price.
    """
    raise NotImplementedError


def compute_weekly_rv(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized weekly realized volatility (RV) for each stock.

    RV for ISO week W = std(daily log returns in W) * sqrt(252).
    Weeks with fewer than 5 trading days are dropped.

    Args:
        log_returns: DataFrame of shape (num_trading_days, num_stocks).

    Returns:
        DataFrame of shape (num_weeks, num_stocks). Index is ISO week start date (Monday).

    Lookahead safety: RV for week W uses only data from within week W — no forward look.
    Shape assertion: result.shape == (num_weeks, num_stocks).
    """
    raise NotImplementedError


def make_target(weekly_rv: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the prediction target by shifting weekly_rv forward by one week.

    Target at row T = RV at week T+1. The last row will have NaN target and is dropped.

    Args:
        weekly_rv: DataFrame of shape (num_weeks, num_stocks).

    Returns:
        DataFrame of shape (num_weeks - 1, num_stocks).

    Lookahead safety: target[T] = rv[T+1] — this is intentional. Features at T must
    use only data strictly before T, so the shift direction is correct.
    Shape assertion: result.shape[0] == weekly_rv.shape[0] - 1.
    """
    raise NotImplementedError


def make_splits(index: pd.Index) -> pd.DataFrame:
    """
    Assign each week in `index` to train, val, or test split based on config dates.

    Args:
        index: DatetimeIndex of weekly RV rows.

    Returns:
        DataFrame with columns ['week', 'split'] where split ∈ {'train', 'val', 'test'}.

    Shape assertion: result.shape[0] == len(index).
    """
    raise NotImplementedError


def download_tbill_rates() -> pd.Series:
    """
    Download FRED DTB3 (3-month T-bill daily rates) via pandas-datareader.
    Saves to data/raw/tbill_rates.parquet.

    Returns:
        Series indexed by date, values are annualized T-bill rates (as decimals, not %).
    """
    raise NotImplementedError
