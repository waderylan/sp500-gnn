"""
Feature engineering: HAR-style RV lags, momentum, turnover, and macro features.
Applies mandatory cross-sectional winsorization then z-scoring.

All features at week T use only data strictly before week T (lookahead-safe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def compute_volatility_features(
    log_returns: pd.DataFrame,
    weekly_rv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute rolling realized volatility features at 5, 10, 21, and 63 trading day lookbacks,
    plus the short/long ratio (rv_5d / rv_63d), aligned to the weekly prediction schedule.

    For each Monday in weekly_rv.index (week T), the feature value is taken from the last
    available trading day before that Monday (typically the prior Friday). This is done by
    reindexing the daily rolling series to the preceding Sunday and forward-filling.

    log_returns: shape (num_trading_days, num_stocks) -- daily log returns.
    weekly_rv: shape (num_weeks, num_stocks) -- index is Monday week-start dates.

    Returns: DataFrame of shape (num_weeks, num_stocks * 5) with MultiIndex columns
             (feature_name, ticker). Feature names: rv_5d, rv_10d, rv_21d, rv_63d, rv_ratio.

    Lookahead safety: each rolling window of N days ends at Friday_{T-1}, strictly before
    Monday_T. The ratio uses already-lagged values. No window touches week T or later.
    Shape assertion: result.shape == (len(weekly_rv), weekly_rv.shape[1] * 5).
    """
    windows = {"rv_5d": 5, "rv_10d": 10, "rv_21d": 21, "rv_63d": 63}

    # Sundays before each Monday: ffill to these dates grabs the preceding Friday
    lookup_dates = weekly_rv.index - pd.Timedelta(days=1)

    aligned: dict[str, pd.DataFrame] = {}
    for name, n in windows.items():
        daily_rv = log_returns.rolling(n).std() * np.sqrt(252)
        weekly = daily_rv.reindex(lookup_dates, method="ffill")
        weekly.index = weekly_rv.index  # relabel Sundays back to Mondays
        aligned[name] = weekly

    # Short/long ratio: replace zero denominators with NaN to avoid inf
    rv_ratio = aligned["rv_5d"] / aligned["rv_63d"].replace(0.0, np.nan)
    aligned["rv_ratio"] = rv_ratio

    result = pd.concat(aligned, axis=1)  # MultiIndex columns: (feature_name, ticker)

    num_weeks = len(weekly_rv)
    num_stocks = weekly_rv.shape[1]
    assert result.shape == (num_weeks, num_stocks * 5), (
        f"compute_volatility_features: expected ({num_weeks}, {num_stocks * 5}), "
        f"got {result.shape}"
    )

    return result


def compute_momentum_features(log_returns: pd.DataFrame,
                               weekly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Compute cross-sectional momentum features aligned to weekly_index.

    Features: 4-week cumulative return, 13-week cumulative return (each ending before week T).

    Args:
        log_returns: Daily log returns, shape (num_trading_days, num_stocks).
        weekly_index: DatetimeIndex of ISO week start dates.

    Returns:
        DataFrame of shape (num_weeks, num_stocks * 2).

    Lookahead safety: cumulative return for week T uses only trading days strictly before
    the start of week T.
    """
    raise NotImplementedError


def winsorize_cross_sectional(df: pd.DataFrame,
                               clip: tuple[float, float] = config.WINSORIZE_CLIP) -> pd.DataFrame:
    """
    Winsorize each feature cross-sectionally at each time step.

    For each row (time step), clip values to [clip[0], clip[1]] quantile of that row's distribution.

    Args:
        df: DataFrame of shape (num_weeks, num_stocks).
        clip: (lower_quantile, upper_quantile) tuple.

    Returns:
        DataFrame of same shape with extreme values clipped.
    """
    raise NotImplementedError


def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature cross-sectionally at each time step (subtract mean, divide by std).

    Must be called AFTER winsorize_cross_sectional. Never before.

    Args:
        df: DataFrame of shape (num_weeks, num_stocks).

    Returns:
        DataFrame of same shape.
    """
    raise NotImplementedError


def build_feature_tensor(weekly_rv: pd.DataFrame,
                          log_returns: pd.DataFrame) -> np.ndarray:
    """
    Assemble all features into a 3D tensor and apply the mandatory normalization pipeline
    (winsorize → z-score, in that order, cross-sectionally).

    Args:
        weekly_rv: DataFrame of shape (num_weeks, num_stocks).
        log_returns: DataFrame of shape (num_trading_days, num_stocks).

    Returns:
        ndarray of shape (num_weeks, num_stocks, num_features).

    Shape assertion: result.shape == (num_weeks, num_stocks, num_features).
    Post-normalization assertions run on 10 random time steps per feature.
    """
    raise NotImplementedError


def save_features(features: np.ndarray, target: pd.DataFrame,
                  tickers: list[str], weeks: pd.DatetimeIndex) -> None:
    """
    Reshape the feature tensor to 2D and save features.parquet and target.parquet.

    Parquet schema: columns = ['week', 'ticker'] + feature_names.
    Shape assertion before save: 2D reshaped array has shape (num_weeks * num_stocks, num_features + 2).
    """
    raise NotImplementedError
