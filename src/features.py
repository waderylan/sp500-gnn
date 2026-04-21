"""
Feature engineering: HAR-style RV lags, momentum, turnover, and macro features.
Applies mandatory cross-sectional winsorization then z-scoring.

All features at week T use only data strictly before week T (lookahead-safe).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


def compute_har_features(weekly_rv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute HAR-style realized volatility lag features for each stock.

    Features: RV_1w (1-week lag), RV_4w (4-week rolling mean), RV_13w (13-week rolling mean).
    These correspond to daily, weekly, and monthly components in the HAR-RV literature.

    Args:
        weekly_rv: DataFrame of shape (num_weeks, num_stocks).

    Returns:
        DataFrame of shape (num_weeks, num_stocks * 3) with multi-level column index.

    Lookahead safety: each feature at row T uses weekly_rv[T-1], weekly_rv[T-4:T], etc.
    No window touches row T or later.
    Shape assertion: result.shape[0] == weekly_rv.shape[0].
    """
    raise NotImplementedError


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
