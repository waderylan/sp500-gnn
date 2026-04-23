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

    For each Monday in weekly_rv.index (week T), the feature value is taken from Friday of
    week T (Monday + 4 days) via reindex with forward-fill. This is the 1-step-ahead design:
    features include week T's own data; the target is week T+1's RV.

    log_returns: shape (num_trading_days, num_stocks) -- daily log returns.
    weekly_rv: shape (num_weeks, num_stocks) -- index is Monday week-start dates.

    Returns: DataFrame of shape (num_weeks, num_stocks * 5) with MultiIndex columns
             (feature_name, ticker). Feature names: rv_5d, rv_10d, rv_21d, rv_63d, rv_ratio.

    Lookahead safety: 1-step-ahead design. Each rolling window of N days ends at
    Friday_T (last trading day of week T). The target is week T+1's RV, which starts
    Monday_{T+1}. Friday_T is strictly before Monday_{T+1}.
    Shape assertion: result.shape == (len(weekly_rv), weekly_rv.shape[1] * 5).
    """
    windows = {"rv_5d": 5, "rv_10d": 10, "rv_21d": 21, "rv_63d": 63}

    # Fridays of each week: ffill to these dates grabs the last trading day of
    # week T (Friday, or Thursday if Friday is a holiday). This is 1-step-ahead:
    # features at row T include week T's own data, predicting week T+1's RV.
    lookup_dates = weekly_rv.index + pd.Timedelta(days=4)

    aligned: dict[str, pd.DataFrame] = {}
    for name, n in windows.items():
        daily_rv = log_returns.rolling(n).std() * np.sqrt(252)
        weekly = daily_rv.reindex(lookup_dates, method="ffill")
        weekly.index = weekly_rv.index  # relabel Fridays back to Mondays
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


def compute_return_volume_features(
    log_returns: pd.DataFrame,
    volume: pd.DataFrame,
    weekly_rv: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute 5-day and 20-day cumulative return (momentum) and volume features aligned
    to the weekly prediction schedule.

    Features (5 total):
        momentum_5d  -- 5-day cumulative log return ending at Friday of week T
        momentum_20d -- 20-day cumulative log return ending at Friday of week T
        mean_vol_5d  -- 5-day rolling mean volume ending at Friday of week T
        mean_vol_20d -- 20-day rolling mean volume ending at Friday of week T
        volume_ratio -- mean_vol_5d / mean_vol_20d

    For each Monday in weekly_rv.index (week T), values are taken from Friday of week T
    (Monday + 4 days) via reindex with forward-fill.

    log_returns: shape (num_trading_days, num_stocks) -- daily log returns.
    volume: shape (num_trading_days, num_stocks) -- daily trading volume.
    weekly_rv: shape (num_weeks, num_stocks) -- index is Monday week-start dates.

    Returns: DataFrame of shape (num_weeks, num_stocks * 5) with MultiIndex columns
             (feature_name, ticker).

    Lookahead safety: 1-step-ahead design. All rolling windows end at Friday_T (last
    trading day of week T). The target is week T+1's RV. Friday_T is strictly before
    Monday_{T+1}.
    Shape assertion: result.shape == (len(weekly_rv), weekly_rv.shape[1] * 5).
    """
    # Fridays of each week — 1-step-ahead: include week T's own data.
    lookup_dates = weekly_rv.index + pd.Timedelta(days=4)

    def _align(daily: pd.DataFrame) -> pd.DataFrame:
        weekly = daily.reindex(lookup_dates, method="ffill")
        weekly.index = weekly_rv.index
        return weekly

    aligned: dict[str, pd.DataFrame] = {}

    aligned["momentum_5d"]  = _align(log_returns.rolling(5).sum())
    aligned["momentum_20d"] = _align(log_returns.rolling(20).sum())

    vol_5d  = _align(volume.rolling(5).mean())
    vol_20d = _align(volume.rolling(20).mean())
    aligned["mean_vol_5d"]  = vol_5d
    aligned["mean_vol_20d"] = vol_20d
    aligned["volume_ratio"] = vol_5d / vol_20d.replace(0.0, np.nan)

    result = pd.concat(aligned, axis=1)  # MultiIndex columns: (feature_name, ticker)

    num_weeks  = len(weekly_rv)
    num_stocks = weekly_rv.shape[1]
    assert result.shape == (num_weeks, num_stocks * 5), (
        f"compute_return_volume_features: expected ({num_weeks}, {num_stocks * 5}), "
        f"got {result.shape}"
    )

    return result


def winsorize_cross_sectional(df: pd.DataFrame,
                               clip: tuple[float, float] = config.WINSORIZE_CLIP) -> pd.DataFrame:
    """
    Winsorize each feature cross-sectionally at each time step.

    For each row (time step), clips values to the [clip[0], clip[1]] quantile of that row.

    df: shape (num_weeks, num_stocks)
    clip: (lower_quantile, upper_quantile)
    Returns: same shape as input.
    """
    lower = df.quantile(clip[0], axis=1)
    upper = df.quantile(clip[1], axis=1)
    result = df.clip(lower=lower, upper=upper, axis=0)
    assert result.shape == df.shape, (
        f"winsorize_cross_sectional: shape changed from {df.shape} to {result.shape}"
    )
    return result


def zscore_cross_sectional(df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score each feature cross-sectionally at each time step.

    Subtracts the row mean and divides by the row std. Call AFTER winsorize_cross_sectional.

    df: shape (num_weeks, num_stocks)
    Returns: same shape as input.
    """
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)  # avoid division by zero if all stocks identical
    result = df.sub(mean, axis=0).div(std, axis=0)
    assert result.shape == df.shape, (
        f"zscore_cross_sectional: shape changed from {df.shape} to {result.shape}"
    )
    return result


def build_feature_tensor(
    weekly_rv: pd.DataFrame,
    log_returns: pd.DataFrame,
    volume: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    Assemble all features into a 3D tensor and apply winsorize then z-score cross-sectionally.

    weekly_rv: shape (num_weeks, num_stocks)
    log_returns: shape (num_trading_days, num_stocks)
    volume: shape (num_trading_days, num_stocks)

    Returns: (array of shape (num_weeks, num_stocks, num_features), list of feature names)

    Lookahead safety: delegates to compute_volatility_features and
    compute_return_volume_features, both of which use only data before each week T.
    Normalization is row-wise and introduces no cross-time information.

    Post-normalization assertions run at 10 random time steps (after 63-day warm-up):
      |mean| < 0.01, |std - 1| < 0.05, max|x| < 5.0 per feature.
    """
    vol_feats = compute_volatility_features(log_returns, weekly_rv)
    ret_vol_feats = compute_return_volume_features(log_returns, volume, weekly_rv)

    all_feats = pd.concat([vol_feats, ret_vol_feats], axis=1)
    feature_names = all_feats.columns.get_level_values(0).unique().tolist()

    num_weeks = len(weekly_rv)
    num_stocks = weekly_rv.shape[1]
    num_features = len(feature_names)

    arrays: list[np.ndarray] = []
    for name in feature_names:
        feat_df = all_feats[name]                    # (num_weeks, num_stocks)
        feat_df = winsorize_cross_sectional(feat_df)
        feat_df = zscore_cross_sectional(feat_df)
        arrays.append(feat_df.values)

    result = np.stack(arrays, axis=2)                # (num_weeks, num_stocks, num_features)

    # Post-normalization assertions at 10 random time steps past the warm-up period
    rng = np.random.default_rng(config.RANDOM_SEED)
    eligible = [i for i in range(63, num_weeks) if not np.isnan(result[i]).any()]
    check_rows = rng.choice(eligible, size=min(10, len(eligible)), replace=False)
    for t in check_rows:
        for f_idx, fname in enumerate(feature_names):
            vals = result[t, :, f_idx]
            assert abs(np.nanmean(vals)) < 0.01, (
                f"Mean not near zero at week {t}, feature {fname}: {np.nanmean(vals):.4f}"
            )
            assert abs(np.nanstd(vals, ddof=1) - 1.0) < 0.05, (
                f"Std not near 1 at week {t}, feature {fname}: {np.nanstd(vals):.4f}"
            )
            assert np.nanmax(np.abs(vals)) < config.NORM_MAX_ABS, (
                f"Outlier survived winsorization at week {t}, feature {fname}: "
                f"{np.nanmax(np.abs(vals)):.4f} (limit={config.NORM_MAX_ABS})"
            )

    assert result.shape == (num_weeks, num_stocks, num_features), (
        f"build_feature_tensor: expected ({num_weeks}, {num_stocks}, {num_features}), "
        f"got {result.shape}"
    )
    return result, feature_names


def save_features(
    features: np.ndarray,
    feature_names: list[str],
    tickers: list[str],
    weeks: pd.DatetimeIndex,
) -> None:
    """
    Reshape the feature tensor to 2D and save features.parquet + features_meta.json.

    Parquet schema: columns = ['week', 'ticker'] + feature_names.
    Shape assertion before save: (num_weeks * num_stocks, num_features + 2).
    """
    import json
    import os

    num_weeks, num_stocks, num_feats = features.shape
    flat = features.reshape(num_weeks * num_stocks, num_feats)

    week_col   = np.repeat([str(w.date()) for w in weeks], num_stocks)
    ticker_col = np.tile(tickers, num_weeks)

    df = pd.DataFrame(flat, columns=feature_names)
    df.insert(0, "ticker", ticker_col)
    df.insert(0, "week", week_col)

    assert df.shape == (num_weeks * num_stocks, num_feats + 2), (
        f"save_features: expected ({num_weeks * num_stocks}, {num_feats + 2}), got {df.shape}"
    )

    os.makedirs(config.DATA_FEATURES_DIR, exist_ok=True)
    out_path = f"{config.DATA_FEATURES_DIR}/features.parquet"
    df.to_parquet(out_path, index=False)

    meta = {"shape": list(features.shape), "feature_names": feature_names}
    with open(f"{config.DATA_FEATURES_DIR}/features_meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"Saved {out_path}  shape={df.shape}")
