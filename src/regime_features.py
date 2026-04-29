"""Market-regime feature engineering for the macro feature upgrade.

This module keeps global market/regime features separate from the stock-level
feature pipeline in ``src.features``. Feature rows are aligned to the existing
weekly prediction schedule: row T may use data through Friday of week T, while
the target row T is realized volatility in week T+1.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

import config


REGIME_FEATURE_VERSION = "regime_features_v1"

FRED_SERIES = {
    "treasury_10y_2y_spread": "T10Y2Y",
    # Moody's Baa corporate yield spread over 10Y Treasury. This is an
    # investment-grade credit-stress proxy with full train-period coverage.
    "ig_credit_spread": "BAA10Y",
}

REGIME_FEATURE_NAMES = [
    "vix_level",
    "vix_change_1w",
    "spy_rv_21d",
    "spy_return_1w",
    "spy_return_1m",
    "treasury_10y_2y_spread",
    "ig_credit_spread",
    "avg_pairwise_stock_correlation",
    "correlation_graph_density",
]


def _read_fred_series(series_id: str) -> pd.Series:
    """Download one FRED series as a daily float Series."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    with urlopen(url, timeout=30) as response:
        raw = pd.read_csv(response)
    raw["observation_date"] = pd.to_datetime(raw["observation_date"])
    values = pd.to_numeric(raw[series_id].replace(".", np.nan), errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float), index=raw["observation_date"], name=series_id)
    return series.loc[config.DATA_START : config.DATA_END].sort_index()


def download_macro_series(force: bool = False) -> pd.DataFrame:
    """Download and cache raw macro series used by regime features.

    Saves ``data/raw/macro_series.parquet``. VIX is converted from index points
    to decimal volatility units, while FRED spread series are converted from
    percentage points to decimal spread units.
    """
    raw_dir = Path(config.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "macro_series.parquet"

    if out_path.exists() and not force:
        cached = pd.read_parquet(out_path)
        cached.index = pd.to_datetime(cached.index)
        expected = {
            "vix_close",
            "spy_close",
            "treasury_10y_2y_spread",
            "ig_credit_spread",
        }
        credit_has_train_data = (
            "ig_credit_spread" in cached.columns
            and cached.loc[: config.TRAIN_END, "ig_credit_spread"].notna().any()
        )
        if expected.issubset(cached.columns) and credit_has_train_data:
            print(f"Cache hit: macro_series.parquet ({len(cached)} rows)")
            return cached

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for VIX and SPY downloads.") from exc

    market = yf.download(
        ["^VIX", "SPY"],
        start=config.DATA_START,
        end=config.DATA_END,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(market.columns, pd.MultiIndex):
        close = market["Close"].copy()
    else:
        close = market[["Close"]].copy()

    macro = pd.DataFrame(index=close.index)
    macro.index = pd.to_datetime(macro.index)
    macro["vix_close"] = close["^VIX"] / 100.0
    macro["spy_close"] = close["SPY"]

    for column, series_id in FRED_SERIES.items():
        macro[column] = _read_fred_series(series_id) / 100.0

    macro = macro.sort_index().ffill()
    macro = macro.loc[config.DATA_START : config.DATA_END]

    assert len(macro) > 0, "Macro series download returned no rows."
    assert not macro[["vix_close", "spy_close"]].isna().all().any(), (
        "VIX or SPY macro series is entirely missing."
    )

    macro.to_parquet(out_path)
    print(f"Saved: macro_series.parquet ({len(macro)} rows) -> {raw_dir}/")
    return macro


def _align_to_feature_fridays(frame: pd.DataFrame, weekly_index: pd.Index) -> pd.DataFrame:
    """Align daily data to Friday of each feature week and relabel to Mondays."""
    lookup_dates = pd.DatetimeIndex(weekly_index) + pd.Timedelta(days=4)
    aligned = frame.reindex(lookup_dates, method="ffill")
    aligned.index = pd.DatetimeIndex(weekly_index)
    return aligned


def _compute_stock_correlation_regimes(
    log_returns: pd.DataFrame,
    weekly_index: pd.Index,
    threshold: float = config.CORR_THRESHOLD,
) -> pd.DataFrame:
    """Compute average pairwise correlation and threshold graph density by week."""
    rows: list[dict[str, float | pd.Timestamp]] = []
    num_stocks = log_returns.shape[1]
    possible_pairs = num_stocks * (num_stocks - 1) / 2
    upper_mask = np.triu(np.ones((num_stocks, num_stocks), dtype=bool), k=1)

    for week in pd.DatetimeIndex(weekly_index):
        friday = week + pd.Timedelta(days=4)
        window = log_returns.loc[:friday].tail(config.CORR_LOOKBACK_DAYS)
        corr = np.corrcoef(window.to_numpy(dtype=float), rowvar=False)
        pair_corr = corr[upper_mask]
        valid = ~np.isnan(pair_corr)
        if valid.any():
            avg_corr = float(np.nanmean(pair_corr[valid]))
            density = float(np.mean(np.abs(pair_corr[valid]) >= threshold))
        else:
            avg_corr = float("nan")
            density = float("nan")
        rows.append(
            {
                "week": week,
                "avg_pairwise_stock_correlation": avg_corr,
                "correlation_graph_density": density if possible_pairs > 0 else float("nan"),
            }
        )

    return pd.DataFrame(rows).set_index("week")


def build_regime_features(
    weekly_rv: pd.DataFrame,
    log_returns: pd.DataFrame,
    macro_series: pd.DataFrame | None = None,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """Build weekly market-regime features aligned to ``weekly_rv.index``."""
    if macro_series is None:
        macro_series = download_macro_series(force=force_download)
    macro_series = macro_series.copy()
    macro_series.index = pd.to_datetime(macro_series.index)

    spy_log_return = np.log(macro_series["spy_close"] / macro_series["spy_close"].shift(1))
    daily = pd.DataFrame(index=macro_series.index)
    daily["vix_level"] = macro_series["vix_close"]
    daily["vix_change_1w"] = macro_series["vix_close"].diff(5)
    daily["spy_rv_21d"] = spy_log_return.rolling(21).std() * np.sqrt(252)
    daily["spy_return_1w"] = spy_log_return.rolling(5).sum()
    daily["spy_return_1m"] = spy_log_return.rolling(21).sum()
    daily["treasury_10y_2y_spread"] = macro_series["treasury_10y_2y_spread"]
    daily["ig_credit_spread"] = macro_series["ig_credit_spread"]

    weekly_macro = _align_to_feature_fridays(daily, weekly_rv.index)
    corr_regimes = _compute_stock_correlation_regimes(log_returns, weekly_rv.index)
    regime = weekly_macro.join(corr_regimes, how="left")
    regime = regime[REGIME_FEATURE_NAMES]

    assert regime.shape == (len(weekly_rv), len(REGIME_FEATURE_NAMES)), (
        f"Expected regime shape {(len(weekly_rv), len(REGIME_FEATURE_NAMES))}, got {regime.shape}"
    )
    assert regime.index.equals(pd.DatetimeIndex(weekly_rv.index)), "Regime feature index misaligned."
    return regime


def save_regime_features(regime_features: pd.DataFrame) -> tuple[Path, Path]:
    """Save weekly regime features and metadata under ``data/features``."""
    features_dir = Path(config.DATA_FEATURES_DIR)
    features_dir.mkdir(parents=True, exist_ok=True)
    features_path = features_dir / "regime_features.parquet"
    meta_path = features_dir / "regime_features_meta.json"

    regime_features.to_parquet(features_path)
    meta = {
        "feature_version": REGIME_FEATURE_VERSION,
        "shape": list(regime_features.shape),
        "feature_names": list(regime_features.columns),
        "lookahead_rule": "Feature row T uses data through Friday of week T; target row T is RV in week T+1.",
        "raw_macro_path": "data/raw/macro_series.parquet",
        "warmup_missing_values": int(regime_features.isna().sum().sum()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {features_path} shape={regime_features.shape}")
    return features_path, meta_path


def build_and_save_regime_features(
    weekly_rv: pd.DataFrame,
    log_returns: pd.DataFrame,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """Convenience entry point for notebooks and scripts."""
    regime = build_regime_features(
        weekly_rv=weekly_rv,
        log_returns=log_returns,
        force_download=force_download,
    )
    save_regime_features(regime)
    return regime
