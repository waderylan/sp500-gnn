"""
Inverse-volatility portfolio construction, weekly rebalancing, and performance metrics.

Weights = 1/predicted_RV, clipped at MAX_WEIGHT, normalized to sum to 1.
Transaction costs deducted at TRANSACTION_COST_BPS per unit of turnover.
Sharpe ratio uses FRED DTB3 T-bill rates as risk-free rate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_tbill_rates(start: str, end: str) -> pd.Series:
    """
    Load 3-month T-bill rates (FRED series DTB3) for the given date range.

    If data/raw/tbill_rates.parquet exists, loads from disk. Otherwise fetches
    from FRED via pandas_datareader and saves to disk.

    start, end: date strings passed to DataReader (e.g. '2024-01-01').
    Returns: daily pd.Series of annualized rates in decimal (e.g. 0.0525 = 5.25%).
             Values are forward-filled so every calendar day has a rate.

    Lookahead safety: pure historical data fetch, no model inputs.
    """
    path = Path(config.DATA_RAW_DIR) / "tbill_rates.parquet"
    if path.exists():
        s = pd.read_parquet(path).squeeze()
        s.name = "DTB3"
        return s

    try:
        import pandas_datareader.data as web
        raw = web.DataReader("DTB3", "fred", start=start, end=end)
        s = raw["DTB3"].dropna() / 100.0  # percent -> decimal
    except Exception as exc:
        # FRED unavailable — fall back to 5% annualized (approximate 2024-2025 average).
        # Document this in the paper if the fallback is used.
        print(f"WARNING: FRED download failed ({exc}). Using fixed 5% annual rate as fallback.")
        date_range = pd.date_range(start=start, end=end, freq="D")
        s = pd.Series(0.05, index=date_range, name="DTB3")
        path.parent.mkdir(parents=True, exist_ok=True)
        s.to_frame().to_parquet(path)
        return s

    s = s.reindex(
        pd.date_range(start=raw.index.min(), end=raw.index.max(), freq="D")
    ).ffill()
    s.name = "DTB3"
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_frame().to_parquet(path)
    return s


def compute_weekly_returns(log_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily log returns to weekly cumulative log returns.

    Uses Monday-anchored weeks (same convention as weekly_rv). Each row is the
    sum of daily log returns for all trading days in that calendar week.

    log_returns_df: shape (num_trading_days, num_stocks), daily log returns.
    Returns: shape (num_weeks, num_stocks), indexed by Monday of each week.

    Shape assertion: output.shape[1] == log_returns_df.shape[1].
    Lookahead safety: pure aggregation within each week, no forward-looking windows.
    """
    weekly = log_returns_df.resample("W-MON", closed="left", label="left").sum()
    # Drop weeks with no trading days (all zeros would be misleading; NaN is cleaner)
    trading_day_counts = log_returns_df.resample(
        "W-MON", closed="left", label="left"
    ).count()
    weekly = weekly.where(trading_day_counts > 0)

    assert weekly.shape[1] == log_returns_df.shape[1], (
        f"Column count changed: {weekly.shape[1]} vs {log_returns_df.shape[1]}"
    )
    return weekly


# ---------------------------------------------------------------------------
# Core portfolio functions
# ---------------------------------------------------------------------------

def compute_weights(predicted_rv: np.ndarray) -> np.ndarray:
    """
    Compute inverse-volatility portfolio weights from predicted RV.

    Pipeline:
        1. raw_weights = 1.0 / predicted_rv  (NaN or non-positive -> 0)
        2. weights = raw_weights / raw_weights.sum()
        3. weights = weights.clip(max=config.MAX_WEIGHT)
        4. weights = weights / weights.sum()  (re-normalize after clip)

    predicted_rv: ndarray of shape (num_stocks,).
    Returns: ndarray of shape (num_stocks,) summing to 1.0.

    Shape assertion: result.shape == predicted_rv.shape, abs(result.sum() - 1.0) < 1e-6.
    Lookahead safety: pure arithmetic on pre-computed predictions.
    """
    raw = np.where(
        np.isnan(predicted_rv) | (predicted_rv <= 0.0),
        0.0,
        1.0 / predicted_rv,
    )
    total = raw.sum()
    if total == 0.0:
        # All stocks have invalid predictions: fall back to equal weight
        raw = np.ones(len(predicted_rv), dtype=float)
        total = raw.sum()

    weights = raw / total
    weights = np.clip(weights, 0.0, config.MAX_WEIGHT)
    weights = weights / weights.sum()

    assert weights.shape == predicted_rv.shape, (
        f"Shape mismatch: {weights.shape} vs {predicted_rv.shape}"
    )
    assert abs(weights.sum() - 1.0) < 1e-6, (
        f"Weights do not sum to 1: {weights.sum()}"
    )
    return weights


def compute_portfolio_returns(
    weights: np.ndarray,
    actual_returns: np.ndarray,
    prev_weights: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute gross and net portfolio return for one week, deducting transaction costs.

    weights: current week's portfolio weights, shape (num_stocks,).
    actual_returns: realized log returns this week, shape (num_stocks,).
    prev_weights: prior week's weights for turnover calculation, or None for week 1.

    Returns dict with keys: 'gross_return', 'net_return', 'turnover', 'max_weight'.

    Lookahead safety: uses only current-week realized returns for performance measurement.
    """
    valid = ~np.isnan(actual_returns)
    gross = float(np.dot(weights[valid], actual_returns[valid]))

    if prev_weights is None:
        turnover = 0.0
    else:
        turnover = float(np.abs(weights - prev_weights).sum())

    cost = turnover * (config.TRANSACTION_COST_BPS / 10_000.0)
    net = gross - cost

    return {
        "gross_return": gross,
        "net_return": net,
        "turnover": turnover,
        "max_weight": float(weights.max()),
    }


def compute_sharpe(
    net_returns: np.ndarray,
    tbill_rates: pd.Series,
    weeks: pd.DatetimeIndex,
) -> float:
    """
    Compute annualized Sharpe ratio net of transaction costs.

    Sharpe = (annualized_return - mean_annualized_tbill) / annualized_volatility.

    net_returns: weekly net portfolio returns, shape (num_weeks,).
    tbill_rates: daily FRED DTB3 rates in decimal, indexed by date.
    weeks: DatetimeIndex of portfolio weeks (Monday of the holding week).

    Returns: annualized Sharpe ratio (float).
    """
    aligned_rf = (
        tbill_rates.reindex(weeks).ffill().bfill().values
    )  # annualized decimal rates aligned to each holding week

    ann_return = float(np.nanmean(net_returns)) * 52
    mean_ann_rf = float(np.nanmean(aligned_rf))
    ann_vol = float(np.nanstd(net_returns, ddof=1)) * np.sqrt(52)

    if ann_vol == 0.0:
        return float("nan")
    return (ann_return - mean_ann_rf) / ann_vol


def run_backtest(
    predicted_rv_by_week: np.ndarray,
    actual_returns_by_week: np.ndarray,
    tbill_rates: pd.Series,
    weeks: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Run the full portfolio backtest over the evaluation period.

    predicted_rv_by_week:   shape (num_weeks, num_stocks) — predictions used to build weights.
    actual_returns_by_week: shape (num_weeks, num_stocks) — realized returns during the holding week.
    tbill_rates:            daily FRED DTB3 rates in decimal.
    weeks:                  DatetimeIndex of the holding weeks (length num_weeks).

    Returns DataFrame with columns: ['week', 'gross_return', 'net_return', 'turnover', 'max_weight'].

    Shape assertion: result.shape == (len(weeks), 5).
    Lookahead safety: weights for holding week i are computed from predicted_rv_by_week[i],
    which was predicted using features from the prior feature week only.
    """
    assert predicted_rv_by_week.shape == actual_returns_by_week.shape, (
        f"Shape mismatch: {predicted_rv_by_week.shape} vs {actual_returns_by_week.shape}"
    )
    assert len(weeks) == predicted_rv_by_week.shape[0], (
        f"weeks length {len(weeks)} != array rows {predicted_rv_by_week.shape[0]}"
    )

    rows: list[dict] = []
    prev_weights: np.ndarray | None = None

    for i, week in enumerate(weeks):
        weights = compute_weights(predicted_rv_by_week[i])
        result = compute_portfolio_returns(
            weights, actual_returns_by_week[i], prev_weights
        )
        rows.append({"week": week, **result})
        prev_weights = weights

    df = pd.DataFrame(rows, columns=["week", "gross_return", "net_return", "turnover", "max_weight"])

    assert df.shape == (len(weeks), 5), (
        f"Expected shape ({len(weeks)}, 5), got {df.shape}"
    )
    return df


# ---------------------------------------------------------------------------
# Summary and comparison
# ---------------------------------------------------------------------------

def summarize_backtest(
    backtest_df: pd.DataFrame,
    tbill_rates: pd.Series,
) -> dict[str, float]:
    """
    Compute all six portfolio metrics from a completed backtest DataFrame.

    backtest_df: output of run_backtest — columns ['week', 'gross_return', 'net_return',
                 'turnover', 'max_weight'].
    tbill_rates: daily FRED DTB3 rates in decimal.

    Returns dict with keys: 'ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
                             'avg_turnover', 'max_single_stock_weight'.
    """
    net = backtest_df["net_return"].values
    weeks = pd.DatetimeIndex(backtest_df["week"])

    ann_return = float(np.mean(net)) * 52
    ann_vol = float(np.std(net, ddof=1)) * np.sqrt(52)
    sharpe = compute_sharpe(net, tbill_rates, weeks)

    # Max drawdown from cumulative wealth curve
    cum = np.cumprod(1.0 + net)
    running_max = np.maximum.accumulate(cum)
    drawdowns = (cum - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    avg_turnover = float(backtest_df["turnover"].mean())
    max_weight = float(backtest_df["max_weight"].max())

    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "max_single_stock_weight": max_weight,
    }


def run_all_model_backtests(
    log_returns_df: pd.DataFrame,
    tbill_rates: pd.Series,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the portfolio backtest for all six models plus the equal-weight benchmark.

    Loads test prediction parquets from config.DATA_RESULTS_DIR. For each model,
    the prediction at test week T is used to set weights for the holding week T+1.
    Actual portfolio returns are the realized stock log returns during week T+1.

    log_returns_df: daily log returns, shape (num_trading_days, num_stocks).
    tbill_rates:    daily FRED DTB3 rates in decimal.
    tickers:        ordered ticker list matching prediction columns.

    Returns:
        portfolio_returns_df: DataFrame with columns
            ['week', 'gross_return', 'net_return', 'turnover', 'max_weight', 'model'],
            concatenated across all seven models.
        metrics_table:        DataFrame indexed by model name with columns
            ['ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
             'avg_turnover', 'max_single_stock_weight'].

    Saves:
        config.DATA_RESULTS_DIR / portfolio_returns.parquet
        config.DATA_RESULTS_DIR / portfolio_metrics_table.csv

    Shape assertions:
        portfolio_returns_df has 7 * num_test_weeks rows.
        metrics_table has 7 rows.

    Lookahead safety:
        Weights for holding week T+1 are computed from predictions at feature week T.
        Features at week T use data only through Friday of week T.
        Actual returns during week T+1 are not observable until after Friday of week T+1.
    """
    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "HAR per-stock":  "test_preds_har.parquet",
        "HAR pooled":     "test_preds_har_pooled.parquet",
        "LSTM":           "test_preds_lstm.parquet",
        "GNN-Correlation": "test_preds_gnn_corr.parquet",
        "GNN-Sector":     "test_preds_gnn_sector.parquet",
        "GNN-Granger":    "test_preds_gnn_granger.parquet",
    }

    # Load predictions for each model; align columns to tickers
    pred_dfs: dict[str, pd.DataFrame] = {}
    for label, fname in model_files.items():
        df = pd.read_parquet(results_dir / fname)
        pred_dfs[label] = df.reindex(columns=tickers)

    # All models share the same test weeks (Monday of feature week T)
    test_weeks = pred_dfs["HAR per-stock"].index  # e.g. 2024-01-01 ... 2025-12-15

    # Compute weekly returns; holding weeks are test_weeks + 7 days
    weekly_ret = compute_weekly_returns(log_returns_df).reindex(columns=tickers)
    holding_weeks = test_weeks + pd.Timedelta(days=7)  # Monday of T+1

    # Actual returns during each holding week, shape (num_test_weeks, num_stocks)
    actual_ret = weekly_ret.reindex(holding_weeks).values

    n_test = len(test_weeks)

    # Add equal-weight benchmark: constant predicted_rv = 1 for all stocks
    # 1/n_stocks < MAX_WEIGHT for 465 stocks, so clip has no effect -> true equal weights
    n_stocks = len(tickers)
    equal_rv = np.ones((n_test, n_stocks), dtype=float)
    pred_dfs["Equal-weight"] = pd.DataFrame(
        equal_rv, index=test_weeks, columns=tickers
    )

    all_backtests: list[pd.DataFrame] = []
    all_summaries: dict[str, dict[str, float]] = {}

    for label, pred_df in pred_dfs.items():
        pred_arr = pred_df.values.astype(float)  # (n_test, n_stocks)
        backtest_df = run_backtest(pred_arr, actual_ret, tbill_rates, holding_weeks)
        summary = summarize_backtest(backtest_df, tbill_rates)

        backtest_df["model"] = label
        all_backtests.append(backtest_df)
        all_summaries[label] = summary

    portfolio_returns_df = pd.concat(all_backtests, ignore_index=True)

    assert len(portfolio_returns_df) == 7 * n_test, (
        f"Expected {7 * n_test} rows, got {len(portfolio_returns_df)}"
    )

    metrics_table = (
        pd.DataFrame(all_summaries)
        .T.rename_axis("model")
        [["ann_return", "ann_vol", "sharpe", "max_drawdown",
          "avg_turnover", "max_single_stock_weight"]]
    )

    assert len(metrics_table) == 7, f"Expected 7 model rows, got {len(metrics_table)}"

    portfolio_returns_df.to_parquet(results_dir / "portfolio_returns.parquet", index=False)
    metrics_table.to_csv(results_dir / "portfolio_metrics_table.csv")

    return portfolio_returns_df, metrics_table
