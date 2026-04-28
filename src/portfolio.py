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


def compute_vol_target_scale(predicted_rv: np.ndarray, vol_target: float) -> float:
    """
    Compute the equity allocation scalar for a vol-targeted portfolio.

    Scale = min(vol_target / predicted_port_vol, 1.0), so the portfolio never
    levers up. The remaining (1 - scale) capital sits in cash earning zero.

    predicted_port_vol is the median predicted RV across all stocks. predicted_rv
    values are already annualized (trained on weekly_rv = std * sqrt(252)), so no
    further scaling is applied. The median is used instead of the mean to reduce
    sensitivity to prediction outliers at the tails.

    predicted_rv: shape (num_stocks,), predicted RV values for one week.
    vol_target:   annualized target volatility in decimal (e.g. 0.10 = 10%).
    Returns:      scalar in (0, 1].

    Lookahead safety: operates on pre-computed predictions only.
    """
    predicted_port_vol = float(np.nanmedian(predicted_rv))
    if predicted_port_vol <= 0.0:
        return 1.0
    return float(min(vol_target / predicted_port_vol, 1.0))


def run_backtest(
    predicted_rv_by_week: np.ndarray,
    actual_returns_by_week: np.ndarray,
    tbill_rates: pd.Series,
    weeks: pd.DatetimeIndex,
    vol_target: float | None = None,
) -> pd.DataFrame:
    """
    Run the full portfolio backtest over the evaluation period.

    predicted_rv_by_week:   shape (num_weeks, num_stocks) — predictions used to build weights.
    actual_returns_by_week: shape (num_weeks, num_stocks) — realized returns during the holding week.
    tbill_rates:            daily FRED DTB3 rates in decimal.
    weeks:                  DatetimeIndex of the holding weeks (length num_weeks).
    vol_target:             annualized target volatility in decimal (e.g. 0.10). When set, each
                            week's inverse-vol weights are scaled by compute_vol_target_scale()
                            before computing returns. The undeployed fraction earns zero (cash).
                            When None (default), behavior is identical to the original backtest.

    Returns DataFrame with columns:
        ['week', 'gross_return', 'net_return', 'turnover', 'max_weight', 'equity_weight'].
    equity_weight is the fraction of capital deployed in equities (1.0 when vol_target is None).

    Shape assertion: result.shape == (len(weeks), 6).
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
        inv_vol_weights = compute_weights(predicted_rv_by_week[i])

        if vol_target is not None:
            scale = compute_vol_target_scale(predicted_rv_by_week[i], vol_target)
            weights = scale * inv_vol_weights
        else:
            scale = 1.0
            weights = inv_vol_weights

        result = compute_portfolio_returns(weights, actual_returns_by_week[i], prev_weights)
        rows.append({"week": week, **result, "equity_weight": scale})
        prev_weights = weights

    df = pd.DataFrame(
        rows,
        columns=["week", "gross_return", "net_return", "turnover", "max_weight", "equity_weight"],
    )

    assert df.shape == (len(weeks), 6), (
        f"Expected shape ({len(weeks)}, 6), got {df.shape}"
    )
    return df


# ---------------------------------------------------------------------------
# Long-short portfolio
# ---------------------------------------------------------------------------

def build_long_short_portfolio(preds_df: pd.DataFrame, q: float = 0.25) -> pd.DataFrame:
    """
    Build a dollar-neutral long-short portfolio from predicted RV.

    At each week t, rank all stocks by predicted RV. Go long the bottom q fraction
    (lowest predicted vol) and short the top q fraction (highest predicted vol).
    Equal weights within each leg: long side sums to +1, short side sums to -1,
    net exposure = 0.

    preds_df: shape (num_weeks, num_stocks), predicted RV values.
    q:        quantile cutoff for each leg (default 0.25 = bottom/top 25%).
    Returns:  shape (num_weeks, num_stocks), weights in [-1/k, 0, +1/k].

    Shape assertion: output.shape == preds_df.shape.
    Lookahead safety: operates only on pre-computed predictions, no date-indexed windows.
    """
    weights = pd.DataFrame(0.0, index=preds_df.index, columns=preds_df.columns)

    for t in preds_df.index:
        row = preds_df.loc[t].dropna()
        n = len(row)
        k = max(1, int(n * q))

        ranked = row.sort_values()
        long_idx  = ranked.index[:k]   # lowest predicted vol -> long
        short_idx = ranked.index[-k:]  # highest predicted vol -> short

        weights.loc[t, long_idx]  =  1.0 / k
        weights.loc[t, short_idx] = -1.0 / k

    assert weights.shape == preds_df.shape, (
        f"Shape mismatch: {weights.shape} vs {preds_df.shape}"
    )
    return weights


def _compute_long_short_return(
    weights: np.ndarray,
    actual_returns: np.ndarray,
    prev_weights: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute gross and net return for one week of a dollar-neutral long-short portfolio.

    Dollar-neutral means weights sum to 0, so standard sum-to-1 assertions do not apply.
    Transaction costs are applied to the same turnover formula as the long-only backtest.
    The Sharpe of the resulting return series measures cross-sectional signal quality,
    not total market return.

    weights:        current weights, shape (num_stocks,), sums to ~0.
    actual_returns: realized log returns, shape (num_stocks,).
    prev_weights:   prior week's weights for turnover calculation, or None for week 1.

    Returns dict with keys: 'gross_return', 'net_return', 'turnover', 'max_long_weight'.
    """
    valid = ~np.isnan(actual_returns)
    gross = float(np.dot(weights[valid], actual_returns[valid]))

    if prev_weights is None:
        turnover = 0.0
    else:
        turnover = float(np.abs(weights - prev_weights).sum())

    cost = turnover * (config.TRANSACTION_COST_BPS / 10_000.0)
    net  = gross - cost

    long_weights = weights[weights > 0]
    max_long = float(long_weights.max()) if len(long_weights) > 0 else 0.0

    return {
        "gross_return":    gross,
        "net_return":      net,
        "turnover":        turnover,
        "max_long_weight": max_long,
    }


def run_all_model_long_short_backtests(
    log_returns_df: pd.DataFrame,
    tbill_rates: pd.Series,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a dollar-neutral long-short backtest for all six models.

    Uses config.LONG_SHORT_QUANTILE for the top/bottom leg cutoff. Loads the same
    test_preds_*.parquet files as run_all_model_backtests(). No equal-weight benchmark
    is included: a dollar-neutral equal-weight portfolio has zero expected return by
    construction.

    log_returns_df: daily log returns, shape (num_trading_days, num_stocks).
    tbill_rates:    daily FRED DTB3 rates in decimal.
    tickers:        ordered ticker list matching prediction columns.

    Returns:
        ls_returns_df:  DataFrame with columns
            ['week', 'gross_return', 'net_return', 'turnover', 'max_long_weight', 'model'],
            concatenated across all six models.
        metrics_table:  DataFrame indexed by model name with columns
            ['ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
             'avg_turnover', 'max_long_weight'].

    Saves:
        config.DATA_RESULTS_DIR / portfolio_ls_returns.parquet
        config.DATA_RESULTS_DIR / portfolio_ls_metrics_table.csv

    Shape assertions:
        ls_returns_df has 6 * num_test_weeks rows.
        metrics_table has 6 rows.

    Lookahead safety: identical to run_all_model_backtests() — weights at week T are
    built from predictions at feature week T; actual returns are from holding week T+1.
    """
    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "GNN-Correlation": "test_preds_gnn_corr.parquet",
        "GNN-Sector":      "test_preds_gnn_sector.parquet",
        "GNN-Granger":     "test_preds_gnn_granger.parquet",
        "GNN-Ensemble":    "test_preds_gnn_ensemble.parquet",
        "HAR per-stock":   "test_preds_har.parquet",
        "HAR pooled":      "test_preds_har_pooled.parquet",
        "LSTM":            "test_preds_lstm.parquet",
    }

    pred_dfs: dict[str, pd.DataFrame] = {}
    for label, fname in model_files.items():
        df = pd.read_parquet(results_dir / fname)
        pred_dfs[label] = df.reindex(columns=tickers)

    test_weeks   = pred_dfs["GNN-Correlation"].index
    weekly_ret   = compute_weekly_returns(log_returns_df).reindex(columns=tickers)
    holding_weeks = test_weeks + pd.Timedelta(days=7)
    actual_ret   = weekly_ret.reindex(holding_weeks).values  # (n_test, n_stocks)

    n_test   = len(test_weeks)
    n_stocks = len(tickers)

    all_backtests: list[pd.DataFrame] = []
    all_summaries: dict[str, dict[str, float]] = {}

    for label, pred_df in pred_dfs.items():
        ls_weights_df = build_long_short_portfolio(pred_df, q=config.LONG_SHORT_QUANTILE)
        weights_arr   = ls_weights_df.values.astype(float)  # (n_test, n_stocks)

        rows: list[dict] = []
        prev_w: np.ndarray | None = None

        for i, week in enumerate(holding_weeks):
            result = _compute_long_short_return(weights_arr[i], actual_ret[i], prev_w)
            rows.append({"week": week, **result})
            prev_w = weights_arr[i]

        backtest_df          = pd.DataFrame(rows)
        backtest_df["model"] = label

        # Reuse summarize_backtest on the same-shaped DataFrame structure
        net      = backtest_df["net_return"].values
        weeks_di = pd.DatetimeIndex(backtest_df["week"])

        ann_return = float(np.mean(net)) * 52
        ann_vol    = float(np.std(net, ddof=1)) * np.sqrt(52)
        sharpe     = compute_sharpe(net, tbill_rates, weeks_di)

        cum         = np.cumprod(1.0 + net)
        running_max = np.maximum.accumulate(cum)
        drawdowns   = (cum - running_max) / running_max
        max_drawdown = float(drawdowns.min())

        all_backtests.append(backtest_df)
        all_summaries[label] = {
            "ann_return":      ann_return,
            "ann_vol":         ann_vol,
            "sharpe":          sharpe,
            "max_drawdown":    max_drawdown,
            "avg_turnover":    float(backtest_df["turnover"].mean()),
            "max_long_weight": float(backtest_df["max_long_weight"].max()),
        }

    ls_returns_df = pd.concat(all_backtests, ignore_index=True)

    assert len(ls_returns_df) == 7 * n_test, (
        f"Expected {7 * n_test} rows, got {len(ls_returns_df)}"
    )

    metrics_table = (
        pd.DataFrame(all_summaries)
        .T.rename_axis("model")
        [["ann_return", "ann_vol", "sharpe", "max_drawdown",
          "avg_turnover", "max_long_weight"]]
    )

    assert len(metrics_table) == 7, f"Expected 7 model rows, got {len(metrics_table)}"

    ls_returns_df.to_parquet(results_dir / "portfolio_ls_returns.parquet", index=False)
    metrics_table.to_csv(results_dir / "portfolio_ls_metrics_table.csv")

    return ls_returns_df, metrics_table


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
        "GNN-Ensemble":   "test_preds_gnn_ensemble.parquet",
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

    assert len(portfolio_returns_df) == 8 * n_test, (
        f"Expected {8 * n_test} rows, got {len(portfolio_returns_df)}"
    )

    metrics_table = (
        pd.DataFrame(all_summaries)
        .T.rename_axis("model")
        [["ann_return", "ann_vol", "sharpe", "max_drawdown",
          "avg_turnover", "max_single_stock_weight"]]
    )

    assert len(metrics_table) == 8, f"Expected 8 model rows, got {len(metrics_table)}"

    portfolio_returns_df.to_parquet(results_dir / "portfolio_returns.parquet", index=False)
    metrics_table.to_csv(results_dir / "portfolio_metrics_table.csv")

    return portfolio_returns_df, metrics_table


# ---------------------------------------------------------------------------
# Volatility-targeted portfolio
# ---------------------------------------------------------------------------

def run_all_model_backtests_vol_target(
    log_returns_df: pd.DataFrame,
    tbill_rates: pd.Series,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a vol-targeted backtest for all six models using config.VOL_TARGET.

    Each week's inverse-vol weights are scaled by compute_vol_target_scale() so the
    portfolio's predicted annualized volatility equals VOL_TARGET. When predicted vol
    is below target, scale = 1.0 (full exposure). When above, scale < 1 and the
    remaining capital sits in cash earning zero.

    The equal-weight benchmark is excluded here. Its dummy predicted_rv = 1.0 gives
    predicted_port_vol = sqrt(52) ≈ 7.2, so scale ≈ 0.014 — essentially all cash.
    That is not a useful benchmark for this construction.

    log_returns_df: daily log returns, shape (num_trading_days, num_stocks).
    tbill_rates:    daily FRED DTB3 rates in decimal.
    tickers:        ordered ticker list matching prediction columns.

    Returns:
        vt_returns_df:  DataFrame with columns
            ['week', 'gross_return', 'net_return', 'turnover', 'max_weight',
             'equity_weight', 'model'],
            concatenated across all six models.
        metrics_table:  DataFrame indexed by model name with columns
            ['ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
             'avg_turnover', 'max_single_stock_weight', 'avg_equity_weight'].

    Saves:
        config.DATA_RESULTS_DIR / portfolio_vt_returns.parquet
        config.DATA_RESULTS_DIR / portfolio_vt_metrics_table.csv

    Shape assertions:
        vt_returns_df has 6 * num_test_weeks rows.
        metrics_table has 6 rows.

    Lookahead safety: identical to run_all_model_backtests(). The scale factor is
    derived from the same pre-computed predictions used to build weights.
    """
    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "HAR per-stock":   "test_preds_har.parquet",
        "HAR pooled":      "test_preds_har_pooled.parquet",
        "LSTM":            "test_preds_lstm.parquet",
        "GNN-Correlation": "test_preds_gnn_corr.parquet",
        "GNN-Sector":      "test_preds_gnn_sector.parquet",
        "GNN-Granger":     "test_preds_gnn_granger.parquet",
        "GNN-Ensemble":    "test_preds_gnn_ensemble.parquet",
    }

    pred_dfs: dict[str, pd.DataFrame] = {}
    for label, fname in model_files.items():
        df = pd.read_parquet(results_dir / fname)
        pred_dfs[label] = df.reindex(columns=tickers)

    test_weeks = pred_dfs["HAR per-stock"].index
    weekly_ret = compute_weekly_returns(log_returns_df).reindex(columns=tickers)
    holding_weeks = test_weeks + pd.Timedelta(days=7)
    actual_ret = weekly_ret.reindex(holding_weeks).values

    n_test = len(test_weeks)

    all_backtests: list[pd.DataFrame] = []
    all_summaries: dict[str, dict[str, float]] = {}

    for label, pred_df in pred_dfs.items():
        pred_arr = pred_df.values.astype(float)
        backtest_df = run_backtest(
            pred_arr, actual_ret, tbill_rates, holding_weeks,
            vol_target=config.VOL_TARGET,
        )
        summary = summarize_backtest(backtest_df, tbill_rates)
        summary["avg_equity_weight"] = float(backtest_df["equity_weight"].mean())

        backtest_df["model"] = label
        all_backtests.append(backtest_df)
        all_summaries[label] = summary

    vt_returns_df = pd.concat(all_backtests, ignore_index=True)

    assert len(vt_returns_df) == 7 * n_test, (
        f"Expected {7 * n_test} rows, got {len(vt_returns_df)}"
    )

    metrics_table = (
        pd.DataFrame(all_summaries)
        .T.rename_axis("model")
        [["ann_return", "ann_vol", "sharpe", "max_drawdown",
          "avg_turnover", "max_single_stock_weight", "avg_equity_weight"]]
    )

    assert len(metrics_table) == 7, f"Expected 7 model rows, got {len(metrics_table)}"

    vt_returns_df.to_parquet(results_dir / "portfolio_vt_returns.parquet", index=False)
    metrics_table.to_csv(results_dir / "portfolio_vt_metrics_table.csv")

    return vt_returns_df, metrics_table


# ---------------------------------------------------------------------------
# Minimum variance portfolio
# ---------------------------------------------------------------------------

def build_minvar_weights(
    predicted_rv: np.ndarray,
    corr_matrix: np.ndarray,
    max_weight: float,
    constraint_matrix: "scipy.sparse.csc_matrix | None" = None,
) -> np.ndarray:
    """
    Solve the minimum variance portfolio given predicted stock volatilities and a
    precomputed realized correlation matrix.

    Covariance matrix: Sigma = D @ C @ D, where D = diag(predicted_rv) and C is
    the precomputed sample correlation matrix. This uses the GNN's predicted
    volatilities as the diagonal and realized pairwise correlations off-diagonal.

    Optimization: min w' Sigma w  s.t. sum(w) = 1, 0 <= w_i <= max_weight.
    Solved via OSQP (operator splitting QP solver). OSQP's per-iteration cost is
    O(n^2) vs O(n^3) for SLSQP's internal QP subproblem factorization, making it
    roughly 100x faster for n=465.

    After the solve, weights are clipped to [0, max_weight] and re-normalized to
    handle tiny bound violations from the solver.

    predicted_rv:        shape (n,), predicted annualized RV for one week.
    corr_matrix:         shape (n, n), precomputed sample correlation matrix ending
                         on Friday of feature week T. Must match ticker order.
    max_weight:          upper bound on any single stock weight.
    constraint_matrix:   optional precomputed sparse constraint matrix A of shape
                         (n+1, n) to avoid rebuilding it every call. When None,
                         builds it from scratch. Pass from the caller to save ~5ms
                         per solve when running many solves in sequence.
    Returns:             shape (n,), weights summing to 1.

    Shape assertion: result.shape == (n,), abs(result.sum() - 1.0) < 1e-6.
    Lookahead safety: corr_matrix must be computed from returns ending on Friday
    of week T; the holding week starts Monday_{T+1}. Friday_T < Monday_{T+1}.
    """
    import osqp
    import scipy.sparse as sp

    n = len(predicted_rv)

    # Replace non-positive or NaN predictions with a small positive floor so the
    # optimizer always has a valid covariance matrix.
    rv = np.where(np.isnan(predicted_rv) | (predicted_rv <= 0.0), 1e-6, predicted_rv)

    # Sigma = D @ C @ D  (element-wise: Sigma[i,j] = rv[i] * C[i,j] * rv[j])
    Sigma = (rv[:, None] * corr_matrix) * rv[None, :]

    # OSQP minimizes (1/2) x' P x + q' x  s.t. l <= A x <= u.
    # Setting P = 2*Sigma gives (1/2) w' (2 Sigma) w = w' Sigma w.
    P = sp.csc_matrix(2.0 * Sigma)
    q = np.zeros(n)

    # Constraint matrix A: first row enforces sum(w)=1, remaining n rows are
    # box bounds 0 <= w_i <= max_weight (encoded as l_i=0, u_i=max_weight).
    if constraint_matrix is None:
        A = sp.vstack([
            sp.csr_matrix(np.ones((1, n))),
            sp.eye(n),
        ]).tocsc()
    else:
        A = constraint_matrix

    l = np.concatenate([[1.0], np.zeros(n)])
    u = np.concatenate([[1.0], np.full(n, max_weight)])

    solver = osqp.OSQP()
    solver.setup(
        P, q, A, l, u,
        verbose=False,
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=4000,
        warm_starting=True,
        polish=True,       # refinement step; adds ~5ms but improves numerical quality
        polish_refine_iter=3,
    )
    res = solver.solve()

    if res.x is None or np.any(np.isnan(res.x)):
        w = np.ones(n) / n
    else:
        w = res.x

    w = np.clip(w, 0.0, max_weight)
    total = w.sum()
    w = w / total if total > 0.0 else np.ones(n) / n

    assert w.shape == (n,), f"Shape mismatch: {w.shape} vs ({n},)"
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights do not sum to 1: {w.sum()}"
    return w


def run_all_model_backtests_minvar(
    log_returns_df: pd.DataFrame,
    tbill_rates: pd.Series,
    tickers: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a minimum variance backtest for all six models.

    At each test week T (indexed by Monday_T), builds the covariance matrix from
    the model's predicted RVs and the trailing CORR_LOOKBACK_DAYS window of daily
    log returns ending on Friday_T = Monday_T + 4 days. Solves the constrained QP
    via build_minvar_weights() and holds the resulting portfolio for week T+1.

    Equal-weight is excluded. Its dummy predicted_rv = 1.0 gives a flat diagonal
    covariance that reduces the QP to an equal-weight solution, which adds nothing
    to the comparison.

    log_returns_df: daily log returns, shape (num_trading_days, num_stocks).
    tbill_rates:    daily FRED DTB3 rates in decimal.
    tickers:        ordered ticker list matching prediction columns.

    Returns:
        mv_returns_df:  DataFrame with columns
            ['week', 'gross_return', 'net_return', 'turnover', 'max_weight',
             'equity_weight', 'model'],
            concatenated across all six models.
        metrics_table:  DataFrame indexed by model name with columns
            ['ann_return', 'ann_vol', 'sharpe', 'max_drawdown',
             'avg_turnover', 'max_single_stock_weight'].

    Saves:
        config.DATA_RESULTS_DIR / portfolio_mv_returns.parquet
        config.DATA_RESULTS_DIR / portfolio_mv_metrics_table.csv

    Shape assertions:
        mv_returns_df has 6 * num_test_weeks rows.
        metrics_table has 6 rows.

    Lookahead safety: log_returns_window for week T ends on Friday_T = Monday_T + 4
    days. The holding week starts Monday_{T+1} = Monday_T + 7 days. Friday_T is
    strictly before Monday_{T+1}.
    """
    import time

    SEP  = "=" * 72
    SEP2 = "-" * 72

    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_files = {
        "HAR per-stock":   "test_preds_har.parquet",
        "HAR pooled":      "test_preds_har_pooled.parquet",
        "LSTM":            "test_preds_lstm.parquet",
        "GNN-Correlation": "test_preds_gnn_corr.parquet",
        "GNN-Sector":      "test_preds_gnn_sector.parquet",
        "GNN-Granger":     "test_preds_gnn_granger.parquet",
        "GNN-Ensemble":    "test_preds_gnn_ensemble.parquet",
    }

    print(SEP)
    print("MINIMUM VARIANCE BACKTEST")
    print(SEP)
    print(f"  MAX_WEIGHT        = {config.MAX_WEIGHT:.0%}")
    print(f"  CORR_LOOKBACK_DAYS= {config.CORR_LOOKBACK_DAYS}")
    print(f"  Models            = {list(model_files.keys())}")
    print()

    print("Loading prediction files...")
    pred_dfs: dict[str, pd.DataFrame] = {}
    for label, fname in model_files.items():
        path = results_dir / fname
        df = pd.read_parquet(path)
        pred_dfs[label] = df.reindex(columns=tickers)
        nan_frac = pred_dfs[label].isna().values.mean()
        print(f"  {label:<20s}  shape={df.shape}  NaN={nan_frac:.3%}  "
              f"pred_range=[{pred_dfs[label].values[~np.isnan(pred_dfs[label].values)].min():.4f}, "
              f"{pred_dfs[label].values[~np.isnan(pred_dfs[label].values)].max():.4f}]")

    test_weeks = pred_dfs["HAR per-stock"].index
    weekly_ret = compute_weekly_returns(log_returns_df).reindex(columns=tickers)
    holding_weeks = test_weeks + pd.Timedelta(days=7)
    actual_ret = weekly_ret.reindex(holding_weeks).values

    n_test   = len(test_weeks)
    n_stocks = len(tickers)
    print()
    print(f"  Test weeks        : {n_test}  ({test_weeks[0].date()} to {test_weeks[-1].date()})")
    print(f"  Holding weeks     : {holding_weeks[0].date()} to {holding_weeks[-1].date()}")
    print(f"  Stocks            : {n_stocks}")
    print(f"  actual_ret shape  : {actual_ret.shape}  NaN={np.isnan(actual_ret).mean():.3%}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: precompute correlation matrices
    # ------------------------------------------------------------------
    log_returns_aligned = log_returns_df.reindex(columns=tickers)

    print(SEP2)
    print("PHASE 1 — Precomputing correlation matrices")
    print(SEP2)
    t0_corr = time.perf_counter()
    corr_matrices: list[np.ndarray] = []
    nan_corr_counts: list[int] = []

    for idx, monday_t in enumerate(test_weeks):
        t_week = time.perf_counter()
        friday_t    = monday_t + pd.Timedelta(days=4)
        window_vals = log_returns_aligned.loc[:friday_t].tail(config.CORR_LOOKBACK_DAYS).values
        C = np.corrcoef(window_vals.T)
        np.fill_diagonal(C, 1.0)
        nan_count = int(np.isnan(C).sum())
        C = np.where(np.isnan(C), 0.0, C)
        corr_matrices.append(C)
        nan_corr_counts.append(nan_count)
        elapsed = time.perf_counter() - t_week

        if idx == 0 or idx == n_test - 1 or (idx + 1) % 20 == 0:
            # Density = fraction of off-diagonal pairs with |corr| > 0.3
            mask = np.ones_like(C, dtype=bool)
            np.fill_diagonal(mask, False)
            density = float((np.abs(C[mask]) > 0.3).mean())
            print(f"  [{idx+1:>3d}/{n_test}]  week={monday_t.date()}  window_rows={window_vals.shape[0]}  "
                  f"C_shape={C.shape}  NaN_zeroed={nan_count}  "
                  f"density(|r|>0.3)={density:.3f}  time={elapsed*1000:.1f}ms")

    t_corr_total = time.perf_counter() - t0_corr
    print()
    print(f"  Correlation phase complete: {n_test} matrices in {t_corr_total:.2f}s  "
          f"({t_corr_total/n_test*1000:.1f}ms/matrix)")
    print(f"  Total NaN cells zeroed: {sum(nan_corr_counts)}  "
          f"(max in any week: {max(nan_corr_counts)})")
    print()

    # ------------------------------------------------------------------
    # Phase 2: per-model QP solves
    # ------------------------------------------------------------------
    import scipy.sparse as sp

    # Constraint matrix A is the same for every solve (depends only on n and
    # max_weight, not on predictions or the correlation window). Build it once.
    A_constraint = sp.vstack([
        sp.csr_matrix(np.ones((1, n_stocks))),
        sp.eye(n_stocks),
    ]).tocsc()

    print(SEP2)
    print("PHASE 2 — Per-model QP solves  (solver: OSQP)")
    print(SEP2)
    print(f"  Constraint matrix A: shape={A_constraint.shape}  "
          f"nnz={A_constraint.nnz}  (precomputed once, reused every solve)")
    print()

    all_backtests: list[pd.DataFrame] = []
    all_summaries: dict[str, dict[str, float]] = {}
    t0_all_models = time.perf_counter()

    for model_idx, (label, pred_df) in enumerate(pred_dfs.items()):
        pred_arr = pred_df.values.astype(float)

        print()
        print(f"  Model {model_idx+1}/7: {label}")
        t0_model = time.perf_counter()

        rows: list[dict] = []
        prev_weights: np.ndarray | None = None
        solve_times: list[float] = []
        n_active_stocks: list[int] = []  # stocks with weight > 1e-4
        failed_solves = 0

        for i, monday_t in enumerate(test_weeks):
            t_solve = time.perf_counter()
            weights = build_minvar_weights(
                pred_arr[i], corr_matrices[i], config.MAX_WEIGHT,
                constraint_matrix=A_constraint,
            )
            dt_solve = time.perf_counter() - t_solve
            solve_times.append(dt_solve)

            active = int((weights > 1e-4).sum())
            n_active_stocks.append(active)

            result = compute_portfolio_returns(weights, actual_ret[i], prev_weights)
            rows.append({"week": holding_weeks[i], **result, "equity_weight": 1.0})
            prev_weights = weights

            # Print every 10th week and the first/last
            if i == 0 or i == n_test - 1 or (i + 1) % 20 == 0:
                elapsed_model = time.perf_counter() - t0_model
                remaining_weeks = n_test - (i + 1)
                eta_s = (elapsed_model / (i + 1)) * remaining_weeks
                print(f"    [{i+1:>3d}/{n_test}]  week={monday_t.date()}  "
                      f"solve={dt_solve*1000:.1f}ms  "
                      f"active_stocks={active}  "
                      f"max_w={weights.max():.4f}  "
                      f"net_ret={result['net_return']:+.4f}  "
                      f"turnover={result['turnover']:.4f}  "
                      f"ETA={eta_s:.0f}s")

        t_model = time.perf_counter() - t0_model
        backtest_df = pd.DataFrame(
            rows,
            columns=["week", "gross_return", "net_return", "turnover", "max_weight", "equity_weight"],
        )
        backtest_df["model"] = label
        summary = summarize_backtest(backtest_df, tbill_rates)
        all_backtests.append(backtest_df)
        all_summaries[label] = summary

        net = backtest_df["net_return"].values
        print()
        print(f"    --- {label} complete in {t_model:.1f}s ---")
        print(f"    solve times:   mean={np.mean(solve_times)*1000:.1f}ms  "
              f"max={np.max(solve_times)*1000:.1f}ms  "
              f"min={np.min(solve_times)*1000:.1f}ms")
        print(f"    active stocks: mean={np.mean(n_active_stocks):.1f}  "
              f"min={np.min(n_active_stocks)}  max={np.max(n_active_stocks)}")
        print(f"    ann_return={summary['ann_return']:.2%}  "
              f"ann_vol={summary['ann_vol']:.2%}  "
              f"sharpe={summary['sharpe']:.3f}  "
              f"max_drawdown={summary['max_drawdown']:.2%}")
        print(f"    avg_turnover={summary['avg_turnover']:.4f}  "
              f"max_weight={summary['max_single_stock_weight']:.4f}")

    t_all_models = time.perf_counter() - t0_all_models

    # ------------------------------------------------------------------
    # Assemble and save
    # ------------------------------------------------------------------
    print()
    print(SEP2)
    print("ASSEMBLING RESULTS")
    print(SEP2)

    mv_returns_df = pd.concat(all_backtests, ignore_index=True)

    assert len(mv_returns_df) == 7 * n_test, (
        f"Expected {7 * n_test} rows, got {len(mv_returns_df)}"
    )

    metrics_table = (
        pd.DataFrame(all_summaries)
        .T.rename_axis("model")
        [["ann_return", "ann_vol", "sharpe", "max_drawdown",
          "avg_turnover", "max_single_stock_weight"]]
    )

    assert len(metrics_table) == 7, f"Expected 7 model rows, got {len(metrics_table)}"

    mv_returns_df.to_parquet(results_dir / "portfolio_mv_returns.parquet", index=False)
    metrics_table.to_csv(results_dir / "portfolio_mv_metrics_table.csv")

    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    print(f"  Correlation phase : {t_corr_total:.2f}s")
    print(f"  QP solve phase    : {t_all_models:.2f}s")
    print(f"  Total             : {t_corr_total + t_all_models:.2f}s")
    print()
    print(metrics_table.to_string())
    print()
    print("Saved: portfolio_mv_returns.parquet, portfolio_mv_metrics_table.csv")
    print(SEP)

    return mv_returns_df, metrics_table
