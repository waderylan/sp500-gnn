from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.diagnostics import compute_calibration_diagnostics
from src.model_artifacts import (
    compute_macro_evaluation_artifacts,
    load_predictions_from_registry,
    load_step7_prediction_registry,
    paired_macro_deltas,
)
from src.portfolio import (
    fetch_tbill_rates,
    run_all_model_backtests,
    run_all_model_backtests_minvar,
    run_all_model_backtests_vol_target,
    run_all_model_long_short_backtests,
)


def main() -> None:
    results_dir = Path(config.DATA_RESULTS_DIR)
    features_dir = Path(config.DATA_FEATURES_DIR)
    raw_dir = Path(config.DATA_RAW_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Refreshing registry-backed ML/ranking artifacts...")
    paths = compute_macro_evaluation_artifacts(
        results_dir=results_dir,
        features_dir=features_dir,
        raw_dir=raw_dir,
    )
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print("\nRefreshing calibration diagnostics...")
    target = pd.read_parquet(features_dir / "target.parquet")
    target.index = pd.to_datetime(target.index)
    registry = load_step7_prediction_registry(results_dir)
    predictions = load_predictions_from_registry(registry)
    calibration, calibration_bins = compute_calibration_diagnostics(predictions, target)
    calibration.to_csv(results_dir / "calibration_summary.csv", index=False)
    calibration_bins.to_csv(results_dir / "calibration_bins.csv", index=False)
    print(f"  calibration rows: {len(calibration)}")
    print(f"  calibration bin rows: {len(calibration_bins)}")

    print("\nRefreshing portfolio artifacts...")
    log_returns = pd.read_parquet(raw_dir / "log_returns.parquet")
    log_returns.index = pd.to_datetime(log_returns.index)
    tickers = target.columns.tolist()
    tbill_rates = fetch_tbill_rates(config.DATA_START, config.DATA_END)
    model_predictions = load_predictions_from_registry(registry, tickers=tickers)

    _, ivol = run_all_model_backtests(log_returns, tbill_rates, tickers, model_predictions=model_predictions)
    _, ls = run_all_model_long_short_backtests(log_returns, tbill_rates, tickers, model_predictions=model_predictions)
    _, vt = run_all_model_backtests_vol_target(log_returns, tbill_rates, tickers, model_predictions=model_predictions)
    _, mv = run_all_model_backtests_minvar(log_returns, tbill_rates, tickers, model_predictions=model_predictions)

    portfolio_delta_tables = []
    for strategy, table in [
        ("inverse_vol", ivol),
        ("long_short", ls),
        ("volatility_targeted", vt),
        ("minimum_variance", mv),
    ]:
        deltas = paired_macro_deltas(
            table,
            ["ann_return", "ann_vol", "sharpe", "max_drawdown", "avg_turnover"],
        )
        deltas.insert(0, "strategy", strategy)
        portfolio_delta_tables.append(deltas)
    portfolio_deltas = pd.concat(portfolio_delta_tables, ignore_index=True)
    portfolio_deltas.to_csv(results_dir / "macro_portfolio_metric_deltas.csv", index=False)

    print("\nSaved refreshed artifacts:")
    for filename in [
        "ml_metrics_table.csv",
        "rank_ic_table.csv",
        "macro_ml_metric_deltas.csv",
        "macro_rank_ic_deltas.csv",
        "calibration_summary.csv",
        "calibration_bins.csv",
        "portfolio_metrics_table.csv",
        "portfolio_ls_metrics_table.csv",
        "portfolio_vt_metrics_table.csv",
        "portfolio_mv_metrics_table.csv",
        "macro_portfolio_metric_deltas.csv",
    ]:
        print(f"  data/results/{filename}")


if __name__ == "__main__":
    main()
