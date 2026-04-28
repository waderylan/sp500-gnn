"""Generate Stage 4 statistical significance artifacts.

Outputs:
    data/results/weekly_model_errors.parquet
    data/results/dm_test_results.csv
    data/results/bootstrap_sharpe_ci.csv
    data/results/significance_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.significance import block_bootstrap_sharpe, run_all_dm_tests


PREDICTION_FILES = {
    "HAR per-stock": "test_preds_har.parquet",
    "HAR pooled": "test_preds_har_pooled.parquet",
    "LSTM": "test_preds_lstm.parquet",
    "GNN-Correlation": "test_preds_gnn_corr.parquet",
    "GNN-Sector": "test_preds_gnn_sector.parquet",
    "GNN-Granger": "test_preds_gnn_granger.parquet",
    "GNN-Ensemble": "test_preds_gnn_ensemble.parquet",
    "Rank-loss GNN-Correlation": "test_preds_gnn_corr_rankloss.parquet",
    "Rank-loss GNN-Sector": "test_preds_gnn_sector_rankloss.parquet",
    "Rank-loss GNN-Granger": "test_preds_gnn_granger_rankloss.parquet",
}

BASELINE_MODELS = ["HAR per-stock", "HAR pooled", "LSTM"]

PORTFOLIO_RETURN_FILES = {
    "long_only_inverse_vol": "portfolio_returns.parquet",
    "long_short": "portfolio_ls_returns.parquet",
    "volatility_targeted": "portfolio_vt_returns.parquet",
    "minimum_variance": "portfolio_mv_returns.parquet",
}


def _load_existing_predictions(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load available prediction files without fabricating missing models."""
    predictions: dict[str, pd.DataFrame] = {}
    for model, filename in PREDICTION_FILES.items():
        path = results_dir / filename
        if path.exists():
            predictions[model] = pd.read_parquet(path)
    if not predictions:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")
    return predictions


def build_weekly_model_errors(results_dir: Path, features_dir: Path) -> pd.DataFrame:
    """Compute weekly cross-sectional MSE for each saved prediction artifact."""
    target = pd.read_parquet(features_dir / "target.parquet")
    predictions = _load_existing_predictions(results_dir)
    rows: list[dict[str, object]] = []

    for model, preds in predictions.items():
        common_weeks = preds.index.intersection(target.index)
        common_tickers = preds.columns.intersection(target.columns)
        if common_weeks.empty or common_tickers.empty:
            raise ValueError(f"{model} has no overlap with target.parquet")

        aligned_preds = preds.loc[common_weeks, common_tickers]
        aligned_target = target.loc[common_weeks, common_tickers]
        squared_error = (aligned_target - aligned_preds) ** 2
        weekly_mse = squared_error.mean(axis=1, skipna=True)
        n_stocks = squared_error.notna().sum(axis=1)

        for week, value in weekly_mse.items():
            rows.append(
                {
                    "week": week,
                    "model": model,
                    "weekly_mse": float(value),
                    "n_stocks": int(n_stocks.loc[week]),
                }
            )

    return pd.DataFrame(rows).sort_values(["week", "model"]).reset_index(drop=True)


def build_dm_results(weekly_errors: pd.DataFrame) -> pd.DataFrame:
    """Run DM tests for available non-baseline models against core baselines."""
    pivot = weekly_errors.pivot(index="week", columns="model", values="weekly_mse").sort_index()
    baselines = {
        model: pivot[model].to_numpy(dtype=float)
        for model in BASELINE_MODELS
        if model in pivot.columns
    }
    if len(baselines) != len(BASELINE_MODELS):
        missing = sorted(set(BASELINE_MODELS) - set(baselines))
        raise FileNotFoundError(f"Missing baseline weekly error series: {missing}")

    comparison_models = {
        model: pivot[model].to_numpy(dtype=float)
        for model in pivot.columns
        if model not in BASELINE_MODELS
    }
    return run_all_dm_tests(comparison_models, baselines)


def _pivot_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Return week-by-model net return matrix from a long-format artifact."""
    required = {"week", "model", "net_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Return artifact is missing columns: {sorted(missing)}")
    return df.pivot(index="week", columns="model", values="net_return").sort_index()


def build_bootstrap_results(
    results_dir: Path,
    *,
    block_size: int,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Generate Sharpe and Sharpe-difference bootstrap intervals."""
    rows: list[dict[str, object]] = []

    for strategy, filename in PORTFOLIO_RETURN_FILES.items():
        path = results_dir / filename
        if not path.exists():
            continue
        returns = _pivot_returns(pd.read_parquet(path))

        benchmark = "Equal-weight" if "Equal-weight" in returns.columns else None
        for model in returns.columns:
            series = returns[model].to_numpy(dtype=float)
            sharpe_ci = block_bootstrap_sharpe(
                series,
                None,
                block_size=block_size,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            rows.append(
                {
                    "strategy": strategy,
                    "model": model,
                    "comparison": "sharpe",
                    "benchmark": "",
                    **sharpe_ci,
                }
            )

            if benchmark is not None and model != benchmark:
                aligned = returns[[model, benchmark]].dropna()
                diff_ci = block_bootstrap_sharpe(
                    aligned[model].to_numpy(dtype=float),
                    aligned[benchmark].to_numpy(dtype=float),
                    block_size=block_size,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                )
                rows.append(
                    {
                        "strategy": strategy,
                        "model": model,
                        "comparison": "sharpe_diff",
                        "benchmark": benchmark,
                        **diff_ci,
                    }
                )

    return pd.DataFrame(rows).sort_values(["strategy", "comparison", "model"]).reset_index(drop=True)


def build_significance_summary(dm_results: pd.DataFrame, bootstrap_results: pd.DataFrame) -> pd.DataFrame:
    """Build a compact mixed summary for notebook and paper-audit review."""
    dm_summary = pd.DataFrame(
        [
            {
                "section": "dm_tests",
                "metric": "fdr_significant_model_vs_baseline",
                "value": int(dm_results["rejected_bh"].sum()) if not dm_results.empty else 0,
                "details": f"{len(dm_results)} DM comparisons at FDR 0.05",
            },
            {
                "section": "dm_tests",
                "metric": "min_bh_adjusted_p",
                "value": float(dm_results["p_value_bh"].min()) if not dm_results.empty else float("nan"),
                "details": "One-sided lower-loss alternative",
            },
        ]
    )

    diff = bootstrap_results[bootstrap_results["comparison"] == "sharpe_diff"].copy()
    if diff.empty:
        bootstrap_summary = pd.DataFrame(
            [
                {
                    "section": "bootstrap",
                    "metric": "positive_sharpe_diff_ci",
                    "value": 0,
                    "details": "No benchmark Sharpe-difference rows available",
                }
            ]
        )
    else:
        positive = diff[diff["ci_lower"] > 0.0]
        bootstrap_summary = pd.DataFrame(
            [
                {
                    "section": "bootstrap",
                    "metric": "positive_sharpe_diff_ci",
                    "value": int(len(positive)),
                    "details": f"{len(diff)} Sharpe-difference intervals vs available benchmark",
                }
            ]
        )

    return pd.concat([dm_summary, bootstrap_summary], ignore_index=True)


def generate_significance_artifacts(
    *,
    results_dir: Path,
    features_dir: Path,
    block_size: int = 8,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate and save all required Stage 4 artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)

    weekly_errors = build_weekly_model_errors(results_dir, features_dir)
    dm_results = build_dm_results(weekly_errors)
    bootstrap_results = build_bootstrap_results(
        results_dir,
        block_size=block_size,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    summary = build_significance_summary(dm_results, bootstrap_results)

    paths = {
        "weekly_errors": results_dir / "weekly_model_errors.parquet",
        "dm_results": results_dir / "dm_test_results.csv",
        "bootstrap_results": results_dir / "bootstrap_sharpe_ci.csv",
        "summary": results_dir / "significance_summary.csv",
    }
    weekly_errors.to_parquet(paths["weekly_errors"], index=False)
    dm_results.to_csv(paths["dm_results"], index=False)
    bootstrap_results.to_csv(paths["bootstrap_results"], index=False)
    summary.to_csv(paths["summary"], index=False)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_significance_artifacts(
        results_dir=Path(config.DATA_RESULTS_DIR),
        features_dir=Path(config.DATA_FEATURES_DIR),
        block_size=args.block_size,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
