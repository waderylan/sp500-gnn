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


FALLBACK_PREDICTION_FILES = {
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

DEFAULT_BASELINE_MODELS = ["HAR per-stock", "HAR pooled", "LSTM"]

KNOWN_PORTFOLIO_RETURN_FILES = {
    "long_only_inverse_vol": "portfolio_returns.parquet",
    "long_short": "portfolio_ls_returns.parquet",
    "volatility_targeted": "portfolio_vt_returns.parquet",
    "minimum_variance": "portfolio_mv_returns.parquet",
}


def _repo_relative_path(path_text: str) -> Path:
    """Resolve a registry artifact path relative to the repository root."""
    path = Path(str(path_text))
    if path.is_absolute():
        return path
    return ROOT / path


def load_prediction_registry(results_dir: Path) -> pd.DataFrame:
    """
    Load registered prediction artifacts, falling back to the frozen roster map.

    New models become eligible for significance tests when their registry row
    includes a non-empty ``prediction_path`` pointing to an existing parquet.
    """
    registry_path = results_dir / "experiment_registry.csv"
    if registry_path.exists():
        registry = pd.read_csv(registry_path).fillna("")
        required = {"experiment_id", "model_name", "prediction_path"}
        missing = required - set(registry.columns)
        if missing:
            raise ValueError(f"Registry is missing columns: {sorted(missing)}")

        rows: list[dict[str, object]] = []
        for row in registry.to_dict("records"):
            prediction_path = str(row.get("prediction_path", "")).strip()
            if not prediction_path:
                continue
            resolved = _repo_relative_path(prediction_path)
            if resolved.exists():
                rows.append({**row, "resolved_prediction_path": resolved})
        if rows:
            return pd.DataFrame(rows)

    fallback_rows = []
    for model, filename in FALLBACK_PREDICTION_FILES.items():
        path = results_dir / filename
        if path.exists():
            fallback_rows.append(
                {
                    "experiment_id": "",
                    "model_name": model,
                    "model_family": "",
                    "graph_type": "",
                    "loss_type": "",
                    "feature_version": "",
                    "graph_version": "",
                    "prediction_path": str(path),
                    "resolved_prediction_path": path,
                }
            )
    if not fallback_rows:
        raise FileNotFoundError(f"No prediction files found in {results_dir}")
    return pd.DataFrame(fallback_rows)


def build_weekly_model_errors(results_dir: Path, features_dir: Path) -> pd.DataFrame:
    """Compute weekly cross-sectional MSE for each saved prediction artifact."""
    target = pd.read_parquet(features_dir / "target.parquet")
    registry = load_prediction_registry(results_dir)
    rows: list[dict[str, object]] = []

    for metadata in registry.to_dict("records"):
        model = str(metadata["model_name"])
        preds = pd.read_parquet(metadata["resolved_prediction_path"])
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
                    "experiment_id": metadata.get("experiment_id", ""),
                    "model_family": metadata.get("model_family", ""),
                    "graph_type": metadata.get("graph_type", ""),
                    "loss_type": metadata.get("loss_type", ""),
                    "feature_version": metadata.get("feature_version", ""),
                    "graph_version": metadata.get("graph_version", ""),
                    "prediction_path": metadata.get("prediction_path", ""),
                    "weekly_mse": float(value),
                    "n_stocks": int(n_stocks.loc[week]),
                }
            )

    return pd.DataFrame(rows).sort_values(["week", "model"]).reset_index(drop=True)


def build_dm_results(
    weekly_errors: pd.DataFrame,
    baseline_models: list[str] | None = None,
) -> pd.DataFrame:
    """Run DM tests for available non-baseline models against core baselines."""
    baseline_models = baseline_models or DEFAULT_BASELINE_MODELS
    pivot = weekly_errors.pivot(index="week", columns="model", values="weekly_mse").sort_index()
    baselines = {
        model: pivot[model].to_numpy(dtype=float)
        for model in baseline_models
        if model in pivot.columns
    }
    if len(baselines) != len(baseline_models):
        missing = sorted(set(baseline_models) - set(baselines))
        raise FileNotFoundError(f"Missing baseline weekly error series: {missing}")

    comparison_models = {
        model: pivot[model].to_numpy(dtype=float)
        for model in pivot.columns
        if model not in baseline_models
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

    known_by_filename = {filename: strategy for strategy, filename in KNOWN_PORTFOLIO_RETURN_FILES.items()}
    candidate_paths = sorted(results_dir.glob("portfolio*_returns.parquet"))
    for path in candidate_paths:
        strategy = known_by_filename.get(path.name, path.stem.removeprefix("portfolio_").removesuffix("_returns"))
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
                    "return_path": str(path),
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
                        "return_path": str(path),
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
    baseline_models: list[str] | None = None,
    block_size: int = 8,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate and save all required Stage 4 artifacts."""
    results_dir.mkdir(parents=True, exist_ok=True)

    weekly_errors = build_weekly_model_errors(results_dir, features_dir)
    dm_results = build_dm_results(weekly_errors, baseline_models=baseline_models)
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
    parser.add_argument(
        "--baseline-model",
        action="append",
        dest="baseline_models",
        help="Model name to use as a DM-test baseline. May be passed multiple times.",
    )
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate_significance_artifacts(
        results_dir=Path(config.DATA_RESULTS_DIR),
        features_dir=Path(config.DATA_FEATURES_DIR),
        baseline_models=args.baseline_models,
        block_size=args.block_size,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
