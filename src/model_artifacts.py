"""Registry-backed model artifact loading and comparison helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import config
from src.evaluate import (
    compute_all_ranking_metrics,
    compute_metrics,
    compute_top_k_hit_rate,
)
from src.experiment_registry import load_experiment_registry


BASE_FEATURE_VERSION = "stock_features_v1"
MACRO_FEATURE_VERSION = "stock_features_plus_regime_v1"
HAR_FEATURE_VERSION = "har_realized_volatility_lags_v1"

MACRO_BASELINE_PAIRS = [
    ("LSTM", "LSTM + Macro"),
    ("GNN-Correlation", "GNN-Correlation + Macro"),
    ("GNN-Correlation", "GNN-Correlation + Macro Tuned"),
    ("GNN-Sector", "GNN-Sector + Macro"),
    ("GNN-Granger", "GNN-Granger + Macro"),
    ("GNN-Ensemble", "GNN-Ensemble + Macro"),
]

STEP7_FEATURE_VERSIONS = {
    HAR_FEATURE_VERSION,
    BASE_FEATURE_VERSION,
    MACRO_FEATURE_VERSION,
}


def repo_path(path_text: str | float | None) -> Path | None:
    """Resolve a registry path relative to the repository root."""
    if path_text is None or pd.isna(path_text):
        return None
    text = str(path_text).strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else config._ROOT / path


def load_prediction_registry(
    results_dir: Path | None = None,
    *,
    feature_versions: Iterable[str] | None = None,
    loss_types: Iterable[str] | None = ("mse",),
    include_har: bool = True,
) -> pd.DataFrame:
    """Load registry rows that have existing test prediction artifacts."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    registry = load_experiment_registry(results_dir).fillna("")

    rows: list[dict[str, object]] = []
    allowed_features = set(feature_versions) if feature_versions is not None else None
    allowed_losses = set(loss_types) if loss_types is not None else None

    for row in registry.to_dict("records"):
        prediction_path = repo_path(row.get("prediction_path"))
        if prediction_path is None or not prediction_path.exists():
            continue

        feature_version = str(row.get("feature_version", ""))
        loss_type = str(row.get("loss_type", ""))
        if allowed_features is not None and feature_version not in allowed_features:
            continue
        if allowed_losses is not None and loss_type not in allowed_losses:
            continue
        if not include_har and feature_version == HAR_FEATURE_VERSION:
            continue

        rows.append({**row, "resolved_prediction_path": prediction_path})

    if not rows:
        raise FileNotFoundError(
            f"No registry prediction rows with existing artifacts found in {results_dir}"
        )
    return pd.DataFrame(rows)


def load_step7_prediction_registry(results_dir: Path | None = None) -> pd.DataFrame:
    """Registry subset for baseline-vs-macro evaluation."""
    registry = load_prediction_registry(
        results_dir,
        feature_versions=STEP7_FEATURE_VERSIONS,
        loss_types=None,
        include_har=True,
    )
    baseline_models = {baseline for baseline, _ in MACRO_BASELINE_PAIRS}
    macro_models = {macro for _, macro in MACRO_BASELINE_PAIRS}
    wanted = baseline_models | macro_models | {
        "HAR per-stock",
        "HAR pooled",
    }
    registry = registry.loc[registry["model_name"].isin(wanted)].copy()
    ordered_models = ["HAR per-stock", "HAR pooled"]
    for baseline, macro in MACRO_BASELINE_PAIRS:
        if baseline not in ordered_models:
            ordered_models.append(baseline)
        if macro not in ordered_models:
            ordered_models.append(macro)
    order = {name: i for i, name in enumerate(
        ordered_models
    )}
    registry["display_order"] = registry["model_name"].map(order).fillna(999).astype(int)
    return registry.sort_values(["display_order", "model_name"]).reset_index(drop=True)


def load_predictions_from_registry(
    registry: pd.DataFrame,
    *,
    tickers: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load prediction parquet files keyed by model name."""
    predictions: dict[str, pd.DataFrame] = {}
    for row in registry.to_dict("records"):
        model = str(row["model_name"])
        path = row.get("resolved_prediction_path") or repo_path(row.get("prediction_path"))
        if path is None:
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        if tickers is not None:
            df = df.reindex(columns=tickers)
        predictions[model] = df
    return predictions


def align_prediction_target(
    preds: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align prediction and target frames to shared weeks and tickers."""
    weeks = preds.index.intersection(target.index)
    tickers = preds.columns.intersection(target.columns)
    return preds.loc[weeks, tickers], target.loc[weeks, tickers]


def compute_ml_metrics_table(
    predictions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    weekly_rv: pd.DataFrame,
) -> pd.DataFrame:
    """Compute point forecast metrics for a registry-selected model set."""
    rows: list[dict[str, object]] = []
    for model, raw_preds in predictions.items():
        preds, actual = align_prediction_target(raw_preds, target)
        rv_now = weekly_rv.reindex(index=preds.index, columns=preds.columns)
        metrics = compute_metrics(actual.to_numpy(dtype=float), preds.to_numpy(dtype=float))
        valid = ~(actual.isna().to_numpy() | preds.isna().to_numpy() | rv_now.isna().to_numpy())
        if valid.any():
            da = float(np.mean(
                np.sign(preds.to_numpy()[valid] - rv_now.to_numpy()[valid])
                == np.sign(actual.to_numpy()[valid] - rv_now.to_numpy()[valid])
            ))
        else:
            da = float("nan")
        rows.append({"model": model, **metrics, "da": da})
    return pd.DataFrame(rows).set_index("model").sort_values("mse")


def compute_ranking_metrics_table(
    predictions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Rank IC, ICIR, hit rate, and pairwise accuracy for each model."""
    rows: list[dict[str, object]] = []
    for model, raw_preds in predictions.items():
        preds, actual = align_prediction_target(raw_preds, target)
        result = compute_all_ranking_metrics(preds, actual, model)
        rows.append(
            {
                "model": model,
                "mean_ic": result.mean_ic,
                "ic_std": result.ic_std,
                "ic_tstat": result.ic_tstat,
                "ic_pvalue": result.ic_pvalue,
                "ic_ir": result.icir,
                "n_weeks": result.n_weeks,
                "pct_positive": result.pct_positive_ic,
                "top_quartile_hit_rate": result.mean_hit_rate,
                "pairwise_accuracy": result.mean_pairwise_acc,
            }
        )
    return pd.DataFrame(rows).set_index("model").sort_values("mean_ic", ascending=False)


def paired_macro_deltas(table: pd.DataFrame, metrics: Iterable[str]) -> pd.DataFrame:
    """Build paired macro-minus-baseline deltas from a model-indexed table."""
    rows: list[dict[str, object]] = []
    for baseline, macro in MACRO_BASELINE_PAIRS:
        if baseline not in table.index or macro not in table.index:
            continue
        row: dict[str, object] = {"baseline_model": baseline, "macro_model": macro}
        for metric in metrics:
            row[f"baseline_{metric}"] = table.loc[baseline, metric]
            row[f"macro_{metric}"] = table.loc[macro, metric]
            row[f"delta_{metric}"] = table.loc[macro, metric] - table.loc[baseline, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_macro_evaluation_artifacts(
    *,
    results_dir: Path | None = None,
    features_dir: Path | None = None,
    raw_dir: Path | None = None,
) -> dict[str, Path]:
    """Compute and save step 7 ML, ranking, and paired comparison artifacts."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    features_dir = Path(features_dir or config.DATA_FEATURES_DIR)
    raw_dir = Path(raw_dir or config.DATA_RAW_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    target = pd.read_parquet(features_dir / "target.parquet")
    weekly_rv = pd.read_parquet(raw_dir / "weekly_rv.parquet")
    target.index = pd.to_datetime(target.index)
    weekly_rv.index = pd.to_datetime(weekly_rv.index)

    registry = load_step7_prediction_registry(results_dir)
    predictions = load_predictions_from_registry(registry)
    ml_table = compute_ml_metrics_table(predictions, target, weekly_rv)
    ranking_table = compute_ranking_metrics_table(predictions, target)
    ml_deltas = paired_macro_deltas(ml_table, ["mse", "mae", "r2", "da"])
    rank_deltas = paired_macro_deltas(
        ranking_table,
        ["mean_ic", "ic_ir", "top_quartile_hit_rate", "pairwise_accuracy"],
    )

    paths = {
        "registry": results_dir / "macro_evaluation_registry.csv",
        "ml_metrics": results_dir / "ml_metrics_table.csv",
        "rank_ic": results_dir / "rank_ic_table.csv",
        "macro_ml_deltas": results_dir / "macro_ml_metric_deltas.csv",
        "macro_rank_deltas": results_dir / "macro_rank_ic_deltas.csv",
    }
    registry.to_csv(paths["registry"], index=False)
    ml_table.to_csv(paths["ml_metrics"])
    ranking_table.to_csv(paths["rank_ic"])
    ml_deltas.to_csv(paths["macro_ml_deltas"], index=False)
    rank_deltas.to_csv(paths["macro_rank_deltas"], index=False)
    return paths
