"""Build and read the project experiment registry.

The registry is the provenance layer for current and future model results. It
records model identity, feature and graph versions, checkpoints, split dates,
hyperparameters, and metric artifact paths without retraining or modifying the
underlying result files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

import config
from src.baseline_freeze import CURRENT_MODEL_ROSTER


REGISTRY_COLUMNS = [
    "experiment_id",
    "model_name",
    "model_family",
    "graph_type",
    "loss_type",
    "feature_version",
    "graph_version",
    "checkpoint_path",
    "train_split",
    "val_split",
    "test_split",
    "hyperparameters",
    "validation_metrics_path",
    "test_metrics_path",
    "portfolio_metrics_path",
    "prediction_path",
    "validation_prediction_path",
    "notes",
]

PORTFOLIO_METRIC_PATHS = [
    "data/results/portfolio_metrics_table.csv",
    "data/results/portfolio_ls_metrics_table.csv",
    "data/results/portfolio_vt_metrics_table.csv",
    "data/results/portfolio_mv_metrics_table.csv",
]


def _json_field(value: Any) -> str:
    """Serialize nested registry values into stable JSON text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _relative(path: Path | str) -> str:
    """Return a repository-relative POSIX path."""
    path = Path(path)
    try:
        return path.relative_to(config._ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _artifact(path: str) -> str:
    """Return an artifact path only when the file exists."""
    candidate = config._ROOT / path
    return path if candidate.exists() else ""


def _load_best_gnn_hparams(results_dir: Path) -> dict[str, Any]:
    """Load the current GNN hyperparameter-search winner if available."""
    path = results_dir / "gnn_hparam_search_results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    best_config = data.get("best_config", {})
    if data.get("best_val_mse") is not None:
        best_config = {**best_config, "best_val_mse": data["best_val_mse"]}
    return best_config


def _base_hparams() -> dict[str, Any]:
    """Collect shared baseline training and split settings."""
    return {
        "random_seed": config.RANDOM_SEED,
        "data_start": config.DATA_START,
        "data_end": config.DATA_END,
        "min_coverage": config.MIN_COVERAGE,
        "dev_universe_size": config.DEV_UNIVERSE_SIZE,
    }


def _registry_specs(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Define artifact paths and hyperparameters for the current roster."""
    best_gnn_hparams = _load_best_gnn_hparams(results_dir)
    base = _base_hparams()
    frozen_gnnmodel_base = {
        **base,
        "architecture": "GNNModel",
        "hidden_dim": 128,
        "num_layers": 2,
        "dropout": config.DROPOUT,
        "learning_rate": config.LEARNING_RATE,
        "max_epochs": config.GNN_MAX_EPOCHS,
        "early_stop_patience": config.EARLY_STOP_PATIENCE,
    }
    tuned_corr_base = {
        **base,
        "architecture": "GNNModelV2",
        "hidden_dim": best_gnn_hparams.get("hidden_dim", config.HIDDEN_DIM),
        "num_layers": best_gnn_hparams.get("num_layers", config.GNN_NUM_LAYERS),
        "dropout": best_gnn_hparams.get("dropout", config.DROPOUT),
        "batch_norm": best_gnn_hparams.get("batch_norm", False),
        "learning_rate": best_gnn_hparams.get("lr", config.GNN_LEARNING_RATE),
        "max_epochs": config.GNN_MAX_EPOCHS,
    }
    rank_loss_base = {
        **frozen_gnnmodel_base,
        "pair_sample_frac": config.RANK_LOSS_PAIR_SAMPLE_FRAC,
    }

    return {
        "HAR per-stock": {
            "experiment_id": "baseline_har_per_stock",
            "feature_version": "har_realized_volatility_lags_v1",
            "graph_version": "none",
            "checkpoint_path": "",
            "hyperparameters": {**base, "estimator": "per_stock_har"},
            "validation_metrics_path": "data/results/validation_summary.json",
            "prediction_path": "data/results/test_preds_har.parquet",
            "validation_prediction_path": "data/results/har_val_preds.parquet",
            "notes": "Frozen baseline row. Classical HAR model has no checkpoint artifact.",
        },
        "HAR pooled": {
            "experiment_id": "baseline_har_pooled",
            "feature_version": "har_realized_volatility_lags_v1",
            "graph_version": "none",
            "checkpoint_path": "",
            "hyperparameters": {**base, "estimator": "pooled_har"},
            "validation_metrics_path": "data/results/validation_summary.json",
            "prediction_path": "data/results/test_preds_har_pooled.parquet",
            "validation_prediction_path": "data/results/har_pooled_val_preds.parquet",
            "notes": "Frozen baseline row. Classical pooled HAR model has no checkpoint artifact.",
        },
        "LSTM": {
            "experiment_id": "baseline_lstm",
            "feature_version": "stock_features_v1",
            "graph_version": "none",
            "checkpoint_path": "data/results/checkpoints/lstm_best.pt",
            "hyperparameters": {
                **base,
                "hidden_dim": config.LSTM_HIDDEN_DIM,
                "seq_len": config.LSTM_SEQ_LEN,
                "learning_rate": config.LEARNING_RATE,
                "max_epochs": config.LSTM_MAX_EPOCHS,
                "early_stop_patience": config.EARLY_STOP_PATIENCE,
            },
            "validation_metrics_path": "data/results/lstm_val_loss.json",
            "prediction_path": "data/results/test_preds_lstm.parquet",
            "validation_prediction_path": "data/results/lstm_val_preds.parquet",
            "notes": "Frozen neural sequence baseline.",
        },
        "GNN-Correlation": {
            "experiment_id": "baseline_gnn_correlation",
            "feature_version": "stock_features_v1",
            "graph_version": f"correlation_threshold_{config.CORR_THRESHOLD}_lookback_{config.CORR_LOOKBACK_DAYS}",
            "checkpoint_path": "data/results/checkpoints/gnn_corr_hparam_best.pt",
            "hyperparameters": {
                **tuned_corr_base,
                "early_stop_patience": config.GNN_HPARAM_PATIENCE,
                "corr_threshold": config.CORR_THRESHOLD,
                "hparam_search_best_config": best_gnn_hparams,
            },
            "validation_metrics_path": "data/results/gnn_corr_th03_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_corr.parquet",
            "validation_prediction_path": "",
            "notes": "Official tuned GNN-Correlation checkpoint selected by validation MSE from gnn_hparam_search_results.json.",
        },
        "GNN-Sector": {
            "experiment_id": "baseline_gnn_sector",
            "feature_version": "stock_features_v1",
            "graph_version": "sector_canonical_gics_labels_v1",
            "checkpoint_path": "data/results/checkpoints/gnn_sector_best.pt",
            "hyperparameters": frozen_gnnmodel_base,
            "validation_metrics_path": "data/results/gnn_sector_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_sector.parquet",
            "validation_prediction_path": "",
            "notes": "Frozen sector-graph baseline using canonical sector labels.",
        },
        "GNN-Granger": {
            "experiment_id": "baseline_gnn_granger",
            "feature_version": "stock_features_v1",
            "graph_version": f"granger_lag_{config.GRANGER_LAG}_{config.GRANGER_CORRECTION}",
            "checkpoint_path": "data/results/checkpoints/gnn_granger_best.pt",
            "hyperparameters": {
                **frozen_gnnmodel_base,
                "granger_lag": config.GRANGER_LAG,
                "granger_correction": config.GRANGER_CORRECTION,
                "granger_min_edges": config.GRANGER_MIN_EDGES,
            },
            "validation_metrics_path": "data/results/gnn_granger_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_granger.parquet",
            "validation_prediction_path": "",
            "notes": "Frozen Granger-graph baseline.",
        },
        "GNN-Ensemble": {
            "experiment_id": "baseline_gnn_ensemble",
            "feature_version": "stock_features_v1",
            "graph_version": "correlation+sector+granger_v1",
            "checkpoint_path": "",
            "hyperparameters": {
                **base,
                "constituents": [
                    "baseline_gnn_correlation",
                    "baseline_gnn_sector",
                    "baseline_gnn_granger",
                ],
                "combination": "average_predictions",
            },
            "validation_metrics_path": "data/results/validation_summary.json",
            "prediction_path": "data/results/test_preds_gnn_ensemble.parquet",
            "validation_prediction_path": "",
            "notes": "Prediction ensemble; no standalone checkpoint artifact.",
        },
        "Rank-loss GNN-Correlation": {
            "experiment_id": "rankloss_gnn_correlation",
            "feature_version": "stock_features_v1",
            "graph_version": f"correlation_threshold_{config.CORR_THRESHOLD}_lookback_{config.CORR_LOOKBACK_DAYS}",
            "checkpoint_path": "data/results/checkpoints/gnn_corr_rankloss_best.pt",
            "hyperparameters": {
                **tuned_corr_base,
                "early_stop_patience": config.GNN_HPARAM_PATIENCE,
                "pair_sample_frac": config.RANK_LOSS_PAIR_SAMPLE_FRAC,
                "corr_threshold": config.CORR_THRESHOLD,
            },
            "validation_metrics_path": "data/results/gnn_corr_rankloss_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_corr_rankloss.parquet",
            "validation_prediction_path": "",
            "notes": "Rank-loss experiment. Current summary metric tables do not yet include this row.",
        },
        "Rank-loss GNN-Sector": {
            "experiment_id": "rankloss_gnn_sector",
            "feature_version": "stock_features_v1",
            "graph_version": "sector_canonical_gics_labels_v1",
            "checkpoint_path": "data/results/checkpoints/gnn_sector_rankloss_best.pt",
            "hyperparameters": rank_loss_base,
            "validation_metrics_path": "data/results/gnn_sector_rankloss_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_sector_rankloss.parquet",
            "validation_prediction_path": "",
            "notes": "Rank-loss experiment. Current summary metric tables do not yet include this row.",
        },
        "Rank-loss GNN-Granger": {
            "experiment_id": "rankloss_gnn_granger",
            "feature_version": "stock_features_v1",
            "graph_version": f"granger_lag_{config.GRANGER_LAG}_{config.GRANGER_CORRECTION}",
            "checkpoint_path": "data/results/checkpoints/gnn_granger_rankloss_best.pt",
            "hyperparameters": {
                **rank_loss_base,
                "granger_lag": config.GRANGER_LAG,
                "granger_correction": config.GRANGER_CORRECTION,
                "granger_min_edges": config.GRANGER_MIN_EDGES,
            },
            "validation_metrics_path": "data/results/gnn_granger_rankloss_val_loss.json",
            "prediction_path": "data/results/test_preds_gnn_granger_rankloss.parquet",
            "validation_prediction_path": "",
            "notes": "Rank-loss experiment. Current summary metric tables do not yet include this row.",
        },
    }


def build_experiment_registry(results_dir: Path | None = None) -> pd.DataFrame:
    """Build the current experiment registry as a DataFrame."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    specs = _registry_specs(results_dir)
    rows = []
    for roster_row in CURRENT_MODEL_ROSTER:
        model_name = roster_row["model_name"]
        spec = specs[model_name]
        row = {
            "experiment_id": spec["experiment_id"],
            "model_name": model_name,
            "model_family": roster_row["model_family"],
            "graph_type": roster_row["graph_type"],
            "loss_type": roster_row["loss_type"],
            "feature_version": spec["feature_version"],
            "graph_version": spec["graph_version"],
            "checkpoint_path": _artifact(spec["checkpoint_path"]) if spec["checkpoint_path"] else "",
            "train_split": f"{config.DATA_START} through {config.TRAIN_END}",
            "val_split": f"{config.TRAIN_END} exclusive through {config.VAL_END}",
            "test_split": f"{config.VAL_END} exclusive through {config.TEST_END}",
            "hyperparameters": _json_field(spec["hyperparameters"]),
            "validation_metrics_path": _artifact(spec["validation_metrics_path"]),
            "test_metrics_path": _json_field(
                [
                    _artifact("data/results/ml_metrics_table.csv"),
                    _artifact("data/results/rank_ic_table.csv"),
                ]
            ),
            "portfolio_metrics_path": _json_field([_artifact(path) for path in PORTFOLIO_METRIC_PATHS]),
            "prediction_path": _artifact(spec["prediction_path"]),
            "validation_prediction_path": _artifact(spec["validation_prediction_path"])
            if spec["validation_prediction_path"]
            else "",
            "notes": spec["notes"],
        }
        rows.append(row)

    registry = pd.DataFrame(rows, columns=REGISTRY_COLUMNS)
    return registry


def write_experiment_registry(results_dir: Path | None = None) -> Path:
    """Write ``data/results/experiment_registry.csv`` and return its path."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    registry = build_experiment_registry(results_dir)
    path = results_dir / "experiment_registry.csv"
    registry.to_csv(path, index=False)
    return path


def load_experiment_registry(results_dir: Path | None = None) -> pd.DataFrame:
    """Load the saved experiment registry."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    path = results_dir / "experiment_registry.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment registry not found at {path}. "
            "Run `uv run python -m src.experiment_registry` first."
        )
    return pd.read_csv(path)


def main() -> None:
    """Command-line entry point for refreshing the experiment registry."""
    parser = argparse.ArgumentParser(description="Create the experiment registry.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Optional results directory. Defaults to config.DATA_RESULTS_DIR.",
    )
    args = parser.parse_args()

    path = write_experiment_registry(args.results_dir)
    registry = load_experiment_registry(args.results_dir)
    print(f"Wrote experiment registry: {_relative(path)}")
    print(f"Rows: {len(registry)}")


if __name__ == "__main__":
    main()
