from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.experiment_registry import register_experiment


WINDOWS = [21, 63, 126]


def _json_field(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _best_macro_corr_config(results_dir: Path) -> dict:
    path = results_dir / "gnn_corr_macro_hparam_search_results.json"
    if not path.exists():
        return {
            "lr": config.GNN_LEARNING_RATE,
            "hidden_dim": config.HIDDEN_DIM,
            "dropout": config.DROPOUT,
            "batch_norm": False,
            "num_layers": config.GNN_NUM_LAYERS,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("best_config", {})


def _artifact(path: str) -> str:
    candidate = config._ROOT / path
    return path if candidate.exists() else ""


def main() -> None:
    results_dir = Path(config.DATA_RESULTS_DIR)
    best_cfg = _best_macro_corr_config(results_dir)
    base_hparams = {
        "random_seed": config.RANDOM_SEED,
        "data_start": config.DATA_START,
        "data_end": config.DATA_END,
        "architecture": "GNNModelV2",
        "hidden_dim": int(best_cfg.get("hidden_dim", config.HIDDEN_DIM)),
        "num_layers": int(best_cfg.get("num_layers", config.GNN_NUM_LAYERS)),
        "dropout": float(best_cfg.get("dropout", config.DROPOUT)),
        "batch_norm": bool(best_cfg.get("batch_norm", False)),
        "learning_rate": float(best_cfg.get("lr", config.GNN_LEARNING_RATE)),
        "corr_threshold": config.CORR_THRESHOLD,
        "max_epochs": config.GNN_MAX_EPOCHS,
        "early_stop_patience": config.GNN_HPARAM_PATIENCE,
        "feature_artifact": "data/features/features_macro.parquet",
        "feature_meta": "data/features/features_macro_meta.json",
    }

    for window in WINDOWS:
        tag = f"w{window:03d}"
        row = {
            "experiment_id": f"window_gnn_correlation_macro_{window}",
            "model_name": f"GNN-Correlation + Macro {window}d",
            "model_family": "GNN",
            "graph_type": "correlation",
            "loss_type": "mse",
            "feature_version": "stock_features_plus_regime_v1",
            "graph_version": f"correlation_threshold_{config.CORR_THRESHOLD}_lookback_{window}",
            "checkpoint_path": _artifact(f"data/results/checkpoints/gnn_corr_macro_{tag}_best.pt"),
            "train_split": f"{config.DATA_START} through {config.TRAIN_END}",
            "val_split": f"{config.TRAIN_END} exclusive through {config.VAL_END}",
            "test_split": f"{config.VAL_END} exclusive through {config.TEST_END}",
            "hyperparameters": _json_field({**base_hparams, "corr_lookback_days": window}),
            "validation_metrics_path": _artifact(f"data/results/gnn_corr_macro_{tag}_val_loss.json"),
            "test_metrics_path": _json_field(
                [
                    _artifact("data/results/ml_metrics_table.csv"),
                    _artifact("data/results/rank_ic_table.csv"),
                ]
            ),
            "portfolio_metrics_path": _json_field(
                [
                    _artifact("data/results/portfolio_metrics_table.csv"),
                    _artifact("data/results/portfolio_ls_metrics_table.csv"),
                    _artifact("data/results/portfolio_vt_metrics_table.csv"),
                    _artifact("data/results/portfolio_mv_metrics_table.csv"),
                ]
            ),
            "prediction_path": _artifact(f"data/results/test_preds_gnn_corr_macro_{tag}.parquet"),
            "validation_prediction_path": _artifact(f"data/results/gnn_corr_macro_{tag}_val_preds.parquet"),
            "notes": (
                "Correlation graph lookback-window experiment using macro/regime features. "
                "Window comparison should be selected on validation metrics only."
            ),
        }
        register_experiment(row, results_dir, overwrite=True)
        print(f"Registered {row['experiment_id']}: {row['model_name']}")


if __name__ == "__main__":
    main()
