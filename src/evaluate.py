"""
Evaluation metrics: MSE, MAE, R², and sector-level breakdowns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import config

if TYPE_CHECKING:
    import torch


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Compute MSE, MAE, and R² between true and predicted RV.

    y_true: ndarray of shape (num_weeks, num_stocks) or (N,).
    y_pred: ndarray of same shape as y_true.
    Returns dict with keys 'mse', 'mae', 'r2'.

    Shape assertion: y_true.shape == y_pred.shape.
    Lookahead safety: pure arithmetic on pre-computed arrays, no date indexing.
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    )
    flat_true = y_true.ravel()
    flat_pred = y_pred.ravel()
    valid = ~(np.isnan(flat_true) | np.isnan(flat_pred))
    flat_true = flat_true[valid]
    flat_pred = flat_pred[valid]
    mse   = float(np.mean((flat_true - flat_pred) ** 2))
    mae   = float(np.mean(np.abs(flat_true - flat_pred)))
    ss_res = float(np.sum((flat_true - flat_pred) ** 2))
    ss_tot = float(np.sum((flat_true - flat_true.mean()) ** 2))
    r2    = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {"mse": mse, "mae": mae, "r2": r2}


def compute_sector_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tickers: list[str],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """
    Compute MSE, MAE, R² broken down by GICS sector.

    y_true: ndarray of shape (num_weeks, num_stocks).
    y_pred: ndarray of same shape.
    tickers: ordered list of ticker symbols (column order of y_true/y_pred).
    sector_map: dict mapping ticker -> sector name. Tickers absent from the map are skipped.
    Returns DataFrame with columns ['sector', 'n_stocks', 'mse', 'mae', 'r2'], one row per sector,
    sorted by MSE descending.

    Shape assertions: y_true.shape == y_pred.shape, y_true.ndim == 2,
                      y_true.shape[1] == len(tickers).
    Lookahead safety: pure arithmetic on pre-computed arrays, no date indexing.
    """
    assert y_true.shape == y_pred.shape, (
        f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
    )
    assert y_true.ndim == 2, f"Expected 2D arrays, got ndim={y_true.ndim}"
    assert y_true.shape[1] == len(tickers), (
        f"y_true has {y_true.shape[1]} columns but len(tickers)={len(tickers)}"
    )

    sectors = sorted({s for s in sector_map.values() if s is not None})
    rows: list[dict] = []
    for sector in sectors:
        col_indices = [i for i, t in enumerate(tickers) if sector_map.get(t) == sector]
        if not col_indices:
            continue
        metrics = compute_metrics(y_true[:, col_indices], y_pred[:, col_indices])
        rows.append({"sector": sector, "n_stocks": len(col_indices), **metrics})

    df = pd.DataFrame(rows, columns=["sector", "n_stocks", "mse", "mae", "r2"])
    df = df.sort_values("mse", ascending=False).reset_index(drop=True)

    assert df.shape[1] == 5, f"Expected 5 columns, got {df.shape[1]}"
    return df


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Aggregate metrics from multiple models into a single comparison table.

    results: dict mapping model_name -> metrics dict (from compute_metrics).
    Returns DataFrame with model names as index, sorted by MSE ascending.

    Shape assertion: output has one row per input key.
    """
    df = pd.DataFrame(results).T.rename_axis("model")
    df = df[["mse", "mae", "r2"]].sort_values("mse")
    assert len(df) == len(results), (
        f"Row count mismatch: {len(df)} vs {len(results)}"
    )
    return df


def compile_validation_summary(
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    splits: pd.DataFrame,
    tickers: list[str],
    sector_graphs: "dict[int, torch.LongTensor]",
    granger_edge_index: "torch.LongTensor",
    device: "torch.device",
) -> tuple[pd.DataFrame, str]:
    """
    Collect val-set metrics for all six models, save validation_summary.json,
    and return a ranked table with a go/no-go decision.

    Loads HAR (per-stock and pooled) and LSTM predictions from saved parquet files.
    Runs GNN inference in-memory for Correlation, Sector, and Granger variants by
    loading each best checkpoint and calling predict_gnn_val().

    GNN-Correlation val graphs are loaded from disk via load_corr_graphs() using
    the best threshold from corr_threshold_ablation.json.
    Run precompute_corr_graphs() before calling this function.

    features:           shape (num_weeks, num_stocks, num_features)
    target:             shape (num_weeks, num_stocks)
    week_index:         DatetimeIndex aligned to features/target rows
    splits:             DataFrame with columns ['week', 'split']
    tickers:            ordered ticker list, length num_stocks
    sector_graphs:      {year: edge_index LongTensor} from build_all_sector_graphs()
    granger_edge_index: static directed LongTensor, shape (2, num_edges)
    device:             torch.device for GNN inference

    Returns:
        ranked_df: DataFrame indexed by model name, columns [rank, mse, mae, r2],
                   sorted by MSE ascending.
        go_nogo:   "GO" if any GNN variant beats HAR per-stock on val MSE, else "NO-GO".

    Output: config.DATA_RESULTS_DIR / "validation_summary.json"

    Shape assertion: summary contains exactly 6 model entries.
    Lookahead safety: inference uses only features and edge indices for val weeks.
    """
    import torch
    from src.models import GNNModel
    from src.train import predict_gnn_val
    from src.graphs import load_corr_graphs

    results_dir = Path(config.DATA_RESULTS_DIR)
    ckpt_dir    = Path(config.CHECKPOINTS_DIR)
    n_feats     = features.shape[2]

    # Build val-period target DataFrame for metric alignment
    val_weeks  = sorted(splits.loc[splits["split"] == "val", "week"])
    target_df  = pd.DataFrame(target, index=week_index, columns=tickers)
    target_val = target_df.loc[target_df.index.isin(val_weeks)]

    # Load pre-saved val predictions for non-GNN models
    har_preds   = pd.read_parquet(results_dir / "har_val_preds.parquet").reindex(target_val.index)
    har_p_preds = pd.read_parquet(results_dir / "har_pooled_val_preds.parquet").reindex(target_val.index)
    lstm_preds  = pd.read_parquet(results_dir / "lstm_val_preds.parquet").reindex(target_val.index)

    # Use the threshold that won the ablation study
    ablation   = json.load(open(results_dir / "corr_threshold_ablation.json"))
    best_theta = ablation["best_threshold"]

    # GNN-Correlation — load precomputed val graphs (no recomputation)
    val_corr_graphs = load_corr_graphs(best_theta, "val")
    corr_model = GNNModel(in_channels=n_feats).to(device)
    corr_model.load_state_dict(
        torch.load(ckpt_dir / "gnn_corr_best.pt", map_location=device, weights_only=True)
    )
    corr_preds = predict_gnn_val(
        model=corr_model,
        features=features,
        target=target,
        week_index=week_index,
        edge_index_fn=lambda week, g=val_corr_graphs: g.get(
            week, torch.zeros(2, 0, dtype=torch.long)
        ),
        splits=splits,
        tickers=tickers,
        device=device,
    )

    # GNN-Sector
    sector_model = GNNModel(in_channels=n_feats).to(device)
    sector_model.load_state_dict(
        torch.load(ckpt_dir / "gnn_sector_best.pt", map_location=device, weights_only=True)
    )
    sector_preds = predict_gnn_val(
        model=sector_model,
        features=features,
        target=target,
        week_index=week_index,
        edge_index_fn=lambda week: sector_graphs[week.year],
        splits=splits,
        tickers=tickers,
        device=device,
    )

    # GNN-Granger
    granger_model = GNNModel(in_channels=n_feats).to(device)
    granger_model.load_state_dict(
        torch.load(ckpt_dir / "gnn_granger_best.pt", map_location=device, weights_only=True)
    )
    granger_preds = predict_gnn_val(
        model=granger_model,
        features=features,
        target=target,
        week_index=week_index,
        edge_index_fn=lambda _week: granger_edge_index,
        splits=splits,
        tickers=tickers,
        device=device,
    )

    def _metrics(preds_df: pd.DataFrame) -> dict[str, float]:
        return compute_metrics(target_val.values, preds_df.reindex(target_val.index).values)

    corr_label = f"GNN-Correlation (θ={best_theta})"
    all_results: dict[str, dict[str, float]] = {
        "HAR per-stock": _metrics(har_preds),
        "HAR pooled":    _metrics(har_p_preds),
        "LSTM":          _metrics(lstm_preds),
        corr_label:      _metrics(corr_preds),
        "GNN-Sector":    _metrics(sector_preds),
        "GNN-Granger":   _metrics(granger_preds),
    }

    assert len(all_results) == 6, f"Expected 6 model entries, got {len(all_results)}"

    # Verify all four neural-model checkpoints load cleanly
    for ckpt_name in ("lstm_best.pt", "gnn_corr_best.pt", "gnn_sector_best.pt", "gnn_granger_best.pt"):
        path = ckpt_dir / ckpt_name
        assert path.exists(), f"Checkpoint missing: {path}"
        torch.load(path, map_location="cpu", weights_only=True)

    ranked_df = compare_models(all_results)
    ranked_df.insert(0, "rank", range(1, len(ranked_df) + 1))

    har_mse    = all_results["HAR per-stock"]["mse"]
    gnn_keys   = [k for k in all_results if k.startswith("GNN")]
    any_better = any(all_results[k]["mse"] < har_mse for k in gnn_keys)
    go_nogo    = "GO" if any_better else "NO-GO"

    best_gnn     = min(gnn_keys, key=lambda k: all_results[k]["mse"])
    best_gnn_mse = all_results[best_gnn]["mse"]

    summary = {
        "go_nogo": go_nogo,
        "har_per_stock_val_mse": har_mse,
        "best_gnn_model": best_gnn,
        "best_gnn_val_mse": best_gnn_mse,
        "dev_universe_size": config.DEV_UNIVERSE_SIZE,
        "note": (
            f"Dev universe ({config.DEV_UNIVERSE_SIZE} stocks). "
            "Rerun on full universe before final go/no-go."
        ),
        "models": all_results,
        "ranked_by_mse": [
            {"rank": int(row["rank"]), "model": name, **{k: v for k, v in all_results[name].items()}}
            for name, row in ranked_df.iterrows()
        ],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "validation_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    return ranked_df, go_nogo
