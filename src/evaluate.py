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


def _load_gnn_model(
    path: "Path",
    in_channels: int,
    device: "torch.device",
) -> "GNNModel":
    """
    Load a GNNModel checkpoint, auto-detecting hidden_dim from the state dict.

    SAGEConv's lin_l.weight has shape (out_channels, in_channels), so
    out_channels = hidden_dim can be read directly without relying on config.HIDDEN_DIM.
    Robust to config.HIDDEN_DIM changes between training and inference sessions.

    path: Path to the .pt checkpoint file.
    in_channels: Number of input features per node.
    device: Device to map weights to.
    Returns: GNNModel with loaded weights.
    """
    import torch
    from src.models import GNNModel
    state = torch.load(path, map_location=device, weights_only=True)
    hidden_dim = next(v.shape[0] for k, v in state.items() if "lin_l.weight" in k)
    model = GNNModel(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    return model


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
    from src.models import GNNModel, GNNModelV2
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

    # GNN-Correlation — use hparam-tuned model if available, else fall back to default.
    val_corr_graphs = load_corr_graphs(best_theta, "val")
    hparam_ckpt  = ckpt_dir / "gnn_corr_hparam_best.pt"
    hparam_json  = results_dir / "gnn_hparam_search_results.json"
    if hparam_ckpt.exists() and hparam_json.exists():
        hparam_cfg = json.load(open(hparam_json))["best_config"]
        corr_model = GNNModelV2(
            in_channels=n_feats,
            hidden_dim=int(hparam_cfg["hidden_dim"]),
            dropout=float(hparam_cfg["dropout"]),
            num_layers=int(hparam_cfg["num_layers"]),
            batch_norm=bool(hparam_cfg["batch_norm"]),
        ).to(device)
        corr_model.load_state_dict(
            torch.load(hparam_ckpt, map_location=device, weights_only=True)
        )
    else:
        corr_model = _load_gnn_model(ckpt_dir / "gnn_corr_best.pt", n_feats, device)
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
    sector_model = _load_gnn_model(ckpt_dir / "gnn_sector_best.pt", n_feats, device)
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
    granger_model = _load_gnn_model(ckpt_dir / "gnn_granger_best.pt", n_feats, device)
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

    # Verify all neural-model checkpoints load cleanly
    corr_ckpt_name = "gnn_corr_hparam_best.pt" if hparam_ckpt.exists() else "gnn_corr_best.pt"
    for ckpt_name in ("lstm_best.pt", corr_ckpt_name, "gnn_sector_best.pt", "gnn_granger_best.pt"):
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


def run_test_evaluation(
    weekly_rv: pd.DataFrame,
    target_df: pd.DataFrame,
    features: np.ndarray,
    week_index: pd.DatetimeIndex,
    splits: pd.DataFrame,
    tickers: list[str],
    sector_history: dict,
    sector_graphs: dict,
    granger_edge_index: "torch.LongTensor",
    device: "torch.device",
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Generate test-set predictions for all six models, compute metrics, and save results.

    Re-fits HAR models on training data (sklearn, no .pt checkpoint exists for these).
    Loads LSTM and GNN checkpoints from config.CHECKPOINTS_DIR. GNN-Correlation uses
    gnn_corr_hparam_best.pt (GNNModelV2) when available, else falls back to
    gnn_corr_best.pt (GNNModel). GNN-Sector and GNN-Granger always use GNNModel with
    hidden_dim auto-detected from the checkpoint via _load_gnn_model().
    Correlation graphs for the test split are loaded from precomputed parquets via
    load_corr_graphs(). Run precompute_corr_graphs() before calling this function.

    weekly_rv:          shape (num_weeks, num_stocks), Monday-indexed weekly RV.
    target_df:          shape (num_target_weeks, num_stocks), Monday-indexed target.
    features:           ndarray of shape (num_weeks, num_stocks, num_features).
    week_index:         DatetimeIndex aligned to features rows.
    splits:             DataFrame with columns ['week', 'split'].
    tickers:            ordered ticker list, length num_stocks.
    sector_history:     {ticker: {str(year): sector_name}}.
    sector_graphs:      {year (int): LongTensor edge_index} for all test-split years.
    granger_edge_index: static directed LongTensor, shape (2, num_edges).
    device:             torch.device for GNN/LSTM inference.

    Returns:
        pooled_df:   DataFrame indexed by model name, columns [mse, mae, r2, da],
                     sorted by MSE ascending.
        sector_dict: {model_name: DataFrame with columns [sector, n_stocks, mse, mae, r2]}.

    Saves:
        config.DATA_RESULTS_DIR / test_preds_{key}.parquet  for 6 models.
        config.DATA_RESULTS_DIR / ml_metrics_table.csv

    Shape assertions:
        Each prediction DataFrame has shape (num_test_weeks, num_stocks).
        pooled_df has exactly 6 rows.
    Lookahead safety: inference accesses only features/graphs at test week T.
    Target values (week T+1 RV) are used only for metric computation, not as inputs.
    """
    import json
    import torch
    from src.models import (
        HARModel, HARPooled, LSTMModel, GNNModelV2,
        compute_har_features, prepare_har_arrays,
    )
    from src.graphs import load_corr_graphs
    from src.train import predict_lstm_split, predict_gnn_split

    results_dir = Path(config.DATA_RESULTS_DIR)
    ckpt_dir    = Path(config.CHECKPOINTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    n_feats  = features.shape[2]
    n_stocks = len(tickers)

    # Sector map at the start of 2024 (first test year); fall back to 2023 if absent.
    def _sector(ticker: str) -> str | None:
        hist = sector_history.get(ticker) or {}
        return hist.get("2024") or hist.get("2023")

    sector_map: dict[str, str] = {t: _sector(t) for t in tickers}

    # -----------------------------------------------------------------------
    # 1. HAR (per-stock and pooled) — re-fit on training data, predict on test
    # -----------------------------------------------------------------------
    rv_1w, rv_4w, rv_13w = compute_har_features(weekly_rv)

    X_tr_dict, y_tr_dict, X_tr_pool, y_tr_pool, _ = prepare_har_arrays(
        rv_1w, rv_4w, rv_13w, target_df, splits, "train"
    )
    har_model = HARModel()
    har_model.fit(X_tr_dict, y_tr_dict)
    har_pool_model = HARPooled()
    har_pool_model.fit(X_tr_pool, y_tr_pool)

    X_te_dict, _, X_te_pool, _, test_valid_index = prepare_har_arrays(
        rv_1w, rv_4w, rv_13w, target_df, splits, "test"
    )
    num_test = len(test_valid_index)

    har_preds_dict = har_model.predict(X_te_dict)
    har_preds_df = pd.DataFrame(
        {ticker: har_preds_dict[ticker] for ticker in tickers},
        index=test_valid_index,
    )

    har_pool_arr = har_pool_model.predict(X_te_pool).reshape(num_test, n_stocks)
    har_pool_preds_df = pd.DataFrame(har_pool_arr, index=test_valid_index, columns=tickers)

    # -----------------------------------------------------------------------
    # 2. LSTM — auto-detect hidden_dim from checkpoint
    # -----------------------------------------------------------------------
    _lstm_state  = torch.load(ckpt_dir / "lstm_best.pt", map_location=device, weights_only=True)
    _lstm_hidden = _lstm_state["lstm.weight_hh_l0"].shape[0] // 4
    lstm_model   = LSTMModel(input_size=n_feats, hidden_dim=_lstm_hidden).to(device)
    lstm_model.load_state_dict(_lstm_state)
    del _lstm_state

    lstm_preds_df = predict_lstm_split(
        model=lstm_model,
        features=features,
        week_index=week_index,
        splits=splits,
        tickers=tickers,
        device=device,
        split_name="test",
    ).reindex(test_valid_index)

    # -----------------------------------------------------------------------
    # 3. GNN-Correlation — hparam-tuned checkpoint if available, else ablation winner
    # -----------------------------------------------------------------------
    ablation_data = json.load(open(results_dir / "corr_threshold_ablation.json"))
    best_theta    = ablation_data["best_threshold"]
    test_corr_graphs = load_corr_graphs(best_theta, "test")

    hparam_ckpt = ckpt_dir / "gnn_corr_hparam_best.pt"
    hparam_json = results_dir / "gnn_hparam_search_results.json"
    if hparam_ckpt.exists() and hparam_json.exists():
        hparam_cfg = json.load(open(hparam_json))["best_config"]
        corr_model = GNNModelV2(
            in_channels=n_feats,
            hidden_dim=int(hparam_cfg["hidden_dim"]),
            dropout=float(hparam_cfg["dropout"]),
            num_layers=int(hparam_cfg["num_layers"]),
            batch_norm=bool(hparam_cfg["batch_norm"]),
        ).to(device)
        corr_model.load_state_dict(
            torch.load(hparam_ckpt, map_location=device, weights_only=True)
        )
        corr_label = "GNN-Correlation (tuned)"
    else:
        corr_model = _load_gnn_model(ckpt_dir / "gnn_corr_best.pt", n_feats, device)
        corr_label = f"GNN-Correlation (θ={best_theta})"

    corr_preds_df = predict_gnn_split(
        model=corr_model,
        features=features,
        week_index=week_index,
        edge_index_fn=lambda week, g=test_corr_graphs: g.get(
            week, torch.zeros(2, 0, dtype=torch.long)
        ),
        splits=splits,
        tickers=tickers,
        device=device,
        split_name="test",
    ).reindex(test_valid_index)

    # -----------------------------------------------------------------------
    # 4. GNN-Sector
    # -----------------------------------------------------------------------
    sector_model = _load_gnn_model(ckpt_dir / "gnn_sector_best.pt", n_feats, device)
    sector_preds_df = predict_gnn_split(
        model=sector_model,
        features=features,
        week_index=week_index,
        edge_index_fn=lambda week: sector_graphs[week.year],
        splits=splits,
        tickers=tickers,
        device=device,
        split_name="test",
    ).reindex(test_valid_index)

    # -----------------------------------------------------------------------
    # 5. GNN-Granger
    # -----------------------------------------------------------------------
    granger_model = _load_gnn_model(ckpt_dir / "gnn_granger_best.pt", n_feats, device)
    granger_preds_df = predict_gnn_split(
        model=granger_model,
        features=features,
        week_index=week_index,
        edge_index_fn=lambda _week: granger_edge_index,
        splits=splits,
        tickers=tickers,
        device=device,
        split_name="test",
    ).reindex(test_valid_index)

    # -----------------------------------------------------------------------
    # Align target and current-week RV to test_valid_index
    # -----------------------------------------------------------------------
    target_test = target_df.reindex(test_valid_index)
    # weekly_rv at test weeks T — reference for directional accuracy
    rv_test = weekly_rv.reindex(test_valid_index)

    # -----------------------------------------------------------------------
    # Save prediction parquets
    # -----------------------------------------------------------------------
    pred_map: dict[str, pd.DataFrame] = {
        "har":         har_preds_df,
        "har_pooled":  har_pool_preds_df,
        "lstm":        lstm_preds_df,
        "gnn_corr":    corr_preds_df,
        "gnn_sector":  sector_preds_df,
        "gnn_granger": granger_preds_df,
    }
    for key, df in pred_map.items():
        assert df.shape == (num_test, n_stocks), (
            f"{key}: expected shape ({num_test}, {n_stocks}), got {df.shape}"
        )
        df.to_parquet(results_dir / f"test_preds_{key}.parquet")

    # -----------------------------------------------------------------------
    # Pooled metrics and directional accuracy
    # -----------------------------------------------------------------------
    y_true = target_test.values   # (num_test, n_stocks)
    y_rv   = rv_test.values       # (num_test, n_stocks) — current-week RV

    def _da(y_pred: np.ndarray) -> float:
        """Fraction of (stock, week) pairs where predicted direction matches actual."""
        valid = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_rv))
        if not valid.any():
            return float("nan")
        return float(np.mean(
            np.sign(y_pred[valid] - y_rv[valid]) == np.sign(y_true[valid] - y_rv[valid])
        ))

    label_map = {
        "har":         "HAR per-stock",
        "har_pooled":  "HAR pooled",
        "lstm":        "LSTM",
        "gnn_corr":    corr_label,
        "gnn_sector":  "GNN-Sector",
        "gnn_granger": "GNN-Granger",
    }

    pooled_rows: list[dict]                   = []
    sector_dict: dict[str, pd.DataFrame] = {}

    for key, pred_df in pred_map.items():
        y_pred  = pred_df.values
        metrics = compute_metrics(y_true, y_pred)
        metrics["da"] = _da(y_pred)
        label = label_map[key]
        pooled_rows.append({"model": label, **metrics})
        sector_dict[label] = compute_sector_metrics(y_true, y_pred, tickers, sector_map)

    pooled_df = (
        pd.DataFrame(pooled_rows)
        .set_index("model")[["mse", "mae", "r2", "da"]]
        .sort_values("mse")
    )

    assert len(pooled_df) == 6, f"Expected 6 model rows, got {len(pooled_df)}"

    pooled_df.to_csv(results_dir / "ml_metrics_table.csv")

    return pooled_df, sector_dict
