"""Diagnostic artifacts for current volatility-forecasting baselines.

The functions in this module operate on saved predictions, targets, graph
artifacts, checkpoints, and portfolio returns. They do not retrain models or
modify frozen baseline artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import config
from src.evaluate import compute_metrics, compute_rank_ic
from src.experiment_registry import load_experiment_registry


MIN_SECTOR_STOCKS = 5
DECILE_COUNT = 10


def _repo_path(path_text: str | float | None) -> Path | None:
    """Convert a registry path into an absolute path, preserving blanks."""
    if path_text is None or pd.isna(path_text):
        return None
    path_string = str(path_text).strip()
    if not path_string:
        return None
    path = Path(path_string)
    return path if path.is_absolute() else config._ROOT / path


def _load_prediction_registry(results_dir: Path) -> pd.DataFrame:
    """Load registry rows with existing prediction artifacts."""
    registry = load_experiment_registry(results_dir)
    rows = []
    for _, row in registry.iterrows():
        prediction_path = _repo_path(row.get("prediction_path"))
        if prediction_path is None or not prediction_path.exists():
            continue
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "model_name": row["model_name"],
                "model_family": row["model_family"],
                "graph_type": row["graph_type"],
                "loss_type": row["loss_type"],
                "feature_version": row["feature_version"],
                "graph_version": row["graph_version"],
                "checkpoint_path": row.get("checkpoint_path", ""),
                "prediction_path": prediction_path,
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No usable prediction_path rows found in {results_dir / 'experiment_registry.csv'}"
        )
    return pd.DataFrame(rows)


def load_target(results_dir: Path | None = None) -> pd.DataFrame:
    """Load the target realized-volatility frame aligned to test predictions."""
    del results_dir
    target = pd.read_parquet(Path(config.DATA_FEATURES_DIR) / "target.parquet")
    target.index = pd.to_datetime(target.index)
    return target


def load_predictions(results_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Load model predictions keyed by model name plus their registry metadata."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    registry = _load_prediction_registry(results_dir)
    predictions: dict[str, pd.DataFrame] = {}
    for _, row in registry.iterrows():
        df = pd.read_parquet(row["prediction_path"])
        df.index = pd.to_datetime(df.index)
        predictions[str(row["model_name"])] = df
    return predictions, registry


def _align_prediction_target(
    preds: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align prediction and target frames to shared weeks and tickers."""
    weeks = preds.index.intersection(target.index)
    tickers = preds.columns.intersection(target.columns)
    return preds.loc[weeks, tickers], target.loc[weeks, tickers]


def _linear_calibration(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit actual_rv = intercept + slope * predicted_rv using valid pairs."""
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 2 or np.nanstd(y_pred[valid]) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred[valid], y_true[valid], deg=1)
    return float(slope), float(intercept)


def compute_calibration_diagnostics(
    predictions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute model-level calibration summaries and decile calibration bins."""
    summary_rows: list[dict] = []
    bin_rows: list[dict] = []

    for model_name, preds_raw in predictions.items():
        preds, actuals = _align_prediction_target(preds_raw, target)
        pred_values = preds.to_numpy(dtype=float)
        actual_values = actuals.to_numpy(dtype=float)
        valid = ~(np.isnan(pred_values) | np.isnan(actual_values))
        pred_flat = pred_values[valid]
        actual_flat = actual_values[valid]

        slope, intercept = _linear_calibration(actual_flat, pred_flat)
        pearson = (
            float(pearsonr(pred_flat, actual_flat).statistic)
            if len(pred_flat) >= 2 and np.std(pred_flat) > 0 and np.std(actual_flat) > 0
            else float("nan")
        )
        weekly_spread = preds.quantile(0.90, axis=1) - preds.quantile(0.10, axis=1)
        rank_ic = compute_rank_ic(preds, actuals).dropna()

        summary_rows.append(
            {
                "model": model_name,
                "n_weeks": int(len(preds)),
                "n_observations": int(valid.sum()),
                "prediction_mean": float(np.nanmean(pred_values)),
                "prediction_std": float(np.nanstd(pred_values, ddof=1)),
                "prediction_min": float(np.nanmin(pred_values)),
                "prediction_max": float(np.nanmax(pred_values)),
                "avg_weekly_prediction_spread_p90_p10": float(weekly_spread.mean()),
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "pearson_corr": pearson,
                "spearman_rank_ic": float(rank_ic.mean()) if len(rank_ic) else float("nan"),
            }
        )

        bin_frame = pd.DataFrame({"predicted_rv": pred_flat, "actual_rv": actual_flat})
        if len(bin_frame) >= DECILE_COUNT and bin_frame["predicted_rv"].nunique() >= DECILE_COUNT:
            bin_frame["decile"] = pd.qcut(
                bin_frame["predicted_rv"],
                DECILE_COUNT,
                labels=False,
                duplicates="drop",
            )
            grouped = bin_frame.groupby("decile", observed=True)
            for decile, group in grouped:
                bin_rows.append(
                    {
                        "model": model_name,
                        "decile": int(decile) + 1,
                        "n_observations": int(len(group)),
                        "predicted_rv_mean": float(group["predicted_rv"].mean()),
                        "actual_rv_mean": float(group["actual_rv"].mean()),
                    }
                )

    summary = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    bins = pd.DataFrame(bin_rows).sort_values(["model", "decile"]).reset_index(drop=True)
    return summary, bins


def compute_prediction_spread_by_week(
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute weekly cross-sectional prediction spread diagnostics."""
    rows: list[dict] = []
    for model_name, preds in predictions.items():
        for week, row in preds.iterrows():
            values = row.to_numpy(dtype=float)
            rows.append(
                {
                    "week": pd.Timestamp(week),
                    "model": model_name,
                    "prediction_spread_p90_p10": float(np.nanpercentile(values, 90) - np.nanpercentile(values, 10)),
                    "prediction_range": float(np.nanmax(values) - np.nanmin(values)),
                    "prediction_std": float(np.nanstd(values, ddof=1)),
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "week"]).reset_index(drop=True)


def compute_correlation_graph_density(
    tickers: list[str],
    log_returns: pd.DataFrame,
    threshold: float = config.CORR_THRESHOLD,
) -> pd.DataFrame:
    """Summarize the official precomputed correlation graph density by week."""
    corr_dir = Path(config.CORR_EDGES_DIR)
    rows: list[dict] = []
    num_nodes = len(tickers)
    possible_edges = num_nodes * (num_nodes - 1) / 2
    threshold_tag = "t" + f"{threshold:.1f}".replace(".", "p")
    log_returns_aligned = log_returns.reindex(columns=tickers)
    avg_abs_cache: dict[pd.Timestamp, float] = {}

    for path in sorted(corr_dir.glob(f"corr_edges_*_{threshold_tag}.parquet")):
        stem = path.stem
        prefix = "corr_edges_"
        split_and_threshold = stem[len(prefix) :]
        split_name, threshold_tag = split_and_threshold.rsplit("_", 1)
        file_threshold = float(threshold_tag[1:].replace("p", "."))
        edges = pd.read_parquet(path)
        edges["week"] = pd.to_datetime(edges["week"])

        for week, group in edges.groupby("week", sort=True):
            week = pd.Timestamp(week)
            if week not in avg_abs_cache:
                friday = week + pd.Timedelta(days=4)
                window = log_returns_aligned.loc[:friday].tail(config.CORR_LOOKBACK_DAYS)
                corr = np.corrcoef(window.to_numpy(dtype=float), rowvar=False)
                upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
                avg_abs_cache[week] = float(np.nanmean(np.abs(corr[upper])))
            rows.append(
                {
                    "week": week,
                    "split": split_name,
                    "threshold": file_threshold,
                    "num_edges": int(len(group) // 2),
                    "density": float((len(group) // 2) / possible_edges),
                    "mean_degree": float(len(group) / num_nodes),
                    "max_degree": int(pd.concat([group["src"], group["dst"]]).value_counts().max()),
                    "avg_abs_correlation": avg_abs_cache[week],
                }
            )

    if not rows:
        raise FileNotFoundError(f"No precomputed correlation edges found in {corr_dir}")
    return pd.DataFrame(rows).sort_values(["threshold", "split", "week"]).reset_index(drop=True)


def _market_return_by_feature_week(log_returns: pd.DataFrame, weeks: pd.Index) -> pd.Series:
    """Compute equal-weight holding-week market return for each feature week."""
    weekly = log_returns.resample("W-MON", closed="left", label="left").sum()
    holding_weeks = pd.DatetimeIndex(weeks) + pd.Timedelta(days=7)
    return weekly.mean(axis=1).reindex(holding_weeks).set_axis(pd.DatetimeIndex(weeks))


def _weekly_metric_row(model: str, preds: pd.DataFrame, actuals: pd.DataFrame, weeks: pd.Index) -> dict:
    """Compute a compact metric row for one model and subset of weeks."""
    p = preds.reindex(weeks)
    a = actuals.reindex(weeks)
    metrics = compute_metrics(a.to_numpy(dtype=float), p.to_numpy(dtype=float))
    ic = compute_rank_ic(p, a).dropna()
    valid = ~(np.isnan(a.to_numpy(dtype=float)) | np.isnan(p.to_numpy(dtype=float)))
    return {
        "model": model,
        "n_weeks": int(len(p)),
        "n_observations": int(valid.sum()),
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "rank_ic": float(ic.mean()) if len(ic) else float("nan"),
    }


def compute_regime_breakdowns(
    predictions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    log_returns: pd.DataFrame,
    graph_density: pd.DataFrame,
    results_dir: Path | None = None,
) -> pd.DataFrame:
    """Compute existing-baseline metrics by non-macro test regimes."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    reference = next(iter(predictions.values()))
    weeks = pd.DatetimeIndex(reference.index)
    market_return = _market_return_by_feature_week(log_returns, weeks)
    density_series = (
        graph_density.loc[graph_density["threshold"] == config.CORR_THRESHOLD]
        .drop_duplicates("week")
        .set_index("week")
        .reindex(weeks)["density"]
    )
    avg_abs_corr = (
        graph_density.loc[graph_density["threshold"] == config.CORR_THRESHOLD]
        .drop_duplicates("week")
        .set_index("week")
        .reindex(weeks)["avg_abs_correlation"]
    )

    regime_masks: dict[tuple[str, str], pd.Index] = {}
    for year in sorted(set(weeks.year)):
        regime_masks[("calendar_year", str(year))] = weeks[weeks.year == year]
    regime_masks[("market_return", "low")] = weeks[market_return <= market_return.median()]
    regime_masks[("market_return", "high")] = weeks[market_return > market_return.median()]
    regime_masks[("graph_density", "low")] = weeks[density_series <= density_series.median()]
    regime_masks[("graph_density", "high")] = weeks[density_series > density_series.median()]
    regime_masks[("avg_abs_correlation", "low")] = weeks[avg_abs_corr <= avg_abs_corr.median()]
    regime_masks[("avg_abs_correlation", "high")] = weeks[avg_abs_corr > avg_abs_corr.median()]

    rows: list[dict] = []
    portfolio_returns = _load_portfolio_returns(results_dir)
    for model_name, preds_raw in predictions.items():
        preds, actuals = _align_prediction_target(preds_raw, target)
        for (regime_type, regime_value), regime_weeks in regime_masks.items():
            subset_weeks = preds.index.intersection(regime_weeks)
            if len(subset_weeks) == 0:
                continue
            row = _weekly_metric_row(model_name, preds, actuals, subset_weeks)
            row.update({"regime_type": regime_type, "regime": regime_value})
            port = portfolio_returns.get(model_name)
            if port is not None:
                port_subset = port.loc[port["feature_week"].isin(subset_weeks)]
                row["portfolio_mean_weekly_return"] = float(port_subset["net_return"].mean())
                row["portfolio_sharpe"] = _annualized_sharpe(port_subset["net_return"].to_numpy(dtype=float))
            else:
                row["portfolio_mean_weekly_return"] = float("nan")
                row["portfolio_sharpe"] = float("nan")
            rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(["regime_type", "regime", "model"])
        .reset_index(drop=True)
    )


def _load_portfolio_returns(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load long-only inverse-vol portfolio returns by model when available."""
    path = results_dir / "portfolio_returns.parquet"
    if not path.exists():
        return {}
    returns = pd.read_parquet(path)
    if not {"week", "model", "net_return"}.issubset(returns.columns):
        return {}
    returns = returns.copy()
    returns["week"] = pd.to_datetime(returns["week"])
    returns["feature_week"] = returns["week"] - pd.Timedelta(days=7)
    return {model: group.copy() for model, group in returns.groupby("model")}


def _annualized_sharpe(returns: np.ndarray) -> float:
    """Compute simple annualized Sharpe for regime subsets."""
    clean = returns[~np.isnan(returns)]
    if len(clean) < 2:
        return float("nan")
    vol = np.std(clean, ddof=1) * np.sqrt(52)
    if vol == 0:
        return float("nan")
    return float(np.mean(clean) * 52 / vol)


def _sector_map_for_year(tickers: list[str], year: int = 2024) -> dict[str, str]:
    """Load canonical sector labels for one year, falling back to nearby years."""
    with open(Path(config.DATA_RAW_DIR) / "sector_history.json", encoding="utf-8") as fh:
        sector_history = json.load(fh)
    sector_map = {}
    for ticker in tickers:
        history = sector_history.get(ticker, {})
        sector = history.get(str(year)) or history.get("2023") or history.get("2025")
        if sector:
            sector_map[ticker] = sector
    return sector_map


def compute_within_sector_rank_ic(
    predictions: dict[str, pd.DataFrame],
    target: pd.DataFrame,
    sector_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute Spearman Rank IC within each sector and average by model."""
    rows: list[dict] = []
    for model_name, preds_raw in predictions.items():
        preds, actuals = _align_prediction_target(preds_raw, target)
        for sector in sorted(set(sector_map.values())):
            tickers = [ticker for ticker in preds.columns if sector_map.get(ticker) == sector]
            if len(tickers) < MIN_SECTOR_STOCKS:
                continue
            sector_ic = compute_rank_ic(preds[tickers], actuals[tickers], min_valid=MIN_SECTOR_STOCKS)
            for week, ic in sector_ic.dropna().items():
                rows.append(
                    {
                        "week": pd.Timestamp(week),
                        "model": model_name,
                        "sector": sector,
                        "rank_ic": float(ic),
                        "n_stocks": len(tickers),
                    }
                )
    by_sector = pd.DataFrame(rows)
    table = (
        by_sector.groupby("model", as_index=False)
        .agg(
            mean_within_sector_rank_ic=("rank_ic", "mean"),
            std_within_sector_rank_ic=("rank_ic", "std"),
            n_sector_weeks=("rank_ic", "count"),
        )
        .sort_values("mean_within_sector_rank_ic", ascending=False)
        .reset_index(drop=True)
    )
    return table, by_sector.sort_values(["model", "sector", "week"]).reset_index(drop=True)


def _features_to_array(tickers: list[str]) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load long feature table into a week x ticker x feature array."""
    features_long = pd.read_parquet(Path(config.DATA_FEATURES_DIR) / "features.parquet")
    feature_cols = [c for c in features_long.columns if c not in {"week", "ticker"}]
    features_long["week"] = pd.to_datetime(features_long["week"])
    weeks = pd.DatetimeIndex(sorted(features_long["week"].unique()))
    frames = []
    for feature in feature_cols:
        wide = (
            features_long.pivot(index="week", columns="ticker", values=feature)
            .reindex(index=weeks, columns=tickers)
        )
        frames.append(wide.to_numpy(dtype=float))
    return np.stack(frames, axis=2), weeks


def _load_sector_graph(year: int) -> "torch.LongTensor":
    """Load a saved sector graph for one calendar year."""
    import torch

    edges = pd.read_parquet(Path(config.DATA_GRAPHS_DIR) / "sector_edges_by_year.parquet")
    group = edges.loc[edges["year"] == year]
    return torch.tensor(group[["src", "dst"]].to_numpy(dtype=np.int64).T, dtype=torch.long)


def _load_granger_graph() -> "torch.LongTensor":
    """Load the saved static Granger graph."""
    import torch

    edges = pd.read_parquet(Path(config.DATA_GRAPHS_DIR) / "granger_edges.parquet")
    return torch.tensor(edges[["src", "dst"]].to_numpy(dtype=np.int64).T, dtype=torch.long)


def _select_audit_weeks(week_index: pd.DatetimeIndex, splits: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """Pick representative weeks requested by the implementation plan."""
    split_by_week = splits.set_index("week")["split"]

    def nearest(label: str, date: str, split: str | None = None) -> tuple[str, pd.Timestamp]:
        candidates = week_index
        if split is not None:
            split_weeks = pd.DatetimeIndex(split_by_week[split_by_week == split].index)
            candidates = candidates.intersection(split_weeks)
        target = pd.Timestamp(date)
        deltas = np.abs((candidates - target).days)
        return label, pd.Timestamp(candidates[int(np.argmin(deltas))])

    return dict(
        [
            nearest("calm_2017", "2017-06-26", "train"),
            nearest("covid_2020", "2020-03-09", "train"),
            nearest("representative_2023_val", "2023-06-26", "val"),
            nearest("representative_2024_test", "2024-06-24", "test"),
            nearest("representative_2025_test", "2025-06-23", "test"),
        ]
    )


def _load_gnn_model_from_checkpoint(path: Path, in_channels: int, device: "torch.device") -> "torch.nn.Module":
    """Instantiate a GNN model matching a saved checkpoint."""
    import torch
    from src.models import GNNModel, GNNModelV2

    state = torch.load(path, map_location=device, weights_only=True)
    if any(key.startswith("convs.") for key in state):
        layer_ids = sorted({int(key.split(".")[1]) for key in state if key.startswith("convs.")})
        first_weight = state["convs.0.lin_l.weight"]
        hidden_dim = int(first_weight.shape[0])
        has_norm = any(key.startswith("bns.") for key in state)
        model = GNNModelV2(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            dropout=config.DROPOUT,
            num_layers=len(layer_ids),
            batch_norm=has_norm,
        ).to(device)
    else:
        hidden_dim = int(next(value.shape[0] for key, value in state.items() if "lin_l.weight" in key))
        model = GNNModel(in_channels=in_channels, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _embedding_layers(
    model: "torch.nn.Module",
    x: "torch.Tensor",
    edge_index: "torch.LongTensor",
) -> list[tuple[str, "torch.Tensor"]]:
    """Return post-activation embeddings after each GraphSAGE layer."""
    import torch

    if torch.isnan(x).any():
        x = torch.nan_to_num(x, nan=0.0)
    layers: list[tuple[str, torch.Tensor]] = []
    if hasattr(model, "convs"):
        for i, conv in enumerate(model.convs):
            x = conv(x, edge_index)
            if getattr(model, "bns", None) is not None:
                x = model.bns[i](x, batch=None)
            x = torch.relu(x)
            layers.append((f"layer_{i + 1}", x.detach()))
    else:
        x = torch.relu(model.conv1(x, edge_index))
        layers.append(("layer_1", x.detach()))
        x = torch.relu(model.conv2(x, edge_index))
        layers.append(("layer_2", x.detach()))
    return layers


def _embedding_dispersion(embedding: "torch.Tensor") -> dict[str, float]:
    """Compute cross-sectional embedding dispersion metrics."""
    import torch

    emb = embedding.detach().cpu()
    pairwise = torch.pdist(emb, p=2)
    center = emb.mean(dim=0, keepdim=True)
    return {
        "mean_pairwise_embedding_distance": float(pairwise.mean().item()) if len(pairwise) else float("nan"),
        "mean_absolute_deviation": float((emb - center).abs().mean().item()),
    }


def compute_oversmoothing_audit(
    registry: pd.DataFrame,
    tickers: list[str],
) -> pd.DataFrame:
    """Capture GNN layer embeddings and summarize cross-sectional dispersion."""
    import torch
    from src.graphs import load_corr_graphs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, week_index = _features_to_array(tickers)
    splits = pd.read_parquet(Path(config.DATA_FEATURES_DIR) / "splits.parquet")
    splits["week"] = pd.to_datetime(splits["week"])
    audit_weeks = _select_audit_weeks(week_index, splits)
    corr_graphs = {
        split_name: load_corr_graphs(config.CORR_THRESHOLD, split_name)
        for split_name in ("train", "val", "test")
    }
    granger_graph = _load_granger_graph().to(device)

    split_lookup = splits.set_index("week")["split"].to_dict()
    rows: list[dict] = []
    gnn_rows = registry.loc[
        registry["model_family"].astype(str).str.contains("GNN", case=False, na=False)
        & registry["checkpoint_path"].astype(str).str.strip().ne("")
        & (registry["feature_version"].astype(str) == "stock_features_v1")
    ]
    for _, row in gnn_rows.iterrows():
        checkpoint = _repo_path(row["checkpoint_path"])
        if checkpoint is None or not checkpoint.exists():
            continue
        model = _load_gnn_model_from_checkpoint(checkpoint, features.shape[2], device)
        graph_type = str(row["graph_type"])

        for week_label, week in audit_weeks.items():
            position = int(np.where(week_index == week)[0][0])
            x = torch.tensor(features[position], dtype=torch.float32, device=device)
            split_name = split_lookup[week]
            if graph_type == "correlation":
                edge_index = corr_graphs[split_name].get(
                    week,
                    torch.zeros(2, 0, dtype=torch.long),
                ).to(device)
            elif graph_type == "sector":
                edge_index = _load_sector_graph(week.year).to(device)
            elif graph_type == "granger":
                edge_index = granger_graph
            else:
                continue

            with torch.no_grad():
                layer_rows = []
                for layer_name, embedding in _embedding_layers(model, x, edge_index):
                    metrics = _embedding_dispersion(embedding)
                    layer_rows.append((layer_name, metrics))

                layer_1_distance = layer_rows[0][1]["mean_pairwise_embedding_distance"]
                final_distance = layer_rows[-1][1]["mean_pairwise_embedding_distance"]
                ratio = final_distance / layer_1_distance if layer_1_distance > 0 else float("nan")
                for layer_name, metrics in layer_rows:
                    rows.append(
                        {
                            "model": row["model_name"],
                            "experiment_id": row["experiment_id"],
                            "graph_type": graph_type,
                            "week_label": week_label,
                            "week": week,
                            "layer": layer_name,
                            "num_edges": int(edge_index.shape[1]),
                            **metrics,
                            "final_to_layer1_distance_ratio": ratio,
                        }
                    )

    return pd.DataFrame(rows).sort_values(["model", "week", "layer"]).reset_index(drop=True)


def _save_diagnostic_figures(
    calibration_bins: pd.DataFrame,
    spread: pd.DataFrame,
    graph_density: pd.DataFrame,
    regime_breakdown: pd.DataFrame,
    within_sector_table: pd.DataFrame,
    oversmoothing: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Render compact diagnostic figures from saved tables."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not calibration_bins.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model, group in calibration_bins.groupby("model"):
            ax.plot(group["predicted_rv_mean"], group["actual_rv_mean"], marker="o", linewidth=1.2, label=model)
        ax.set_xlabel("Mean predicted RV by decile")
        ax.set_ylabel("Mean realized RV")
        ax.set_title("Calibration by predicted-RV decile")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = figures_dir / "calibration_deciles.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    spread_summary = spread.groupby("model", as_index=False)["prediction_spread_p90_p10"].mean()
    ax.barh(spread_summary["model"], spread_summary["prediction_spread_p90_p10"])
    ax.set_xlabel("Average weekly p90-p10 predicted RV spread")
    ax.set_title("Prediction spread by model")
    fig.tight_layout()
    path = figures_dir / "prediction_spread_by_week.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    density = graph_density.loc[graph_density["threshold"] == config.CORR_THRESHOLD]
    ax.plot(density["week"], density["density"], linewidth=1.2)
    ax.set_xlabel("Week")
    ax.set_ylabel("Density")
    ax.set_title(f"Correlation graph density, threshold={config.CORR_THRESHOLD}")
    fig.autofmt_xdate()
    fig.tight_layout()
    path = figures_dir / "correlation_graph_density.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_regime = regime_breakdown.loc[regime_breakdown["regime_type"].isin(["calendar_year", "graph_density"])]
    pivot = plot_regime.pivot_table(index="model", columns=["regime_type", "regime"], values="mse")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MSE")
    ax.set_title("Regime breakdown: MSE")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = figures_dir / "regime_breakdown_mse.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(within_sector_table["model"], within_sector_table["mean_within_sector_rank_ic"])
    ax.set_xlabel("Mean within-sector Rank IC")
    ax.set_title("Within-sector ranking")
    fig.tight_layout()
    path = figures_dir / "within_sector_rank_ic.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    if not oversmoothing.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        final_layers = oversmoothing.sort_values("layer").groupby(["model", "week_label"], as_index=False).tail(1)
        pivot = final_layers.pivot_table(
            index="model",
            columns="week_label",
            values="final_to_layer1_distance_ratio",
        )
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Final layer / layer 1 distance")
        ax.set_title("GNN oversmoothing audit")
        ax.legend(fontsize=7)
        fig.tight_layout()
        path = figures_dir / "oversmoothing_audit.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    return paths


def generate_diagnostic_artifacts(
    results_dir: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Path]:
    """Generate all Step 5 diagnostic tables and figures."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    figures_dir = Path(figures_dir or config.FIGURES_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions, registry = load_predictions(results_dir)
    target = load_target(results_dir)
    first_predictions = next(iter(predictions.values()))
    tickers = list(first_predictions.columns)
    log_returns = pd.read_parquet(Path(config.DATA_RAW_DIR) / "log_returns.parquet").reindex(columns=tickers)
    log_returns.index = pd.to_datetime(log_returns.index)

    calibration_summary, calibration_bins = compute_calibration_diagnostics(predictions, target)
    prediction_spread = compute_prediction_spread_by_week(predictions)
    graph_density = compute_correlation_graph_density(tickers, log_returns)
    regime_breakdown = compute_regime_breakdowns(predictions, target, log_returns, graph_density, results_dir)
    sector_map = _sector_map_for_year(tickers)
    within_sector_table, within_sector_by_sector = compute_within_sector_rank_ic(predictions, target, sector_map)
    oversmoothing = compute_oversmoothing_audit(registry, tickers)

    outputs = {
        "calibration_summary": results_dir / "calibration_summary.csv",
        "calibration_bins": results_dir / "calibration_bins.csv",
        "prediction_spread_by_week": results_dir / "prediction_spread_by_week.csv",
        "correlation_graph_density": results_dir / "correlation_graph_density.csv",
        "regime_breakdown_metrics": results_dir / "regime_breakdown_metrics.csv",
        "oversmoothing_audit": results_dir / "oversmoothing_audit.csv",
        "within_sector_rank_ic_table": results_dir / "within_sector_rank_ic_table.csv",
        "within_sector_rank_ic_by_sector": results_dir / "within_sector_rank_ic_by_sector.csv",
    }

    calibration_summary.to_csv(outputs["calibration_summary"], index=False)
    calibration_bins.to_csv(outputs["calibration_bins"], index=False)
    prediction_spread.to_csv(outputs["prediction_spread_by_week"], index=False)
    graph_density.to_csv(outputs["correlation_graph_density"], index=False)
    regime_breakdown.to_csv(outputs["regime_breakdown_metrics"], index=False)
    oversmoothing.to_csv(outputs["oversmoothing_audit"], index=False)
    within_sector_table.to_csv(outputs["within_sector_rank_ic_table"], index=False)
    within_sector_by_sector.to_csv(outputs["within_sector_rank_ic_by_sector"], index=False)

    figure_paths = _save_diagnostic_figures(
        calibration_bins,
        prediction_spread,
        graph_density,
        regime_breakdown,
        within_sector_table,
        oversmoothing,
        figures_dir,
    )
    outputs.update({f"figure_{path.stem}": path for path in figure_paths})
    return outputs


def main() -> None:
    """Command-line entry point."""
    outputs = generate_diagnostic_artifacts()
    print("Generated diagnostic artifacts:")
    for name, path in outputs.items():
        try:
            display_path = path.relative_to(config._ROOT)
        except ValueError:
            display_path = path
        print(f"  {name}: {display_path}")


if __name__ == "__main__":
    main()
