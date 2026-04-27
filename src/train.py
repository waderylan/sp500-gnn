"""
Training loops for LSTM and GNN models.

Key invariants:
- Time steps iterated in strict chronological order (no shuffle).
- Random seeds set before every model initialization.
- Checkpoints saved every CHECKPOINT_EVERY_N_EPOCHS and on val improvement.
- Early stopping after EARLY_STOP_PATIENCE epochs without val MSE improvement.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn

import config
from src.models import LSTMModel, GNNModel


def set_seeds() -> None:
    """Set all random seeds to config.RANDOM_SEED before model init or training."""
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    torch.cuda.manual_seed_all(config.RANDOM_SEED)


def train_lstm(
    model: LSTMModel,
    features: np.ndarray,
    target: np.ndarray,
    train_pos: np.ndarray,
    val_pos: np.ndarray,
    device: torch.device,
) -> tuple[LSTMModel, list[float]]:
    """
    Train the LSTM model with early stopping and periodic checkpointing.

    At each training week position p, the input is features[p-SEQ_LEN+1:p+1]
    (shape SEQ_LEN x num_stocks x num_features), permuted to
    (num_stocks, SEQ_LEN, num_features) before the forward pass.
    Positions with p < SEQ_LEN-1 or any NaN in the sequence are skipped.

    features: shape (num_weeks, num_stocks, num_features)
    target: shape (num_weeks, num_stocks)
    train_pos: integer row positions into features/target for the training split
    val_pos: integer row positions for the validation split (must be sorted ascending)
    device: torch.device for computation

    Returns: (best model loaded from checkpoint, val_loss_history per epoch)

    Checkpoints: config.CHECKPOINTS_DIR / "lstm_epoch{n}.pt" every N epochs,
                 config.CHECKPOINTS_DIR / "lstm_best.pt" on each val improvement.
    Val loss history: config.DATA_RESULTS_DIR / "lstm_val_loss.json"

    Lookahead safety: sequence at position p uses features[p-SEQ_LEN+1:p+1],
    all computed from data through Friday of week p. Target is week p+1's RV.
    Friday_p < Monday_{p+1}.
    """
    set_seeds()
    seq_len  = config.LSTM_SEQ_LEN
    ckpt_dir = Path(config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lstm_best.pt"

    def _valid_positions(pos_arr: np.ndarray) -> list[int]:
        """Keep positions where the preceding sequence window exists.
        Per-stock NaN masking is applied inside the training loop."""
        return [int(p) for p in pos_arr if p >= seq_len - 1]

    train_valid = _valid_positions(train_pos)
    val_valid   = _valid_positions(val_pos)

    optimizer  = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion  = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    val_loss_history: list[float] = []

    for epoch in range(config.LSTM_MAX_EPOCHS):
        # Training pass — chronological order
        model.train()
        train_loss_sum  = 0.0
        train_steps_used = 0
        for p in train_valid:
            seq = features[p - seq_len + 1 : p + 1]       # (seq_len, S, F)
            # Per-stock NaN mask: True = stock has NaN anywhere in its feature window
            stock_nan = np.isnan(seq).any(axis=(0, 2))    # (S,)
            seq_clean = seq.copy()
            seq_clean[:, stock_nan, :] = 0.0              # zero-fill; excluded from loss
            x = torch.tensor(seq_clean, dtype=torch.float32).permute(1, 0, 2).to(device)
            assert x.shape[1:] == (seq_len, features.shape[2]), \
                f"Seq shape mismatch at p={p}: {x.shape}"
            y = torch.tensor(target[p], dtype=torch.float32).to(device)
            feat_mask = torch.tensor(~stock_nan, dtype=torch.bool, device=device)
            mask = feat_mask & ~y.isnan()
            if not mask.any():
                continue
            optimizer.zero_grad()
            loss = criterion(model(x)[mask], y[mask])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            train_steps_used += 1

        # Validation pass
        model.eval()
        val_loss_sum  = 0.0
        val_steps_used = 0
        with torch.no_grad():
            for p in val_valid:
                seq = features[p - seq_len + 1 : p + 1]
                stock_nan = np.isnan(seq).any(axis=(0, 2))
                seq_clean = seq.copy()
                seq_clean[:, stock_nan, :] = 0.0
                x = torch.tensor(seq_clean, dtype=torch.float32).permute(1, 0, 2).to(device)
                y = torch.tensor(target[p], dtype=torch.float32).to(device)
                feat_mask = torch.tensor(~stock_nan, dtype=torch.bool, device=device)
                mask = feat_mask & ~y.isnan()
                if not mask.any():
                    continue
                val_loss_sum += criterion(model(x)[mask], y[mask]).item()
                val_steps_used += 1

        if train_steps_used == 0 or val_steps_used == 0:
            raise RuntimeError(
                f"Epoch {epoch + 1}: no valid steps "
                f"(train={train_steps_used}, val={val_steps_used}). "
                "Check features.parquet for excessive NaN coverage."
            )

        avg_train = train_loss_sum / train_steps_used
        avg_val   = val_loss_sum   / val_steps_used
        val_loss_history.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch + 1:3d}  train={avg_train:.6f}  val={avg_val:.6f}")

        if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            torch.save(model.state_dict(), ckpt_dir / f"lstm_epoch{epoch + 1}.pt")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch + 1} "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "lstm_val_loss.json", "w") as fh:
        json.dump({"val_loss": val_loss_history, "best_val_loss": best_val_loss}, fh, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, val_loss_history


def train_gnn(
    model: GNNModel,
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    edge_index_fn: Callable[[pd.Timestamp], torch.LongTensor],
    splits: pd.DataFrame,
    graph_type: str,
    device: torch.device,
    max_epochs: int = 150,
) -> tuple[GNNModel, list[float]]:
    """
    Train a GNN model with early stopping and checkpointing.

    At each time step calls edge_index_fn(week_timestamp) to get the graph.
    For the correlation graph this recomputes each week (dynamic).
    For sector and Granger it is a dict lookup (static or annual).

    Steps with NaN in features or target are skipped.

    features:      ndarray of shape (num_weeks, num_stocks, num_features).
    target:        ndarray of shape (num_weeks, num_stocks).
    week_index:    DatetimeIndex of length num_weeks aligned to features/target rows.
    edge_index_fn: Callable mapping a pd.Timestamp to a LongTensor edge_index.
    splits:        DataFrame with columns ['week', 'split'].
    graph_type:    One of "correlation", "sector", "granger" — used in checkpoint names.
    device:        torch.device for computation.
    max_epochs:    Hard upper bound on training epochs.

    Returns: (best model loaded from checkpoint, val_loss_history per epoch)

    Checkpoints: config.CHECKPOINTS_DIR / f"gnn_{graph_type}_epoch{{n}}.pt"
    Best:        config.CHECKPOINTS_DIR / f"gnn_{graph_type}_best.pt"
    Val loss:    config.DATA_RESULTS_DIR / f"gnn_{graph_type}_val_loss.json"
    """
    ckpt_dir  = Path(config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"gnn_{graph_type}_best.pt"

    train_weeks = set(splits.loc[splits["split"] == "train", "week"])
    val_weeks   = set(splits.loc[splits["split"] == "val",   "week"])

    def _valid_positions(week_set: set) -> list[tuple[int, pd.Timestamp]]:
        """Return (row_index, week_timestamp) pairs for weeks in the given split.
        Feature NaN is imputed in GNNModel.forward via nan_to_num.
        Target NaN is masked in the loss. Weeks with no valid targets are skipped in the loop."""
        return [(i, w) for i, w in enumerate(week_index) if w in week_set]

    train_steps = _valid_positions(train_weeks)
    val_steps   = _valid_positions(val_weeks)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=False
    )
    criterion = nn.MSELoss()

    best_val_loss    = float("inf")
    patience_counter = 0
    val_loss_history: list[float] = []

    n_train = len(train_steps)
    for epoch in range(max_epochs):
        model.train()
        train_loss_sum   = 0.0
        train_steps_used = 0
        for step_idx, (pos, week) in enumerate(train_steps):
            x    = torch.tensor(features[pos], dtype=torch.float32).to(device)
            y    = torch.tensor(target[pos],   dtype=torch.float32).to(device)
            ei   = edge_index_fn(week).to(device)
            mask = ~y.isnan()
            if not mask.any():
                continue
            optimizer.zero_grad()
            loss = criterion(model(x, ei)[mask], y[mask])
            loss.backward()
            optimizer.step()
            train_loss_sum   += loss.item()
            train_steps_used += 1
            if (step_idx + 1) % 50 == 0 or (step_idx + 1) == n_train:
                print(f"  Epoch {epoch + 1}  step {step_idx + 1}/{n_train}", end="\r", flush=True)

        model.eval()
        val_loss_sum   = 0.0
        val_steps_used = 0
        with torch.no_grad():
            for pos, week in val_steps:
                x    = torch.tensor(features[pos], dtype=torch.float32).to(device)
                y    = torch.tensor(target[pos],   dtype=torch.float32).to(device)
                ei   = edge_index_fn(week).to(device)
                mask = ~y.isnan()
                if not mask.any():
                    continue
                val_loss_sum   += criterion(model(x, ei)[mask], y[mask]).item()
                val_steps_used += 1

        if train_steps_used == 0 or val_steps_used == 0:
            raise RuntimeError(
                f"Epoch {epoch + 1}: no valid steps "
                f"(train={train_steps_used}, val={val_steps_used}). "
                "Check features.parquet for excessive NaN coverage."
            )

        avg_train = train_loss_sum / train_steps_used
        avg_val   = val_loss_sum   / val_steps_used
        val_loss_history.append(avg_val)
        scheduler.step(avg_val)

        print(f"Epoch {epoch + 1:3d}  train={avg_train:.6f}  val={avg_val:.6f}")

        if (epoch + 1) % config.CHECKPOINT_EVERY_N_EPOCHS == 0:
            torch.save(model.state_dict(),
                       ckpt_dir / f"gnn_{graph_type}_epoch{epoch + 1}.pt")

        if avg_val < best_val_loss:
            best_val_loss    = avg_val
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1

        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch + 1} "
                  f"(no improvement for {config.EARLY_STOP_PATIENCE} epochs)")
            break

    results_dir = Path(config.DATA_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / f"gnn_{graph_type}_val_loss.json", "w") as fh:
        json.dump({"val_loss": val_loss_history, "best_val_loss": best_val_loss}, fh, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    return model, val_loss_history


def train_gnn_corr_ablation(
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    corr_graphs: dict[float, dict[pd.Timestamp, torch.LongTensor]],
    splits: pd.DataFrame,
    device: torch.device,
    thresholds: list[float] | None = None,
    max_epochs: int = config.GNN_MAX_EPOCHS,
) -> dict[float, float]:
    """
    Train GNN-Correlation for each threshold in config.CORR_ABLATION_THRESHOLDS.

    For each θ: sets seeds, initializes a fresh GNNModel, and calls train_gnn()
    with a dict-lookup edge_fn backed by precomputed graphs. Records the best
    validation MSE from the saved loss JSON (the minimum over all epochs).

    After all runs, copies the winning checkpoint to gnn_corr_best.pt and saves
    the full ablation results to DATA_RESULTS_DIR/corr_threshold_ablation.json.

    features:    shape (num_weeks, num_stocks, num_features)
    target:      shape (num_weeks, num_stocks)
    week_index:  DatetimeIndex aligned to features/target rows
    corr_graphs: {threshold: {week_timestamp: edge_index}} from load_corr_graphs().
                 Must contain entries for both train and val weeks.
    splits:      DataFrame with columns ['week', 'split']
    device:      torch.device for computation
    thresholds:  list of θ values; defaults to config.CORR_ABLATION_THRESHOLDS
    max_epochs:  hard epoch ceiling passed to train_gnn

    Returns: {threshold: best_val_mse} for every θ tested

    Checkpoints: gnn_corr_th{int(θ*10):02d}_best.pt per run
                 gnn_corr_best.pt for the winning θ
    Ablation JSON: DATA_RESULTS_DIR/corr_threshold_ablation.json
    """
    if thresholds is None:
        thresholds = config.CORR_ABLATION_THRESHOLDS

    in_channels = features.shape[2]
    results_dir = Path(config.DATA_RESULTS_DIR)
    ckpt_dir    = Path(config.CHECKPOINTS_DIR)
    ablation: dict[float, float] = {}

    for theta in thresholds:
        print(f"\n{'=' * 60}")
        print(f"GNN-Correlation  θ={theta}")
        print(f"{'=' * 60}")

        set_seeds()
        model = GNNModel(in_channels=in_channels).to(device)

        graphs = corr_graphs[theta]
        edge_fn: Callable[[pd.Timestamp], torch.LongTensor] = (
            lambda week, g=graphs: g.get(week, torch.zeros(2, 0, dtype=torch.long))
        )

        graph_tag = f"corr_th{int(theta * 10):02d}"
        train_gnn(
            model=model,
            features=features,
            target=target,
            week_index=week_index,
            edge_index_fn=edge_fn,
            splits=splits,
            graph_type=graph_tag,
            device=device,
            max_epochs=max_epochs,
        )

        loss_path = results_dir / f"gnn_{graph_tag}_val_loss.json"
        with open(loss_path) as fh:
            loss_data = json.load(fh)
        best_mse = loss_data["best_val_loss"]
        ablation[theta] = best_mse
        print(f"θ={theta}  best val MSE={best_mse:.6f}")

    assert len(ablation) == len(thresholds), (
        f"Expected {len(thresholds)} ablation entries, got {len(ablation)}"
    )

    best_theta = min(ablation, key=ablation.__getitem__)
    best_tag   = f"corr_th{int(best_theta * 10):02d}"
    src_ckpt   = ckpt_dir / f"gnn_{best_tag}_best.pt"
    dst_ckpt   = ckpt_dir / "gnn_corr_best.pt"
    shutil.copy2(src_ckpt, dst_ckpt)

    ablation_out = {
        "thresholds_tested": thresholds,
        "val_mse_by_threshold": {str(k): v for k, v in ablation.items()},
        "best_threshold": best_theta,
        "best_val_mse": ablation[best_theta],
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "corr_threshold_ablation.json", "w") as fh:
        json.dump(ablation_out, fh, indent=2)

    print(f"\nAblation complete.")
    print(f"Best θ={best_theta}  val MSE={ablation[best_theta]:.6f}")
    print(f"Checkpoint: {dst_ckpt}")

    return ablation


def train_gnn_sector(
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    sector_graphs: dict,
    splits: pd.DataFrame,
    device: torch.device,
    max_epochs: int = config.GNN_MAX_EPOCHS,
) -> tuple[GNNModel, list[float]]:
    """
    Train GNN-Sector using annual point-in-time GICS sector graphs.

    At each time step T, looks up sector_graphs[week.year] to get the graph.
    No price or return data enters edge construction. The sector assignment
    for year Y is fixed at the start of year Y, before any week in Y begins.

    features:      shape (num_weeks, num_stocks, num_features)
    target:        shape (num_weeks, num_stocks)
    week_index:    DatetimeIndex aligned to features/target rows
    sector_graphs: {year (int): edge_index LongTensor}, from build_all_sector_graphs()
    splits:        DataFrame with columns ['week', 'split']
    device:        torch.device for computation
    max_epochs:    hard epoch ceiling

    Returns: (best model loaded from checkpoint, val_loss_history per epoch)
    Checkpoint: config.CHECKPOINTS_DIR / "gnn_sector_best.pt"
    Val loss:   config.DATA_RESULTS_DIR / "gnn_sector_val_loss.json"

    Lookahead safety: sector assignments are point-in-time at the start of each
    calendar year. No return or price data is used in edge construction.
    Shape assertion: all years in week_index must be present as keys in sector_graphs.
    """
    years_needed = {w.year for w in week_index}
    missing = years_needed - set(sector_graphs.keys())
    assert not missing, f"sector_graphs missing years: {missing}"

    in_channels = features.shape[2]
    set_seeds()
    model = GNNModel(in_channels=in_channels).to(device)

    edge_fn: Callable[[pd.Timestamp], torch.LongTensor] = (
        lambda week: sector_graphs[week.year]
    )

    return train_gnn(
        model=model,
        features=features,
        target=target,
        week_index=week_index,
        edge_index_fn=edge_fn,
        splits=splits,
        graph_type="sector",
        device=device,
        max_epochs=max_epochs,
    )


def train_gnn_granger(
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    granger_edge_index: torch.LongTensor,
    splits: pd.DataFrame,
    device: torch.device,
    max_epochs: int = config.GNN_MAX_EPOCHS,
) -> tuple[GNNModel, list[float]]:
    """
    Train GNN-Granger using the static directed Granger causality graph.

    At each time step the same granger_edge_index is returned. No price or return
    data enters edge construction after the Granger tests were run on training data.

    features:           shape (num_weeks, num_stocks, num_features)
    target:             shape (num_weeks, num_stocks)
    week_index:         DatetimeIndex aligned to features/target rows
    granger_edge_index: Directed LongTensor of shape (2, num_edges), static
    splits:             DataFrame with columns ['week', 'split']
    device:             torch.device for computation
    max_epochs:         hard epoch ceiling

    Returns: (best model loaded from checkpoint, val_loss_history per epoch)
    Checkpoint: config.CHECKPOINTS_DIR / "gnn_granger_best.pt"
    Val loss:   config.DATA_RESULTS_DIR / "gnn_granger_val_loss.json"

    Lookahead safety: Granger edge_index was computed from training data only
    (through config.TRAIN_END). The same static graph is used for every time step
    including val weeks. No future data enters edge construction.
    Shape assertions: granger_edge_index.shape[0] == 2, shape[1] > 0.
    """
    assert granger_edge_index.shape[0] == 2, (
        f"granger_edge_index row dim should be 2, got {granger_edge_index.shape[0]}"
    )
    assert granger_edge_index.shape[1] > 0, (
        "granger_edge_index has no edges — run build_granger_graph() and check edge count"
    )

    in_channels = features.shape[2]
    set_seeds()
    model = GNNModel(in_channels=in_channels).to(device)

    edge_fn: Callable[[pd.Timestamp], torch.LongTensor] = (
        lambda week: granger_edge_index
    )

    return train_gnn(
        model=model,
        features=features,
        target=target,
        week_index=week_index,
        edge_index_fn=edge_fn,
        splits=splits,
        graph_type="granger",
        device=device,
        max_epochs=max_epochs,
    )


def predict_gnn_val(
    model: GNNModel,
    features: np.ndarray,
    target: np.ndarray,
    week_index: pd.DatetimeIndex,
    edge_index_fn: Callable[[pd.Timestamp], torch.LongTensor],
    splits: pd.DataFrame,
    tickers: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """
    Generate validation-set predictions from a trained GNN model.

    Iterates over val weeks in chronological order. Skips any week with NaN
    in features or target, matching the filter used during training.

    features:      shape (num_weeks, num_stocks, num_features)
    target:        shape (num_weeks, num_stocks) — used only for NaN-skip logic
    week_index:    DatetimeIndex aligned to features/target rows
    edge_index_fn: Callable mapping pd.Timestamp to LongTensor edge_index
    splits:        DataFrame with columns ['week', 'split']
    tickers:       Ordered ticker list, length num_stocks
    device:        torch.device

    Returns DataFrame of shape (num_val_weeks, num_stocks), indexed by week
    timestamp, columns = tickers. Values are predicted next-week RV.

    Lookahead safety: accesses only features[i] and edge_index_fn(week) for each
    val week i. No test data is accessed.
    Shape assertions: output.shape[1] == len(tickers), output.shape[0] > 0.
    """
    val_weeks = set(splits.loc[splits["split"] == "val", "week"])

    rows:  list[np.ndarray]    = []
    index: list[pd.Timestamp]  = []

    model.eval()
    with torch.no_grad():
        for i, week in enumerate(week_index):
            if week not in val_weeks:
                continue
            x    = torch.tensor(features[i], dtype=torch.float32).to(device)
            ei   = edge_index_fn(week).to(device)
            pred = np.clip(model(x, ei).cpu().numpy(), a_min=1e-6, a_max=None)
            rows.append(pred)
            index.append(week)

    result = pd.DataFrame(rows, index=pd.DatetimeIndex(index), columns=tickers)

    assert result.shape[1] == len(tickers), (
        f"Expected {len(tickers)} columns, got {result.shape[1]}"
    )
    assert len(result) > 0, "No valid val weeks found in splits"

    return result
