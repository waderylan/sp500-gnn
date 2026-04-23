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
from src.graphs import build_correlation_graph


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
    seq_len  = config.LSTM_SEQ_LEN
    ckpt_dir = Path(config.CHECKPOINTS_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "lstm_best.pt"

    def _valid_positions(pos_arr: np.ndarray) -> list[int]:
        """Keep positions with a full sequence and no NaN in features or target."""
        out = []
        for p in pos_arr:
            if p < seq_len - 1:
                continue
            if np.isnan(features[p - seq_len + 1 : p + 1]).any():
                continue
            if np.isnan(target[p]).any():
                continue
            out.append(int(p))
        return out

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
        train_loss_sum = 0.0
        for p in train_valid:
            seq = features[p - seq_len + 1 : p + 1]       # (seq_len, S, F)
            x   = torch.tensor(seq, dtype=torch.float32).permute(1, 0, 2).to(device)
            assert x.shape[1:] == (seq_len, features.shape[2]), \
                f"Seq shape mismatch at p={p}: {x.shape}"
            y = torch.tensor(target[p], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        # Validation pass
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for p in val_valid:
                seq = features[p - seq_len + 1 : p + 1]
                x   = torch.tensor(seq, dtype=torch.float32).permute(1, 0, 2).to(device)
                y   = torch.tensor(target[p], dtype=torch.float32).to(device)
                val_loss_sum += criterion(model(x), y).item()

        avg_train = train_loss_sum / len(train_valid)
        avg_val   = val_loss_sum   / len(val_valid)
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
        out = []
        for i, w in enumerate(week_index):
            if w not in week_set:
                continue
            if np.isnan(features[i]).any() or np.isnan(target[i]).any():
                continue
            out.append((i, w))
        return out

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

    for epoch in range(max_epochs):
        model.train()
        train_loss_sum = 0.0
        for pos, week in train_steps:
            x   = torch.tensor(features[pos], dtype=torch.float32).to(device)
            y   = torch.tensor(target[pos],   dtype=torch.float32).to(device)
            ei  = edge_index_fn(week).to(device)
            optimizer.zero_grad()
            loss = criterion(model(x, ei), y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for pos, week in val_steps:
                x  = torch.tensor(features[pos], dtype=torch.float32).to(device)
                y  = torch.tensor(target[pos],   dtype=torch.float32).to(device)
                ei = edge_index_fn(week).to(device)
                val_loss_sum += criterion(model(x, ei), y).item()

        avg_train = train_loss_sum / len(train_steps)
        avg_val   = val_loss_sum   / len(val_steps)
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
    log_returns: pd.DataFrame,
    splits: pd.DataFrame,
    device: torch.device,
    thresholds: list[float] | None = None,
    max_epochs: int = config.GNN_MAX_EPOCHS,
) -> dict[float, float]:
    """
    Train GNN-Correlation for each threshold in config.CORR_ABLATION_THRESHOLDS.

    For each θ: sets seeds, initializes a fresh GNNModel, and calls train_gnn()
    with edge_index_fn = build_correlation_graph(..., threshold=θ). Records the
    best validation MSE from the saved loss JSON (the minimum over all epochs).

    After all runs, copies the winning checkpoint to gnn_corr_best.pt and saves
    the full ablation results to DATA_RESULTS_DIR/corr_threshold_ablation.json.

    features:    shape (num_weeks, num_stocks, num_features)
    target:      shape (num_weeks, num_stocks)
    week_index:  DatetimeIndex aligned to features/target rows
    log_returns: daily log returns, used by build_correlation_graph
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

        # Default arg captures theta at definition time — avoids loop closure bug
        edge_fn = lambda week, th=theta: build_correlation_graph(
            log_returns, week, config.CORR_LOOKBACK_DAYS, th
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
