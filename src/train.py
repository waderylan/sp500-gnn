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
from pathlib import Path

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


def train_gnn(model: GNNModel,
              features: np.ndarray,
              target: np.ndarray,
              edge_index_fn: object,
              splits: object,
              graph_type: str,
              device: torch.device) -> GNNModel:
    """
    Train a GNN model with early stopping and checkpointing.

    At each time step, calls edge_index_fn(week) to get the graph for that week.
    For the correlation graph this is dynamic (recomputed each week).
    For sector and Granger graphs this is a lookup.

    Args:
        model: Initialized GNNModel (seeds must be set before calling this).
        features: ndarray of shape (num_weeks, num_stocks, num_features).
        target: ndarray of shape (num_weeks, num_stocks).
        edge_index_fn: Callable mapping a week timestamp to a LongTensor edge_index.
        splits: DataFrame with 'week' and 'split' columns.
        graph_type: One of "correlation", "sector", "granger" — used in checkpoint names.
        device: torch.device to run training on.

    Returns:
        Best model (loaded from checkpoint at best val epoch).

    Checkpoints saved to: config.CHECKPOINTS_DIR / f"gnn_{graph_type}_epoch{{n}}.pt"
    Best checkpoint: config.CHECKPOINTS_DIR / f"gnn_{graph_type}_best.pt"
    """
    raise NotImplementedError
