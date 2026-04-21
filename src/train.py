"""
Training loops for LSTM and GNN models.

Key invariants:
- Time steps iterated in strict chronological order (no shuffle).
- Random seeds set before every model initialization.
- Checkpoints saved every CHECKPOINT_EVERY_N_EPOCHS and on val improvement.
- Early stopping after EARLY_STOP_PATIENCE epochs without val MSE improvement.
"""

from __future__ import annotations

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


def train_lstm(model: LSTMModel,
               features: np.ndarray,
               target: np.ndarray,
               splits: object,
               device: torch.device) -> LSTMModel:
    """
    Train the LSTM model with early stopping and checkpointing.

    Iterates over training weeks in chronological order. At each epoch,
    evaluates on the validation set to track early stopping.

    Args:
        model: Initialized LSTMModel (seeds must be set before calling this).
        features: ndarray of shape (num_weeks, num_stocks, num_features).
        target: ndarray of shape (num_weeks, num_stocks).
        splits: DataFrame with 'week' and 'split' columns.
        device: torch.device to run training on.

    Returns:
        Best model (loaded from checkpoint at best val epoch).

    Checkpoints saved to: config.CHECKPOINTS_DIR / "lstm_epoch{n}.pt"
    Best checkpoint: config.CHECKPOINTS_DIR / "lstm_best.pt"
    """
    raise NotImplementedError


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
