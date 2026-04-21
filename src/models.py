"""
Model definitions: HAR (per-stock and pooled), LSTM, and GNN (GraphSAGE).

GNNModel is a single class used for all three graph variants (correlation, sector, Granger).
The graph is passed at forward time, not stored in the model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from sklearn.linear_model import LinearRegression

import config


class HARModel:
    """
    Per-stock HAR-RV model using sklearn LinearRegression.

    Fits one model per ticker. Input features: RV_1w, RV_4w, RV_13w.
    HAR uses only RV features computed directly from weekly_rv — NOT from features.parquet.
    """

    def fit(self, X: dict[str, object], y: dict[str, object]) -> None:
        """
        Fit one LinearRegression per ticker.

        Args:
            X: Dict mapping ticker -> ndarray of shape (num_train_weeks, 3).
            y: Dict mapping ticker -> ndarray of shape (num_train_weeks,).
        """
        raise NotImplementedError

    def predict(self, X: dict[str, object]) -> dict[str, object]:
        """
        Predict for each ticker.

        Returns:
            Dict mapping ticker -> ndarray of shape (num_weeks,).
        """
        raise NotImplementedError


class HARPooled:
    """
    Pooled HAR-RV model: single LinearRegression fit across all stocks simultaneously.

    Input features: RV_1w, RV_4w, RV_13w (same as HARModel but pooled).
    """

    def fit(self, X: object, y: object) -> None:
        """
        Args:
            X: ndarray of shape (num_train_weeks * num_stocks, 3).
            y: ndarray of shape (num_train_weeks * num_stocks,).
        """
        raise NotImplementedError

    def predict(self, X: object) -> object:
        """
        Returns:
            ndarray of shape (num_weeks * num_stocks,).
        """
        raise NotImplementedError


class LSTMModel(nn.Module):
    """
    2-layer LSTM for per-stock volatility forecasting.

    Architecture: LSTM(input_size, hidden=64, num_layers=2, dropout=0.3) → Linear(64, 1).
    Input: 4-week sequence of feature vectors per stock.
    Output: scalar prediction per stock.
    """

    def __init__(self, input_size: int,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (num_stocks, seq_len, input_size).

        Returns:
            Tensor of shape (num_stocks,).
        """
        raise NotImplementedError


class GNNModel(nn.Module):
    """
    Two-layer GraphSAGE model for cross-sectional volatility prediction.

    Architecture:
        SAGEConv(in_channels, 64, flow=SAGE_FLOW) → ReLU → Dropout(0.3)
        SAGEConv(64, 64, flow=SAGE_FLOW)           → ReLU → Dropout(0.3)
        Linear(64, 1)

    Output: shape (num_stocks,) — one scalar per stock.

    Used for all three graph variants. Graph is passed at forward time.
    flow=config.SAGE_FLOW must be set on every SAGEConv layer.
    """

    def __init__(self, in_channels: int,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: Node features, shape (num_stocks, in_channels).
            edge_index: Graph connectivity, shape (2, num_edges).

        Returns:
            Tensor of shape (num_stocks,).
        """
        raise NotImplementedError
