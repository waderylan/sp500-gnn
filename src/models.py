"""
Model definitions: HAR (per-stock and pooled), LSTM, and GNN (GraphSAGE).

GNNModel is a single class used for all three graph variants (correlation, sector, Granger).
The graph is passed at forward time, not stored in the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.norm import GraphNorm
from sklearn.linear_model import LinearRegression

import config


# ---------------------------------------------------------------------------
# HAR feature helpers
# ---------------------------------------------------------------------------

def compute_har_features(
    weekly_rv: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute HAR-RV input features from weekly realized volatility.

    Forward-fills isolated NaN gaps along the time axis before computing rolling
    windows. ffill at time T uses the last known value from at or before T,
    so this is lookahead-safe.

    This is a 1-step-ahead setup: features at row T include week T's own RV
    (and rolling means ending at T). The target is week T+1's RV. Week T's RV
    is fully observed by Friday of week T, strictly before week T+1 begins.

    weekly_rv: shape (num_weeks, num_stocks), Monday-indexed weekly RV.
    Returns: (rv_1w, rv_4w, rv_13w), each shape (num_weeks, num_stocks).

    Lookahead safety: at row T, rv_1w = weekly_rv[T]; rv_4w = mean of weeks T-3
    to T; rv_13w = mean of weeks T-12 to T. All data through Friday of week T.
    First 12 rows are NaN (rv_13w needs 13 weeks of history).
    """
    rv = weekly_rv.ffill()

    rv_1w  = rv
    rv_4w  = rv.rolling(4).mean()
    rv_13w = rv.rolling(13).mean()

    assert rv_1w.shape  == weekly_rv.shape, f"rv_1w shape mismatch: {rv_1w.shape}"
    assert rv_4w.shape  == weekly_rv.shape, f"rv_4w shape mismatch: {rv_4w.shape}"
    assert rv_13w.shape == weekly_rv.shape, f"rv_13w shape mismatch: {rv_13w.shape}"

    return rv_1w, rv_4w, rv_13w


def prepare_har_arrays(
    rv_1w: pd.DataFrame,
    rv_4w: pd.DataFrame,
    rv_13w: pd.DataFrame,
    target: pd.DataFrame,
    splits: pd.DataFrame,
    split_name: str,
) -> tuple[dict, dict, np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Align HAR features to target's index, drop NaN weeks, and filter to one split.

    The first 13 rows of rv_13w are NaN (insufficient rolling history) and are dropped.
    Pooled arrays stack over (week, stock) in row-major order: week 0 all stocks,
    week 1 all stocks, etc. — matching the reshape used to recover predictions.

    rv_1w, rv_4w, rv_13w: DataFrames of shape (num_weeks, num_stocks).
    target: DataFrame of shape (num_target_weeks, num_stocks), Monday week-start index.
    splits: DataFrame with columns ['week', 'split'].
    split_name: one of 'train', 'val', 'test'.

    Returns:
        X_dict:      {ticker: ndarray(num_valid_weeks, 3)} for HARModel.
        y_dict:      {ticker: ndarray(num_valid_weeks,)} for HARModel.
        X_pooled:    ndarray(num_valid_weeks * num_stocks, 3) for HARPooled.
        y_pooled:    ndarray(num_valid_weeks * num_stocks,) for HARPooled.
        valid_index: DatetimeIndex of weeks included (use to reconstruct DataFrames).
    """
    # Align feature rows to target index (weekly_rv has one extra row at the end)
    rv_1w  = rv_1w.loc[target.index]
    rv_4w  = rv_4w.loc[target.index]
    rv_13w = rv_13w.loc[target.index]

    # Weeks belonging to the requested split
    split_weeks = set(splits.loc[splits["split"] == split_name, "week"])
    split_index = target.index[target.index.isin(split_weeks)]

    # Drop weeks where any stock has NaN rv_13w (the first ~13 weeks of the dataset)
    valid_mask  = rv_13w.loc[split_index].notna().all(axis=1)
    valid_index = split_index[valid_mask]

    X_1w  = rv_1w.loc[valid_index].values    # (num_valid_weeks, num_stocks)
    X_4w  = rv_4w.loc[valid_index].values
    X_13w = rv_13w.loc[valid_index].values
    Y     = target.loc[valid_index].values    # (num_valid_weeks, num_stocks)

    tickers         = list(target.columns)
    num_valid_weeks = len(valid_index)
    num_stocks      = len(tickers)

    # Per-ticker dicts for HARModel
    X_dict: dict[str, np.ndarray] = {
        ticker: np.column_stack([X_1w[:, i], X_4w[:, i], X_13w[:, i]])
        for i, ticker in enumerate(tickers)
    }
    y_dict: dict[str, np.ndarray] = {
        ticker: Y[:, i] for i, ticker in enumerate(tickers)
    }

    # Pooled arrays (week-major flatten: all stocks for week 0, then week 1, ...)
    X_pooled = np.column_stack([X_1w.reshape(-1), X_4w.reshape(-1), X_13w.reshape(-1)])
    y_pooled = Y.reshape(-1)

    assert X_pooled.shape == (num_valid_weeks * num_stocks, 3), (
        f"X_pooled shape mismatch: {X_pooled.shape}"
    )
    assert y_pooled.shape == (num_valid_weeks * num_stocks,), (
        f"y_pooled shape mismatch: {y_pooled.shape}"
    )

    return X_dict, y_dict, X_pooled, y_pooled, valid_index


# ---------------------------------------------------------------------------
# HAR model classes
# ---------------------------------------------------------------------------

class HARModel:
    """
    Per-stock HAR-RV model using sklearn LinearRegression.

    Fits one model per ticker. Input features: RV_1w, RV_4w, RV_13w.
    HAR uses only RV features computed directly from weekly_rv — NOT from features.parquet.
    """

    def __init__(self) -> None:
        self._models: dict[str, LinearRegression] = {}

    def fit(self, X: dict[str, np.ndarray], y: dict[str, np.ndarray]) -> None:
        """
        Fit one LinearRegression per ticker.

        X: {ticker: ndarray(num_train_weeks, 3)}
        y: {ticker: ndarray(num_train_weeks,)}
        """
        for ticker, X_tick in X.items():
            model = LinearRegression()
            model.fit(X_tick, y[ticker])
            self._models[ticker] = model

    def predict(self, X: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Predict for each ticker.

        X: {ticker: ndarray(num_weeks, 3)}
        Returns: {ticker: ndarray(num_weeks,)}
        """
        preds: dict[str, np.ndarray] = {}
        for ticker, X_tick in X.items():
            assert ticker in self._models, f"No model fitted for ticker: {ticker}"
            pred = self._models[ticker].predict(X_tick)
            assert pred.shape == (X_tick.shape[0],), (
                f"{ticker}: prediction shape {pred.shape} != ({X_tick.shape[0]},)"
            )
            preds[ticker] = pred
        return preds


class HARPooled:
    """
    Pooled HAR-RV model: single LinearRegression fit across all stocks simultaneously.

    Input features: RV_1w, RV_4w, RV_13w (same as HARModel but pooled).
    """

    def __init__(self) -> None:
        self._model: LinearRegression = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: ndarray(num_train_weeks * num_stocks, 3)
        y: ndarray(num_train_weeks * num_stocks,)
        """
        assert X.ndim == 2 and X.shape[1] == 3, (
            f"Expected X shape (N, 3), got {X.shape}"
        )
        assert y.ndim == 1 and y.shape[0] == X.shape[0], (
            f"y shape {y.shape} does not match X rows {X.shape[0]}"
        )
        self._model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: ndarray(num_weeks * num_stocks, 3)
        Returns: ndarray(num_weeks * num_stocks,)
        """
        assert X.ndim == 2 and X.shape[1] == 3, (
            f"Expected X shape (N, 3), got {X.shape}"
        )
        preds = self._model.predict(X)
        assert preds.shape == (X.shape[0],), (
            f"Prediction shape {preds.shape} != ({X.shape[0]},)"
        )
        return preds


# ---------------------------------------------------------------------------
# LSTM and GNN (stubs — implemented in Tasks 4.2 and 4.3)
# ---------------------------------------------------------------------------

class LSTMModel(nn.Module):
    """
    2-layer LSTM for per-stock volatility forecasting.

    Each stock's feature sequence is processed independently — batch dimension is stocks.
    Architecture: LSTM(input_size, hidden=64, num_layers=2, dropout=0.3) → Linear(64, 1).

    input_size: number of features per stock per week (e.g. 10)
    hidden_dim: LSTM hidden size
    dropout: applied between the two LSTM layers
    """

    def __init__(self, input_size: int,
                 hidden_dim: int = config.HIDDEN_DIM,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (num_stocks, seq_len, input_size).
        Returns: Tensor of shape (num_stocks,).
        """
        out, _ = self.lstm(x)             # (num_stocks, seq_len, hidden_dim)
        last   = out[:, -1, :]            # (num_stocks, hidden_dim)
        pred   = self.fc(last).squeeze(-1)  # (num_stocks,)
        assert pred.shape == (x.shape[0],), f"Expected ({x.shape[0]},), got {pred.shape}"
        return pred


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
        self.conv1   = SAGEConv(in_channels, hidden_dim, flow=config.SAGE_FLOW)
        self.conv2   = SAGEConv(hidden_dim,  hidden_dim, flow=config.SAGE_FLOW)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: Node features, shape (num_stocks, in_channels).
        edge_index: Graph connectivity, shape (2, num_edges).
        Returns: Tensor of shape (num_stocks,).
        """
        num_nodes = x.shape[0]
        if torch.isnan(x).any():
            warnings.warn(
                "GNNModel.forward: input features contain NaN values. "
                "Imputing with 0 (z-score mean). Expected during warm-up weeks."
            )
            x = torch.nan_to_num(x, nan=0.0)
        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = self.dropout(torch.relu(self.conv2(x, edge_index)))
        pred = self.fc(x).squeeze(-1)   # (num_nodes,)
        assert pred.shape == (num_nodes,), f"Expected ({num_nodes},), got {pred.shape}"
        assert not torch.isnan(pred).any(), "NaN in GNN output"
        return pred


class GNNModelV2(nn.Module):
    """
    Flexible GraphSAGE model for hyperparameter search.

    Extends GNNModel with configurable depth and optional GraphNorm.
    Forward interface is identical to GNNModel: forward(x, edge_index) → (num_stocks,).

    in_channels: number of input features per node
    hidden_dim:  hidden layer width for all SAGEConv layers
    dropout:     dropout probability applied after each conv layer
    num_layers:  number of SAGEConv layers (2 or 3)
    batch_norm:  if True, GraphNorm is applied after each conv, before activation.
                 GraphNorm preserves cross-sectional variance between node embeddings
                 (critical for ranking tasks) via a learnable mean_scale (alpha) per layer.
                 BatchNorm1d was used previously but normalizes away cross-sectional
                 variance, harming rank IC. GraphNorm replaces it with batch=None,
                 treating all 465 stocks as one graph per time step.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = config.HIDDEN_DIM,
        dropout: float = config.DROPOUT,
        num_layers: int = 2,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.convs: nn.ModuleList = nn.ModuleList()
        self.bns: nn.ModuleList | None = nn.ModuleList() if batch_norm else None

        for i in range(num_layers):
            in_ch = in_channels if i == 0 else hidden_dim
            self.convs.append(SAGEConv(in_ch, hidden_dim, flow=config.SAGE_FLOW))
            if batch_norm:
                self.bns.append(GraphNorm(hidden_dim))  # type: ignore[union-attr]

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor) -> torch.Tensor:
        """
        x: shape (num_stocks, in_channels)
        edge_index: shape (2, num_edges)
        Returns: shape (num_stocks,)
        """
        num_nodes = x.shape[0]
        if torch.isnan(x).any():
            warnings.warn(
                "GNNModelV2.forward: input features contain NaN values. "
                "Imputing with 0 (z-score mean). Expected during warm-up weeks."
            )
            x = torch.nan_to_num(x, nan=0.0)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.bns is not None:
                # batch=None: all nodes belong to one graph snapshot (465 stocks at week T)
                x = self.bns[i](x, batch=None)
            x = self.dropout(torch.relu(x))

        pred = self.fc(x).squeeze(-1)
        assert pred.shape == (num_nodes,), f"Expected ({num_nodes},), got {pred.shape}"
        assert not torch.isnan(pred).any(), "NaN in GNNModelV2 output"
        return pred
