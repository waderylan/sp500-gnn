"""
Graph constructors for three graph types: correlation (dynamic), sector (annual),
and Granger causality (static, train-period only).

Also provides make_pyg_data() to assemble PyTorch Geometric Data objects.
"""

from __future__ import annotations

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data

import config


def build_correlation_graph(log_returns: pd.DataFrame,
                             date: pd.Timestamp,
                             lookback: int = config.CORR_LOOKBACK_DAYS,
                             threshold: float = config.CORR_THRESHOLD) -> torch.LongTensor:
    """
    Build an undirected correlation graph for a given date.

    Computes pairwise Pearson correlations over the `lookback` trading days ending
    at `date` (inclusive). Adds an edge between stock i and j if |corr(i,j)| >= threshold.
    Both directions included (A→B and B→A). Self-loops excluded.

    Args:
        log_returns: Daily log returns, shape (num_trading_days, num_stocks).
        date: Last trading day before the prediction week starts (inclusive end of window).
        lookback: Number of trading days to look back.
        threshold: Absolute correlation threshold for edge inclusion.

    Returns:
        edge_index: LongTensor of shape (2, num_edges).

    Lookahead safety: window ends AT `date`, which is before the prediction week starts.
    Shape assertion: edge_index.shape[0] == 2.
    """
    raise NotImplementedError


def build_sector_graph(tickers: list[str],
                        year: int,
                        sector_history: dict) -> torch.LongTensor:
    """
    Build an undirected sector graph for a given calendar year.

    Connects all pairs of stocks sharing the same GICS sector in `year`,
    using point-in-time sector assignments from sector_history.
    Both directions included. Self-loops excluded.

    Args:
        tickers: Ordered list of ticker symbols (defines node indices).
        year: Calendar year for sector lookup.
        sector_history: Dict of {ticker: {str(year): sector_name}}.

    Returns:
        edge_index: LongTensor of shape (2, num_edges).

    Shape assertion: edge_index.shape[0] == 2, edge_index.max() < len(tickers).
    """
    raise NotImplementedError


def build_granger_graph(log_returns: pd.DataFrame,
                         tickers: list[str],
                         train_end: str = config.TRAIN_END,
                         lag: int = config.GRANGER_LAG) -> torch.LongTensor:
    """
    Build a directed Granger causality graph using only the training period.

    For each ordered pair (A, B), run a Granger-causality F-test at `lag` lags.
    Apply Bonferroni correction (fallback to BH if <500 edges survive).
    Edge A→B means A's past returns Granger-cause B's returns.

    Args:
        log_returns: Daily log returns, shape (num_trading_days, num_stocks).
        tickers: Ordered list of ticker symbols.
        train_end: End date of training period (data after this date is not used).
        lag: Number of lags for the Granger F-test.

    Returns:
        edge_index: LongTensor of shape (2, num_edges), directed.

    Lookahead safety: uses only data up to and including train_end.
    Shape assertion: edge_index.shape[0] == 2.
    """
    raise NotImplementedError


def build_all_sector_graphs(tickers: list[str],
                             sector_history: dict,
                             years: range) -> dict[int, torch.LongTensor]:
    """
    Pre-build sector graphs for all years and return as a dict {year: edge_index}.
    Also saves results to data/graphs/sector_edges_by_year.parquet.

    Args:
        tickers: Ordered list of ticker symbols.
        sector_history: Dict of {ticker: {str(year): sector_name}}.
        years: Range of years to build graphs for (e.g., range(2015, 2026)).

    Returns:
        Dict mapping year (int) to edge_index (LongTensor).
    """
    raise NotImplementedError


def make_pyg_data(features_t: torch.Tensor,
                   edge_index: torch.LongTensor,
                   target_t: torch.Tensor) -> Data:
    """
    Construct a PyTorch Geometric Data object for one time step.

    Args:
        features_t: Node feature matrix, shape (num_stocks, num_features).
        edge_index: Graph connectivity, shape (2, num_edges) — LongTensor.
        target_t: Target values, shape (num_stocks,).

    Returns:
        Data(x=features_t, edge_index=edge_index, y=target_t).

    Shape assertions:
        - features_t.shape[0] == target_t.shape[0]
        - edge_index.max() < features_t.shape[0]
    """
    raise NotImplementedError
