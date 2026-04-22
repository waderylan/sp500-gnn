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
    Both directions included (A->B and B->A). Self-loops excluded.

    log_returns: Daily log returns, shape (num_trading_days, num_stocks).
    date: Last trading day before the prediction week starts (inclusive end of window).
    lookback: Number of trading days in the rolling window.
    threshold: Absolute Pearson correlation threshold for edge inclusion.

    Returns edge_index of shape (2, num_edges) as LongTensor.
    Returns (2, 0) tensor if no pairs exceed the threshold.

    Lookahead safety: window ends AT `date`, which must be the last trading day
    before the prediction week. No data from the prediction week is used.
    Shape assertions: edge_index.shape[0] == 2, edge_index.max() < num_stocks when edges > 0.
    """
    num_stocks = log_returns.shape[1]

    # Last `lookback` rows up to and including `date`
    window = log_returns.loc[:date].iloc[-lookback:]

    # pandas .corr() uses pairwise deletion -- NaN pairs produce NaN, which
    # fails the >= threshold check and are simply skipped (no edge added)
    corr = window.corr().values  # shape (num_stocks, num_stocks)

    # Upper triangle only (k=1 excludes diagonal); NaN comparisons return False
    mask = np.triu(np.abs(corr) >= threshold, k=1)
    rows, cols = np.where(mask)

    if len(rows) > 0:
        src = np.concatenate([rows, cols]).astype(np.int64)
        dst = np.concatenate([cols, rows]).astype(np.int64)
    else:
        src = np.empty(0, dtype=np.int64)
        dst = np.empty(0, dtype=np.int64)

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    assert edge_index.shape[0] == 2, \
        f"edge_index row dim should be 2, got {edge_index.shape[0]}"
    if edge_index.shape[1] > 0:
        assert int(edge_index.max()) < num_stocks, \
            f"edge_index max {int(edge_index.max())} >= num_stocks {num_stocks}"

    return edge_index


def save_correlation_samples(log_returns: pd.DataFrame,
                              tickers: list[str]) -> dict[str, torch.LongTensor]:
    """
    Build and save correlation graphs for the three sample dates in config.CORR_SAMPLE_DATES.

    Saves each edge list as a parquet to data/graphs/corr_sample/corr_{label}.parquet.
    Prints edge count for each sample date.

    log_returns: Daily log returns, shape (num_trading_days, num_stocks).
    tickers: Ordered ticker list (used for column count verification only).

    Returns dict mapping label -> edge_index LongTensor.
    """
    import os

    out_dir = os.path.join(config.DATA_GRAPHS_DIR, "corr_sample")
    os.makedirs(out_dir, exist_ok=True)

    graphs: dict[str, torch.LongTensor] = {}
    for label, date_str in config.CORR_SAMPLE_DATES.items():
        date = pd.Timestamp(date_str)
        edge_index = build_correlation_graph(log_returns, date)
        num_edges = edge_index.shape[1]
        unique_pairs = num_edges // 2

        df = pd.DataFrame({
            "src": edge_index[0].numpy().astype("int32"),
            "dst": edge_index[1].numpy().astype("int32"),
        })
        out_path = os.path.join(out_dir, f"corr_{label}.parquet")
        df.to_parquet(out_path, index=False)

        graphs[label] = edge_index
        print(f"  {label:8s} ({date_str}): {num_edges:6d} directed edges  "
              f"({unique_pairs:5d} unique pairs)")

    return graphs


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

    Lookahead safety: no price/return data used. Sector assignments are
    point-in-time for the start of `year`, known before any week in that year.
    Shape assertions: edge_index.shape[0] == 2, edge_index.max() < len(tickers).
    """
    year_key = str(year)

    # Map each ticker to its sector for this year; None if missing
    sectors = [sector_history.get(t, {}).get(year_key) for t in tickers]

    # Group ticker indices by sector
    sector_to_indices: dict[str, list[int]] = {}
    for idx, sector in enumerate(sectors):
        if sector is not None:
            sector_to_indices.setdefault(sector, []).append(idx)

    # Emit both directions for every within-sector pair
    src_list: list[int] = []
    dst_list: list[int] = []
    for indices in sector_to_indices.values():
        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    src_list.append(indices[i])
                    dst_list.append(indices[j])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)

    assert edge_index.shape[0] == 2, f"edge_index row dim should be 2, got {edge_index.shape[0]}"
    assert edge_index.shape[1] > 0, f"Sector graph for {year} has no edges"
    assert int(edge_index.max()) < len(tickers), "edge_index references out-of-bounds node index"

    return edge_index


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

    Columns saved: [year, src, dst] — all integer indices into `tickers`.

    Args:
        tickers: Ordered list of ticker symbols.
        sector_history: Dict of {ticker: {str(year): sector_name}}.
        years: Range of years to build graphs for (e.g., range(2015, 2026)).

    Returns:
        Dict mapping year (int) to edge_index (LongTensor).
    """
    import os

    graphs: dict[int, torch.LongTensor] = {}
    rows: list[dict] = []

    for year in years:
        edge_index = build_sector_graph(tickers, year, sector_history)
        graphs[year] = edge_index
        src = edge_index[0].tolist()
        dst = edge_index[1].tolist()
        for s, d in zip(src, dst):
            rows.append({"year": year, "src": s, "dst": d})

    df = pd.DataFrame(rows, columns=["year", "src", "dst"])
    df = df.astype({"year": "int32", "src": "int32", "dst": "int32"})

    out_path = os.path.join(config.DATA_GRAPHS_DIR, "sector_edges_by_year.parquet")
    os.makedirs(config.DATA_GRAPHS_DIR, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} total edges across {len(graphs)} years to {out_path}")

    assert df.shape[1] == 3, f"Expected 3 columns, got {df.shape[1]}"

    return graphs


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
