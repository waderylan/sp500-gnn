"""
Graph constructors for three graph types: correlation (dynamic), sector (annual),
and Granger causality (static, train-period only).

Also provides make_pyg_data() to assemble PyTorch Geometric Data objects.

Granger computation has two backends:
  run_granger_tests_gpu  — batched OLS on CUDA, ~5-15 min on A100 for full universe
  run_granger_tests_cpu  — statsmodels via multiprocessing, ~6-12h on Colab CPU
  run_granger_tests      — dispatcher: picks GPU if CUDA is available, else CPU

Always call run_granger_tests() first to produce granger_pvalues.parquet, then
call build_granger_graph() to apply the multiple-comparison correction and return
the directed edge index.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

import config


# ── Correlation graph ─────────────────────────────────────────────────────────

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
    date: Last trading day of the current week T (Friday of week T). The prediction
          target is week T+1's RV, so Friday_T is strictly before the target period.
    lookback: Number of trading days in the rolling window.
    threshold: Absolute Pearson correlation threshold for edge inclusion.

    Returns edge_index of shape (2, num_edges) as LongTensor.
    Returns (2, 0) tensor if no pairs exceed the threshold.

    Lookahead safety: 1-step-ahead design. Window ends AT `date` (Friday of week T).
    Target is week T+1's RV (starts Monday_{T+1}). Friday_T < Monday_{T+1}.
    Shape assertions: edge_index.shape[0] == 2, edge_index.max() < num_stocks when edges > 0.
    """
    num_stocks = log_returns.shape[1]

    # Last `lookback` rows up to and including `date`
    window = log_returns.loc[:date].iloc[-lookback:]
    if len(window) < lookback:
        warnings.warn(
            f"build_correlation_graph: only {len(window)} rows available before "
            f"{date.date()} (requested {lookback}). "
            "Correlation estimated on a shorter window.",
            stacklevel=2,
        )

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


# ── Sector graph ──────────────────────────────────────────────────────────────

def build_sector_graph(tickers: list[str],
                        year: int,
                        sector_history: dict) -> torch.LongTensor:
    """
    Build an undirected sector graph for a given calendar year.

    Connects all pairs of stocks sharing the same GICS sector in `year`,
    using point-in-time sector assignments from sector_history.
    Both directions included. Self-loops excluded.

    tickers: Ordered list of ticker symbols (defines node indices).
    year: Calendar year for sector lookup.
    sector_history: Dict of {ticker: {str(year): sector_name}}.

    Returns edge_index of shape (2, num_edges) as LongTensor.

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


def build_all_sector_graphs(tickers: list[str],
                             sector_history: dict,
                             years: range) -> dict[int, torch.LongTensor]:
    """
    Pre-build sector graphs for all years and return as a dict {year: edge_index}.
    Also saves results to data/graphs/sector_edges_by_year.parquet.

    Columns saved: [year, src, dst] — all integer indices into `tickers`.

    tickers: Ordered list of ticker symbols.
    sector_history: Dict of {ticker: {str(year): sector_name}}.
    years: Range of years to build graphs for (e.g., range(2015, 2026)).

    Returns dict mapping year (int) to edge_index (LongTensor).
    """
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


# ── Granger causality graph ───────────────────────────────────────────────────
#
# Two-step workflow:
#   1. run_granger_tests()    — heavy computation, saves granger_pvalues.parquet
#   2. build_granger_graph()  — loads saved p-values, applies correction, saves edges
#
# Keeping these separate means you can re-run the correction step with different
# settings (e.g., switch Bonferroni -> BH) without re-running the 6-12h test.


# Module-level globals used by the CPU multiprocessing worker.
# Must be at module level so they survive pickling across processes.
_GRANGER_RETURNS: np.ndarray | None = None


def _init_granger_worker(returns_array: np.ndarray) -> None:
    """Copy training returns into each worker process once at Pool startup."""
    global _GRANGER_RETURNS
    _GRANGER_RETURNS = returns_array


def _granger_pair(args: tuple) -> tuple:
    """
    Test whether stock a_idx Granger-causes stock b_idx at the given lag.

    args: (b_idx, a_idx, lag)
    Returns: (b_idx, a_idx, pvalue)

    Uses statsmodels grangercausalitytests, which includes an intercept and runs
    the F-test comparing restricted (B lags only) vs unrestricted (B + A lags) OLS.
    Returns pvalue=1.0 for degenerate pairs (constant series, NaN, etc.).
    """
    from statsmodels.tsa.stattools import grangercausalitytests
    b_idx, a_idx, lag = args
    # grangercausalitytests expects column 0 = target (B), column 1 = cause (A)
    data = np.column_stack([_GRANGER_RETURNS[:, b_idx], _GRANGER_RETURNS[:, a_idx]])
    try:
        res = grangercausalitytests(data, maxlag=lag, verbose=False)
        pvalue = float(res[lag][0]["ssr_ftest"][1])
    except Exception:
        pvalue = 1.0
    return b_idx, a_idx, pvalue


def run_granger_tests_cpu(
    log_returns: pd.DataFrame,
    tickers: list[str],
    train_end: str = config.TRAIN_END,
    lag: int = config.GRANGER_LAG,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Compute Granger causality p-values for all N*(N-1) ordered pairs using CPU multiprocessing.

    Wraps statsmodels grangercausalitytests. Each worker process receives the full
    training returns array once via Pool initializer (avoids re-serialization per pair).

    log_returns: Daily log returns, shape (num_trading_days, num_stocks). Columns = tickers.
    tickers: Ordered ticker list, length N. Must match columns of log_returns.
    train_end: Last date of training data. Default = config.TRAIN_END = "2022-12-31".
    lag: Number of lags for the Granger F-test. Default = config.GRANGER_LAG = 5.
    n_workers: CPU cores. None = all available (os.cpu_count()).

    Returns DataFrame of shape (N, N) — p-values. Diagonal = NaN (no self-test).
    Saves result to data/graphs/granger_pvalues.parquet.

    Lookahead safety: slices log_returns to [:train_end]. No val/test data enters.
    Runtime: ~30-60s for N=50 (dev), ~6-12h for N=462 (full) on Colab CPU.
    Shape assertion: output DataFrame is (N, N).
    """
    from multiprocessing import Pool

    train_data = log_returns.loc[:train_end].dropna().values.astype(np.float64)
    T, N = train_data.shape
    n_workers = n_workers or os.cpu_count()
    pairs = [(b, a, lag) for b in range(N) for a in range(N) if a != b]

    print(f"CPU Granger tests: {N} stocks, {T} training days, lag={lag}")
    print(f"Total pairs: {len(pairs):,}  |  Workers: {n_workers}")

    pval_matrix = np.full((N, N), np.nan)
    print_every = max(1, len(pairs) // 20)

    with Pool(
        processes=n_workers,
        initializer=_init_granger_worker,
        initargs=(train_data,),
    ) as pool:
        for i, (b_idx, a_idx, pval) in enumerate(
            pool.imap_unordered(_granger_pair, pairs, chunksize=50)
        ):
            pval_matrix[b_idx, a_idx] = pval
            if (i + 1) % print_every == 0:
                print(f"  {i+1:>7,}/{len(pairs):,} pairs done", end="\r")
    print()

    pval_df = pd.DataFrame(pval_matrix, index=tickers, columns=tickers)
    assert pval_df.shape == (N, N), f"Expected ({N}, {N}), got {pval_df.shape}"

    os.makedirs(config.DATA_GRAPHS_DIR, exist_ok=True)
    out_path = os.path.join(config.DATA_GRAPHS_DIR, "granger_pvalues.parquet")
    pval_df.to_parquet(out_path)
    print(f"Saved {pval_df.shape} p-value matrix -> {out_path}")
    return pval_df


def run_granger_tests_gpu(
    log_returns: pd.DataFrame,
    tickers: list[str],
    train_end: str = config.TRAIN_END,
    lag: int = config.GRANGER_LAG,
) -> pd.DataFrame:
    """
    Compute Granger causality p-values using GPU-batched OLS in PyTorch (float64).

    Implements the same Granger F-test as statsmodels, but vectorized over all N-1
    causing stocks A for each target stock B in a single GPU pass.

    Math:
      For each target B, we fit two OLS models over T_eff = T - lag observations:
        Restricted:   B_t = c + sum_{l=1}^{lag} b_l * B_{t-l}   + eps
        Unrestricted: B_t = c + sum_{l=1}^{lag} b_l * B_{t-l}
                               + sum_{l=1}^{lag} g_l * A_{t-l}  + eps

      F = ((RSS_r - RSS_ur) / lag) / (RSS_ur / (T_eff - 2*lag - 1))
      p-value = P(F_{lag, T_eff-2*lag-1} > F_obs)   [scipy.stats.f.sf]

    The unrestricted fits for all N-1 causing stocks A are batched as:
      X_ur: (N-1, T_eff, 2*lag+1)  — [intercept | B_lags | A_lags]
      Normal equations: beta = solve(X^T X, X^T y) for each A simultaneously.

    This matches statsmodels p-values within float64 numerical precision.

    log_returns: Daily log returns, shape (num_trading_days, num_stocks).
    tickers: Ordered ticker list, length N.
    train_end: Last date of training data. Default = config.TRAIN_END = "2022-12-31".
    lag: Number of lags. Default = config.GRANGER_LAG = 5.

    Returns DataFrame of shape (N, N) — p-values. Diagonal = NaN.
    Saves result to data/graphs/granger_pvalues.parquet.

    Lookahead safety: slices log_returns to [:train_end].
    Runtime: ~5-15 min on A100 (N=462); seconds for N=50 (dev).
    Shape assertion: output DataFrame is (N, N).

    Raises RuntimeError if CUDA is not available.
    """
    from scipy.stats import f as f_dist

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA not available. Call run_granger_tests_cpu() or "
            "run_granger_tests(use_gpu=False) instead."
        )

    train_data = log_returns.loc[:train_end].dropna().values.astype(np.float64)
    T, N = train_data.shape
    T_eff = T - lag
    df1 = lag                       # restrictions = number of A-lag coefficients
    df2 = T_eff - 2 * lag - 1      # T_eff rows minus (2*lag + 1) unrestricted params

    print(f"GPU Granger tests: {N} stocks, {T} training days, lag={lag}")
    print(f"T_eff={T_eff}, df1={df1}, df2={df2}, device={torch.cuda.get_device_name(0)}")

    device = torch.device("cuda")
    # R[t, n] = daily log return of stock n on day t
    R = torch.tensor(train_data, dtype=torch.float64, device=device)  # (T, N)

    # Pre-build all lag slices (shared across the loop over target stocks B).
    # A_lags_all[n, t, l] = R[t + lag - 1 - l, n]  for t in [0, T_eff), l in [0, lag)
    # Equivalently: for lag offset l, the time slice of all stocks is R[lag-1-l : T-1-l, :]
    A_lags_all = torch.stack(
        [R[lag - 1 - l : T - 1 - l, :] for l in range(lag)], dim=2
    )  # (T_eff, N, lag)
    A_lags_all = A_lags_all.permute(1, 0, 2).contiguous()  # (N, T_eff, lag)

    ones = torch.ones(T_eff, 1, dtype=torch.float64, device=device)
    all_idx = torch.arange(N, device=device)
    pval_matrix = np.full((N, N), np.nan)
    print_every = max(1, N // 10)

    for b_idx in range(N):
        a_indices = np.where(np.arange(N) != b_idx)[0]
        try:
            y = R[lag:, b_idx]  # (T_eff,)

            # Restricted design matrix: [1, B_{t-1}, ..., B_{t-lag}]
            B_lags = torch.stack(
                [R[lag - 1 - l : T - 1 - l, b_idx] for l in range(lag)], dim=1
            )  # (T_eff, lag)
            X_r = torch.cat([ones, B_lags], dim=1)  # (T_eff, lag+1)

            # Restricted OLS via normal equations
            XtX_r = X_r.T @ X_r                                          # (lag+1, lag+1)
            Xty_r = X_r.T @ y                                            # (lag+1,)
            beta_r = torch.linalg.solve(XtX_r, Xty_r.unsqueeze(1)).squeeze(1)  # (lag+1,)
            RSS_r = ((y - X_r @ beta_r) ** 2).sum()                      # scalar

            # Unrestricted design matrices for all A != b_idx
            # Shape: (N-1, T_eff, 2*lag+1) = [intercept | B_lags | A_lags]
            mask = all_idx != b_idx
            A_lags = A_lags_all[mask]                                            # (N-1, T_eff, lag)
            X_r_exp = X_r.unsqueeze(0).expand(N - 1, -1, -1)                    # (N-1, T_eff, lag+1)
            X_ur = torch.cat([X_r_exp, A_lags], dim=2)                          # (N-1, T_eff, 2*lag+1)

            # Batched OLS via normal equations: solve (X^T X) beta = X^T y
            y_exp = y.view(1, T_eff, 1).expand(N - 1, -1, -1)                   # (N-1, T_eff, 1)
            XtX_ur = torch.bmm(X_ur.transpose(1, 2), X_ur)                      # (N-1, 2*lag+1, 2*lag+1)
            Xty_ur = torch.bmm(X_ur.transpose(1, 2), y_exp)                     # (N-1, 2*lag+1, 1)
            beta_ur = torch.linalg.solve(XtX_ur, Xty_ur)                        # (N-1, 2*lag+1, 1)
            RSS_ur = ((y_exp - torch.bmm(X_ur, beta_ur)) ** 2).sum(dim=(1, 2))  # (N-1,)

            # F-statistics. Clamp to 0 to guard against floating-point rounding where
            # RSS_ur > RSS_r by a tiny epsilon (theoretically impossible, but can occur).
            F = torch.clamp(
                ((RSS_r - RSS_ur) / df1) / (RSS_ur / df2), min=0.0
            ).cpu().numpy()  # (N-1,)
            pvals = f_dist.sf(F, df1, df2)  # survival function = 1 - CDF
        except RuntimeError:
            # Degenerate target stock (near-constant series or singular XtX).
            # Match CPU backend behavior: treat as no Granger causality.
            pvals = np.ones(len(a_indices))

        pval_matrix[b_idx, a_indices] = pvals

        if (b_idx + 1) % print_every == 0 or b_idx == N - 1:
            print(f"  {b_idx+1:>4}/{N} target stocks done", end="\r")

    print()

    pval_df = pd.DataFrame(pval_matrix, index=tickers, columns=tickers)
    assert pval_df.shape == (N, N), f"Expected ({N}, {N}), got {pval_df.shape}"

    os.makedirs(config.DATA_GRAPHS_DIR, exist_ok=True)
    out_path = os.path.join(config.DATA_GRAPHS_DIR, "granger_pvalues.parquet")
    pval_df.to_parquet(out_path)
    print(f"Saved {pval_df.shape} p-value matrix -> {out_path}")
    return pval_df


def run_granger_tests(
    log_returns: pd.DataFrame,
    tickers: list[str],
    train_end: str = config.TRAIN_END,
    lag: int = config.GRANGER_LAG,
    use_gpu: bool | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """
    Run Granger causality tests — dispatcher that picks GPU or CPU automatically.

    On Colab with A100, GPU is ~30-60x faster than CPU multiprocessing:
      GPU: ~5-15 min for full universe (N=462)
      CPU: ~6-12h for full universe (N=462)
    Both produce numerically identical p-values within float64 precision.

    Step 1 of 2: call this function once to produce granger_pvalues.parquet.
    Step 2 of 2: call build_granger_graph() to apply correction and get edge_index.

    log_returns: Daily log returns, shape (num_trading_days, num_stocks).
    tickers: Ordered ticker list, length N.
    train_end: Last date of training data. Default = config.TRAIN_END = "2022-12-31".
    lag: Lags for F-test. Default = config.GRANGER_LAG = 5.
    use_gpu: True = force GPU, False = force CPU, None = auto-detect via torch.cuda.is_available().
    n_workers: CPU cores when use_gpu=False. None = all available cores.

    Returns DataFrame (N, N) of p-values with diagonal = NaN.
    Saves to data/graphs/granger_pvalues.parquet.

    Lookahead safety: both backends slice to [:train_end]. No val/test data used.
    """
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    if use_gpu:
        print("CUDA detected — using GPU batched OLS (run_granger_tests_gpu).")
        return run_granger_tests_gpu(log_returns, tickers, train_end, lag)
    else:
        print("No CUDA — using CPU multiprocessing with statsmodels (run_granger_tests_cpu).")
        return run_granger_tests_cpu(log_returns, tickers, train_end, lag, n_workers)


def build_granger_graph(
    tickers: list[str],
    correction: str = config.GRANGER_CORRECTION,
) -> tuple[torch.LongTensor, str]:
    """
    Build a directed Granger causality edge index from the saved p-value matrix.

    Step 2 of 2: loads granger_pvalues.parquet (produced by run_granger_tests()),
    applies multiple-comparison correction, and saves the directed edge list.

    Correction logic:
      1. Try Bonferroni at alpha = 0.05 / N*(N-1).
      2. If fewer than 500 edges survive, fall back to Benjamini-Hochberg FDR
         (statsmodels multipletests, method='fdr_bh') and log the switch.
    The correction method actually used is returned so callers can document it.

    Edge semantics: edge A -> B means stock A's past returns Granger-cause stock B's.
    Row 0 of edge_index = source (cause), row 1 = destination (target).

    tickers: Ordered ticker list, length N. Must match the p-value matrix index.
    correction: Starting method — "bonferroni" (default) or "bh". Overridden to "bh"
                automatically if Bonferroni yields fewer than 500 edges.

    Returns (edge_index, correction_used):
        edge_index: LongTensor of shape (2, num_edges).
        correction_used: "bonferroni" or "bh".

    Saves directed edges to data/graphs/granger_edges.parquet (columns: src, dst).
    Shape assertions: edge_index.shape[0] == 2, edge_index.max() < N when edges > 0.

    Raises FileNotFoundError if granger_pvalues.parquet does not exist.
    """
    from statsmodels.stats.multitest import multipletests

    pval_path = os.path.join(config.DATA_GRAPHS_DIR, "granger_pvalues.parquet")
    if not os.path.exists(pval_path):
        raise FileNotFoundError(
            f"P-value file not found: {pval_path}\n"
            "Run run_granger_tests() first to compute Granger causality p-values."
        )

    pval_df = pd.read_parquet(pval_path)
    N = len(tickers)
    assert pval_df.shape == (N, N), (
        f"P-value matrix shape {pval_df.shape} does not match len(tickers)={N}"
    )
    pval_matrix = pval_df.values  # (N, N)

    # Extract all off-diagonal pairs
    off_diag = ~np.eye(N, dtype=bool)
    src_all, dst_all = np.where(off_diag)
    pvals_flat = pval_matrix[src_all, dst_all]  # (N*(N-1),)
    num_pairs = len(pvals_flat)

    # Bonferroni
    bonf_alpha = 0.05 / num_pairs
    bonf_reject = pvals_flat < bonf_alpha

    if bonf_reject.sum() >= config.GRANGER_MIN_EDGES:
        reject = bonf_reject
        correction_used = "bonferroni"
    else:
        print(
            f"Bonferroni yielded {int(bonf_reject.sum())} edges "
            f"(< GRANGER_MIN_EDGES={config.GRANGER_MIN_EDGES}). "
            "Falling back to Benjamini-Hochberg FDR."
        )
        reject, _, _, _ = multipletests(pvals_flat, alpha=0.05, method="fdr_bh")
        correction_used = "bh"

    n_edges = int(reject.sum())
    print(f"Correction: {correction_used}  |  Edges: {n_edges:,} / {num_pairs:,} pairs")

    meta_path = os.path.join(config.DATA_GRAPHS_DIR, "granger_meta.json")
    os.makedirs(config.DATA_GRAPHS_DIR, exist_ok=True)
    with open(meta_path, "w") as _fh:
        json.dump({"correction_used": correction_used, "n_edges": n_edges,
                   "n_pairs": num_pairs}, _fh, indent=2)

    # pval_matrix[b, a] = P(A does NOT Granger-cause B), so src_all=row=B=target,
    # dst_all=col=A=cause. Edge semantics: cause A -> target B, so cause goes in row 0.
    src_edges = dst_all[reject].astype(np.int64)   # cause A (col index)
    dst_edges = src_all[reject].astype(np.int64)   # target B (row index)
    edge_index = torch.tensor(np.stack([src_edges, dst_edges]), dtype=torch.long)

    assert edge_index.shape[0] == 2, \
        f"edge_index row dim should be 2, got {edge_index.shape[0]}"
    if edge_index.shape[1] > 0:
        assert int(edge_index.max()) < N, \
            f"edge_index max {int(edge_index.max())} >= N={N}"

    # Save edge list
    edges_df = pd.DataFrame({
        "src": src_edges.astype("int32"),
        "dst": dst_edges.astype("int32"),
    })
    out_path = os.path.join(config.DATA_GRAPHS_DIR, "granger_edges.parquet")
    os.makedirs(config.DATA_GRAPHS_DIR, exist_ok=True)
    edges_df.to_parquet(out_path, index=False)
    print(f"Saved {n_edges:,} directed edges -> {out_path}")

    return edge_index, correction_used


# ── PyG Data helper ───────────────────────────────────────────────────────────

def verify_sageconv_directionality(
    num_nodes: int = 10,
    in_channels: int = 8,
    out_channels: int = config.HIDDEN_DIM,
) -> dict:
    """
    Verify that SAGEConv with flow=config.SAGE_FLOW respects edge direction.

    Constructs a small asymmetric directed graph, runs a forward pass, then
    reverses all edges and runs again. Asserts the two outputs differ.

    If the assertion fails, SAGEConv is symmetrizing the adjacency and the
    Granger graph cannot be used as directed. Do not proceed with GNN-Granger
    training until the issue is resolved.

    num_nodes: Number of nodes in the test graph.
    in_channels: Feature dimension per node.
    out_channels: SAGEConv output dimension (default = config.HIDDEN_DIM).

    Returns dict with keys: flow, num_nodes, num_edges, max_abs_diff, passed.

    Shape assertions:
        out_forward.shape  == (num_nodes, out_channels)
        out_reversed.shape == (num_nodes, out_channels)
    """
    import random as _random
    import numpy as _np
    from torch_geometric.nn import SAGEConv

    _random.seed(config.RANDOM_SEED)
    _np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # Asymmetric directed edges: a chain 0->1->2->...->9 plus a few skip
    # connections, all one-directional. No edge goes both ways.
    src = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 2, 4], dtype=torch.long)
    dst = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 5, 7, 9], dtype=torch.long)
    edge_index          = torch.stack([src, dst])     # (2, 12)
    edge_index_reversed = edge_index[[1, 0]]          # flip src <-> dst

    x = torch.randn(num_nodes, in_channels)

    conv = SAGEConv(in_channels, out_channels, flow=config.SAGE_FLOW)
    conv.eval()

    with torch.no_grad():
        out_forward  = conv(x, edge_index)           # (num_nodes, out_channels)
        out_reversed = conv(x, edge_index_reversed)  # (num_nodes, out_channels)

    assert out_forward.shape == (num_nodes, out_channels), \
        f"Expected ({num_nodes}, {out_channels}), got {tuple(out_forward.shape)}"
    assert out_reversed.shape == (num_nodes, out_channels), \
        f"Expected ({num_nodes}, {out_channels}), got {tuple(out_reversed.shape)}"

    max_abs_diff = float((out_forward - out_reversed).abs().max())

    assert not torch.allclose(out_forward, out_reversed), (
        "SAGEConv is symmetrizing edges — directionality not working. "
        "Do not proceed with GNN-Granger training until this is resolved."
    )

    return {
        "flow":         config.SAGE_FLOW,
        "num_nodes":    num_nodes,
        "num_edges":    int(edge_index.shape[1]),
        "max_abs_diff": max_abs_diff,
        "passed":       True,
    }


# ── Graph statistics ──────────────────────────────────────────────────────────

def compute_graph_stats(
    edge_index: torch.LongTensor,
    num_nodes: int,
    directed: bool = False,
) -> dict:
    """
    Compute summary statistics for a graph given its edge index.

    edge_index: Shape (2, num_edges). For undirected graphs, both directions
                must be present (A->B and B->A), as is the PyG convention.
    num_nodes: Total node count, including nodes with zero degree.
    directed: If True, reports separate in/out-degree stats and counts every
              directed edge. If False, counts unique pairs (divides raw edge
              count by 2) and reports a single degree per node.

    Returns a dict with: num_nodes, num_edges, density, and degree statistics.
    Undirected keys: mean_degree, max_degree.
    Directed keys:   mean_out_degree, max_out_degree, mean_in_degree, max_in_degree.

    Shape assertion: edge_index.shape[0] == 2.
    """
    assert edge_index.shape[0] == 2, \
        f"edge_index row dim should be 2, got {edge_index.shape[0]}"

    num_raw = edge_index.shape[1]

    if not directed:
        num_edges = num_raw // 2
        possible  = num_nodes * (num_nodes - 1) // 2
    else:
        num_edges = num_raw
        possible  = num_nodes * (num_nodes - 1)

    density = num_edges / possible if possible > 0 else 0.0

    if num_raw > 0:
        src_np  = edge_index[0].numpy()
        dst_np  = edge_index[1].numpy()
        out_deg = np.bincount(src_np, minlength=num_nodes)
        in_deg  = np.bincount(dst_np, minlength=num_nodes)
    else:
        out_deg = np.zeros(num_nodes, dtype=np.int64)
        in_deg  = np.zeros(num_nodes, dtype=np.int64)

    if directed:
        return {
            "num_nodes":       num_nodes,
            "num_edges":       num_edges,
            "density":         density,
            "mean_out_degree": float(out_deg.mean()),
            "max_out_degree":  int(out_deg.max()),
            "mean_in_degree":  float(in_deg.mean()),
            "max_in_degree":   int(in_deg.max()),
        }
    return {
        "num_nodes":   num_nodes,
        "num_edges":   num_edges,
        "density":     density,
        "mean_degree": float(out_deg.mean()),
        "max_degree":  int(out_deg.max()),
    }


def make_pyg_data(features_t: torch.Tensor,
                   edge_index: torch.LongTensor,
                   target_t: torch.Tensor) -> Data:
    """
    Construct a PyTorch Geometric Data object for one time step.

    features_t: Node feature matrix, shape (num_stocks, num_features).
    edge_index: Graph connectivity, shape (2, num_edges) — LongTensor.
    target_t: Target values, shape (num_stocks,).

    Returns Data(x=features_t, edge_index=edge_index, y=target_t).

    Shape assertions:
        - features_t.shape[0] == target_t.shape[0]
        - edge_index.max() < features_t.shape[0] (when edges > 0)
    """
    num_nodes = features_t.shape[0]
    assert features_t.shape[0] == target_t.shape[0], (
        f"features_t has {features_t.shape[0]} nodes but target_t has {target_t.shape[0]}"
    )
    if edge_index.shape[1] > 0:
        assert int(edge_index.max()) < num_nodes, (
            f"edge_index max {int(edge_index.max())} >= num_nodes {num_nodes}"
        )
    return Data(x=features_t, edge_index=edge_index, y=target_t)
