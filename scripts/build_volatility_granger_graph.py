"""
Build the experimental volatility-Granger graph.

This script creates separate artifacts from the frozen return-Granger graph:

  data/graphs/granger_vol_pvalues.parquet
  data/graphs/granger_vol_edges.parquet
  data/graphs/granger_vol_meta.json

It expects the usual project data artifacts to exist in data/raw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.graphs import build_volatility_granger_graph, run_volatility_granger_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Only rebuild edges from existing granger_vol_pvalues.parquet.",
    )
    parser.add_argument(
        "--use-gpu",
        choices=["auto", "true", "false"],
        default="auto",
        help="Granger backend selection. Default auto uses CUDA when available.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="CPU worker count when --use-gpu=false or CUDA is unavailable.",
    )
    parser.add_argument(
        "--lag",
        type=int,
        default=config.GRANGER_VOL_LAG,
        help="Number of weekly realized-volatility lags for the F-test.",
    )
    return parser.parse_args()


def _use_gpu_arg(value: str) -> bool | None:
    if value == "auto":
        return None
    return value == "true"


def main() -> None:
    args = parse_args()
    raw_dir = Path(config.DATA_RAW_DIR)
    graph_dir = Path(config.DATA_GRAPHS_DIR)
    log_returns_path = raw_dir / "log_returns.parquet"
    tickers_path = raw_dir / "tickers.json"

    if not log_returns_path.exists():
        raise FileNotFoundError(f"Missing {log_returns_path}")
    if not tickers_path.exists():
        raise FileNotFoundError(f"Missing {tickers_path}")

    with open(tickers_path) as fh:
        tickers: list[str] = json.load(fh)
    if config.DEV_UNIVERSE_SIZE is not None:
        tickers = tickers[: config.DEV_UNIVERSE_SIZE]

    log_returns = pd.read_parquet(log_returns_path).reindex(columns=tickers)
    print(f"Loaded log returns: {log_returns.shape[0]} days x {log_returns.shape[1]} stocks")
    print(f"Volatility-Granger lag: {args.lag} weekly RV lags")

    if not args.skip_tests:
        run_volatility_granger_tests(
            log_returns=log_returns,
            tickers=tickers,
            lag=args.lag,
            use_gpu=_use_gpu_arg(args.use_gpu),
            n_workers=args.n_workers,
        )
    else:
        pvalue_path = graph_dir / config.GRANGER_VOL_PVALUES_FILE
        if not pvalue_path.exists():
            raise FileNotFoundError(f"Missing {pvalue_path}; rerun without --skip-tests")
        print(f"Using existing p-values: {pvalue_path}")

    edge_index, correction_used = build_volatility_granger_graph(tickers)
    n_possible = len(tickers) * (len(tickers) - 1)
    print(
        f"Volatility-Granger graph: {edge_index.shape[1]:,} directed edges "
        f"({edge_index.shape[1] / n_possible:.4f} density), correction={correction_used}"
    )


if __name__ == "__main__":
    main()
