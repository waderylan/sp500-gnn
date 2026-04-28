"""Refresh lightweight reproducibility artifacts without retraining models."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.data import canonicalize_sector_label
from src.graphs import build_all_sector_graphs


RAW_DIR = Path(config.DATA_RAW_DIR)
RESULTS_DIR = Path(config.DATA_RESULTS_DIR)


def _load_json(path: Path):
    with path.open() as fh:
        return json.load(fh)


def canonicalize_sector_history() -> dict[str, dict[str, str]]:
    """Rewrite saved sector history with canonical GICS-style labels."""
    path = RAW_DIR / "sector_history.json"
    sector_history = _load_json(path)
    canonical = {
        ticker: {
            year: canonicalize_sector_label(sector)
            for year, sector in year_map.items()
        }
        for ticker, year_map in sector_history.items()
    }
    with path.open("w") as fh:
        json.dump(canonical, fh, indent=2)
    return canonical


def write_universe_reproducibility_table(tickers: list[str]) -> Path:
    """Record the current-constituent universe definition in machine-readable form."""
    out_path = RESULTS_DIR / "universe_reproducibility.csv"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "universe_source",
                "candidate_count",
                "final_count",
                "min_coverage",
                "symbol_format",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "universe_source": "current Wikipedia S&P 500 constituents",
                "candidate_count": "",
                "final_count": len(tickers),
                "min_coverage": config.MIN_COVERAGE,
                "symbol_format": "yfinance-compatible symbols",
                "notes": (
                    "Candidate count was not persisted by the original download; "
                    "final_count is read from data/raw/tickers.json."
                ),
            }
        )
    return out_path


def main() -> None:
    tickers = _load_json(RAW_DIR / "tickers.json")
    sector_history = canonicalize_sector_history()
    build_all_sector_graphs(tickers, sector_history, range(2015, 2026))
    universe_path = write_universe_reproducibility_table(tickers)
    print(f"Canonicalized sector history: {RAW_DIR / 'sector_history.json'}")
    print(f"Rebuilt sector graphs: {Path(config.DATA_GRAPHS_DIR) / 'sector_edges_by_year.parquet'}")
    print(f"Wrote universe reproducibility table: {universe_path}")


if __name__ == "__main__":
    main()
