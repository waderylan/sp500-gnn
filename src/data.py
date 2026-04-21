"""
Price download, log-return computation, weekly realized-volatility calculation,
and train/val/test split generation.

All functions are lookahead-safe: no window extends past the prediction week start.
Outputs are saved to data/raw/ and data/features/ as parquet files.
"""

from __future__ import annotations

import json
import time
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# GICS reclassification tables (point-in-time sector correction)
# ---------------------------------------------------------------------------
# Communication Services sector was created September 2018. These sets identify
# which current CommSvcs tickers migrated from each prior sector, enabling us
# to assign the correct point-in-time sector for years 2015–2018.

_COMM_FROM_IT: frozenset[str] = frozenset({
    "GOOGL", "GOOG", "META", "ATVI", "EA", "TTWO",
})
_COMM_FROM_CONSUMER_DISC: frozenset[str] = frozenset({
    "DIS", "NFLX", "CHTR", "CMCSA", "FOXA", "FOX", "PARA", "PARAA", "WBD",
})
_COMM_FROM_TELECOM: frozenset[str] = frozenset({
    "T", "VZ", "TMUS", "LUMN",
})
# Real Estate was separated from Financials in August 2016.
# Any ticker currently in "Real Estate" belonged to "Financials" at start of
# 2015 and 2016.  No separate set needed — handled by sector string check.


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_sp500_universe() -> list[str]:
    """
    Scrape Wikipedia for a point-in-time S&P 500 universe.

    Inclusion rules (per project spec):
      - Added to the index before 2016-01-01 (or original member with no date).
      - Not removed before 2024-12-01.

    Returns:
        Sorted list of ticker symbols in yfinance format (dots replaced with hyphens).
    """
    try:
        import requests
        resp = requests.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30,
        )
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch S&P 500 list from Wikipedia. Check connectivity."
        ) from exc

    current: pd.DataFrame = tables[0].copy()
    changes: pd.DataFrame = tables[1].copy()

    # Wikipedia uses dots (BRK.B); yfinance uses hyphens (BRK-B)
    current["Symbol"] = current["Symbol"].str.replace(".", "-", regex=False)

    # --- Addition-date filter ---
    current["Date added"] = pd.to_datetime(current["Date added"], errors="coerce")
    cutoff_added = pd.Timestamp("2016-01-01")
    mask_early = current["Date added"].isna() | (current["Date added"] < cutoff_added)
    candidates: set[str] = set(current.loc[mask_early, "Symbol"].tolist())

    # --- Removal-date filter (best-effort; falls back gracefully) ---
    try:
        # Flatten MultiIndex columns produced by pd.read_html on the changes table
        if isinstance(changes.columns, pd.MultiIndex):
            changes.columns = [
                "_".join(str(c) for c in col if str(c) not in ("", "nan")).strip("_")
                for col in changes.columns
            ]

        date_col = changes.columns[0]
        removed_col = next(
            (c for c in changes.columns if "Removed" in c and "Symbol" in c),
            None,
        )

        if removed_col is not None:
            changes[date_col] = pd.to_datetime(changes[date_col], errors="coerce")
            changes[removed_col] = (
                changes[removed_col]
                .astype(str)
                .str.replace(".", "-", regex=False)
                .replace("nan", pd.NA)
            )
            cutoff_removed = pd.Timestamp("2024-12-01")
            early_removals = set(
                changes.loc[changes[date_col] < cutoff_removed, removed_col]
                .dropna()
                .tolist()
            )
            candidates -= early_removals
    except Exception:
        # If the changes table can't be parsed, rely on coverage filter downstream
        pass

    return sorted(candidates)


def _build_sector_history(tickers: list[str]) -> dict[str, dict[str, str]]:
    """
    Build a point-in-time GICS sector mapping for each ticker across 2015–2025.

    Fetches the current sector from yfinance, then applies two historical corrections:
      1. Real Estate tickers → "Financials" for years 2015–2016 (split Aug 2016).
      2. Communication Services tickers → their pre-2018 sector for years 2015–2018
         (CommSvcs created Sep 2018 from Telecom, Consumer Disc, and IT).

    Args:
        tickers: List of ticker symbols.

    Returns:
        {ticker: {"2015": sector, "2016": sector, ..., "2025": sector}}
    """
    import yfinance as yf  # imported here to keep module-level imports minimal

    years = list(range(2015, 2026))
    sector_history: dict[str, dict[str, str]] = {}

    for ticker in tickers:
        try:
            current_sector: str = yf.Ticker(ticker).info.get("sector") or "Unknown"
        except Exception:
            current_sector = "Unknown"

        year_map: dict[str, str] = {}
        for year in years:
            sector = current_sector

            # Real Estate correction
            if current_sector == "Real Estate" and year <= 2016:
                sector = "Financials"

            # Communication Services correction
            if current_sector == "Communication Services" and year <= 2018:
                if ticker in _COMM_FROM_IT:
                    sector = "Information Technology"
                elif ticker in _COMM_FROM_CONSUMER_DISC:
                    sector = "Consumer Discretionary"
                elif ticker in _COMM_FROM_TELECOM:
                    sector = "Telecommunication Services"
                else:
                    # Default: assume legacy telecom (catches any unlisted CommSvcs name)
                    sector = "Telecommunication Services"

            year_map[str(year)] = sector

        sector_history[ticker] = year_map
        time.sleep(0.05)  # light rate limiting to avoid yfinance 429s

    return sector_history


# ---------------------------------------------------------------------------
# Public API — Task 1.2
# ---------------------------------------------------------------------------

def download_prices() -> pd.DataFrame:
    """
    Download daily adjusted close prices for the filtered S&P 500 universe.

    Pipeline:
      1. Fetch point-in-time constituent list from Wikipedia.
      2. Batch-download adjusted close prices via yfinance (2015-01-01 → 2025-12-31).
      3. Drop tickers with < config.MIN_COVERAGE non-NaN days.
      4. Build point-in-time sector history and apply GICS reclassification corrections.
      5. Save prices.parquet, tickers.json, sector_history.json to DATA_RAW_DIR.

    Lookahead safety: no rolling windows; filters use only static constituent data.

    Returns:
        DataFrame of shape (num_trading_days, num_stocks) — adjusted close prices.
        Index: DatetimeIndex of trading days (business days only).
        Columns: ticker symbols in sorted alphabetical order.

    Shape assertions:
        prices.shape[0] > 0
        prices.shape[1] > 0
        list(prices.columns) == valid_tickers
        prices.shape[1] >= 300  (only when DEV_UNIVERSE_SIZE is None)
    """
    import yfinance as yf

    raw_dir = Path(config.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Universe ---
    print("Fetching S&P 500 constituent list from Wikipedia...")
    all_tickers = _get_sp500_universe()

    if config.DEV_UNIVERSE_SIZE is not None:
        all_tickers = all_tickers[: config.DEV_UNIVERSE_SIZE]

    print(f"Candidate universe: {len(all_tickers)} tickers")

    # --- 2. Price download ---
    print("Downloading prices (2015-01-01 → 2025-12-31)...")
    raw = yf.download(
        all_tickers,
        start="2015-01-01",
        end="2025-12-31",
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    # yfinance returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices: pd.DataFrame = raw["Close"].copy()
    else:
        # Single-ticker edge case
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    # Restrict to tickers that were actually returned
    available = [t for t in all_tickers if t in prices.columns]
    prices = prices[available]

    # --- 3. Coverage filter ---
    n_days = len(prices)
    coverage = prices.notna().sum() / n_days
    valid_tickers = coverage[coverage >= config.MIN_COVERAGE].index.tolist()
    prices = prices[valid_tickers]

    print(
        f"After coverage filter (≥ {config.MIN_COVERAGE:.0%}): "
        f"{len(valid_tickers)} tickers"
    )

    # --- 4. Sector history ---
    print(f"Fetching sector assignments for {len(valid_tickers)} tickers...")
    sector_history = _build_sector_history(valid_tickers)

    # --- 5. Persist ---
    prices.to_parquet(raw_dir / "prices.parquet")

    with open(raw_dir / "tickers.json", "w") as fh:
        json.dump(valid_tickers, fh, indent=2)

    with open(raw_dir / "sector_history.json", "w") as fh:
        json.dump(sector_history, fh, indent=2)

    # --- 6. Shape assertions ---
    assert prices.shape[0] > 0, "prices has zero rows — no trading days downloaded"
    assert prices.shape[1] > 0, "prices has zero columns — no tickers survived filters"
    assert list(prices.columns) == valid_tickers, "Column/ticker order mismatch"
    if config.DEV_UNIVERSE_SIZE is None:
        assert prices.shape[1] >= 300, (
            f"Universe unexpectedly small: {prices.shape[1]} tickers. "
            "Check MIN_COVERAGE threshold and Wikipedia scrape output."
        )

    print(
        f"Saved: prices.parquet ({prices.shape[1]} stocks × {prices.shape[0]} days), "
        f"tickers.json, sector_history.json → {raw_dir}/"
    )
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from adjusted close prices.

    prices: shape (num_trading_days, num_stocks) — adjusted close prices.
    Returns: shape (num_trading_days - 1, num_stocks) — first row dropped (no prior day).

    Lookahead safety: return[t] = log(close[t] / close[t-1]). Uses only current and
    prior day prices. No rolling window, no forward look.
    """
    raw_dir = Path(config.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    log_returns = np.log(prices / prices.shift(1)).iloc[1:]

    assert log_returns.shape[0] == prices.shape[0] - 1, (
        f"Expected {prices.shape[0] - 1} rows, got {log_returns.shape[0]}"
    )
    assert log_returns.shape[1] == prices.shape[1], (
        f"Column count changed: {log_returns.shape[1]} != {prices.shape[1]}"
    )
    assert not log_returns.isna().all(axis=0).any(), (
        "At least one ticker is entirely NaN in log_returns"
    )

    log_returns.to_parquet(raw_dir / "log_returns.parquet")
    print(
        f"Saved: log_returns.parquet "
        f"({log_returns.shape[1]} stocks × {log_returns.shape[0]} days) → {raw_dir}/"
    )
    return log_returns


def compute_weekly_rv(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute annualized weekly realized volatility (RV) for each stock.

    RV for ISO week W = std(daily log returns in W) * sqrt(252).
    Weeks with fewer than 5 trading days are dropped.

    Args:
        log_returns: DataFrame of shape (num_trading_days, num_stocks).

    Returns:
        DataFrame of shape (num_weeks, num_stocks). Index is ISO week start date (Monday).

    Lookahead safety: RV for week W uses only data from within week W — no forward look.
    Shape assertion: result.shape == (num_weeks, num_stocks).
    """
    raise NotImplementedError


def make_target(weekly_rv: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the prediction target by shifting weekly_rv forward by one week.

    Target at row T = RV at week T+1. The last row will have NaN target and is dropped.

    Args:
        weekly_rv: DataFrame of shape (num_weeks, num_stocks).

    Returns:
        DataFrame of shape (num_weeks - 1, num_stocks).

    Lookahead safety: target[T] = rv[T+1] — this is intentional. Features at T must
    use only data strictly before T, so the shift direction is correct.
    Shape assertion: result.shape[0] == weekly_rv.shape[0] - 1.
    """
    raise NotImplementedError


def make_splits(index: pd.Index) -> pd.DataFrame:
    """
    Assign each week in `index` to train, val, or test split based on config dates.

    Args:
        index: DatetimeIndex of weekly RV rows.

    Returns:
        DataFrame with columns ['week', 'split'] where split ∈ {'train', 'val', 'test'}.

    Shape assertion: result.shape[0] == len(index).
    """
    raise NotImplementedError


def download_tbill_rates() -> pd.Series:
    """
    Download FRED DTB3 (3-month T-bill daily rates) via pandas-datareader.
    Saves to data/raw/tbill_rates.parquet.

    Returns:
        Series indexed by date, values are annualized T-bill rates (as decimals, not %).
    """
    raise NotImplementedError
