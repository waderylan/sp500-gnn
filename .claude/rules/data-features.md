# Rules: src/data.py and src/features.py

Loaded when working in the data pipeline or feature engineering modules.

---

## Lookahead Safety Checklist

Every function in these files must pass this checklist before being considered complete.
State the answers explicitly in a comment at the top of each function.

1. **What is the prediction week T?** (the row index; target is T+1's RV)
2. **What data does this function use?** (date range, columns)
3. **Does any window extend into week T+1 or later?** (must be NO)
4. **Where does the rolling window end relative to the target week?** (must end no later than Friday of week T)

**1-step-ahead design:** `target.parquet` stores RV for week T+1 at row index T.
Features at row T may include data through Friday of week T (the last trading day
of week T). Friday_T is strictly before Monday_{T+1} (the start of the predicted week).
Do NOT use any data from week T+1 or later.

---

## Weekly RV Formula

```python
# RV for a given week = std(daily_log_returns_in_that_week) * sqrt(252)
# Exactly 5 trading days per ISO week. Weeks with fewer than 5 days are dropped.
rv = daily_returns.groupby(iso_week).std() * np.sqrt(252)
```

Verify: mean annualized RV across all stocks should be 0.15–0.25. Week of 2020-03-16 should be 0.80–1.50+.

---

## Feature Engineering Rules

**Order of operations is mandatory:**
1. Compute raw feature values (rolling windows, ratios)
2. Winsorize cross-sectionally at config.WINSORIZE_CLIP = (0.01, 0.99)
3. Z-score cross-sectionally (subtract mean, divide by std across stocks at each time step)

Never z-score before winsorizing. Never skip winsorization.

**Post-normalization assertions (run programmatically):**
```python
# At 10 random time steps, for each feature:
assert abs(features[t, :, f].mean()) < 0.01, "Mean not near zero after normalization"
assert abs(features[t, :, f].std() - 1.0) < 0.05, "Std not near 1 after normalization"
assert features[t, :, f].abs().max() < 5.0, "Outlier survived winsorization"
```

---

## Universe Filtering Rules

Inclusion criteria (both must be satisfied):
- >= config.MIN_COVERAGE (95%) data coverage across 2015–2025
- Was an S&P 500 constituent for >= 80% of total sample weeks
- Entered index before January 2016 AND was not removed before December 2024

These filters are applied once in `download_prices()` and the final ticker list is saved to `data/raw/tickers.json`. Do not re-filter downstream.

---

## Sector History

Sector assignments are point-in-time, stored in `data/raw/sector_history.json`.
Format: `{ticker: {year: sector_name}}` for years 2015–2025.

Key reclassification events to handle:
- 2016: Real Estate separated from Financials (11 sectors from 2017 onward)
- 2018: Telecom → Communication Services (absorbed some Consumer Disc. and IT names)

When building the sector graph for a given week, look up `sector_history[ticker][str(week.year)]`.
Never use a single static sector mapping.

---

## Required Output Shapes

| Function | Output Shape |
|---|---|
| `compute_weekly_rv()` | `(num_weeks, num_stocks)` |
| `compute_log_returns()` | `(num_trading_days, num_stocks)` |
| `make_target()` | `(num_weeks, num_stocks)` — identical shape to weekly_rv, shifted by 1 |
| Feature tensor (stacked) | `(num_weeks, num_stocks, num_features)` |
| Parquet storage (2D reshape) | `(num_weeks * num_stocks, num_features + 2)` with week and stock index columns |

Assert these shapes before saving any parquet file.
