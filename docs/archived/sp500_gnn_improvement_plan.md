# Implementation Plan — Improving Portfolio Results

Version 1.3 · Focus: Extracting Existing Signal Before Model Changes

---

# Objective

The current results indicate that all models produce highly similar portfolios due to the combination of:

- Large universe size (~465 stocks)  
- Inverse-volatility weighting  
- Weight caps (MAX_WEIGHT = 0.05)  

This construction collapses model differences into near-equal-weight behavior, masking any cross-sectional predictive signal.

The goal of this phase is to **extract and measure that signal directly**, without retraining models.

---

# Execution Order

The following tasks must be completed in sequence:

- [x] 1. Long-short portfolio construction  
- [x] 2. Volatility-targeted portfolio  
- [x] 3. Minimum variance portfolio  
- [x] 4. Rank IC evaluation metric  
- [x] 5. GNN ensemble (prediction averaging)  
- [x] 6. Rank-based training loss  
- [ ] 7. Mixed rank-MSE loss  
- [ ] 8. Ex-sector robustness check (IT + Communication Services exclusion)  
- [ ] 9. Sector-neutral portfolio construction  

No model retraining should occur before Tasks 1–5 are completed and analyzed.

---

# Phase 1 — Long-Short Portfolio

## Goal

Replace long-only inverse-volatility portfolio with a **cross-sectional long-short strategy** that isolates ranking signal.

---

## Construction

At each week t:

- Rank all stocks by predicted volatility  
- Define:  
  - Long portfolio = bottom 25% (lowest predicted vol)  
  - Short portfolio = top 25% (highest predicted vol)  
- Assign equal weights within each side:  
  - Long weights sum to +1  
  - Short weights sum to -1  
- Net exposure = 0 (dollar neutral)  

---

## Implementation

### File: `src/portfolio.py`

Add:

```python
def build_long_short_portfolio(preds, q=0.25):
    weights = pd.DataFrame(0.0, index=preds.index, columns=preds.columns)

    for t in preds.index:
        row = preds.loc[t]
        n = len(row)
        k = int(n * q)

        ranked = row.sort_values()
        long_idx = ranked.index[:k]
        short_idx = ranked.index[-k:]

        weights.loc[t, long_idx] = 1.0 / k
        weights.loc[t, short_idx] = -1.0 / k

    return weights
```

---

# Phase 2 — Volatility-Targeted Portfolio

## Goal

Use the GNN's predicted volatility to dynamically scale overall portfolio exposure each week. When predicted vol is high, reduce equity weight and hold the remainder in cash. When predicted vol is low, take full exposure. This converts cross-sectional vol forecasts into a risk-budgeting decision and addresses the 2024-2025 test-period problem where the inverse-vol portfolio took full exposure during a high-vol, high-return environment.

---

## Construction

At each week t:

- Compute `predicted_port_vol` as the median predicted RV across all stocks, scaled to annualized terms: `median(predicted_rv) * sqrt(52)`
- Compute a scale factor: `scale = min(VOL_TARGET / predicted_port_vol, 1.0)` (no leverage)
- Apply scale to the existing inverse-volatility weights: `final_weights = scale * inv_vol_weights`
- The remaining `1 - scale` of capital sits in cash (zero return)

---

## Implementation

### `config.py`

Add:

```python
VOL_TARGET = 0.10  # annualized target portfolio volatility (10%)
```

### `src/portfolio.py`

Add:

```python
def compute_vol_target_scale(predicted_rv: np.ndarray, vol_target: float) -> float:
    predicted_port_vol = float(np.nanmedian(predicted_rv)) * np.sqrt(52)
    if predicted_port_vol <= 0:
        return 1.0
    return min(vol_target / predicted_port_vol, 1.0)
```

Modify `run_backtest()` to accept an optional `vol_target` parameter. When set, multiply each week's `compute_weights()` output by `compute_vol_target_scale()` before computing returns.

Add `run_all_model_backtests_vol_target()` that runs the same six models plus equal-weight under the vol-targeted construction. Saves to `portfolio_vt_returns.parquet` and `portfolio_vt_metrics_table.csv`.

### `notebooks/06_portfolio.ipynb`

Add cells after the long-short section to call `run_all_model_backtests_vol_target`, display the metrics table, and plot cumulative returns.

---

# Phase 3 — Minimum Variance Portfolio

## Goal

Use predicted individual stock volatilities as the diagonal of the covariance matrix, combined with the realized correlation matrix from the GNN's own correlation graph, to construct the true minimum variance portfolio. This ties the GNN's graph structure directly into the optimizer and is a stronger paper contribution than inverse-vol weighting.

---

## Construction

At each week t:

- Build the diagonal vol matrix `D` from predicted RV: `D = diag(predicted_rv)`
- Estimate the correlation matrix `C` from the trailing `CORR_LOOKBACK_DAYS` window of log returns (same window used to build the correlation graph)
- Form the covariance matrix: `Sigma = D @ C @ D`
- Solve: `w* = argmin w' Sigma w` subject to `sum(w) = 1`, `0 <= w_i <= MAX_WEIGHT`
- Use `scipy.optimize.minimize` with the SLSQP solver, or `cvxpy` for a cleaner formulation

---

## Implementation

### `src/portfolio.py`

Add:

```python
def build_minvar_weights(
    predicted_rv: np.ndarray,
    log_returns_window: pd.DataFrame,
    max_weight: float,
) -> np.ndarray:
    D = np.diag(predicted_rv)
    C = log_returns_window.corr().values
    Sigma = D @ C @ D

    n = len(predicted_rv)
    w0 = np.ones(n) / n

    from scipy.optimize import minimize
    result = minimize(
        fun=lambda w: w @ Sigma @ w,
        x0=w0,
        method="SLSQP",
        bounds=[(0, max_weight)] * n,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 500},
    )
    w = result.x
    w = np.clip(w, 0, max_weight)
    w /= w.sum()
    return w
```

Add `run_all_model_backtests_minvar()` that calls `build_minvar_weights` at each test week for each model. The `log_returns_window` at week t is the trailing `CORR_LOOKBACK_DAYS` days of `log_returns_df` ending on Friday of week t (same lookahead-safety rule as the correlation graph). Saves to `portfolio_mv_returns.parquet` and `portfolio_mv_metrics_table.csv`.

### `notebooks/06_portfolio.ipynb`

Add cells after the vol-target section. Note: min-var optimization over 465 stocks runs for ~0.5s per week; expect ~50s total per model over the test period.

---

## Lookahead Safety

`log_returns_window` ends on Friday of week t. The target is week t+1's RV (starts Monday of t+1). Friday_t < Monday_{t+1}. Lookahead-free.

---

# Phase 4 — Rank IC (Information Coefficient)

## Implementation

```python
from scipy.stats import spearmanr

def compute_rank_ic(preds, actuals):
    ic_series = []

    for t in preds.index:
        p = preds.loc[t]
        a = actuals.loc[t]

        valid = p.notna() & a.notna()

        if valid.sum() < 10:
            ic_series.append(np.nan)
            continue

        ic, _ = spearmanr(p[valid], a[valid])
        ic_series.append(ic)

    return pd.Series(ic_series, index=preds.index)
```

---

# Phase 5 — GNN Ensemble

```python
mses = {
    "corr": mse_corr,
    "sector": mse_sector,
    "granger": mse_granger
}

inv = {k: 1/v for k, v in mses.items()}
total = sum(inv.values())

weights = {k: v / total for k, v in inv.items()}

preds_ensemble = (
    weights["corr"] * preds_corr +
    weights["sector"] * preds_sector +
    weights["granger"] * preds_granger
)
```

---

# Phase 6 — Rank-Based Loss

```python
def rank_loss(preds, targets):
    preds_rank = torch.argsort(torch.argsort(preds))
    targets_rank = torch.argsort(torch.argsort(targets))
    return torch.mean((preds_rank.float() - targets_rank.float())**2)
```

---

# Phase 7 — Mixed Rank-MSE Loss

## Goal

Pure rank loss (Phase 6) improved Rank IC, ICIR, and top-k hit rate across all three graph types, but broke prediction calibration: val MSE degraded 6-10x relative to MSE-trained models, and predictions lost their absolute scale. This makes rank loss models unsuitable for inverse-volatility or minimum variance portfolio constructions, which require predictions in the correct RV range.

Pure MSE preserves calibration but produces weaker ICIR (3.4-3.7 vs. 3.9-4.3 under rank loss), meaning the cross-sectional signal is noisier week to week.

A mixed loss combines both objectives: the MSE term keeps predictions on the correct absolute scale, and the rank loss term pushes the model to prioritize cross-sectional ordering. A single scalar weight controls the tradeoff.

---

## Construction

At each training step:

```
loss = (1 - α) * MSE + α * BPR_rank_loss
```

Where:
- `α = config.MIXED_LOSS_RANK_WEIGHT` (value in [0, 1])
- MSE is the standard masked mean squared error over non-NaN targets
- BPR rank loss is `compute_rank_loss()` from Phase 6 with the same 10% pair sampling

At `α = 0` the loss is pure MSE (reproduces Phase 5 models). At `α = 1` it is pure rank loss (reproduces Phase 6 models). The goal is to find the smallest `α` that recovers the ICIR gains from Phase 6 while keeping val MSE within a reasonable range of the pure MSE models.

A sweep over `α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}` is run on GNN-Correlation only. The winning `α` is then used to train all three graph types.

The key diagnostic is the ICIR vs. val MSE tradeoff curve as `α` increases. If ICIR plateaus at `α = 0.3` while val MSE is still acceptable, there is no reason to go higher. The paper reports the result at the winning `α` alongside the pure MSE and pure rank loss results as a three-way comparison.

---

## Implementation

### `config.py`

Add:

```python
MIXED_LOSS_RANK_WEIGHT     = 0.5    # default α for mixed loss training
MIXED_LOSS_ALPHA_SWEEP     = [0.1, 0.3, 0.5, 0.7, 0.9]  # values tested in sweep
```

### `src/train.py`

Add:

```python
def compute_mixed_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    rank_weight: float = config.MIXED_LOSS_RANK_WEIGHT,
) -> torch.Tensor:
    """
    Weighted combination of masked MSE and pairwise BPR rank loss.

    rank_weight=0.0 reduces to masked MSE.
    rank_weight=1.0 reduces to compute_rank_loss.
    """
    mse  = _masked_mse(preds, targets)
    rank = compute_rank_loss(preds, targets)
    return (1.0 - rank_weight) * mse + rank_weight * rank
```

Add `train_gnn_corr_mixed_sweep()` that iterates over `config.MIXED_LOSS_ALPHA_SWEEP`, trains one GNN-Correlation model per value using `loss_fn=lambda p, t: compute_mixed_loss(p, t, alpha)`, and records val MSE and val IC for each. Results saved to `DATA_RESULTS_DIR/mixed_loss_alpha_sweep.json`.

Add `train_gnn_corr_mixed()`, `train_gnn_sector_mixed()`, and `train_gnn_granger_mixed()` wrapper functions that call `train_gnn()` with `loss_fn=compute_mixed_loss` using the winning `α`. Checkpoints use `_mixed` suffix. Predictions saved as `test_preds_gnn_{variant}_mixed.parquet`.

### `notebooks/04c_rank_loss_models.ipynb`

Add a new section after the rank loss models. Run the alpha sweep on GNN-Correlation, plot the ICIR vs. val MSE tradeoff curve across `α` values, and select the winner. Train all three graph types with the winning `α`. Add to the ranking metrics summary table alongside the MSE and pure rank loss results so all three families are compared in one place.

---

## Anticipated Outcomes

The mixed loss should recover most of the ICIR improvement from pure rank loss at values of `α` between 0.3 and 0.5, while keeping val MSE within 2-3x of the pure MSE models (compared to 6-10x for pure rank loss). The hit rate and mean IC improvements should partially persist. Predictions should remain calibrated enough for inverse-vol and min-var portfolio constructions.

If the ICIR gain requires `α > 0.7` before it appears, the signal from the rank loss component is fragile and the paper will report this as a null result on mixed training. If ICIR converges at `α = 0.3`, that is the cleanest result: a small rank loss weight provides measurable consistency gains at minimal calibration cost.

---

# Key Constraint

Do not proceed to model changes until:

"Do the current models contain usable cross-sectional signal?"

---

# Phase 7 — Ex-Sector Robustness Check

## Goal

Test whether GNN performance during the 2024-2025 test period is genuinely driven by volatility prediction skill or is instead an artifact of sector-level concentration in Information Technology and Communication Services — two sectors that experienced an AI-driven structural expansion that has no analog in the 2015-2023 training period.

The 2024-2025 regime is characterized by a narrow set of large-cap tech names (the "Magnificent 7") generating outsized returns alongside persistently elevated realized volatility. An inverse-volatility portfolio that over- or under-weights these names will exhibit performance that is determined by sector exposure, not cross-sectional ranking within sectors. This phase isolates whether the GNN's predictions add value outside of that regime distortion.

This analysis is reported as a robustness check alongside the primary full-universe results. It does not replace the primary results.

---

## Which Sectors to Exclude

Exclude two sectors for the portfolio backtest only (not for model training or ML metrics):

- **Information Technology** (GICS code: 45): Dominated by semiconductor and software names that drove the 2024-2025 rally. The GNN's predicted vol for these names will be systematically lower than realized vol during the AI expansion period, causing the inverse-vol portfolio to over-weight them precisely when they are most richly valued.
- **Communication Services** (GICS code: 50): Contains Alphabet, Meta, and Netflix, which reclassified from Telecom in 2018 and behaved similarly to IT names during 2024-2025. Including these alongside IT captures the full scope of the AI-driven sector.

Do not exclude any other sectors. The exclusion is pre-specified based on the structural argument above, not by observing test-period returns. The paper must state this justification explicitly.

The sector assignments for each week are read from `data/raw/sector_history.json` using the year-specific sector lookup, consistent with the rest of the pipeline.

---

## Construction

At each test week t:

- Look up each ticker's sector using `sector_history[ticker][str(week.year)]`
- Remove tickers assigned to "Information Technology" or "Communication Services"
- Run the existing inverse-volatility weighting on the reduced universe using the same `compute_weights()` function
- Apply the same `MAX_WEIGHT` cap and re-normalization

The number of stocks in the reduced universe will vary by week due to reclassifications, but should remain approximately 340-380 stocks across the test period.

---

## Implementation

### `config.py`

Add:

```python
EXCLUDED_SECTORS_ROBUSTNESS = ["Information Technology", "Communication Services"]
```

### `src/portfolio.py`

Add a helper that filters predictions and the ticker list by sector:

```python
def filter_universe_by_sector(
    preds_row: pd.Series,
    sector_map: dict,
    excluded_sectors: list,
) -> pd.Series:
    """
    Remove tickers belonging to any sector in excluded_sectors.

    preds_row: predicted RV for one time step, indexed by ticker
    sector_map: {ticker: sector_name} for the relevant year
    excluded_sectors: list of sector names to remove

    Returns a filtered Series with excluded tickers dropped.
    """
    keep = [t for t in preds_row.index if sector_map.get(t, "") not in excluded_sectors]
    return preds_row.loc[keep]
```

Add `run_all_model_backtests_ex_sector()` that wraps the existing `run_backtest()` but applies `filter_universe_by_sector` at each test week before passing predictions to `compute_weights()`. The function accepts `excluded_sectors=config.EXCLUDED_SECTORS_ROBUSTNESS`. Saves to `portfolio_exsector_returns.parquet` and `portfolio_exsector_metrics_table.csv`.

The sector_map at week t is built by calling:

```python
year = week_date.year
sector_map = {ticker: sector_history[ticker][str(year)] for ticker in tickers if str(year) in sector_history[ticker]}
```

Pass this into `filter_universe_by_sector` at each time step inside the backtest loop.

### `notebooks/06_portfolio.ipynb`

Add a section after the min-var section. Call `run_all_model_backtests_ex_sector`, display the metrics table side-by-side with the full-universe metrics, and plot cumulative returns for both on the same axes. Print the number of stocks in the reduced universe at the first and last test week to confirm the filter is working.

---

## Anticipated Outcomes

**If GNN performance improves in the ex-sector universe:** This is evidence that IT/Communication Services concentration was masking the model's signal. The paper can argue that the GNN's volatility predictions are predictive within the broader market but that the 2024-2025 tech regime created a sector-timing problem that the inverse-vol weighting could not handle. This is a constructive finding.

**If GNN performance is unchanged or worsens:** This is also informative. It means the GNN's ranking signal does not depend on sector composition, which is a stronger robustness claim. The paper reports this as evidence that performance generalizes across sector regimes.

**Expected direction:** The equal-weight benchmark will likely underperform more in the ex-sector universe (losing exposure to the tech rally), while the GNN-based portfolios may show relatively better Sharpe ratios due to lower realized drawdowns. The Rank IC numbers, which are computed over the full universe regardless of portfolio construction, should be unaffected by this phase and serve as the clean ML-level comparison.

---

# Phase 8 — Sector-Neutral Portfolio Construction

## Goal

Reweight each model's predictions so that the portfolio holds equal gross exposure across GICS sectors at every time step. This construction isolates the GNN's ability to rank stocks within sectors, eliminating any sector-timing component entirely.

This is a stronger paper contribution than Phase 7 because it makes no assumption about which sectors are problematic. Instead, it neutralizes sector effects by construction, making the GNN's within-sector cross-sectional signal the only source of active return. If GNN-based portfolios outperform equal-weight under sector-neutral construction, the result is attributable to stock-level volatility prediction rather than sector allocation.

---

## Construction

At each week t:

1. Determine which tickers are in each GICS sector using `sector_history[ticker][str(week.year)]`
2. For each sector s, compute inverse-volatility weights for the stocks in that sector:
   - `raw_w_s = 1.0 / predicted_rv[stocks_in_s]`
   - `raw_w_s = raw_w_s.clip(max=implied_sector_max)` (see note below on weight cap)
   - `w_s = raw_w_s / raw_w_s.sum()` (weights sum to 1.0 within sector)
3. Scale each sector's weights by `1.0 / num_active_sectors` so that all sectors receive equal gross weight and the total portfolio sums to 1.0
4. Sectors with fewer than `MIN_SECTOR_SIZE` stocks (see config) are dropped from that week and the remaining sectors are re-normalized to sum to 1.0

The within-sector MAX_WEIGHT cap must be applied at the sector level before the cross-sector scaling step. If a sector has only 5 stocks, the per-stock cap should be `1.0 / MIN_SECTOR_SIZE`, not `config.MAX_WEIGHT`, to avoid degenerate equal-weight behavior inside tiny sectors. Use `max(config.MAX_WEIGHT, 1.0 / len(stocks_in_s))` as the effective cap for each sector.

---

## Implementation

### `config.py`

Add:

```python
MIN_SECTOR_SIZE = 5  # sectors with fewer stocks than this are excluded from the sector-neutral portfolio
```

### `src/portfolio.py`

Add:

```python
def build_sector_neutral_weights(
    preds_row: pd.Series,
    sector_map: dict,
    max_weight: float,
    min_sector_size: int,
) -> pd.Series:
    """
    Construct a sector-neutral inverse-volatility portfolio for one time step.

    Assigns equal gross weight to each active GICS sector, then allocates
    within each sector using inverse-volatility weighting. Sectors with
    fewer than min_sector_size stocks are excluded for that week.

    preds_row: predicted RV indexed by ticker, one time step
    sector_map: {ticker: sector_name} for the relevant year
    max_weight: per-stock MAX_WEIGHT from config (applied at sector level)
    min_sector_size: minimum stocks required to include a sector

    Returns a pd.Series of portfolio weights indexed by ticker, summing to 1.0.
    Lookahead safety: uses only preds_row and sector_map, both available at week T.
    """
    from collections import defaultdict

    sector_buckets = defaultdict(list)
    for ticker in preds_row.index:
        sector = sector_map.get(ticker)
        if sector is not None:
            sector_buckets[sector].append(ticker)

    active_sectors = {s: tickers for s, tickers in sector_buckets.items()
                      if len(tickers) >= min_sector_size}

    if len(active_sectors) == 0:
        n = len(preds_row)
        return pd.Series(1.0 / n, index=preds_row.index)

    sector_weight = 1.0 / len(active_sectors)
    all_weights = pd.Series(0.0, index=preds_row.index)

    for sector, tickers in active_sectors.items():
        rv_s = preds_row.loc[tickers]
        effective_cap = max(max_weight, 1.0 / len(tickers))
        raw_w = 1.0 / rv_s
        raw_w = raw_w.clip(upper=effective_cap * raw_w.sum())  # pre-clip before normalization
        w = raw_w / raw_w.sum()
        w = w.clip(upper=effective_cap)
        w = w / w.sum()  # re-normalize after clip
        all_weights.loc[tickers] = w * sector_weight

    total = all_weights.sum()
    assert abs(total - 1.0) < 1e-6, f"Sector-neutral weights sum to {total}, expected 1.0"
    return all_weights
```

Add `run_all_model_backtests_sector_neutral()` that calls `build_sector_neutral_weights` at each test week for each model. The `sector_map` at each week is built from `sector_history` using `week_date.year`. Saves to `portfolio_sn_returns.parquet` and `portfolio_sn_metrics_table.csv`.

Include a sector-neutral equal-weight benchmark: equal-weight within each sector, then equal-weight across sectors (this is `build_sector_neutral_weights` with a flat `1.0 / len(tickers)` per-sector allocation rather than inverse-vol). Name this `"equal_weight_sn"` in the output.

### `notebooks/06_portfolio.ipynb`

Add a section after the ex-sector section. Show the sector-neutral metrics table, the cumulative return plot, and a breakdown of mean per-sector weight allocation across the test period (a stacked bar chart or table showing that sector weights are approximately equal). The latter confirms the construction is working as intended and can be included as a supplementary figure in the paper.

---

## Anticipated Outcomes

**Primary expectation:** Sharpe ratios will likely compress for all models relative to the full-universe results, because sector-neutral construction removes the beta to high-return sectors. This is expected and should be stated in the paper. The relevant comparison is GNN models vs. the sector-neutral equal-weight benchmark, not vs. the full-universe equal-weight benchmark.

**If GNN outperforms sector-neutral equal-weight:** This is the strongest possible result for the paper. It means the GNN predicts within-sector volatility ranks well enough to generate alpha net of sector effects and transaction costs. The paper can present this as evidence that the GNN captures genuine stock-level signal.

**If GNN and equal-weight perform similarly under sector-neutral construction:** This suggests the GNN's training-period signal is primarily cross-sector (e.g., it learned that tech stocks have different vol dynamics than utilities) rather than within-sector. This is still a publishable finding: it identifies the mechanism of the model's signal and points to within-sector training as a direction for improvement. It also motivates Phase 6's rank-based loss.

**Rank IC interaction:** The sector-neutral construction should be paired with a within-sector Rank IC computation in `notebooks/05_evaluate.ipynb`: compute Spearmanr separately within each sector at each week and report the mean within-sector IC. If the within-sector IC is near zero while the full-universe IC is positive, that confirms the hypothesis above.

---

# Change Log

- v1.0: Original plan (long-short, Rank IC, GNN ensemble, rank-based loss)
- v1.1: Added Phase 2 (volatility-targeted portfolio) and Phase 3 (minimum variance portfolio); renumbered Rank IC to Phase 4, GNN ensemble to Phase 5, rank-based loss to Phase 6
- v1.2: Added Phase 7 (ex-sector robustness check) and Phase 8 (sector-neutral portfolio construction)
