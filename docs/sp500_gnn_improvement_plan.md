# Implementation Plan — Improving Portfolio Results

Version 1.1 · Focus: Extracting Existing Signal Before Model Changes

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
- [ ] 5. GNN ensemble (prediction averaging)  
- [ ] 6. (Optional, next phase) Rank-based training loss  

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

# Key Constraint

Do not proceed to model changes until:

"Do the current models contain usable cross-sectional signal?"

---

# Change Log

- v1.0: Original plan (long-short, Rank IC, GNN ensemble, rank-based loss)
- v1.1: Added Phase 2 (volatility-targeted portfolio) and Phase 3 (minimum variance portfolio); renumbered Rank IC to Phase 4, GNN ensemble to Phase 5, rank-based loss to Phase 6
