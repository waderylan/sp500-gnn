# Rules: src/models.py · src/train.py · src/portfolio.py · src/significance.py

Loaded when working on model definitions, training loops, portfolio simulation, or significance tests.

---

## Models Overview

Six models total. All are defined in src/models.py.

| Model | Class | Notes |
|---|---|---|
| HAR per-stock | `HARModel` | sklearn LinearRegression, one fit per ticker |
| HAR pooled | `HARPooled` | sklearn LinearRegression, single fit across all stocks |
| LSTM | `LSTMModel` | 2-layer, hidden=64, dropout=0.3, input = 4-week sequence per stock |
| GNN-Correlation | `GNNModel` | GraphSAGE, correlation graph |
| GNN-Sector | `GNNModel` | GraphSAGE, sector graph |
| GNN-Granger | `GNNModel` | GraphSAGE, Granger graph, directed |

HAR uses only RV features (5, 21, 63-day means) computed directly from weekly_rv — NOT from features.parquet. This is intentional and must not be changed.

GNNModel is a single class used for all three graph variants. The graph is passed in at each forward call, not stored in the model.

---

## GNNModel Architecture

```python
class GNNModel(torch.nn.Module):
    # SAGEConv(in_channels, 64, flow=config.SAGE_FLOW) + ReLU + Dropout(0.3)
    # SAGEConv(64, 64, flow=config.SAGE_FLOW) + ReLU + Dropout(0.3)
    # Linear(64, 1)
    # Output: shape (num_stocks,) — one scalar prediction per stock
```

`flow=config.SAGE_FLOW` must be passed to every SAGEConv layer. Do not hardcode it.

---

## Training Rules

**Seeds before every model initialization:**
```python
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
torch.cuda.manual_seed_all(config.RANDOM_SEED)
```

**Chronological iteration — never shuffle:**
Training iterates over time steps in forward chronological order. Do not use a DataLoader with shuffle=True. The order is: train weeks → val weeks → test weeks (test is never seen during training).

**Checkpoint every N epochs:**
Save checkpoint to `data/results/checkpoints/{model_name}_epoch{n}.pt` every `config.CHECKPOINT_EVERY_N_EPOCHS` epochs. Also save `{model_name}_best.pt` whenever val loss improves. Colab sessions can drop without warning.

**Early stopping:**
Track validation MSE. If it doesn't improve for `config.EARLY_STOP_PATIENCE` (10) consecutive epochs, stop and load the best checkpoint.

**Correlation graph in train loop:**
At each time step, call `build_correlation_graph(log_returns, date, config.CORR_LOOKBACK_DAYS, config.CORR_THRESHOLD)`. This is dynamic — recomputed every week. It is intentionally slower than the static graphs.

---

## Portfolio Rules

**Inverse-volatility weighting:**
```python
raw_weights = 1.0 / predicted_rv  # shape (num_stocks,)
raw_weights = raw_weights.clip(max=...)  # implied by MAX_WEIGHT after normalization
weights = raw_weights / raw_weights.sum()
weights = weights.clip(max=config.MAX_WEIGHT)
weights = weights / weights.sum()  # re-normalize after clipping
```

Always re-normalize after the MAX_WEIGHT clip. Weights must sum to 1.0 (assert this).

**Transaction costs:**
```python
turnover = abs(weights_t - weights_t_minus_1).sum()
cost = turnover * (config.TRANSACTION_COST_BPS / 10_000)
net_return = gross_return - cost
```

**Risk-free rate:**
Load from `data/raw/tbill_rates.parquet` (sourced from FRED series DTB3).
Sharpe = (annualized_return - mean_annualized_tbill) / annualized_volatility.
Do not assume risk-free rate = 0.

**Log each week:** max single-stock weight, turnover, gross return, net return.
Report max single-stock weight in the paper.

---

## Significance Testing Rules

**Diebold-Mariano test:**
- One-sided: H₀ = GNN is not better than baseline
- Use HLN small-sample correction
- Input: per-week mean squared errors (averaged across all stocks), shape (num_test_weeks,)
- Run for every (GNN variant, baseline) pair = 3 variants × 3 baselines = 9 tests minimum
- Also run per GICS sector = 9 × 11 = 99 additional tests

**FDR correction:**
Apply Benjamini-Hochberg across all p-values within each baseline group.
Report both uncorrected and FDR-corrected p-values.
The FDR-corrected result is the primary claim in the paper.

**Block bootstrap for Sharpe:**
- Block size: 8 weeks
- n_bootstrap: 5000
- Report 95% CI as [2.5th, 97.5th] percentile of bootstrap distribution
- For pairwise comparison vs equal-weight: bootstrap the *difference* in Sharpe

**Honesty rule:**
If no improvement survives FDR correction, report this clearly. Do not search for a different test that gives significance. The result is what it is.
