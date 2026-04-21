# SP500-GNN — Claude Code Project Memory

Research project: GNN-based cross-sectional volatility prediction for S&P 500.
Target venue: FinML Workshop at NeurIPS 2026.
Full spec: @docs/project_outline.md

---

## Stack

- Python 3.10, PyTorch 2.4, PyTorch Geometric, statsmodels, sklearn, yfinance
- Google Colab A100 GPU. All data persisted to Google Drive.
- pandas-datareader for FRED T-bill rates

---

## Non-Negotiable Rules

### 1. config.py is the only place constants live
Never hardcode a number, date, path, or threshold anywhere in src/ or notebooks.
If a constant doesn't exist yet, add it to config.py first, then reference it.

`DEV_UNIVERSE_SIZE = 50` limits the ticker universe for fast iteration and debugging. Set to `None` to use the full universe. Every function that loads tickers must respect this: slice the ticker list to `tickers[:config.DEV_UNIVERSE_SIZE]` if it is not `None`.

### 2. No logic in notebooks
Notebooks call src/ functions and display results only. If a cell has more than ~5 lines of computation, it belongs in src/.

### 3. Lookahead bias is the critical failure mode
Before writing or modifying any feature or target computation, state explicitly which dates are used and confirm they are strictly prior to the prediction week start. The target for week T is RV from week T+1 — the shift goes forward, not backward. Any rolling window for a feature at week T must end before week T begins.

### 4. The test set is sealed
Do not write any code that reads from data/results/test_preds_*.parquet or generates test-period predictions until Task 5.1 is explicitly reached.

### 5. Random seeds before any model initialization
```python
import random, numpy as np, torch
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
torch.cuda.manual_seed_all(config.RANDOM_SEED)
```

### 6. Fail loudly on shape mismatches
Every function that produces a tensor or dataframe must assert its output shape before returning.

---

## Code Style

- All src/ functions have typed signatures and a docstring covering: what it does, inputs, outputs, and any lookahead-safety notes
- No classes in src/ unless state is genuinely needed. Prefer plain functions.
- Parquet over CSV for all data artifacts

---

## Before Writing Any Code

State:
1. Which src/ module this belongs in
2. What config.py constants it uses
3. Whether it touches any date-indexed data — if yes, confirm the lookahead safety of every window
4. What shape assertions will be added to the output

---

## Task Checklist

Full task specs in @docs/project_outline.md.

### Phase 1 — Data Pipeline
- [x] 1.1 Environment setup: directory structure, config.py, deps confirmed
- [ ] 1.2 Price download, universe construction → `data/raw/prices.parquet`, `tickers.json`, `sector_history.json`
- [ ] 1.3 Log returns → `data/raw/log_returns.parquet`
- [ ] 1.4 Weekly realized volatility → `data/raw/weekly_rv.parquet`
- [ ] 1.5 Target construction + lookahead audit → `data/features/target.parquet`
- [ ] 1.6 Train/val/test splits → `data/features/splits.parquet`

### Phase 2 — Feature Engineering
- [ ] 2.1 Volatility features (RV at 5/10/21/63d lookbacks, short/long ratio)
- [ ] 2.2 Return and volume features
- [ ] 2.3 Stack → winsorize → z-score → `data/features/features.parquet`

### Phase 3 — Graph Construction
- [ ] 3.1 Sector graph → `data/graphs/sector_edges_by_year.parquet`
- [ ] 3.2 Correlation graph builder (`build_correlation_graph()`)
- [ ] 3.3 Granger causality computation → `data/graphs/granger_pvalues.parquet`, `granger_edges.parquet`
- [ ] 3.4 SAGEConv directionality verification (assert reversed edges produce different output)
- [ ] 3.5 Graph stats, visualization → `figures/graph_comparison.png`

### Phase 4 — Model Training
- [ ] 4.1 HAR baselines (per-stock + pooled) → `har_val_preds.parquet`, `har_pooled_val_preds.parquet`
- [ ] 4.2 LSTM baseline → `checkpoints/lstm_best.pt`
- [ ] 4.3 GNN implementation + forward-pass verification on all three graph types
- [ ] 4.4 Train GNN-Correlation (ablate θ ∈ {0.3, 0.5, 0.7}) → `gnn_corr_best.pt`, `corr_threshold_ablation.json`
- [ ] 4.5 Train GNN-Sector → `gnn_sector_best.pt`
- [ ] 4.6 Train GNN-Granger → `gnn_granger_best.pt`
- [ ] 4.7 Validation summary + go/no-go checkpoint → `validation_summary.json`

### Phase 5 — Evaluation
- [ ] 5.1 Test set ML evaluation (unseal test set) → `test_preds_*.parquet`, `ml_metrics_table.csv`
- [ ] 5.2 Portfolio backtest + FRED T-bill rates → `portfolio_returns.parquet`, `portfolio_metrics_table.csv`
- [ ] 5.3 Significance tests (DM + BH FDR + block bootstrap) → `dm_test_results.csv`, `bootstrap_sharpe_ci.csv`, `significance_summary.csv`
- [ ] 5.4 Figure generation (8 publication figures)

### Phase 6 — Writing
- [ ] 6.1 Related work section
- [ ] 6.2 Methodology section
- [ ] 6.3 Experiments section
- [ ] 6.4 Introduction and abstract
- [ ] 6.5 Conclusion, limitations, future work
- [ ] 6.6 Revision, LaTeX compilation, arXiv submission
