# SP500-GNN — Claude Code Project Memory

Research project: GNN-based cross-sectional volatility prediction for S&P 500.
Target venue: FinML Workshop at NeurIPS 2026.

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

Each task is complete only when: (1) src/ code written, (2) notebook cells filled and runnable, (3) output file(s) saved and verified.

### Phase 1 — Data Pipeline (`notebooks/01_data.ipynb`)
- [x] 1.1 Environment setup: directory structure, config.py, deps confirmed
- [x] 1.2 Price download, universe construction → `data/raw/prices.parquet`, `tickers.json`, `sector_history.json` · notebook: run + verify cell, coverage stats, 5-ticker price plot, sector spot-check
- [x] 1.3 Log returns → `data/raw/log_returns.parquet` · notebook: print mean/std, verify March 2020 spike
- [x] 1.4 Weekly realized volatility → `data/raw/weekly_rv.parquet` · notebook: print mean RV, verify March 2020 week, plot 5 RV series
- [x] 1.5 Target construction + lookahead audit → `data/features/target.parquet` · notebook: written audit trail tracing 5 rows from raw price to target value
- [x] 1.6 Train/val/test splits → `data/features/splits.parquet` · notebook: print week counts per split

### Phase 2 — Feature Engineering (`notebooks/02_features.ipynb`)
- [x] 2.1 Volatility features (RV at 5/10/21/63d lookbacks, short/long ratio) · notebook: spot-check values are positive, ratio spikes in volatile regimes
- [x] 2.2 Return and volume features · notebook: verify lagging, spot-check volume ratio
- [x] 2.3 Stack → winsorize → z-score → `data/features/features.parquet` · notebook: assert mean≈0 / std≈1 at 10 random time steps, feature correlation heatmap

### Phase 3 — Graph Construction (`notebooks/03_graphs.ipynb`)
- [x] 3.1 Sector graph → `data/graphs/sector_edges_by_year.parquet` · notebook: networkx plot for 2016 (10 clusters) and 2017 (11 clusters), node counts per sector
- [x] 3.2 Correlation graph builder (`build_correlation_graph()`) · notebook: visualize calm/COVID/recent samples, print edge counts
- [x] 3.3 Granger causality computation → `data/graphs/granger_pvalues.parquet`, `granger_edges.parquet` · notebook: print edge count, in/out-degree distribution plot
- [x] 3.4 SAGEConv directionality verification · notebook: named cell with assert that reversed edges produce different output
- [ ] 3.5 Graph stats, visualization → `figures/graph_comparison.png` · notebook: density table, side-by-side graph figure saved to disk

### Phase 4 — Model Training (`notebooks/04_models.ipynb`)
- [ ] 4.1 HAR baselines (per-stock + pooled) → `har_val_preds.parquet`, `har_pooled_val_preds.parquet` · notebook: print val MSE/MAE for both, R² distribution
- [ ] 4.2 LSTM baseline → `checkpoints/lstm_best.pt` · notebook: plot val loss curve
- [ ] 4.3 GNN implementation + forward-pass verification on all three graph types · notebook: print output shapes and confirm no NaNs
- [ ] 4.4 Train GNN-Correlation (ablate θ ∈ {0.3, 0.5, 0.7}) → `gnn_corr_best.pt`, `corr_threshold_ablation.json` · notebook: print ablation table
- [ ] 4.5 Train GNN-Sector → `gnn_sector_best.pt` · notebook: print val MSE vs GNN-Correlation
- [ ] 4.6 Train GNN-Granger → `gnn_granger_best.pt` · notebook: print val MSE, confirm correction method
- [ ] 4.7 Validation summary + go/no-go checkpoint → `validation_summary.json` · notebook: print ranked model table, document go/no-go decision

### Phase 5 — Evaluation
- [ ] 5.1 Test set ML evaluation (unseal test set) → `test_preds_*.parquet`, `ml_metrics_table.csv` · `notebooks/05_evaluate.ipynb`: display full metrics table
- [ ] 5.2 Portfolio backtest + FRED T-bill rates → `portfolio_returns.parquet`, `portfolio_metrics_table.csv` · `notebooks/06_portfolio.ipynb`: display portfolio metrics table, cumulative return plot
- [ ] 5.3 Significance tests (DM + BH FDR + block bootstrap) → `dm_test_results.csv`, `bootstrap_sharpe_ci.csv`, `significance_summary.csv` · `notebooks/07_significance.ipynb`: display significance summary table
- [ ] 5.4 Figure generation (8 publication figures) · `notebooks/07_significance.ipynb`: confirm all 8 PNGs saved to `data/results/figures/`

### Phase 6 — Writing
- [ ] 6.1 Related work section
- [ ] 6.2 Methodology section
- [ ] 6.3 Experiments section
- [ ] 6.4 Introduction and abstract
- [ ] 6.5 Conclusion, limitations, future work
- [ ] 6.6 Revision, LaTeX compilation, arXiv submission
