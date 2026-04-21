# SP500-GNN — Claude Code Project Memory

Research project: GNN-based cross-sectional volatility prediction for S&P 500.
Target venue: FinML Workshop at NeurIPS 2026.
Full project spec: project_plan.md

---

## Stack

- Python 3.10, PyTorch 2.4, PyTorch Geometric, statsmodels, sklearn, yfinance
- Google Colab A100 GPU. All data persisted to Google Drive.
- pandas-datareader for FRED T-bill rates
- All dependencies listed in requirements.txt

---

## Repository Layout

```
sp500-gnn/
├── config.py                  # Single source of truth for ALL constants
├── src/
│   ├── data.py                # Price download, log returns, weekly RV, splits
│   ├── features.py            # Feature engineering, winsorization, z-scoring
│   ├── graphs.py              # Three graph constructors + make_pyg_data()
│   ├── models.py              # HAR (per-stock + pooled), LSTM, GNN classes
│   ├── train.py               # Training loops for LSTM and GNN
│   ├── evaluate.py            # MSE, MAE, R², sector breakdowns
│   ├── portfolio.py           # Inverse-vol portfolio, rebalancing, Sharpe
│   └── significance.py        # DM test, block bootstrap, FDR correction
├── docs/
│   └── project_outline.md     # Full spec — reference with @docs/project_outline.md
├── data/
│   ├── raw/                   # prices.parquet, log_returns.parquet, weekly_rv.parquet
│   ├── features/              # features.parquet, target.parquet, splits.parquet
│   ├── graphs/                # sector_edges_by_year.parquet, granger_*.parquet, corr_sample/
│   └── results/               # checkpoints/, figures/, *.csv, *.json
└── notebooks/
    ├── 01_data.ipynb
    ├── 02_features.ipynb
    ├── 03_graphs.ipynb
    ├── 04_models.ipynb
    ├── 05_evaluate.ipynb
    ├── 06_portfolio.ipynb
    └── 07_significance.ipynb
```

---

## Non-Negotiable Rules

These apply to every file you touch. Do not proceed without confirming them.

### 1. config.py is the only place constants live
Never hardcode a number, date, path, or threshold anywhere in src/ or notebooks.
Every magic number belongs in config.py. If a constant doesn't exist yet, add it there first, then reference it.

Key constants already defined:
- RANDOM_SEED = 42
- TRAIN_END = "2022-12-31"
- VAL_END = "2023-12-31"
- TEST_END = "2025-12-31"
- MIN_COVERAGE = 0.95
- WINSORIZE_CLIP = (0.01, 0.99)
- MAX_WEIGHT = 0.05
- CORR_THRESHOLD = 0.5
- CORR_LOOKBACK_DAYS = 252
- GRANGER_LAG = 5
- SAGE_FLOW = "source_to_target"
- HIDDEN_DIM = 64
- DROPOUT = 0.3
- LEARNING_RATE = 0.001
- EARLY_STOP_PATIENCE = 10
- CHECKPOINT_EVERY_N_EPOCHS = 5
- TRANSACTION_COST_BPS = 10

### 2. No logic in notebooks
Notebooks call src/ functions and display results. They do not contain loops, model definitions, or data transformations. If a notebook cell has more than ~5 lines of computation, it belongs in src/.

### 3. Lookahead bias is the critical failure mode
Before writing or modifying any feature or target computation, state explicitly which dates are used and confirm they are strictly prior to the prediction week start. The target for week T is RV from week T+1 — the shift goes forward, not backward. Any rolling window for a feature at week T must end before week T begins.

### 4. The test set is sealed
Do not write any code that reads from data/results/test_preds_*.parquet or generates test-period predictions until Task 5.1 is explicitly reached. If asked to evaluate on test data before that point, refuse and explain why.

### 5. Random seeds must be set before any model initialization
At the top of any script or notebook cell that initializes a model or runs training:
```python
import random, numpy as np, torch
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
torch.manual_seed(config.RANDOM_SEED)
torch.cuda.manual_seed_all(config.RANDOM_SEED)
```

### 6. Fail loudly on shape mismatches
Every function that produces a tensor or dataframe must assert its output shape before returning. Silent shape errors are the second most common source of bugs after lookahead bias.

---

## Code Style

- All src/ functions have typed signatures and a one-paragraph docstring covering: what it does, inputs, outputs, and any lookahead-safety notes
- No classes in src/ unless state is genuinely needed. Prefer plain functions.
- Parquet over CSV for all data artifacts
- When in doubt, explain your reasoning before writing code — especially for anything that touches dates, rolling windows, or train/val/test indexing

---

## Before Writing Any Code

State:
1. Which src/ module this belongs in
2. What config.py constants it uses
3. Whether it touches any date-indexed data — if yes, confirm the lookahead safety of every window
4. What shape assertions will be added to the output
