# SP500-GNN

**Graph Neural Networks for Cross-Sectional Volatility Prediction**

*S&P 500 Universe · 2015–2025 · Three Graph Construction Strategies*

Version 1.1 · April 2026

---

# Executive Summary

This document defines the full scope, architecture, methodology, and execution plan for SP500-GNN, a research project targeting publication at the FinML Workshop at NeurIPS 2026. The project investigates whether explicitly modeling inter-stock relationships via graph neural networks improves weekly realized volatility prediction over time-series baselines, and whether the choice of graph construction strategy materially affects predictive accuracy and portfolio performance.

**The research contribution is threefold:**

- A systematic empirical comparison of three graph construction strategies — correlation-based, sector-based, and Granger causality-based — applied to the same GNN backbone under identical training conditions.

- A rigorous financial evaluation layer that translates statistical accuracy improvements into portfolio-level risk-adjusted performance metrics, including explicit transaction cost modeling. This evaluation framework is absent from the majority of existing FinML literature.

- An open-source, reproducible codebase structured for extensibility, enabling downstream research to build on the graph construction comparison without re-implementing the data pipeline.

| **Target Venue** | FinML Workshop at NeurIPS 2026. Secondary: arXiv preprint (cs.LG + q-fin.ST) to be posted upon completion of experiments. |
|---|---|
| **Timeline** | 3 weeks of focused execution. All phases are sequenced with explicit dependencies and risk flags. |
| **Stack** | Python 3.10, PyTorch 2.4, PyTorch Geometric, Google Colab (A100), Google Drive for persistence. |
| **Risk** | Granger causality computation at 462-stock scale is the highest-risk task. Mitigation strategy defined in Section 5. |

---

# Problem Statement

## Why Volatility, Not Returns

Equity return prediction is a well-studied and largely intractable problem. The efficient market hypothesis, supported by decades of empirical evidence, implies that price changes should be close to a random walk — and observed model performance confirms this. State-of-the-art deep learning models applied to return prediction routinely achieve directional accuracy in the 51–53% range, barely above chance, and frequently fail to generalize out-of-sample.

Volatility — the magnitude of price movement independent of direction — exhibits a fundamentally different statistical character. Realized volatility is strongly autocorrelated across time (a property known as volatility clustering), mean-reverting over longer horizons, and exhibits well-documented cross-sectional spillover effects across related assets. These properties make it amenable to supervised learning in a way that returns are not. The HAR (Heterogeneous Autoregressive) model, which is a simple linear regression of next-period volatility on past volatility at three timescales, achieves R-squared values of 0.4–0.6 on equity volatility — a signal strength that makes meaningful improvement feasible and measurable.

## Why Graph Structure Matters

Most volatility forecasting models treat each stock as an isolated time series. This ignores a well-established empirical regularity: volatility is contagious. Turbulence in one stock spreads to its sector peers, its supply chain partners, and its market cap neighbors. During the March 2020 COVID shock, volatility spiked simultaneously across correlated asset groups — not independently and randomly. A model that can incorporate this cross-sectional information at prediction time has access to signal that univariate models structurally cannot use.

Graph Neural Networks (GNNs) provide the natural framework for this. Each stock is represented as a node. Relationships between stocks are encoded as edges. The GNN propagates information across edges during a message-passing process, allowing each stock's volatility prediction to be informed by the recent behavior of its graph neighbors. The central unresolved question — which this project directly addresses — is: what is the right way to define those edges?

## Research Question

Does graph structure improve cross-sectional volatility prediction for S&P 500 equities, and if so, which graph construction strategy — correlation-based, sector-based, or Granger causality-based — produces the best predictive and portfolio performance?

---

# Methodology

## 1. Data

### Universe

S&P 500 constituent stocks with complete daily price and volume data from January 2015 through December 2025. The universe is restricted to stocks that were S&P 500 constituents for the majority of the sample period, using point-in-time constituent lists to avoid survivorship bias. Stocks are included if they have at least 95% data coverage AND were index constituents for at least 80% of the total sample weeks. Stocks that entered the index after January 2016 or were removed before December 2024 are excluded. This exclusion criteria must be documented explicitly in the paper as a limitation, as it biases the universe toward more stable, established companies. Expected universe size after filtering: approximately 400–462 tickers.

> **NOTE ON GICS SECTOR CHANGES:** GICS sector classifications changed materially during the sample period. Most notably, Real Estate was separated from Financials in August 2016, and the Telecommunication Services sector was restructured into Communication Services in September 2018, absorbing media and internet companies from Consumer Discretionary and Information Technology. The sector graph must use point-in-time sector assignments rather than current assignments. Stocks should be assigned the sector they held at the start of each calendar year. This requires maintaining a sector assignment history table rather than a single static mapping.

### Source

Daily OHLCV data from yfinance (free, reliable for historical S&P 500 data). Point-in-time sector membership from Wikipedia GICS table cross-referenced against SEC filings for reclassification dates. No premium data sources required.

### Target Variable: Weekly Realized Volatility

For each stock and each calendar week, realized volatility (RV) is computed as the annualized standard deviation of the five daily log returns within that week:

| **Formula** | RV(t) = std(r_t1, r_t2, r_t3, r_t4, r_t5) × sqrt(252) where r = daily log return |
|---|---|

The prediction target for week T is RV computed from week T+1 — strictly future data. This shift must be implemented and audited carefully to prevent lookahead bias. Every row in the target dataframe is verified by hand-tracing 5 observations from raw price to target value.

### Train / Validation / Test Split

| **Split** | **Date Range** | **Purpose** |
|---|---|---|
| Train | 2015 – 2022 | Model fitting and weight optimization |
| Validation | 2023 | Hyperparameter selection, early stopping, threshold tuning |
| Test | 2024 – 2025 | Final evaluation only. Sealed until all development is complete. |

> **CRITICAL:** The test set is sealed. It is evaluated exactly once, after all model selection and hyperparameter tuning is finalized on the validation set. Any modification to model design after viewing test results invalidates the experiment.

> **NOTE ON VALIDATION SET USAGE:** Multiple hyperparameter decisions are made using the validation set: the correlation threshold θ, the Granger correction method (Bonferroni vs. BH), and early stopping for all neural models. While each individual decision is standard practice, the cumulative effect of repeated validation set consultation should be acknowledged in the paper as a limitation. Report the number of distinct hyperparameter configurations evaluated on validation before final test evaluation.

## 2. Feature Engineering

All features are computed per stock per week using only data available prior to the start of that prediction week. Every rolling window ends at least one trading day before the prediction date.

### Volatility Features

- Realized volatility over 5-day lookback (1 week)
- Realized volatility over 10-day lookback (2 weeks)
- Realized volatility over 21-day lookback (1 month)
- Realized volatility over 63-day lookback (1 quarter)
- Short-to-long volatility ratio: RV(5) / RV(63) — captures vol spikes relative to baseline

### Return Features

- Price momentum over 5 days (5-day cumulative log return)
- Price momentum over 20 days (20-day cumulative log return)

### Volume Features

- Rolling mean volume over 5 days
- Rolling mean volume over 20 days
- Volume ratio: 5-day mean volume / 20-day mean volume

### Cross-Sectional Normalization

At each weekly time step, all features are z-scored cross-sectionally: for each feature, subtract the mean across all stocks and divide by the standard deviation across all stocks. This removes market-wide level effects, improves stationarity across the 10-year sample, and is standard practice in quantitative factor research. Post-normalization, the mean of any feature across stocks at any time step should be approximately 0 with standard deviation approximately 1. This is verified programmatically.

> **NOTE ON WINSORIZATION:** Before z-scoring, features should be winsorized at the 1st and 99th percentile cross-sectionally at each time step. This prevents outlier stocks (e.g., a single stock experiencing a halt or data error) from distorting the normalization for all other stocks. Add WINSORIZE_CLIP = (0.01, 0.99) to config.py and apply before z-scoring in features.py.

## 3. Graph Construction Strategies

Three distinct graphs are constructed from the same underlying data. Each feeds into an identical GNN architecture. All other model components are held constant so that differences in performance are attributable to graph structure, not architecture.

### Graph 1: Correlation Graph (Dynamic, Undirected)

At each weekly time step, a rolling 252-trading-day window of daily log returns is used to compute the pairwise Pearson correlation matrix across all stocks. An edge is drawn between two stocks if their correlation exceeds a threshold θ. Self-loops are excluded. The graph is recomputed each week as the rolling window advances, making it dynamic — the graph structure changes over time in response to evolving market relationships.

- Lookback window: 252 trading days (1 year)
- Threshold θ: tuned on validation set across {0.3, 0.5, 0.7}
- Storage format: sparse edge index tensor (2 × num_edges) per time step
- Expected density: moderate; increases during high-correlation regimes (e.g., 2020 COVID)

Limitation: Pearson correlation is symmetric and captures linear co-movement only. At 462 stocks it can identify spurious relationships. Threshold selection materially affects graph density and model performance.

### Graph 2: Sector Graph (Static Per Year, Undirected)

Two stocks are connected if and only if they share the same GICS sector at the start of each calendar year. There are 11 GICS sectors. The graph is updated annually to reflect any reclassifications (see Section 1 note on GICS changes). No return data is used in its construction; edges encode fundamental economic relationships (shared customers, suppliers, regulatory environment, macro sensitivities).

- Source: Point-in-time GICS sector assignments, updated annually
- Structure: 11 fully-connected cliques (one per sector) per calendar year
- Expected density: lower than correlation graph; coarser but stable within each year

Limitation: Sector classification is a coarse proxy. Companies within the same sector can have very different volatility drivers. The graph may be too blunt to capture fine-grained relationships.

### Graph 3: Granger Causality Graph (Static, Directed)

Granger causality tests whether past values of stock A's returns contain statistically significant predictive information about stock B's returns, beyond what B's own history provides. Unlike correlation, this produces a directed graph: an edge from A to B does not imply an edge from B to A. This is the most theoretically principled of the three graphs as it attempts to identify genuine predictive relationships rather than co-movement.

- Test: statsmodels grangercausalitytests, lag=5 trading days
- Universe: all ordered pairs of 462 stocks = 462 × 461 = 213,082 tests
- Multiple comparison correction: Bonferroni correction (α / 213,082 ≈ 2.35 × 10⁻⁷)
- Computation: parallelized across CPU cores via Python multiprocessing
- Estimated runtime: 6–12 hours on Colab CPU; initiated overnight
- Static: computed once over the training period; fixed for all time steps

> **IMPORTANT — DIRECTIONALITY IN PyG:** Verify explicitly that SAGEConv is receiving and using directed edges as intended. Some PyG aggregation implementations symmetrize the adjacency by default, which would make the Granger graph behaviorally identical to an undirected graph. Add an assertion in the forward pass that checks whether removing reverse edges changes the output. Document the aggregation direction (source-to-target vs. target-to-source) in the methodology section.

Limitation: Computationally expensive and sensitive to lag selection. Bonferroni correction is conservative and may produce a very sparse graph. The graph is static and computed over 2015–2022 training data — pre-COVID relationships may differ from post-COVID ones, and this stationarity assumption should be acknowledged as a limitation. This is the highest-risk component of the project — see Risk Register in Section 5.

## 4. Model Architecture

### GNN Backbone: GraphSAGE

All three graph variants use GraphSAGE (Graph Sample and Aggregate) as the GNN layer. GraphSAGE operates via neighborhood aggregation: for each node, it computes the mean of neighbor feature vectors, concatenates this aggregate with the node's own features, and passes the result through a learned linear transformation followed by a ReLU nonlinearity. Two such layers are stacked, enabling information propagation up to 2 hops from each node.

GraphSAGE is selected over alternatives (GCN, GAT, GIN) for the following reasons:

- Speed and scalability: handles 462 nodes efficiently without sampling
- Inductive capability: generalizes to new graph structures, relevant for the dynamic correlation graph
- Directional support: PyTorch Geometric's SAGEConv natively handles directed edges, required for the Granger graph
- Reviewer familiarity: well-understood baseline in the GNN literature

### Full Architecture

| **Layer** | **Output Shape** | **Notes** |
|---|---|---|
| Input | (462, F) | F = number of features per stock per week |
| SAGEConv 1 + ReLU + Dropout | (462, 64) | Hidden dim = 64, dropout = 0.3 |
| SAGEConv 2 + ReLU + Dropout | (462, 64) | Second message-passing layer |
| Linear | (462, 1) | Per-stock scalar prediction |
| Output | (462,) | Predicted next-week RV per stock |

### Training Configuration

- Loss function: Mean Squared Error (MSE) against next-week realized volatility
- Optimizer: Adam, learning rate = 0.001
- LR scheduler: ReduceLROnPlateau — halve LR if validation loss does not improve for 5 epochs
- Regularization: Dropout 0.3 after each GNN layer
- Early stopping: patience = 10 epochs based on validation MSE
- Batch structure: one time step (all 462 stocks) = one forward pass. Iterate chronologically over all training weeks.
- Hardware: Google Colab A100 GPU (40GB VRAM)
- Random seeds: fix random seed = 42 for Python, NumPy, and PyTorch at the start of every training script. Store RANDOM_SEED = 42 in config.py. This applies to all five models.

## 5. Baselines

| **Model** | **Type** | **Role** |
|---|---|---|
| HAR (per-stock) | Linear regression, one model per stock | Primary finance baseline. Industry standard for RV forecasting. Must be beaten. |
| HAR (pooled) | Linear regression, single model across all stocks | Secondary baseline. Shares parameters across stocks, making it a fairer structural comparison to GNN. |
| LSTM | Deep learning, no graph | Isolates the contribution of graph structure from deep learning alone. |
| GNN-Correlation | GraphSAGE + dynamic correlation graph | GNN variant 1. |
| GNN-Sector | GraphSAGE + static sector graph | GNN variant 2. |
| GNN-Granger | GraphSAGE + directed Granger graph | GNN variant 3. |

> **NOTE:** A pooled HAR model has been added as a secondary baseline. Because the GNN shares parameters across all stocks, per-stock HAR is not a fully fair comparison — it has more total parameters (one model per stock). Pooled HAR fits a single linear model using all stocks' data and generates predictions for all stocks, making the parameter-sharing structure comparable to the GNN. Both HAR variants should be reported in the final metrics table.

| **Minimum Bar** | Any model that does not beat HAR (per-stock) on at least one primary metric (MSE or Sharpe) does not constitute a publishable contribution. HAR performance is the threshold, not the floor. |
|---|---|

## 6. Evaluation Framework

### ML Metrics

- Mean Squared Error (MSE) — primary accuracy metric
- Mean Absolute Error (MAE) — interpretable in volatility units
- R-squared — proportion of variance explained
- All metrics reported: pooled across all stocks, and broken down by GICS sector

### Portfolio Evaluation

Each week, model predictions are used to construct an inverse-volatility weighted portfolio across all 462 stocks. Weight for stock i at time t is proportional to 1 / predicted_RV(i, t), normalized to sum to 1. Weights are capped at a maximum of 5% per stock (MAX_WEIGHT = 0.05 in config.py) to prevent extreme concentration in stocks predicted to have near-zero volatility. This overweights stocks predicted to be calm and underweights stocks predicted to be turbulent — a risk parity approach standard in quantitative fund management.

Rebalancing is weekly. At each rebalance, turnover is computed as the sum of absolute weight changes. Transaction costs are applied at 10 basis points (0.10%) per unit of turnover. This is conservative and defensible for liquid S&P 500 names. Sharpe ratios are computed using the actual risk-free rate (3-month US T-bill weekly rate) sourced from FRED (Federal Reserve Economic Data) rather than assuming zero, since the test period (2024–2025) had meaningfully positive interest rates. All portfolio metrics are computed on the 2024–2025 test period only.

| **Portfolio Metric** | **Definition** |
|---|---|
| Annualized Return | Mean weekly portfolio return × 52 |
| Annualized Volatility | Std of weekly portfolio returns × sqrt(52) |
| Sharpe Ratio | (Annualized Return − Risk-Free Rate) / Annualized Volatility |
| Maximum Drawdown | Worst peak-to-trough loss over the test period |
| Average Weekly Turnover | Mean of absolute weight changes per rebalance |
| Maximum Single-Stock Weight | Highest weight assigned to any stock in any week (monitors concentration) |

All six models plus an equal-weight benchmark are evaluated on all portfolio metrics. Results are presented as a comparative table in the paper.

### Statistical Significance Testing

Claimed improvements over baselines must be accompanied by formal significance tests. Performance differences that appear numerically meaningful over 104 test weeks may not be statistically distinguishable from noise. All significance tests are one-sided (testing whether the GNN variant is *better* than the baseline) and use α = 0.05. Results must report test statistics, p-values, and confidence intervals alongside point estimates in all tables. See Task 5.4 for implementation details.

---

# Codebase Architecture

The project uses a hybrid notebook + module structure. Notebooks serve as orchestration and reporting layers only — they call functions, display results, and tell the experimental story. All computation lives in plain Python modules under src/. This design ensures reproducibility, enables function-level testing, and makes the codebase navigable by external readers and future collaborators.

## Repository Structure

| **Root** | sp500-gnn/ |
|---|---|

| **Path** | **Purpose** |
|---|---|
| config.py | Single source of truth for all hyperparameters, thresholds, file paths, and constants. Changing one number here propagates everywhere. Includes: RANDOM_SEED, WINSORIZE_CLIP, MAX_WEIGHT, CORR_THRESHOLD, all date boundaries. |
| src/data.py | Price download, log return computation, weekly RV computation, train/val/test splitting. |
| src/features.py | All feature engineering: volatility windows, return features, volume features, winsorization, cross-sectional z-scoring. |
| src/graphs.py | Three graph constructors: build_correlation_graph(), build_sector_graph(), build_granger_graph(). Each returns a PyG-compatible edge index. Includes make_pyg_data() helper. |
| src/models.py | HAR model (sklearn, per-stock and pooled), LSTM model (PyTorch), GNN model (PyTorch Geometric GraphSAGE). All model class definitions. |
| src/train.py | Training loops for LSTM and GNN. Handles chronological iteration, early stopping, checkpoint saving, and validation logging. |
| src/evaluate.py | MSE, MAE, R-squared computation. Sector-level breakdown. Results table generation. |
| src/portfolio.py | Inverse-volatility portfolio construction, weight capping, weekly rebalancing simulation, transaction cost application, Sharpe/drawdown computation. Fetches T-bill rates from FRED for risk-free rate. |
| src/significance.py | Diebold-Mariano test for forecast comparison, bootstrapped confidence intervals for portfolio metrics, block bootstrap for time-series-aware resampling. |
| 01_data.ipynb | Runs data.py. Verifies target construction. Shows RV distribution plots and COVID spike. |
| 02_features.ipynb | Runs features.py. Verifies normalization and winsorization. Shows feature correlation heatmap. |
| 03_graphs.ipynb | Runs graphs.py. Visualizes all three graphs for a sample week. Reports density statistics. Verifies SAGEConv directionality behavior. |
| 04_models.ipynb | Trains all five models. Saves checkpoints. Logs validation loss curves. |
| 05_evaluate.ipynb | Loads checkpoints. Runs final test set evaluation. Produces ML metrics table. |
| 06_portfolio.ipynb | Runs portfolio backtest. Produces all portfolio figures and metrics table. |
| 07_significance.ipynb | Runs all statistical significance tests. Produces significance summary table for paper. |
| data/raw/ | Downloaded price parquets. Never modified after creation. |
| data/features/ | Computed feature tensors. Regenerated if features.py changes. |
| data/graphs/ | Granger causality p-value matrix. Sector edge list. Sample correlation edge lists. |
| data/results/ | Model checkpoints, predictions, metrics JSON, figures. |

## Design Principles

- No logic in notebooks. If a cell contains a loop with more than 5 lines, it belongs in src/.
- No classes in src/ unless an object genuinely needs mutable state. All graph constructors, feature pipelines, and evaluation functions are plain functions with typed signatures and docstrings.
- config.py is the only place hyperparameters live. No magic numbers in src/ modules.
- Parquet over CSV. All data artifacts are saved as parquet files. Faster to read, smaller on disk, preserves dtypes.
- Fail loudly. Every data loading function asserts expected shapes. Every feature function verifies post-normalization statistics. Silent shape mismatches are the leading cause of subtle bugs in ML pipelines.

---

# Execution Plan

Tasks are ordered by dependency. Each task has an explicit input, output, and verification criterion. A task is not complete until its verification criterion is met. No task should be skipped to save time — each feeds the next.

## Phase 1: Data Pipeline

| **Goal** | A clean, verified, lookahead-free dataset of prices, log returns, weekly realized volatility, and train/val/test splits, all saved to Google Drive. |
|---|---|

### Task 1.1 — Environment Setup

- Create sp500-gnn/ directory structure in Google Drive as specified in Section 3
- Initialize config.py with all constants from Section 3 (dates, thresholds, hyperparameters, paths). Include: RANDOM_SEED = 42, WINSORIZE_CLIP = (0.01, 0.99), MAX_WEIGHT = 0.05, CORR_THRESHOLD = 0.5 (initial), MIN_COVERAGE = 0.95, all date boundaries
- Verify Colab connects to Drive and can read/write to sp500-gnn/
- Install all dependencies: yfinance, PyTorch Geometric, statsmodels, pyarrow, pandas-datareader (for FRED T-bill data)
- Confirm PyTorch detects A100 GPU via torch.cuda.is_available()
- Set random seeds at top of script: random.seed(42), np.random.seed(42), torch.manual_seed(42), torch.cuda.manual_seed_all(42)

| **Output** | Initialized repo. config.py populated. GPU confirmed. Seeds set. |
|---|---|

### Task 1.2 — Price Download and Universe Construction

- Implement src/data.py: download_prices() function using yfinance batch download for S&P 500 tickers
- Source point-in-time constituent list: use the Wikipedia historical S&P 500 changes table to identify which tickers were members during the study period. Exclude tickers added after January 2016 or removed before December 2024.
- Filter to stocks with >= 95% data coverage across 2015–2025 (threshold in config.py)
- Save filtered daily OHLCV as data/raw/prices.parquet
- Save final ticker list as data/raw/tickers.json
- Build and save a sector_history.json file mapping each ticker to its GICS sector by year (2015–2025), capturing the 2016 Real Estate split and the 2018 Communication Services restructure
- Verify: plot 5 random stock price series. Confirm no all-NaN columns. Print final universe size.

| **Output** | data/raw/prices.parquet. data/raw/sector_history.json. Universe of ~400-462 tickers confirmed. |
|---|---|

### Task 1.3 — Log Returns

- Implement compute_log_returns() in src/data.py
- Compute daily log return as log(close_t / close_t-1) for all stocks
- Save as data/raw/log_returns.parquet
- Verify: mean daily return should be near 0. Std should be ~0.01–0.02 for most stocks. Check March 2020 — returns should spike to ±0.10+.

| **Output** | data/raw/log_returns.parquet |
|---|---|

### Task 1.4 — Weekly Realized Volatility

- Implement compute_weekly_rv() in src/data.py
- Group daily log returns by ISO week. For each week and each stock, compute std of 5 daily returns × sqrt(252)
- Result shape: (num_weeks, num_stocks)
- Save as data/raw/weekly_rv.parquet
- Verify: annualized RV should average 0.15–0.25 for most stocks. Week of March 16 2020 should show RV of 0.80–1.50+ for most names. Plot RV time series for 5 stocks.

| **Output** | data/raw/weekly_rv.parquet |
|---|---|

### Task 1.5 — Target Construction and Lookahead Audit

- Implement make_target() in src/data.py: shift weekly_rv forward by 1 week
- Target row for week T contains RV from week T+1
- Save as data/features/target.parquet
- **LOOKAHEAD AUDIT:** manually select 5 random rows. For each row, record the week index, the target RV value, and the date range used to compute it. Confirm the date range is strictly after the week index. Document this audit in 01_data.ipynb.

| **Output** | data/features/target.parquet. Written audit trail in notebook. |
|---|---|

### Task 1.6 — Train/Val/Test Split

- Implement split_data() in src/data.py using date boundaries from config.py
- Split applies to weekly_rv, target, and all subsequently computed features
- Save split index arrays (train_idx, val_idx, test_idx) as data/features/splits.parquet
- Verify: print number of weeks in each split. Train should be ~400 weeks, val ~52, test ~104.

| **Output** | data/features/splits.parquet |
|---|---|

---

## Phase 2: Feature Engineering

| **Goal** | A normalized feature tensor of shape (num_weeks, num_stocks, num_features) saved to Drive, with verified zero-mean cross-sectional normalization and winsorization applied before z-scoring. |
|---|---|

### Task 2.1 — Volatility Features

- Implement in src/features.py: rolling RV at 5, 10, 21, 63 trading day lookbacks using daily log returns
- Implement short-to-long vol ratio: RV_5 / RV_63
- All windows must end at least 1 trading day before the prediction week start
- Align to weekly index by taking the value on the last trading day of each week
- Verify: feature values should be positive. RV_5 should be more variable than RV_63. Ratio should spike in volatile regimes.

| **Output** | 5 volatility feature arrays aligned to weekly index |
|---|---|

### Task 2.2 — Return and Volume Features

- Download and save volume data to data/raw/volume.parquet via download_volume() in src/data.py
- Implement price momentum over 5 and 20 days (5-day and 20-day cumulative log return)
- Implement rolling mean volume over 5 and 20 days
- Implement volume ratio: 5-day mean volume / 20-day mean volume
- All windows lagged: values at week T use only data from trading days strictly before Monday_T

| **Output** | 5 return and volume feature arrays aligned to weekly index |
|---|---|

### Task 2.3 — Stack, Winsorize, and Cross-Sectional Normalization

- Stack all features into tensor of shape (num_weeks, num_stocks, num_features)
- For each week t and each feature f: winsorize cross-sectionally at the 1st and 99th percentile (clip values outside these bounds to the boundary values). This prevents outlier stocks from distorting normalization.
- After winsorizing: subtract mean across stocks, divide by std across stocks
- Save normalized tensor as data/features/features.parquet (reshaped to 2D for parquet storage) with metadata JSON recording shape and feature names
- Verify: at 10 random time steps, compute mean and std of each feature across stocks. Mean should be within 0.01 of 0. Std should be within 0.05 of 1. Assert these bounds programmatically. Also verify that max absolute feature value does not exceed ~4.0 after winsorization (confirms clipping is working).

| **Output** | data/features/features.parquet. Shape: (num_weeks × num_stocks, num_features) |
|---|---|

---

## Phase 3: Graph Construction

| **Goal** | Three graph edge structures, verified and visualized, ready for PyTorch Geometric. |
|---|---|

### Task 3.1 — Sector Graph

- Load sector assignments from data/raw/sector_history.json. Use the sector assignment for the start of each calendar year.
- Map each ticker in the universe to its sector per year. Handle tickers with dot notation (BRK.B → BRK-B).
- Build edge list per year: for each pair of stocks in the same sector in that year, add a bidirectional edge. Store as a dict mapping year → edge_index tensor.
- Save as data/graphs/sector_edges_by_year.parquet: columns [year, src, dst] as integer indices
- Verify: visualize graph using networkx for 2016 (pre-Real Estate split) and 2017 (post-split). Should show 10 clusters in 2016 and 11 clusters in 2017. Print node count per sector per year.

| **Output** | data/graphs/sector_edges_by_year.parquet |
|---|---|

### Task 3.2 — Correlation Graph Builder

- Implement build_correlation_graph(log_returns, date, lookback, threshold) in src/graphs.py
- Function takes the full log_returns DataFrame, a date, and config parameters
- Slices the lookback window ending at date, computes Pearson correlation matrix, applies threshold, returns edge_index as LongTensor
- Remove self-loops. Return only upper triangle edges for undirected graph (add reverse edges for PyG).
- Save sample edge lists for 3 dates (calm period, COVID peak, recent) to data/graphs/corr_sample/
- Verify: visualize sample graphs. Graph density should increase during COVID period. Print edge count for each sample date.

| **Output** | build_correlation_graph() function. Sample edge lists saved. |
|---|---|

### Task 3.3 — Granger Causality Computation

- Implement run_granger_tests() in src/graphs.py using statsmodels grangercausalitytests at lag=5
- Parallelize over all 213,082 ordered stock pairs using Python multiprocessing.Pool
- For each pair (A, B), extract the F-test p-value for lag 5
- Save full p-value matrix (462 × 462) as data/graphs/granger_pvalues.parquet
- **THIS TASK RUNS OVERNIGHT.** Initiate before end of workday. Verify multiprocessing is active before leaving.
- Next session: apply Bonferroni correction. Threshold = 0.05 / 213082. Draw directed edges where p < threshold.
- Save Granger edge index as data/graphs/granger_edges.parquet
- Verify: print number of directed edges. Visualize in-degree and out-degree distributions. Large-cap stocks should have higher out-degree.

| **Output** | data/graphs/granger_pvalues.parquet, data/graphs/granger_edges.parquet |
|---|---|

| **Risk** | If parallelization fails or computation exceeds 14 hours, fall back to computing Granger on a 100-stock subset (top 100 by market cap) with full documentation of this scope change in the paper. At 100 stocks, 9,900 ordered pairs are tested — manageable within 1–2 hours. Verify the resulting graph has at least 200 directed edges before proceeding; if Bonferroni is too conservative at this scale, switch to Benjamini-Hochberg FDR correction (statsmodels multipletests with method='fdr_bh'). |
|---|---|

### Task 3.4 — SAGEConv Directionality Verification

- Create a minimal test in 03_graphs.ipynb: construct a small directed graph (10 nodes, asymmetric edges) and verify that SAGEConv produces different outputs when edges are reversed
- If outputs are identical with reversed edges, SAGEConv is symmetrizing internally — switch to using flow='source_to_target' parameter explicitly and re-verify
- Document the aggregation direction used and confirm it is applied consistently in train_gnn() for the Granger variant
- Save this test and its output as a named cell in 03_graphs.ipynb for reproducibility and reviewer reference

| **Output** | Written verification of SAGEConv directionality behavior. Configuration documented in config.py as SAGE_FLOW. |
|---|---|

### Task 3.5 — Graph Verification and Visualization

- For all three graphs, compute and log: number of nodes, number of edges, graph density (edges / possible edges), degree distribution statistics
- Visualize all three graphs side by side for the same representative week using networkx with sector-colored nodes
- Confirm PyTorch Geometric Data objects can be constructed from each edge index without errors
- Save visualization as data/results/figures/graph_comparison.png for use in paper

| **Output** | Graph statistics table. graph_comparison.png. |
|---|---|

---

## Phase 4: Model Implementation and Training

| **Goal** | All six models trained, validated, and checkpointed. Validation metrics logged for all models. |
|---|---|

### Task 4.1 — HAR Baselines

- Implement HARModel (per-stock) class in src/models.py using sklearn LinearRegression
- Features: mean RV over past 5, 21, and 63 trading days (computed fresh from weekly_rv — do not use features.parquet for HAR)
- Fit one model per stock using training split only. Save all per-stock models.
- Implement HARPooled in src/models.py: single sklearn LinearRegression fit on all (stock, week) pairs in the training set with the same three features. This model generates predictions for all stocks using shared coefficients.
- Generate validation predictions for both HAR variants. Compute and log MSE and MAE across all stocks.
- Save predictions as data/results/har_val_preds.parquet and data/results/har_pooled_val_preds.parquet
- Verify: HAR R-squared on validation set should be 0.3–0.6 for most stocks. If lower, check feature alignment.

| **Output** | data/results/har_val_preds.parquet, data/results/har_pooled_val_preds.parquet. Validation MSE/MAE logged for both. |
|---|---|

### Task 4.2 — LSTM Baseline

- Implement LSTMModel in src/models.py: 2-layer LSTM, hidden_size=64, dropout=0.3, final linear layer
- Input: sequence of 4 weeks of features per stock, processed independently (no cross-stock information)
- Implement train_lstm() in src/train.py: chronological iteration over training weeks, Adam optimizer, early stopping. Set random seeds before initialization.
- Train on 50-stock subset first. Confirm loss decreases smoothly before scaling to full universe.
- Train on full 462-stock universe. Save best checkpoint to data/results/checkpoints/lstm_best.pt
- Log validation loss curve to data/results/lstm_val_loss.json
- Verify: validation loss should decrease monotonically for at least 15 epochs. If oscillating, reduce learning rate.

| **Output** | lstm_best.pt. Validation MSE logged. Loss curve saved. |
|---|---|

### Task 4.3 — GNN Implementation

- Implement GNNModel in src/models.py: 2-layer GraphSAGE (SAGEConv), hidden_size=64, dropout=0.3, linear output
- Model accepts node features and edge_index as inputs. Pass flow=config.SAGE_FLOW explicitly to SAGEConv.
- Implement train_gnn() in src/train.py: chronological iteration, dynamic graph construction at each step for correlation variant, annual sector graph lookup for sector variant, static graph lookup for Granger variant
- Implement PyG Data object construction helper in src/graphs.py: make_pyg_data(features_t, edge_index, target_t)
- Test full forward pass on 50-stock subset with each of the three edge structures. Confirm output shape (50,) and no NaN values.

| **Output** | GNNModel class and train_gnn() function. Forward pass verified on all three graph types. |
|---|---|

### Task 4.4 — Train GNN-Correlation

- Run train_gnn() with correlation graph. Threshold = 0.5 (initial). Use config.py CORR_THRESHOLD.
- Monitor validation loss. If not converging after 20 epochs, reduce learning rate to 0.0005.
- After initial training: retrain with θ ∈ {0.3, 0.5, 0.7}. Select threshold with best validation MSE. Record all three validation MSE values in corr_threshold_ablation.json — these are reported in the paper's ablation section.
- Save best checkpoint to data/results/checkpoints/gnn_corr_best.pt
- Log threshold comparison results to data/results/corr_threshold_ablation.json

| **Output** | gnn_corr_best.pt. Best threshold documented. |
|---|---|

### Task 4.5 — Train GNN-Sector

- Run train_gnn() with annual sector edge index, selecting the correct year's graph at each time step. Same hyperparameters as Task 4.4.
- Training should be faster than correlation variant as the graph changes only annually.
- Save checkpoint to data/results/checkpoints/gnn_sector_best.pt
- Compare validation MSE to GNN-Correlation. Note which sectors show largest prediction error.

| **Output** | gnn_sector_best.pt. Validation MSE logged. |
|---|---|

### Task 4.6 — Train GNN-Granger

- Run train_gnn() with directed Granger edge index. Confirm SAGEConv is using SAGE_FLOW direction from config.py.
- If Granger graph is very sparse (< 500 edges), experiment with relaxing Bonferroni to FDR correction (Benjamini-Hochberg). Document any correction method changes in the paper — note this as a secondary analysis prompted by sparsity, not as a hyperparameter tuned for performance.
- Save checkpoint to data/results/checkpoints/gnn_granger_best.pt

| **Output** | gnn_granger_best.pt. Validation MSE logged. Correction method documented. |
|---|---|

### Task 4.7 — Validation Summary

- Compile validation MSE and MAE for all 6 models into a single JSON: data/results/validation_summary.json
- Rank models by validation MSE. Document ranking.
- Verify all 6 checkpoints exist and can be loaded without errors.
- This is a go/no-go checkpoint. If no GNN variant beats HAR (per-stock) on validation MSE, review feature construction and graph density before proceeding to test evaluation.

| **Output** | validation_summary.json. Go/no-go decision documented. |
|---|---|

---

## Phase 5: Evaluation

| **Goal** | Final test set evaluation complete. All ML metrics, portfolio metrics, and statistical significance tests computed. All paper figures generated. |
|---|---|

### Task 5.1 — Test Set ML Evaluation

- Load all 6 model checkpoints
- Generate test set predictions for 2024–2025 from each model
- Compute MSE, MAE, R-squared pooled across all stocks and broken down by GICS sector
- Save all predictions as data/results/test_preds_{model}.parquet
- Build final ML metrics table: rows = models, columns = MSE / MAE / R² (pooled + per sector)
- Save table as data/results/ml_metrics_table.csv
- **THIS IS THE ONLY TIME THE TEST SET IS EVALUATED.** Do not modify any model after this point.

| **Output** | test_preds_*.parquet for all 6 models. ml_metrics_table.csv. |
|---|---|

### Task 5.2 — Portfolio Backtest

- Fetch weekly 3-month T-bill rates from FRED (series DTB3) for 2024–2025 using pandas-datareader. Align to weekly test period. Save as data/raw/tbill_rates.parquet.
- Implement build_portfolio() in src/portfolio.py: accepts model predictions, computes inverse-volatility weights, applies MAX_WEIGHT cap, normalizes to sum to 1. Log the maximum single-stock weight observed across the test period.
- Implement simulate_rebalancing() in src/portfolio.py: iterates weekly over test period, computes turnover, applies 10bps transaction cost, computes weekly portfolio return
- Compute Sharpe ratio as (annualized return − mean annualized T-bill rate) / annualized volatility
- Run simulation for all 6 models plus equal-weight benchmark
- Compute annualized return, annualized volatility, Sharpe ratio, maximum drawdown, average weekly turnover, and maximum single-stock weight for each
- Save portfolio returns series as data/results/portfolio_returns.parquet
- Build portfolio metrics table. Save as data/results/portfolio_metrics_table.csv

| **Output** | portfolio_returns.parquet. portfolio_metrics_table.csv. tbill_rates.parquet. |
|---|---|

### Task 5.3 — Statistical Significance Testing

This task implements formal hypothesis tests for all claimed performance improvements. Results are incorporated directly into the ML metrics table and portfolio metrics table as additional columns. No model improvement should be reported in the paper without an accompanying p-value and confidence interval.

**Implement src/significance.py with the following functions:**

#### 5.3.1 — Diebold-Mariano Test for Forecast Accuracy

The Diebold-Mariano (DM) test assesses whether two forecasting models have statistically different predictive accuracy. It is designed specifically for comparing time-series forecasts and accounts for serial correlation in forecast errors.

- For each GNN variant and each baseline (HAR per-stock, HAR pooled, LSTM), run a one-sided DM test with null hypothesis H₀: the GNN variant's MSE is equal to or worse than the baseline's MSE
- Use squared error loss differential: d_t = e²_baseline(t) − e²_GNN(t) where e_t is the mean squared prediction error across all stocks at week t
- Use Harvey, Leybourne & Newbold (1997) small-sample correction for the DM statistic, implemented in statsmodels or via manual computation
- Report: DM statistic, p-value (one-sided), and whether the null is rejected at α = 0.05
- Run this for pooled errors and separately for each GICS sector
- Save results as data/results/dm_test_results.csv: columns [model_a, model_b, dm_statistic, p_value, reject_null, sector]

```python
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
from scipy import stats

def diebold_mariano_test(errors_a, errors_b, h=1):
    """
    One-sided DM test: H0: model_a MSE >= model_b MSE (model_b is not better).
    errors_a, errors_b: arrays of per-week mean squared errors, shape (num_test_weeks,)
    h: forecast horizon (1 for one-step-ahead)
    Returns: dm_stat, p_value (one-sided, lower = model_b is better)
    """
    d = errors_a - errors_b  # positive d means model_b is better
    T = len(d)
    d_mean = np.mean(d)
    # Newey-West variance with h-1 lags
    gamma_0 = np.var(d, ddof=1)
    gamma_sum = sum(
        2 * (1 - k / h) * np.cov(d[k:], d[:-k])[0, 1]
        for k in range(1, h)
    ) if h > 1 else 0
    var_d = (gamma_0 + gamma_sum) / T
    dm_stat = d_mean / np.sqrt(var_d)
    # HLN small-sample correction
    k = ((T + 1 - 2*h + h*(h-1)/T) / T) ** 0.5
    dm_stat_corrected = dm_stat * k
    p_value = 1 - stats.t.cdf(dm_stat_corrected, df=T-1)  # one-sided
    return dm_stat_corrected, p_value
```

#### 5.3.2 — Block Bootstrap for Portfolio Metrics

Sharpe ratios and other portfolio statistics computed over a 104-week test window are not straightforward to test with parametric methods due to the autocorrelation in weekly returns. Use a circular block bootstrap to generate confidence intervals.

- Implement circular_block_bootstrap(returns_series, n_bootstrap=5000, block_size=8) in src/significance.py
- Block size of 8 weeks (~2 months) preserves weekly return autocorrelation structure
- For each bootstrap sample: resample blocks of weekly returns with replacement, compute the Sharpe ratio of the resampled series
- Report 95% confidence interval as the 2.5th and 97.5th percentiles of bootstrap distribution
- For pairwise comparison (GNN vs. equal-weight benchmark): bootstrap the *difference* in Sharpe ratios. Reject H₀ if the 95% CI for the difference excludes zero.
- Run for all 6 models plus equal-weight benchmark. Save results as data/results/bootstrap_sharpe_ci.csv: columns [model, sharpe_point_estimate, ci_lower, ci_upper, vs_ewb_diff_lower, vs_ewb_diff_upper]

```python
def circular_block_bootstrap(returns, n_bootstrap=5000, block_size=8, metric_fn=None):
    """
    Circular block bootstrap for time series.
    returns: array of shape (T,) — weekly portfolio returns
    metric_fn: function(returns) -> scalar metric. Default: annualized Sharpe.
    Returns: array of shape (n_bootstrap,) — bootstrap distribution of metric
    """
    T = len(returns)
    n_blocks = int(np.ceil(T / block_size))
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        start_indices = np.random.randint(0, T, size=n_blocks)
        resampled = np.concatenate([
            np.roll(returns, -start)[: block_size] for start in start_indices
        ])[:T]
        bootstrap_metrics.append(metric_fn(resampled))
    return np.array(bootstrap_metrics)
```

#### 5.3.3 — Multiple Comparison Correction Across Models

When comparing multiple models against the same baseline, the probability of a false positive increases with the number of comparisons. Apply Benjamini-Hochberg FDR correction across the set of DM test p-values for each baseline.

- Collect all DM test p-values for comparisons of the form (GNN variant vs. HAR per-stock), (GNN variant vs. HAR pooled), (GNN variant vs. LSTM)
- Apply statsmodels multipletests(pvalues, method='fdr_bh') to each group of comparisons
- Report both uncorrected and FDR-corrected p-values in data/results/dm_test_results.csv
- Add a column reject_fdr_corrected to the results table
- In the paper, use FDR-corrected p-values as the primary significance criterion; report uncorrected p-values in an appendix table

#### 5.3.4 — Significance Summary Table

- Compile a single summary table: data/results/significance_summary.csv
- Rows: all pairwise model comparisons (GNN vs. each baseline)
- Columns: dm_statistic, p_value_uncorrected, p_value_fdr_corrected, reject_fdr, sharpe_diff_ci_lower, sharpe_diff_ci_upper, sharpe_improvement_significant
- This table is directly referenced in the Experiments section of the paper and reproduced in full or in abbreviated form as a standalone exhibit

| **Output** | dm_test_results.csv, bootstrap_sharpe_ci.csv, significance_summary.csv. All statistical tests documented with test statistics, p-values, and confidence intervals. |
|---|---|

### Task 5.4 — Figure Generation

All figures must have consistent styling: Arial font, figure size (10, 6) or (12, 8), labeled axes with units, titles, and legends. Save all as 300dpi PNG to data/results/figures/.

- **fig1_graph_comparison.png** — side-by-side visualization of all three graphs for one representative week
- **fig2_rv_distribution.png** — histogram of weekly RV across all stocks and weeks. Shows non-normality.
- **fig3_val_loss_curves.png** — validation loss curves for LSTM and all three GNN variants
- **fig4_ml_metrics_bar.png** — bar chart of test MSE by model, sorted ascending. Include error bars showing 95% bootstrap CI around MSE estimates.
- **fig5_portfolio_cumulative.png** — cumulative portfolio return over 2024–2025 for all models vs equal-weight benchmark
- **fig6_sharpe_comparison.png** — bar chart of Sharpe ratio by model. Include error bars from bootstrap CI. Mark bars where improvement over equal-weight is statistically significant (FDR-corrected p < 0.05) with an asterisk.
- **fig7_sector_heatmap.png** — heatmap of per-sector MSE improvement of best GNN vs HAR
- **fig8_dm_significance.png** — heatmap of DM test p-values (FDR-corrected) for all pairwise model comparisons. Color scale: green = significant improvement, white = no significant difference, red = significant degradation.

| **Output** | 8 publication-ready figures in data/results/figures/ |
|---|---|

---

## Phase 6: Writing

| **Goal** | Complete paper draft ready for arXiv submission. Target: 8–10 pages in NeurIPS format. |
|---|---|

### Task 6.1 — Related Work

- Survey volatility forecasting literature: HAR (Corsi 2009), HAR extensions, GARCH family. 3–4 citations.
- Survey GNNs for finance: HATS, RSHN, Temporal GNN papers. Identify how this work differs. 4–5 citations.
- Survey graph construction for financial networks: correlation thresholding, MST, sector graphs. 2–3 citations.
- Write 1.5 pages. Every cited paper must be read, not just title-scanned.

| **Output** | Related work section draft. Bibliography .bib file started. |
|---|---|

### Task 6.2 — Methodology Section

- Write data subsection: universe construction criteria (including survivorship discussion), RV computation formula, train/val/test split rationale, GICS sector history handling
- Write features subsection: feature list with brief motivation for each group. Mention winsorization before normalization.
- Write graph construction subsection: one paragraph per graph type, covering construction procedure, directionality, stationarity assumptions, and key limitation. Include SAGEConv directionality verification note for Granger graph.
- Write model architecture subsection: GraphSAGE description, architecture table, training configuration. Note pooled HAR as additional baseline.
- Write evaluation framework subsection: ML metrics, portfolio construction (with weight cap and T-bill risk-free rate), transaction cost modeling, significance testing framework
- Target: 3 pages

| **Output** | Methodology section draft. |
|---|---|

### Task 6.3 — Experiments Section

- Write experimental setup paragraph: hardware, software versions, random seeds, reproducibility statement, number of hyperparameter configurations evaluated on validation set
- Write ML results subsection: present ml_metrics_table with FDR-corrected DM test p-values. Discuss which graph type performs best and on which sectors.
- Write portfolio results subsection: present portfolio_metrics_table with bootstrap Sharpe CIs. Discuss Sharpe ratios relative to equal-weight benchmark and to each other. Note maximum single-stock weight statistics.
- Write ablation subsection: correlation threshold sensitivity results (three θ values and their validation MSE), impact of cross-sectional normalization and winsorization
- Write significance subsection: summarize significance_summary.csv results. Note which improvements survive FDR correction and which do not. Be honest if no improvement is statistically significant after correction.
- Target: 2.5 pages

| **Output** | Experiments section draft. |
|---|---|

### Task 6.4 — Introduction and Abstract

- Write introduction last (after results are known): motivate volatility prediction, state the graph construction gap, preview your findings, state contributions as 3 explicit bullet points
- Write abstract: 150 words. Problem, method, key quantitative result, implication. No vague language. Include whether claimed improvements are statistically significant.
- Target: 1 page

| **Output** | Introduction and abstract drafts. |
|---|---|

### Task 6.5 — Conclusion, Limitations, Future Work

- Write conclusion: summarize findings, state what they imply for practitioners
- Write limitations honestly: Granger causality sensitivity to lag selection, static feature set, single equity market, no fundamental data, survivorship bias in universe construction, multiple validation set consultations, stationarity assumption in static Granger graph
- Write future work: temporal GNN component, macro-conditioned graphs, multi-market extension, dynamic Granger graph with rolling window recomputation
- Target: 0.5 page

| **Output** | Conclusion section draft. |
|---|---|

### Task 6.6 — Revision and Submission

- Read full paper aloud. Fix all awkward sentences.
- Verify every number in every table matches the corresponding output file in data/results/
- Verify every figure has a self-contained caption that explains what is shown without requiring the surrounding text
- Verify all significance claims cite the correct test, statistic, and corrected p-value
- Send to advisor or peer reviewer. Incorporate feedback.
- Format for arXiv: compile LaTeX (NeurIPS workshop format), verify PDF renders correctly
- Submit to arXiv under cs.LG and q-fin.ST cross-listing
- Add arXiv link to CV and LinkedIn immediately upon confirmation of arXiv ID

| **Output** | Paper submitted to arXiv. arXiv ID on CV. |
|---|---|

---

# Risk Register

| **Risk** | **Severity** | **Mitigation** | **Trigger** |
|---|---|---|---|
| Granger computation exceeds available time or crashes | HIGH | Fall back to 100-stock subset (top market cap). Document in paper as scope limitation. Verify resulting graph has ≥200 edges before proceeding. | Still running after 14 hours |
| No GNN variant beats HAR on validation MSE | HIGH | Review feature construction for lookahead bias. Adjust correlation threshold. Review graph density. | After Task 4.7 go/no-go checkpoint |
| Lookahead bias discovered after models trained | CRITICAL | Fix pipeline and retrain from scratch. This invalidates all prior results. | Any time before submission |
| Colab session timeout during long training run | MEDIUM | Save checkpoint every 5 epochs. Resume from last checkpoint. | During Task 4.4–4.6 |
| Granger graph too sparse after Bonferroni correction | MEDIUM | Try Benjamini-Hochberg FDR correction instead. Document correction method chosen. | After Task 3.3 edge count check |
| SAGEConv symmetrizes directed edges internally | MEDIUM | Verify in Task 3.4. If confirmed, set flow='source_to_target' explicitly. If behavior cannot be verified, report Granger results as undirected with a note. | During Task 3.4 |
| No improvement survives FDR correction in significance tests | MEDIUM | Report results honestly. Reframe contribution around methodology and reproducibility rather than performance claims. Smaller workshops often accept negative or null results with strong methodology. | After Task 5.3 |
| T-bill rate data unavailable from FRED | LOW | Fall back to using 5% annualized (approximate 2024–2025 average) as a fixed rate. Document this in the paper. | During Task 5.2 |
| Writing takes longer than allocated time | LOW | Methodology section can be drafted in parallel with model training. Deprioritize related work polish — reviewers care most about methods and results. | End of Phase 5 |

---

# Success Criteria

The project is considered complete and ready for arXiv submission when all of the following criteria are met:

### Technical

- All six models trained and evaluated on the held-out test set (2024–2025) exactly once
- At least one GNN variant outperforms HAR on test MSE
- No lookahead bias present — confirmed by written audit trail in 01_data.ipynb
- All results reproducible from saved checkpoints and data artifacts on Google Drive
- Random seeds documented in config.py and set at the start of all training scripts

### Portfolio

- At least one model produces a higher Sharpe ratio than the equal-weight benchmark after transaction costs
- Portfolio simulation covers full test period with weekly rebalancing
- Sharpe ratios computed using actual T-bill risk-free rate, not zero
- Maximum single-stock weight reported for all models

### Statistical Validity

- Diebold-Mariano test run for all GNN-vs-baseline comparisons
- FDR correction applied across multiple comparisons
- Bootstrap confidence intervals reported for all Sharpe ratio estimates
- All significance test results (including non-significant ones) reported transparently in the paper

### Paper

- Complete draft of all sections: abstract, introduction, related work, methodology, experiments, conclusion
- All figures are publication quality: labeled axes, consistent styling, self-contained captions
- All numbers in tables verified against output files
- All performance claims accompanied by p-values and confidence intervals
- Paper submitted to arXiv under cs.LG and q-fin.ST

### Codebase

- All src/ modules documented with function-level docstrings
- config.py is the single source of truth for all hyperparameters including RANDOM_SEED, WINSORIZE_CLIP, MAX_WEIGHT, SAGE_FLOW
- README.md explains how to reproduce all results from scratch

---

# Paper Outline

Target: 8–10 pages in NeurIPS workshop format. Submitted to FinML Workshop at NeurIPS 2026.

| **Section** | **Target Length** | **Key Content** |
|---|---|---|
| Abstract | 150 words | Problem, method, key result with significance statement, implication |
| 1. Introduction | 1 page | Motivation, gap, 3 explicit contributions, results preview |
| 2. Related Work | 1.5 pages | Volatility forecasting, GNNs for finance, graph construction methods |
| 3. Methodology | 3 pages | Data (with universe construction details), features, three graph types, GNN architecture, portfolio construction, significance testing framework |
| 4. Experiments | 2.5 pages | ML metrics table (with DM test p-values), portfolio metrics table (with bootstrap CIs), ablation studies, significance summary |
| 5. Conclusion | 0.5 pages | Findings, limitations (including survivorship bias and validation set usage), future work |
| References | ~0.5 pages | 10–15 citations |

## Key Citations to Include

- Corsi (2009) — HAR model. The primary volatility forecasting baseline.
- Andersen & Bollerslev (1998) — Realized volatility foundation.
- Hamilton et al. (2017) — GraphSAGE. The GNN architecture used.
- HATS (Kim & Han 2019) — GNN for stock prediction. Most cited predecessor.
- Granger (1969) — Granger causality original paper.
- Engle (1982) — ARCH model. Context for volatility clustering.
- Diebold & Mariano (1995) — Forecast comparison test. Required citation for significance testing methodology.
- Harvey, Leybourne & Newbold (1997) — Small-sample DM correction. Required citation if HLN statistic is used.
- Benjamini & Hochberg (1995) — FDR correction. Required citation if BH correction is applied.
- At least 2 recent (2022+) FinML workshop papers to show currency of approach.
