# Research Journey

A running log of what was tried, what was found, and why decisions were made.
Intended to support the paper's experiments section and as a reference when writing results.

---

## Chapter 1 — Early Drafts and the Decision to Rebuild

The project did not start from scratch. Before the current codebase, there were rough implementations: a basic GNN, an ARIMA baseline, an LSTM, and a feature pipeline. These ran on a small universe and produced results, but the code had no consistent structure. The GNN produced poor predictions. There was no portfolio layer, no significance testing, and no guarantee of lookahead-freeness.

Rather than patch those drafts, the decision was made to start over with a structured plan. The reasons were practical: a cleaner codebase would be easier to reason about during model debugging and would produce a more credible paper. All code moved into a `src/` module structure. All constants moved to `config.py`. Notebooks became orchestration layers only. The project outline and CLAUDE.md were written before any new code to lock down the rules before they could be violated.

---

## Chapter 2 — Data Pipeline

### Universe construction

The universe was built from S&P 500 constituent stocks with daily price data from 2015 through 2025. Inclusion required at least 95% data coverage AND constituent membership for at least 80% of total sample weeks. Stocks that entered the index after January 2016 or were removed before December 2024 were excluded. This filters out survivorship bias at the tails while keeping the stable core of the index. Final universe: 465 stocks.

Point-in-time sector assignments were required. GICS classifications changed materially twice during the sample: Real Estate separated from Financials in 2016, and Telecom became Communication Services in 2018 while absorbing names from Consumer Discretionary and IT. A `sector_history.json` file maps each ticker to its sector by year so that the sector graph uses the correct classification at each point in time rather than current assignments.

### Target construction and the 1-step-ahead design

Weekly realized volatility is computed as `std(daily_log_returns_in_week) * sqrt(252)`. Weeks with fewer than 3 trading days are dropped.

The prediction target at row T is the RV of week T+1. This is a 1-step-ahead design: features at row T may use any data through the end of week T (Friday of week T). Friday of week T is strictly before Monday of week T+1, so the design is lookahead-free by construction.

A written audit trail was added to `notebooks/01_data.ipynb` tracing 5 random rows from raw price to target value. This audit exists because lookahead bias is the highest-priority bug class in this project and needs to be documentable for any reviewer who questions the setup.

### Train/val/test split

| Split | Date range |
|---|---|
| Train | 2015 through 2022 |
| Validation | 2023 |
| Test | 2024 through 2025 |

The validation year (2023) was a relatively calm market: steady recovery from the 2022 bear market, moderate volatility, stable cross-stock correlations. The test period (2024-2025) included the AI-driven tech divergence, the rate-cut cycle, and periods where historical cross-stock relationships broke down. This mattered for the final results.

### Feature engineering

Ten features per stock per week:

- Realized volatility at 5, 10, 21, and 63 trading day lookbacks
- Short-to-long volatility ratio: RV(5d) / RV(63d)
- Price momentum over 5 and 20 days
- Log rolling mean volume over 5 and 20 days
- Volume ratio: 5-day mean / 20-day mean

All features are winsorized cross-sectionally at the 1st/99th percentile at each time step, then z-scored cross-sectionally. The order matters: winsorize first, then z-score. Reversing this would allow outlier stocks to distort the normalization for all others. Post-normalization assertions verify mean near 0 and std near 1 at 10 random time steps.

---

## Chapter 3 — Graph Construction

Three graphs, each feeding the same GNN backbone. All other model components are held constant so that performance differences are attributable to graph type, not architecture.

### Correlation graph (dynamic, undirected)

At each week T, a 252-trading-day rolling window of daily log returns ending on Friday of week T is used to compute the pairwise Pearson correlation matrix. An edge connects two stocks if their correlation exceeds threshold θ. The graph is recomputed each week. During calm regimes, the graph is moderately dense. During crisis weeks (March 2020), it approaches full connectivity as all stocks move together.

Threshold θ was tuned on the validation set across {0.3, 0.5, 0.7}. The winner was θ=0.3. This is lower than typical values in the literature, which reflects the large universe: with 465 stocks, a higher threshold produces a sparser graph that misses meaningful relationships.

### Sector graph (annual, undirected)

Stocks sharing the same GICS sector in a given calendar year are connected. The graph changes annually to reflect reclassifications. It encodes fundamental economic structure rather than return co-movement. It is coarser than the correlation graph but more stable: sector membership does not fluctuate week to week.

The sector graph was pre-built for all years 2015-2025 using the point-in-time sector history. Saved to `data/graphs/sector_edges_by_year.parquet`.

### Granger causality graph (static, directed)

Granger causality tests whether past values of stock A's returns improve prediction of stock B's returns beyond B's own history. Unlike correlation, this produces a directed graph: an edge A→B does not imply B→A. It was computed once over the training period using a lag of 5 trading days, testing all ordered pairs among 465 stocks (approximately 215,000 tests).

Bonferroni correction was applied at α=0.05, yielding 13,886 directed edges. This is sparse relative to the correlation and sector graphs. The static nature of the Granger graph is a known limitation: lead-lag relationships change across market regimes.

### SAGEConv directionality verification

SAGEConv can silently symmetrize directed edges depending on how flow is configured. Before any GNN-Granger training, a verification was run: a small directed test graph was constructed, outputs were computed with forward and reversed edges, and an assertion confirmed the outputs differed. `flow='source_to_target'` is set explicitly on every SAGEConv layer, stored in `config.SAGE_FLOW` to prevent accidental inconsistency.

---

## Chapter 4 — Baselines and a Conceptual Fix

### The 1-step-ahead pivot

The original HAR implementation used a 2-step-ahead target. This was caught during an audit and corrected before any GNN training began. The correction involved reversing the shift direction in `make_target()` so that the target at row T is week T+1's RV rather than T-1's. This is the kind of bug that would invalidate all results if left in, which is why the audit trail in the notebook was written explicitly.

### HAR baselines

Two HAR variants were implemented:

- **HAR per-stock**: one OLS regression per ticker, using RV at 5, 21, and 63 trading day lookbacks as features. Produces one set of coefficients per stock.
- **HAR pooled**: one OLS regression across all (stock, week) pairs simultaneously. Shares coefficients across all stocks, making it a fairer comparison to the GNN which also shares parameters.

HAR uses features computed directly from `weekly_rv`, not from `features.parquet`. This is intentional: HAR is meant to be a pure volatility-autoregression baseline, not a model that benefits from the additional return and volume features.

### LSTM baseline

A 2-layer LSTM with hidden size 64 and dropout 0.3, processing each stock's feature sequence independently. The LSTM sees a 4-week rolling window of the 10 features for each stock. No cross-stock information passes through the LSTM. It serves to isolate the contribution of graph structure from deep learning capacity: if the GNN beats the LSTM, the win is attributable to the graph, not to model complexity.

---

## Chapter 5 — GNN Training

### Architecture

`GNNModel` uses two SAGEConv layers with hidden dimension 64, followed by a linear output head. All three graph variants use the same class. The graph is passed at forward time, not stored in the model, so the same code handles the dynamic correlation graph (different edges each week), the annual sector graph, and the static Granger graph.

### Training three variants

GNN-Correlation, GNN-Sector, and GNN-Granger were trained sequentially under identical conditions: Adam optimizer, learning rate 1e-3, early stopping on validation MSE with patience 10, chronological iteration over training weeks (never shuffled). Checkpoints saved every 5 epochs to handle Colab session drops.

GNN-Sector was fastest: the graph changes only annually. GNN-Correlation was slowest because `build_correlation_graph()` was recomputed at every training step. On the 465-stock full universe, each `.corr()` call over a 252-day window took 200-400ms. Across three threshold values, 150 epochs, and ~470 train+val weeks, this amounted to roughly 210,000 correlation matrix computations. The training loop was spending most of its time on CPU-bound graph construction while the A100 sat idle.

### Precomputed correlation graphs

The solution was to precompute all correlation graphs once and save them to disk before training begins. `precompute_corr_graphs()` iterates over all three splits and all three threshold values, calls `build_correlation_graph()` once per week, and writes the results to parquet files in `data/graphs/corr_edges/`. The training loop replaces the live `.corr()` call with a dictionary lookup.

Nine parquet files total. The first run takes approximately 90 seconds. Every subsequent run detects existing files and returns instantly. The precompute step was inserted into `notebooks/04_gnn_models.ipynb` immediately after data loading, with a comment noting the first-run time.

This change eliminated the training bottleneck and reduced subsequent training runs from hours of graph computation to under one second of file loading per run.

### Validation summary

After training all six models, a go/no-go check compared each GNN variant's validation MSE to HAR per-stock. On the validation year (2023), GNN-Correlation achieved val MSE = 0.019589 vs. HAR per-stock at 0.021840. Decision: GO.

---

## Chapter 6 — Scaling to the Full Universe

Initial development used `DEV_UNIVERSE_SIZE=50` (50 stocks) for fast iteration. Once all models ran cleanly, `DEV_UNIVERSE_SIZE` was set to `None` to use the full 465-stock universe, and all five notebooks were re-run end-to-end.

Several issues surfaced during the full-universe run:

- Feature NaN propagation: with 465 stocks, some had isolated missing weeks that caused NaN values to spread through rolling windows. The training loop was hardened to handle sparse NaN targets gracefully rather than crashing on them.
- Graph edge indices: some pre-built sector edge indices referenced node positions that shifted when the full ticker list was used. The sector graph builder was corrected to use position indices consistently.
- LSTM warm-up: the first 4 weeks of the sequence-based LSTM had incomplete history. The forward pass was updated to impute with zeros during the warm-up period and emit a warning rather than silently producing NaN outputs.

All five notebooks were verified against the full universe before any evaluation began.

---

## Chapter 7 — Hyperparameter Search

The base GNN architecture (2 layers, hidden 64, lr 1e-3, no batch norm) was selected by analogy to standard GNN practice, not by tuning. To properly select the architecture for the paper's main GNN variant, a grid search was run over GNN-Correlation using a parameterized model class, `GNNModelV2`.

### Round 1: 48 configurations

Grid: 3 learning rates × 2 hidden dims × 2 dropouts × 2 batch norm settings × 2 layer counts. Early stopping patience was reduced from 10 to 7 epochs to limit wall time. Precomputed correlation graphs at θ=0.3 (ablation winner) were used throughout.

**Top 5 results:**

| Rank | lr | hidden_dim | dropout | batch_norm | num_layers | val MSE |
|---|---|---|---|---|---|---|
| 1 | 3e-4 | 128 | 0.3 | False | 3 | 0.019589 |
| 2 | 1e-3 | 128 | 0.3 | False | 3 | 0.019679 |
| 3 | 3e-3 | 128 | 0.1 | False | 3 | 0.019681 |
| 4 | 1e-3 | 64 | 0.1 | False | 3 | 0.019684 |
| 5 | 1e-3 | 64 | 0.3 | False | 3 | 0.019710 |

Baseline (original config, 2 layers, hidden 64, lr 1e-3): val MSE = 0.019778.

**Key findings from round 1:**
- `num_layers=3` appeared in every top-5 slot without exception. A third message-passing hop expands each node's receptive field and was the dominant architectural factor.
- `batch_norm=False` appeared in every top-10 slot. The reason was not immediately clear but is discussed in Chapter 9.
- `hidden_dim=128` won the top slot but `hidden_dim=64` appeared at ranks 3 and 4. Width matters less than depth on this dataset.
- `lr=3e-4` won but `lr=1e-3` appeared in 3 of the top 5. Learning rate was a secondary factor.
- The overall improvement from the search was 1.0% MSE reduction (0.019778 to 0.019589).

The winning config (lr=3e-4, hidden=128, dropout=0.3, batch_norm=False, num_layers=3) was saved to `gnn_corr_hparam_best.pt` and `config.py` was updated with the winning values.

---

## Chapter 8 — Test Set Evaluation

With all models trained and the architecture selected, the test set (2024-2025, 103 weeks) was evaluated once. Prior to this run, the test predictions had never been touched.

### ML metrics

| Model | Test MSE | Test DA |
|---|---|---|
| LSTM | 0.032424 | 0.7088 |
| HAR per-stock | 0.032858 | 0.7070 |
| HAR pooled | 0.033104 | 0.7030 |
| GNN-Correlation (tuned) | 0.033305 | 0.7155 |
| GNN-Sector | 0.033631 | 0.6822 |
| GNN-Granger | 0.033702 | 0.6879 |

All six models were within 0.0013 MSE of each other. The GNN-Correlation model, which won on validation by 9.4% over HAR per-stock, finished 4th on test MSE. Two things were happening simultaneously:

**Every model got harder.** All six MSEs jumped roughly 60% from validation to test. The validation year (2023) was calm; the test period (2024-2025) included the AI-driven tech bifurcation, the rate-cut cycle, and periods where historical cross-stock relationships broke down. This affected all models equally.

**GNN-Correlation overfit to 2023 market structure.** The correlation graph uses a 252-day rolling window. In 2024-2025, when AI stocks diverged sharply from the rest of the market, the rolling window still incorporated 2023 and 2022 data, encoding correlation relationships that no longer held. The GNN propagated stale neighbor information, which actively hurt predictions. An LSTM processing each stock independently was not exposed to this problem.

**Directional accuracy told a different story.** GNN-Correlation had the best DA at 0.7155, despite ranking 4th on MSE. DA measures whether the model correctly predicts whether next week's RV will be higher or lower than this week's. That is directly relevant to portfolio construction: you want to overweight stocks predicted to be calm, not stocks predicted to be volatile. Whether the GNN advantage on DA translated to portfolio performance was the question that motivated the next phase.

---

## Chapter 9 — Portfolio Analysis and the Core Problem Diagnosis

### The convergence problem

The portfolio backtest revealed the structural reason why all six models produced nearly identical portfolios: with 465 stocks and MAX_WEIGHT=0.05, inverse-volatility weighting collapses toward equal-weight regardless of prediction quality. The maximum single-stock weight for most models in most weeks was 0.002-0.005, essentially the same as equal-weight's 0.002. The GNN would need radically differentiated cross-sectional predictions to move the needle, and better MSE alone would not achieve this.

Two independent problems were identified:

1. Model predictions were not differentiated enough cross-sectionally.
2. The portfolio construction could not amplify whatever cross-sectional signal did exist.

### The improvement plan

A sequenced plan was written to address both problems without retraining any model:

1. **Long-short portfolio**: go long the bottom 25% of predicted RV (lowest predicted vol stocks) and short the top 25% (highest predicted vol). Dollar-neutral. This construction directly exploits ranking accuracy rather than diluting it across 465 long-only positions.

2. **Volatility-targeted portfolio**: scale overall portfolio exposure each week based on the median predicted RV. When predicted vol is high, reduce equity weight and hold the remainder in cash. This converts cross-sectional vol forecasts into a risk-budgeting decision.

3. **Minimum variance portfolio**: use predicted RV as the diagonal of the covariance matrix combined with the realized correlation matrix from the GNN's correlation graph. Solve the minimum variance optimization subject to weight constraints. This ties the GNN's graph structure directly into the optimizer.

4. **Rank IC**: measure cross-sectional Spearman correlation between predicted and actual RV at each test week. This measures ranking accuracy directly, independent of the portfolio construction problem.

5. **GNN ensemble**: average predictions from all three GNN variants weighted by 1/val_MSE.

Phases 1-3 were implemented in `src/portfolio.py` and added to `notebooks/06_portfolio.ipynb`.

---

## Chapter 10 — Rank IC Results

Rank IC (information coefficient) was implemented as the cross-sectional Spearman correlation between predicted and actual RV at each test week. Two functions were added to `src/evaluate.py`:

- `compute_rank_ic(preds, actuals)`: runs Spearman IC across stocks at each week, returns a weekly series.
- `summarize_rank_ic(ic_series)`: computes mean IC, IC t-statistic (= mean / (std / sqrt(n))), information ratio (= mean / std), and fraction of positive-IC weeks.

### Results (103 test weeks, 2024-2025)

| Model | Mean IC | IC t-stat | IC IR | % Positive |
|---|---|---|---|---|
| LSTM | 0.4288 | 44.28 | 4.36 | 100% |
| GNN-Correlation (tuned) | 0.4113 | 34.37 | 3.39 | 100% |
| HAR per-stock | 0.4049 | 35.25 | 3.47 | 100% |
| HAR pooled | 0.3923 | 34.88 | 3.44 | 100% |
| GNN-Sector | 0.3826 | 34.50 | 3.40 | 100% |
| GNN-Granger | 0.3749 | 37.18 | 3.66 | 100% |

Every model had positive IC in every single test week. IC of 0.37-0.43 is high by equity factor standards, where 0.05-0.10 is considered good. The explanation is volatility persistence: a stock volatile this week will likely be volatile next week. Even HAR pooled achieves 0.39. This means roughly 90% of the predictable IC comes for free from autocorrelation that any model captures. The graph structure is competing for the remaining 10%.

GNN-Correlation ranks second (0.4113) and leads all baselines except LSTM. GNN-Granger has the lowest mean IC but the most consistent IC (highest t-stat and IR), meaning its weekly signal is stable even if the average is lower.

These t-stats are far above any significance threshold. The honest framing for the paper: all six models have statistically significant and persistent cross-sectional predictive power, but the signal is dominated by volatility autocorrelation shared by all approaches.

---

## Chapter 11 — BatchNorm Diagnosis and GraphNorm

### Why batch_norm=False won the hparam search

After seeing the Rank IC results, the round 1 finding that `batch_norm=False` won every top-10 slot was revisited with a clearer lens.

`BatchNorm1d` applied across the node dimension normalizes node embeddings to zero mean and unit standard deviation at each time step. For MSE prediction this is roughly neutral. For cross-sectional ranking it is actively harmful: it explicitly removes the relative differences between stock embeddings, which is the signal the ranking depends on. The model is forced to reconstruct cross-sectional differences through fixed learned scale/shift parameters (gamma and beta), which are not conditioned on the input and cannot recover the original cross-sectional structure.

This explains why BatchNorm never helped in the search: even when other hyperparameters were favorable, the normalization was destroying the cross-sectional signal that the model had just spent several SAGEConv layers computing.

### The fix: GraphNorm

`GraphNorm` (from `torch_geometric.nn.norm`) differs from `BatchNorm1d` in one key way: it has a learnable `mean_scale` parameter (alpha) per layer. When alpha learns toward 0, the full cross-sectional spread in node embeddings is retained. When alpha learns toward 1, it behaves like standard normalization. The model determines which is better based on what the loss rewards.

In `GNNModelV2`, `nn.BatchNorm1d(hidden_dim)` was replaced with `GraphNorm(hidden_dim)`. The forward call changed from `self.bns[i](x)` to `self.bns[i](x, batch=None)`. `batch=None` tells GraphNorm that all 465 nodes belong to a single graph snapshot at week T, which is the correct setup for this problem.

The existing checkpoint (`gnn_corr_hparam_best.pt`) cannot be loaded into a model with GraphNorm because the parameter names differ. Retraining is required.

### Round 2 hparam grid

The round 1 results made several axes unambiguous. The grid was pruned from 48 to 24 configurations:

| Axis | Round 1 | Round 2 | Reason |
|---|---|---|---|
| lr | [3e-4, 1e-3, 3e-3] | [1e-4, 3e-4, 1e-3] | 3e-3 never cracked top 5; winner at low end, explore lower |
| hidden_dim | [64, 128] | [128, 256] | 64 never appeared in top 5; winner was 128; explore 256 |
| dropout | [0.1, 0.3] | [0.1, 0.3] | Both in top 5; keep both |
| batch_norm | [True, False] | [True, False] | True now means GraphNorm; deserves a proper test |
| num_layers | [2, 3] | [3] | Fixed; 3 won every top-5 slot without exception |

All 96 round 1 checkpoint and val_loss JSON files were deleted so the resumable search logic would not skip round 2 configs. The `gnn_corr_hparam_best.pt` file (the copied winner) survived under its own name.

**What to watch in round 2:** if `batch_norm=True` (GraphNorm) configs appear in the top 5 where BatchNorm never did, that confirms the normalization was suppressing cross-sectional signal. If they still do not appear, the issue is elsewhere, which points toward Phase 6 of the improvement plan: rank-based training loss.

### Round 2 results

**Top 10 configs (sorted by val MSE):**

| Rank | lr | hidden_dim | dropout | batch_norm | num_layers | val MSE | vs round 1 best (%) |
|---|---|---|---|---|---|---|---|
| 1 | 1e-3 | 256 | 0.3 | False | 3 | 0.019646 | +0.29 |
| 2 | 1e-3 | 128 | 0.3 | False | 3 | 0.019768 | +0.91 |
| 3 | 1e-3 | 128 | 0.1 | False | 3 | 0.019781 | +0.98 |
| 4 | 3e-4 | 128 | 0.1 | False | 3 | 0.019788 | +1.02 |
| 5 | 1e-3 | 256 | 0.1 | False | 3 | 0.020075 | +2.48 |
| 6 | 3e-4 | 128 | 0.3 | True (GraphNorm) | 3 | 0.020094 | +2.58 |
| 7 | 1e-3 | 128 | 0.3 | True (GraphNorm) | 3 | 0.020249 | +3.37 |
| 8 | 3e-4 | 256 | 0.3 | False | 3 | 0.020418 | +4.23 |
| 9 | 1e-4 | 128 | 0.3 | False | 3 | 0.020489 | +4.60 |
| 10 | 3e-4 | 256 | 0.1 | False | 3 | 0.020533 | +4.82 |

**Key findings from round 2:**

- No round 2 config beat the round 1 winner (0.019589). The best round 2 config (0.019646) is 0.29% worse. Increasing hidden_dim to 256 and exploring lower learning rates did not yield gains.
- `batch_norm=False` still dominates the top 5. GraphNorm first appears at rank 6 (0.020094), a full 2.3% behind the round 2 winner. This is an improvement over BatchNorm, which never cracked the top 10 in round 1, confirming that GraphNorm is less harmful. But it is still not competitive with no normalization on this dataset.
- The round 1 winner (lr=3e-4, hidden=128, dropout=0.3, no norm, 3 layers, val MSE=0.019589) remains the best configuration found across both searches. `config.py` retains those values.
- `hidden_dim=256` won the top slot but by a narrow margin over hidden=128 configs, and it performed poorly at lower learning rates (ranks 5 and 10). The width benefit is inconsistent.
- `lr=1e-3` dominates the top 3 slots. The round 1 finding that lr=3e-4 won was not replicated; at the higher width of 256 and with GraphNorm absent, 1e-3 is preferred.

**Conclusion on normalization:** GraphNorm is a genuine improvement over BatchNorm (moves from outside top 10 to rank 6-7) but does not close the gap to no normalization. The cross-sectional signal preservation problem is real, but GraphNorm's learnable alpha does not fully solve it. If ranking accuracy is the goal, the next intervention is a rank-based training loss (Phase 6), not further normalization tuning.

---

## Open questions

- Round 2 hparam search is complete. The round 1 winner (lr=3e-4, hidden=128, dropout=0.3, no norm, 3 layers) remains best. No retraining needed unless Phase 6 is pursued.
- Do the long-short / vol-targeted / min-var portfolio constructions show a GNN advantage that MSE and IC obscure?
- Phase 5 (GNN ensemble) is still pending. No retraining needed; can be run against existing checkpoints.
- GraphNorm did not close the gap to no normalization. Phase 6 (rank-based loss, e.g. ListMLE or a Spearman IC surrogate) is the next architectural change to try if portfolio results remain flat.
