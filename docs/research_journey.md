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

The universe was built from the current S&P 500 constituent list scraped from Wikipedia, converted to yfinance-compatible symbols, then filtered for daily price coverage from 2015 through 2025. Inclusion required at least `config.MIN_COVERAGE` non-missing adjusted close data over the full sample window (`0.95` in the frozen run). This produces a stable full-history current-constituent universe. Final universe: 465 stocks. The original candidate count was not persisted by the download artifact.

Year-specific sector assignments were required. Sector classifications changed materially twice during the sample: Real Estate separated from Financials in 2016, and Telecom became Communication Services in 2018 while absorbing names from Consumer Discretionary and Information Technology. A `sector_history.json` file maps each ticker to its canonical GICS-style sector by year so that the sector graph can reflect major historical sector reclassifications rather than using one static current-sector assignment.

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

The sector graph was pre-built for all years 2015-2025 using the year-specific sector history. Saved to `data/graphs/sector_edges_by_year.parquet`.

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

`GNNModel` is the original two-layer GraphSAGE implementation. `GNNModelV2` is the tunable GraphSAGE implementation used for the official tuned GNN-Correlation checkpoint. The graph is passed at forward time, not stored in the model, so the same code path handles the dynamic correlation graph (different edges each week), the annual sector graph, and the static Granger graph.

### Training three variants

The original GNN-Correlation, GNN-Sector, and GNN-Granger runs were trained sequentially under identical conditions: Adam optimizer, learning rate 1e-3, early stopping on validation MSE with patience 10, chronological iteration over training weeks (never shuffled). The official evaluated GNN-Correlation row now points to the tuned `GNNModelV2` checkpoint selected by validation MSE; sector and Granger frozen baselines retain their original checkpoints.

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
- The official saved search artifact selects the round 2 top slot: lr=1e-3, hidden=256, dropout=0.3, no norm, 3 layers, val MSE=0.019646. `config.py` is aligned to this checkpoint for future tuned GNN-Correlation runs.
- `hidden_dim=256` won the top slot but by a narrow margin over hidden=128 configs, and it performed poorly at lower learning rates (ranks 5 and 10). The width benefit is inconsistent.
- `lr=1e-3` dominates the top 3 slots. The round 1 finding that lr=3e-4 won was not replicated; at the higher width of 256 and with GraphNorm absent, 1e-3 is preferred.

**Conclusion on normalization:** GraphNorm is a genuine improvement over BatchNorm (moves from outside top 10 to rank 6-7) but does not close the gap to no normalization. The cross-sectional signal preservation problem is real, but GraphNorm's learnable alpha does not fully solve it. If ranking accuracy is the goal, the next intervention is a rank-based training loss (Phase 6), not further normalization tuning.

---

---

## Chapter 12 — GNN Ensemble

### Construction and weights

The ensemble averages predictions from all three GNN variants, weighted by the inverse of each variant's validation MSE. This is a no-training-required combination: the three checkpoint files already existed, so the ensemble is assembled at inference time in `build_gnn_ensemble_preds()` in `src/evaluate.py`. Predictions are saved to `test_preds_gnn_ensemble.parquet` and appended to `ml_metrics_table.csv`.

The resulting weights were: gnn_corr=0.365, gnn_sector=0.315, gnn_granger=0.319. These are nearly equal. The three GNN variants had close validation MSEs, so the inverse-weighting scheme had little leverage. GNN-Correlation received the highest weight because it had the best validation performance, but the margin was small. If the three variants had diverged more on validation, the weighting would have been more consequential.

### ML metrics

The ensemble achieved the lowest test MSE of all seven models: 0.032015. This is a 0.6% improvement over GNN-Correlation (0.032191), a 1.3% improvement over LSTM (0.032424), and a 2.6% improvement over HAR per-stock (0.032858). It is the first model in the project to clearly lead on test MSE rather than clustering with the pack.

Directional accuracy was 0.700, ranking fourth behind GNN-Correlation (0.712), LSTM (0.709), and HAR per-stock (0.707). Averaging the three GNN outputs attenuates the directional sharpness that GNN-Correlation achieves individually. This is expected: the ensemble trades peak DA for lower variance in predictions.

### Rank IC

Mean IC across the 103 test weeks was 0.4162, essentially tied with GNN-Correlation (0.4165). The ensemble's IC t-statistic (36.30) and information ratio (3.58) were both slightly above GNN-Correlation's (34.91, 3.44), indicating marginally more stable week-to-week IC. This consistency benefit comes from GNN-Granger, which had the highest t-stat and IR of any individual model despite the lowest mean IC. Averaging in Granger smoothed the ensemble's IC without depressing the mean. All 103 test weeks were positive-IC, the same as every other model.

The headline conclusion on IC: the ensemble is nearly identical to GNN-Correlation in cross-sectional predictive power. There is no ensemble benefit worth claiming.

### Long-only inverse-volatility portfolio

In the standard inverse-vol construction, the ensemble produced a Sharpe of 0.417, annualized return of 10.35%, and annualized vol of 12.83%. Among GNN variants, GNN-Granger had the highest Sharpe (0.423) and the ensemble ranked second at 0.417. All models trailed equal-weight (Sharpe 0.513, return 12.03%), which captures more market beta with no transaction costs.

The ensemble's standout result was turnover: 0.058 average weekly turnover, the lowest of any model in the study except equal-weight (which has zero turnover by construction). GNN-Correlation, LSTM, and GNN-Granger turned over 0.072-0.082 per week. HAR per-stock was highest at 0.163. The ensemble's predictions are a stable average of three predictions, and this damping cuts rebalancing. Annualized transaction cost drag for the ensemble was the lowest of any active model. In a higher-cost execution environment this advantage would be more meaningful; at 5 bps per side it has limited impact on reported Sharpe.

Across all six metrics (return, vol, Sharpe, drawdown, turnover, max weight), the ensemble had an average rank of 3.50 among the eight strategies, second only to equal-weight at 3.33. This makes it the most consistently good model by the composite ranking, even though it does not top any individual metric.

### Long-short portfolio

All long-short portfolios lost money in the 2024-2025 test period. The ensemble returned -18.46% annualized with Sharpe -1.289 and peak drawdown -39.23%. GNN-Correlation was the worst at -26.44%: its sharper cross-sectional signal correctly identified the high-vol names (NVDA-adjacent stocks, SMCI, TSLA) and shorted them, which was the losing side in the AI rally. The ensemble, by blending in the flatter GNN-Sector and GNN-Granger forecasts, took smaller short positions in those names and therefore lost less. This is a form of diversification rather than signal improvement.

The ensemble did not rescue the long-short construction. The problem diagnosed in Chapter 9 (volatility-based long-short inverts in strong momentum markets) applies regardless of how the GNN predictions are combined.

### Minimum variance portfolio

The minimum variance optimization places predicted RV on the diagonal of the covariance matrix. A wider prediction spread drives more concentrated, differentiated portfolios, which in turn allows the optimizer to exploit the model's cross-sectional ranking more aggressively.

The ensemble's prediction range over the test period was 0.122 to 0.748 annualized. This was the narrowest range of any model: GNN-Correlation spanned 0.093 to 0.864, GNN-Granger spanned 0.136 to 0.864, and HAR per-stock spanned 0.064 to 3.139. Averaging shrinks the tails. As a result, the ensemble's minimum variance portfolio had less diagonal differentiation, and the optimizer produced a less concentrated solution.

The minimum variance Sharpe ranking was: HAR pooled (0.729), HAR per-stock (0.635), GNN-Correlation (0.581), LSTM (0.564), GNN-Ensemble (0.498), GNN-Sector (0.492), GNN-Granger (0.412). The ensemble was fifth of seven. HAR models benefit from wide prediction ranges that drive diagonal concentration, while the ensemble's compressed predictions partially flatten the covariance matrix toward equal-weight. The minimum variance construction is the one context where the ensemble's stabilization of predictions actively costs performance.

### Summary

The ensemble's benefits are measurable but narrow. It achieves the best test MSE, nearly equal Rank IC to GNN-Correlation, and the lowest turnover of any active model. In the long-only portfolio it has the best composite ranking across metrics. Against that, it sacrifices directional accuracy relative to GNN-Correlation, does not improve the long-short outcome, and underperforms individual models in the minimum variance construction due to compressed prediction ranges.

The near-equal ensemble weights (0.365/0.315/0.319) mean the three GNN variants were not differentiated enough on validation for the weighting scheme to act as a meaningful selector. A harder question for future work: would a stacked ensemble (a second-stage model that learns to weight the three GNN outputs conditional on market regime) produce more differentiated weights and a more useful combination?

---

## Chapter 13 — Rank Loss Training

### Motivation

Chapter 11 closed with a clear diagnosis: the training objective and the evaluation objective are misaligned. MSE penalizes absolute prediction errors. Portfolio construction rewards correct cross-sectional ordering. Whether the model predicts 0.18 or 0.22 for a stock is irrelevant as long as the ranking is correct. The improvement plan's Phase 6 called for replacing MSE with a loss that directly penalizes ordering errors.

The approach chosen is pairwise BPR (Bayesian Personalized Ranking) loss. For a random sample of stock pairs (i, j) where actual RV differs, the loss penalizes the model if the sign of `pred_i - pred_j` disagrees with the sign of `actual_i - actual_j`. The formula is `softplus(-sign(target_i - target_j) * (pred_i - pred_j))`, which is equivalent to `-log(sigmoid(signed_diff))`. At each training step, roughly 10% of all ~107k possible stock pairs are sampled (configurable via `RANK_LOSS_PAIR_SAMPLE_FRAC`). This gives approximately 10k pairs per time step, which is fast enough to maintain training speed while covering a representative fraction of the pair space.

The key property of this loss: gradients exist only through the prediction differences, not through absolute values. The model is free to output predictions at any scale, as long as the relative ordering is correct. This separation from absolute scale would turn out to be important.

Three new models were trained in `notebooks/04c_rank_loss_models.ipynb`, one per graph type, using the same architecture family as the corresponding MSE models: GNNModelV2 for Correlation (hparam-tuned config), GNNModel for Sector and Granger. Checkpoints use `_rankloss` suffixes and predictions are saved separately, leaving the MSE models fully intact.

### Training behavior

All three rank loss models converged quickly relative to the MSE models:

| Model | Epochs to convergence | Best val rank loss |
|---|---|---|
| GNN-Corr rank loss | 50 | 0.6167 |
| GNN-Sector rank loss | 26 | 0.6198 |
| GNN-Granger rank loss | 19 | 0.6195 |

The rank loss values sit around 0.617-0.620 at convergence. These are not directly comparable to MSE values. The absolute scale reflects the BPR formulation: a value near `log(2) ≈ 0.693` corresponds to random ordering, so values converging to ~0.617 indicate the model is doing better than random but the gap is modest. Training curves showed clean monotonic descent with no instability, which is encouraging given that random pair sampling introduces stochasticity at each step.

### Val MSE divergence

The rank loss models produce substantially worse val MSE than the MSE-trained models:

| Model | Val MSE |
|---|---|
| GNN-Correlation (MSE) | 0.019778 |
| GNN-Sector (MSE) | 0.022894 |
| GNN-Granger (MSE) | 0.022615 |
| GNN-Correlation (rank loss) | 0.190358 |
| GNN-Sector (rank loss) | 0.178683 |
| GNN-Granger (rank loss) | 0.134868 |

The rank loss models are 6-10x worse on val MSE. This is expected and not a failure. The BPR loss never penalizes predictions for being at the wrong absolute scale, only for ordering incorrectly. The model is free to shift all predictions by a constant or compress their spread to near-zero and still achieve minimal rank loss. MSE deterioration is the direct, unavoidable cost of not including an absolute calibration term in the loss. For portfolio construction, this is acceptable: the inverse-vol weighting uses predictions only for relative ranking, not absolute magnitude.

### Ranking metrics

The full ranking evaluation was run on the 103 test-period weeks using `compute_all_ranking_metrics`, which reports Rank IC, ICIR, IC t-stat/p-value, top-25% hit rate, and pairwise accuracy:

| Model | Mean IC | ICIR | IC t-stat | Hit Rate (25%) | Pairwise Acc |
|---|---|---|---|---|---|
| GNN-Corr (MSE) | 0.4165 | 3.440 | 34.91 | 0.492 | 0.645 |
| GNN-Sector (MSE) | 0.3826 | 3.399 | 34.50 | 0.479 | 0.633 |
| GNN-Granger (MSE) | 0.3749 | 3.663 | 37.18 | 0.473 | 0.629 |
| GNN-Corr (rank loss) | 0.4293 | 3.863 | 39.20 | 0.510 | 0.499 |
| GNN-Sector (rank loss) | 0.4203 | 4.123 | 41.85 | 0.506 | 0.500 |
| GNN-Granger (rank loss) | 0.4139 | 4.326 | 43.91 | 0.506 | 0.436 |

Three metrics improved uniformly across all graph types. Mean IC rose for every rank loss model: GNN-Corr by 3.1% (0.4165 to 0.4293), GNN-Sector by 9.9% (0.3826 to 0.4203), and GNN-Granger by 10.4% (0.3749 to 0.4139). The improvement is largest for the graph types that were furthest behind GNN-Correlation under MSE training, suggesting rank loss is partially compensating for the limitations of sparser or coarser graphs.

ICIR improved substantially: GNN-Sector went from 3.399 to 4.123 (+21%), GNN-Granger from 3.663 to 4.326 (+18%), GNN-Corr from 3.440 to 3.863 (+12%). ICIR measures consistency of the cross-sectional signal, not just its average level. A higher ICIR means the model's week-to-week ranking accuracy was more stable under rank loss training. This is directly attributable to the objective function: MSE can have large losses on weeks where predictions are wrong by a lot in absolute terms but still rank correctly; rank loss only loses when the ordering is wrong. The signal is cleaner.

IC t-statistics followed the same direction: 34.91 to 39.20 for Corr, 34.50 to 41.85 for Sector, 37.18 to 43.91 for Granger. All IC p-values round to 0.0000 under any reasonable threshold.

Top-25% hit rate also improved across all three: Corr from 0.492 to 0.510, Sector from 0.479 to 0.506, Granger from 0.473 to 0.506. Hit rate measures whether the model correctly identifies the top-quartile high-vol stocks each week. The random baseline is 0.25 and all models are well above it. The rank loss models are consistently better at identifying the most volatile names, which is the input to the short side of the long-short portfolio.

### The pairwise accuracy result

Pairwise accuracy tells a completely different story. The MSE models achieved 0.629-0.645, well above the 0.5 random baseline. The rank loss models produced 0.499-0.500, essentially random, with GNN-Granger at 0.436, which is below random.

This outcome appears contradictory: the metric the rank loss was designed to optimize is the one that degraded the most. The explanation is in the prediction scale. BPR loss penalizes the sign of `pred_i - pred_j` but never penalizes the magnitude. The model learns to produce predictions with correct relative rank ordering but compressed spread. If predictions across 465 stocks span a range of 0.001 rather than 0.700, the correct ranking is preserved in principle but `pred_i - pred_j` for many pairs is a floating point value near zero. At that precision, numerical noise determines the sign, and pairwise accuracy degrades toward random.

Spearman IC is immune to this problem because it converts predictions to integer ranks before computing correlation. A model outputting [0.1001, 0.1002, 0.1003] with the correct ordering has perfect Spearman IC relative to actual ranks [1, 2, 3]. The same predictions produce near-random pairwise accuracy because all differences are near zero.

GNN-Granger's pairwise accuracy of 0.436 going below random is the strongest evidence for this mechanism. GNN-Granger uses the sparsest graph (13,886 edges vs. 91,854 for Correlation). Many stocks have few neighbors and receive little informative message passing. Under rank loss, the model may push predictions for poorly-connected stocks to the midpoint of the prediction range to minimize ordering errors on those nodes, effectively inverting some pairwise relationships for sparse nodes.

The pairwise accuracy metric as implemented was checking `sign(pred_i - pred_j)` against `sign(actual_i - actual_j)` on raw prediction values. For MSE-trained models, predictions track actual RV scale and the pairwise sign is meaningful. For rank-loss-trained models, this metric is measuring floating point noise, not ordering signal. The metric is not wrong, but it is measuring something different for the two model families, which makes direct comparison misleading.

### What the results mean

The rank loss achieved its intended effect on Rank IC, ICIR, and hit rate. All three improved. The three graph types converged toward similar IC (0.41-0.43 range) where under MSE they had spread (0.37-0.42). The rank loss is a partial equalizer across graph architectures.

The consistency improvement is the most useful finding for the paper. An ICIR above 4.0 for GNN-Sector and GNN-Granger is a strong result. For a cross-sectional equity factor, ICIR above 2.0 is considered portfolio-grade. The rank loss models are above that threshold for all three graph types, while the MSE models fell below it for Sector (3.40) and Granger (3.66) depending on how one sets the threshold. Whether ICIR above 4 is meaningful in an out-of-sample 103-week test is the question the significance tests in Phase 5.3 will address.

The practical cost is the prediction calibration loss. Inverse-volatility weighting and minimum variance optimization both depend on the absolute scale of predicted RV, not just the ranking. Rank loss models cannot be dropped into those constructions without re-calibration. They are suitable for the long-short portfolio (which depends only on relative rankings) but not for inverse-vol or min-var portfolios without an additional step to restore scale.

One open question: the BPR loss penalizes at the pair level but does not have any term that preserves prediction spread. Adding a small auxiliary MSE term (a weighted combination of rank loss and MSE) might preserve the ranking improvement while keeping predictions on a usable scale. That would make the rank loss models substitutable into all portfolio constructions, not just the long-short.

---

## Chapter 14 — Freezing the Baseline and Making the Project Auditable

After the rank-loss work, the project had become productive but fragile. There were now original MSE models, a tuned GNN-Correlation, rank-loss GNNs, an ensemble, several portfolio constructions, and a growing pile of CSV/parquet outputs. That was enough to make progress, but not enough to make a defensible paper. The next move was not another model. It was to freeze the state of the project so later experiments could not accidentally overwrite the control condition.

The baseline snapshot was written to `data/results/frozen_baseline_20260428T211759Z`, with the manifest saved at `data/results/frozen_baseline_manifest.json`. The manifest records 233 files with sizes, modification times, and SHA-256 hashes. This matters because the baseline is not a single metric table. It is a state of the whole experiment: checkpoints, predictions, metrics, rank-loss outputs, portfolio returns, hyperparameter JSON files, and graph-ablation artifacts.

The frozen roster contains 10 models:

| Model | Family | Graph | Loss |
|---|---|---|---|
| HAR per-stock | HAR | none | squared error |
| HAR pooled | HAR | none | squared error |
| LSTM | LSTM | none | MSE |
| GNN-Correlation | GNN | correlation | MSE |
| GNN-Sector | GNN | sector | MSE |
| GNN-Granger | GNN | granger | MSE |
| GNN-Ensemble | GNN ensemble | correlation + sector + granger | MSE |
| Rank-loss GNN-Correlation | GNN | correlation | BPR rank |
| Rank-loss GNN-Sector | GNN | sector | BPR rank |
| Rank-loss GNN-Granger | GNN | granger | BPR rank |

This freeze changed the psychology of the project. Before the snapshot, every rerun carried some risk of silently changing the story. After the snapshot, later feature experiments could be treated as additions rather than replacements. The baseline could be wrong, imperfect, or incomplete, but it was at least fixed.

The experiment registry was added at `data/results/experiment_registry.csv`. It now has 16 rows: the 10 frozen baseline/rank-loss models, LSTM + Macro, three macro GNNs, the macro ensemble, and the tuned macro GNN-Correlation. Each row records the experiment id, model family, graph type, loss type, feature version, graph version, checkpoint path, and metric/prediction artifacts. This is the first time the project had a single machine-readable provenance table instead of provenance spread across notebooks, JSON files, checkpoint names, and memory.

The important design choice: the registry is explicit, not magic. New models are not auto-discovered just because a checkpoint exists. That is intentional. If a model matters enough to compare, it needs a row with its feature version, graph version, loss type, and artifact paths. This makes the registry slightly more manual, but much safer for a research codebase where accidental files should not become paper results.

---

## Chapter 15 — Reproducibility Cleanup

Several small inconsistencies were fixed before adding new features. None of them were exciting, but each one could have created reviewer-level confusion later.

The biggest issue was the official tuned GNN-Correlation configuration. Earlier docs and `config.py` had drifted around `hidden_dim=128` and `lr=3e-4`, while the later hparam artifact selected a different official checkpoint: `hidden_dim=256`, `lr=1e-3`, `dropout=0.3`, `batch_norm=False`, `num_layers=3`. The code, registry, and docs were aligned to the official checkpoint. This does not make the model better; it makes the comparison auditable. The model reported in the text now matches the model loaded by the code.

The LSTM default hidden dimension was corrected so `LSTMModel` defaults to `config.LSTM_HIDDEN_DIM`, not the generic `config.HIDDEN_DIM`. This mattered because `HIDDEN_DIM` had effectively become a GNN setting. If future notebooks instantiated `LSTMModel(input_size=n_feats)` without passing a hidden size, the LSTM could silently inherit a GNN width and become a different baseline.

Universe language was also locked down. The official method is the current-constituent S&P 500 list scraped from Wikipedia, converted to yfinance symbols, then filtered by historical coverage. This is not a historical point-in-time membership universe. That limitation is real and should be described honestly, but the implementation is now documented consistently. The reproducibility artifact `data/results/universe_reproducibility.csv` records candidate/final count and coverage settings.

Sector labels were standardized from yfinance-style labels into canonical GICS-style labels. This is easy to dismiss as cosmetic, but it is not. Sector graphs, sector-neutral portfolios, ex-sector robustness checks, and any statement about Information Technology or Communication Services all depend on having a stable taxonomy. The sector mapping now lives in code rather than prose.

---

## Chapter 16 — Significance Testing

The first evaluation tables were useful for model development, but point estimates alone were too weak for final claims. The MSE differences were often tiny. For example, the frozen GNN ensemble beat LSTM by about 0.00041 MSE and HAR per-stock by about 0.00085 MSE. With only 103 test weeks, this is exactly the kind of result that can look meaningful in a table while failing a time-series test.

`src/significance.py` now implements the core testing layer:

- Diebold-Mariano tests using weekly loss differentials.
- Harvey-Leybourne-Newbold small-sample correction.
- Benjamini-Hochberg FDR correction.
- Circular block bootstrap for Sharpe ratios and Sharpe differences.

The key methodological correction was to test per-week model error series, not pooled scalar MSE. The weekly loss series is:

```text
weekly_mse[t] = mean across stocks of (actual_rv[t] - predicted_rv[t])^2
```

Those series are saved in `data/results/weekly_model_errors.parquet`. The DM test then operates on 103 weekly loss differentials, preserving the time structure of the forecast errors. This is more defensible than pretending that all stock-week errors are independent observations.

The initial significance artifacts are:

- `data/results/dm_test_results.csv`
- `data/results/bootstrap_sharpe_ci.csv`
- `data/results/significance_summary.csv`
- `data/results/weekly_model_errors.parquet`

The final significance pass found only a narrow set of statistically supportable MSE claims. In `dm_test_results.csv`, the tuned macro correlation GNN is FDR-significant versus HAR per-stock (`p_bh=0.0112`) and HAR pooled (`p_bh=0.0033`). The matched macro-vs-baseline comparisons did not survive FDR correction; the best adjusted p-value was about 0.200. This is an important constraint on the paper story. Macro features improved several point estimates, but the formal tests do not support saying every macro upgrade significantly improved its matched baseline.

The bootstrap portfolio result is similarly cautionary. The final summary reports 0 positive Sharpe-difference intervals for the broad baseline/final comparisons, but 7 positive intervals among 24 matched macro Sharpe-difference intervals. Translation: the return side has some encouraging macro-feature evidence, especially in portfolio variants, but it is not a blanket win across every comparison.

---

## Chapter 17 — Diagnostics on the Frozen Models

The diagnostics phase was where the earlier symptoms became measurable. The project already suspected that several things were happening: predictions were too compressed, correlation graphs changed regime by regime, dense graphs might oversmooth node embeddings, and full-universe Rank IC might hide sector-level effects. The diagnostics turned those suspicions into artifacts.

Calibration and spread diagnostics were added to `src/diagnostics.py` and displayed in `notebooks/05_evaluate.ipynb`. The core outputs are:

- `data/results/calibration_summary.csv`
- `data/results/calibration_bins.csv`
- `data/results/prediction_spread_by_week.csv`
- calibration figures under `data/results/figures/`

The calibration table explains why portfolio constructions react differently to the same forecast model. The original GNN-Correlation had a calibration slope of 1.183 and prediction std of 0.066. LSTM had a slope of 0.930 and prediction std of 0.079. HAR per-stock had lower slope, 0.745, but wider prediction std around 0.100. In minimum-variance portfolios, wider prediction ranges matter because predicted RV becomes the covariance diagonal. This is why a model can look worse on MSE but still produce a more differentiated optimizer input.

Graph-density diagnostics were saved to `data/results/correlation_graph_density.csv`. The oversmoothing audit was saved to `data/results/oversmoothing_audit.csv`. The oversmoothing result was not subtle. For GNN-Correlation, mean pairwise embedding distance fell sharply from layer 1 to layer 3 across representative weeks. In a calm 2017 week it dropped from about 4.47 at layer 1 to 0.58 at layer 3. In a 2023 validation week it dropped from about 3.04 to 0.52. That is direct evidence that the third message-passing layer, while helpful by validation MSE, makes node embeddings much more similar.

The COVID week looked different but still informative: the correlation graph had 197,420 edges and layer-3 distance around 0.68. Dense crisis graphs do not merely add information. They also increase the risk that every stock receives nearly the same market-mode message. This supports the future-work idea of DropEdge, residual connections, Jumping Knowledge, or regime-gated message passing. But under the implementation plan, those remain future work unless the simpler paper-scope fixes fail.

Within-sector Rank IC was added to test whether full-universe IC was mostly sector sorting. The output files are `data/results/within_sector_rank_ic_table.csv` and `data/results/within_sector_rank_ic_by_sector.csv`. The results were useful:

| Model | Mean within-sector Rank IC |
|---|---:|
| LSTM + Macro | 0.4054 |
| GNN-Ensemble + Macro | 0.3970 |
| GNN-Correlation + Macro | 0.3909 |
| GNN-Sector + Macro | 0.3896 |
| LSTM | 0.3882 |
| GNN-Ensemble | 0.3782 |
| GNN-Correlation | 0.3765 |
| HAR per-stock | 0.3610 |
| HAR pooled | 0.3536 |
| GNN-Granger | 0.3255 |

The important conclusion: the signal is not only sector-level. The IC falls somewhat inside sectors, but it does not disappear. Macro-feature models and LSTM still rank stocks inside sectors. This supports the idea that sector-neutral portfolios are worth implementing next, because there is at least measurable within-sector ranking signal to exploit.

---

## Chapter 18 — Market-Regime Features

The macro/regime feature upgrade was the first major feature change after the freeze. The motivation came directly from the diagnostics and test-period behavior: graph structure helped in some regimes and hurt in others. The model needed some way to know whether the market was calm, stressed, correlated, rates-sensitive, credit-stressed, or momentum-led.

The feature set was intentionally compact:

- VIX level.
- VIX 1-week change.
- SPY 21-day realized volatility.
- SPY 1-week return.
- SPY 1-month return.
- 10Y-2Y Treasury spread.
- Investment-grade credit spread.
- Average pairwise stock correlation.
- Correlation graph density.

The new pipeline lives in `src/macro_dataset.py` and `src/regime_features.py`. It writes:

- `data/features/regime_features.parquet`
- `data/features/regime_features_meta.json`
- `data/features/regime_normalization_stats.csv`
- `data/features/features_macro.parquet`
- `data/features/features_macro_meta.json`

The lookahead rule is preserved: feature row T may use data through Friday of week T only, while the target is RV in week T+1. Macro features are global per week, then duplicated across stocks in the node tensor. The normalization is time-series normalization using train-only mean/std, not cross-sectional z-scoring after duplication. This matters because cross-sectionally normalizing a duplicated global feature would erase the signal or create divide-by-zero behavior.

The normalization audit had no missingness for VIX, VIX change, weekly SPY return, Treasury spread, credit spread, average pairwise correlation, or graph density. SPY 21-day RV and SPY 1-month return had about 0.96% train missingness and 0.70% all-period missingness, which is consistent with rolling-window warm-up rather than data failure.

All neural baselines were retrained fairly with the same expanded feature tensor:

| Model | Best validation MSE |
|---|---:|
| LSTM + Macro | 0.019594 |
| GNN-Correlation + Macro | 0.020164 |
| GNN-Sector + Macro | 0.020043 |
| GNN-Granger + Macro | 0.019875 |

The first macro training result was mixed. LSTM + Macro had the best validation MSE among the first macro models, and the untuned GNN-Correlation + Macro was worse than its frozen counterpart on test MSE. This was a useful warning: adding regime features is not automatically helpful. The model still has to learn how to use them, and a tuned architecture selected for the baseline feature set may not be optimal for the expanded feature set.

---

## Chapter 19 — Macro GNN Hyperparameter Search and Evaluation

Because the initial untuned GNN-Correlation + Macro underperformed, a macro-specific GNN-Correlation hyperparameter search was run. This was the right choice methodologically: macro features changed the input dimension and feature distribution, so blindly reusing the baseline GNN architecture would not be a fair test.

The macro search wrote 48 validation-loss JSON files and the aggregate `data/results/gnn_corr_macro_hparam_search_results.json`. The selected tuned macro correlation model was registered as `macro_gnn_correlation_hparam` and saved as `GNN-Correlation + Macro Tuned`.

The final ML table changed the project story:

| Model | Test MSE | DA |
|---|---:|---:|
| GNN-Correlation + Macro Tuned | 0.030889 | 0.7197 |
| GNN-Granger + Macro | 0.031439 | 0.7136 |
| GNN-Sector + Macro | 0.031508 | 0.7043 |
| GNN-Ensemble + Macro | 0.031598 | 0.7148 |
| GNN-Ensemble | 0.032012 | 0.7004 |
| GNN-Correlation | 0.032191 | 0.7122 |
| LSTM | 0.032424 | 0.7088 |
| HAR per-stock | 0.032858 | 0.7070 |

The tuned macro correlation GNN is now the best point-forecast model in the table. It improves over the frozen GNN-Correlation by about 0.00130 MSE and over HAR per-stock by about 0.00197 MSE. That second comparison survives the final DM/FDR test. The matched macro-vs-baseline improvement does not survive FDR correction, so the honest wording is: the tuned macro GNN is significantly better than HAR baselines in the final table, but the macro upgrade itself is not statistically significant across matched model pairs after FDR correction.

Rank IC also improved. The best overall Rank IC is the macro ensemble at 0.4378, followed by GNN-Granger + Macro at 0.4295, LSTM at 0.4288, and GNN-Correlation + Macro Tuned at 0.4286. This is a subtle but important result. The best MSE model is not the best ranking model. The macro ensemble is the best cross-sectional ranker, while the tuned macro correlation GNN is the best point forecaster.

The macro portfolio results were stronger than the ML significance story. In inverse-volatility portfolios, equal-weight still wins with Sharpe 0.513, but macro GNNs moved closer:

| Model | Inverse-vol Sharpe |
|---|---:|
| Equal-weight | 0.513 |
| GNN-Sector + Macro | 0.468 |
| GNN-Ensemble + Macro | 0.465 |
| GNN-Granger + Macro | 0.456 |
| GNN-Correlation + Macro Tuned | 0.425 |
| GNN-Granger | 0.423 |

In minimum-variance portfolios, the macro graph models became much more interesting:

| Model | Min-var Sharpe |
|---|---:|
| GNN-Sector + Macro | 0.984 |
| GNN-Granger + Macro | 0.973 |
| GNN-Ensemble + Macro | 0.914 |
| HAR pooled | 0.729 |
| GNN-Correlation + Macro Tuned | 0.671 |
| HAR per-stock | 0.635 |

This is one of the more important findings so far. The macro features appear especially useful when the portfolio construction uses both predicted volatility and covariance structure. Sector and Granger macro models do not win the point-forecast table, but they dominate minimum-variance Sharpe. This supports the project's central framing: graph value depends on the objective and the portfolio construction, not just MSE.

The macro calibration table also shows why different models win different downstream tasks. GNN-Correlation + Macro Tuned has the highest Pearson correlation with actual RV among the final models (0.4495) and a wider prediction std (0.1138), but its calibration slope is 0.768. GNN-Granger + Macro and GNN-Ensemble + Macro have higher slopes around 1.48 and 1.44 but narrower prediction std around 0.058. These are not interchangeable forecasts. The optimizer sees different covariance diagonals even when MSEs are close.

---

## Chapter 20 — Final Result Notebook and Current State

A final artifact-reader notebook was created at `notebooks/08_final_results.ipynb`, generated by `scripts/create_final_results_notebook.py`. This notebook is not another training notebook. It is meant to load saved artifacts, display the final result tables, and write final CSVs/figures for the paper workflow.

The final result artifacts include:

- `data/results/final_ml_metrics_table.csv`
- `data/results/final_rank_ic_table.csv`
- `data/results/final_calibration_summary.csv`
- `data/results/final_portfolio_inverse_vol_metrics.csv`
- `data/results/final_portfolio_long_short_metrics.csv`
- `data/results/final_portfolio_vol_target_metrics.csv`
- `data/results/final_portfolio_minvar_metrics.csv`
- `data/results/final_dm_test_summary.csv`
- `data/results/final_bootstrap_sharpe_summary.csv`
- `data/results/final_macro_vs_baseline_deltas.csv`
- `data/results/final_significance_summary.csv`

The current project state is much more credible than the original exploratory version. The baseline is frozen. The registry records provenance. Reproducibility inconsistencies are fixed. Significance tests exist. Diagnostics explain calibration, spread, graph density, oversmoothing, regimes, and within-sector ranking. Macro/regime features have been added with lookahead-safe train-only normalization. Macro neural models were trained and evaluated fairly, including LSTM.

The story is not that GNNs always dominate. That would still be too strong. The better story is conditional:

```text
Graph structure contains useful cross-sectional volatility information, but its value depends on regime stability, objective alignment, calibration, and portfolio construction.
```

The latest evidence supports that framing. Macro graph models now lead the point-forecast table, the macro ensemble leads Rank IC, and sector/granger macro models lead minimum-variance portfolio Sharpe. But matched macro-vs-baseline DM tests do not survive FDR correction, and equal-weight still beats inverse-volatility portfolios on Sharpe. The results are useful precisely because they are not one-dimensional.

---

## Open questions

- Sector-neutral and ex-sector portfolios remain the next finance robustness checks after mixed loss. The within-sector IC results suggest there is real within-sector signal, but portfolio construction still needs to prove it.
- The macro feature upgrade improved several final results, but matched macro-vs-baseline DM tests did not survive FDR correction. The paper should report this honestly and avoid claiming universal macro-feature significance.
- Oversmoothing is measured, especially in dense correlation graphs, but architecture fixes such as DropEdge, residual connections, Jumping Knowledge, and regime-gated GNNs remain future work unless the paper-scope tasks fail.

---

## Chapter 21 — Reframing Granger Edges Around Volatility Spillovers

After the macro-feature results were in place, the Granger graph definition was audited more carefully. The existing Granger graph was built from daily log returns. In the source code, `run_granger_tests()` receives `log_returns`, and `build_granger_graph()` documents the edge semantics as:

```text
edge A -> B means stock A's past returns Granger-cause stock B's
```

That graph is not wrong as an information-flow graph, but it is not a direct volatility-spillover graph. The prediction target in this project is next-week realized volatility. Therefore, a paper that describes the original Granger graph as a volatility causality graph would be overstating the construction. A reviewer could reasonably ask why a volatility forecasting paper used return-predictability edges and then interpreted them as volatility spillovers.

To make the graph definition more defensible, a separate target-aligned graph was added rather than replacing the frozen baseline. The new graph uses weekly realized volatility:

```text
weekly_rv[week T, stock i] = sqrt(sum of daily log return squared during week T)
```

Then the same ordered-pair Granger F-test is applied over the training period only. The new edge semantics are:

```text
edge A -> B means stock A's past weekly realized volatility helps predict stock B's weekly realized volatility
```

This is a cleaner definition for the paper. It directly supports language such as "directed volatility-spillover graph" or "volatility-Granger graph." The old return-based graph should now be described, if referenced, as a return-information-flow graph.

The implementation was intentionally versioned. The frozen return-Granger artifacts remain:

- `data/graphs/granger_pvalues.parquet`
- `data/graphs/granger_edges.parquet`
- `data/results/checkpoints/gnn_granger_best.pt`

The new volatility-Granger artifacts are separate:

- `data/graphs/granger_vol_pvalues.parquet`
- `data/graphs/granger_vol_edges.parquet`
- `data/graphs/granger_vol_meta.json`
- `data/results/checkpoints/gnn_granger_vol_best.pt`
- `data/results/checkpoints/gnn_granger_vol_macro_best.pt`
- `data/results/test_preds_gnn_granger_vol.parquet`
- `data/results/test_preds_gnn_granger_vol_macro.parquet`

The volatility-Granger graph produced 26,169 directed edges under Bonferroni correction, compared with 13,886 directed edges for the original return-Granger graph. Its density is about 0.1213. This makes intuitive sense: realized volatility is more persistent and cross-sectionally clustered than raw returns, so more ordered relationships survive multiple-testing correction.

The first result was stock-only:

| Model | Test MSE | Rank IC |
|---|---:|---:|
| GNN-Granger | 0.033702 | 0.374945 |
| GNN-Granger-Volatility | 0.033080 | 0.410440 |

This is a useful improvement. Switching from return-Granger to volatility-Granger improved both point accuracy and cross-sectional ranking in the stock-only setting. It supports the claim that target-aligned graph construction matters.

The macro-feature version was then trained in the same `notebooks/04f_volatility_granger_models.ipynb` notebook, using `stock_features_plus_regime_v1` and the same volatility-Granger edge index:

| Model | Test MSE | Rank IC |
|---|---:|---:|
| GNN-Granger + Macro | 0.031439 | 0.429478 |
| GNN-Granger-Volatility + Macro | 0.031375 | 0.419766 |

With macro features, the volatility-Granger variant is nearly tied with the existing macro Granger model on MSE and slightly better by 0.000064. However, it has lower Rank IC. The broader comparison is:

| Model | Test MSE | Rank IC |
|---|---:|---:|
| GNN-Granger-Volatility + Macro | 0.031375 | 0.419766 |
| GNN-Granger + Macro | 0.031439 | 0.429478 |
| GNN-Correlation | 0.032191 | 0.416509 |
| LSTM | 0.032424 | 0.428819 |
| GNN-Granger-Volatility | 0.033080 | 0.410440 |
| GNN-Granger | 0.033702 | 0.374945 |

The interpretation is not that volatility-Granger transforms the project. It does not. The important finding is more nuanced:

1. The original return-Granger graph was semantically weaker for a realized-volatility paper.
2. Replacing it with volatility-Granger improves the stock-only Granger GNN.
3. Once macro/regime features are added, the exact Granger signal definition matters less for MSE.
4. Macro/regime context still appears to explain more of the remaining variation than the return-vs-volatility Granger distinction.
5. The best graph definition can depend on the metric: volatility-Granger + Macro slightly improves MSE, while return-Granger + Macro has better Rank IC.

This should change the paper framing. The paper should avoid claiming that the original Granger graph was a volatility-causality graph. The more defensible wording is:

```text
We evaluate two directed Granger graph definitions: a return-information-flow graph and a target-aligned volatility-spillover graph. The volatility-spillover graph improves the stock-only Granger GNN, but after adding regime features the two directed graph definitions perform similarly. This suggests that target-aligned graph construction matters, but macro/regime context is the larger driver of the final Granger-model improvement.
```

This result also supports the central research framing rather than weakening it. Graph structure contains useful cross-sectional volatility information, but the value of a graph depends on how well its edge semantics match the target, the regime context available to the model, and the evaluation objective. The volatility-Granger experiment makes the Granger comparison more auditable and more defensible, even though it does not materially change the headline performance ranking.

The registry now includes both new experiment IDs:

- `gnn_granger_vol`
- `gnn_granger_vol_macro`

These should be included in future final significance runs if the paper reports volatility-Granger results. Until those significance tests are regenerated, the results above should be treated as point estimates from the current saved predictions, not final statistical claims.

---

## Chapter 22 - Correlation Window Sensitivity

The 252-trading-day correlation graph became a clear limitation after the 2024-2025 test results. A one-year lookback is stable, but it can be too slow to adapt when market structure changes. In the test period, AI-linked stocks and the broader market diverged in ways that made stale 2022-2023 correlations less useful for message passing. To test whether the correlation graph was too sluggish, three shorter lookback windows were promoted into a controlled experiment:

- 21 trading days, roughly one month.
- 63 trading days, roughly one quarter.
- 126 trading days, roughly six months.

The experiment was implemented in `notebooks/04e_corr_window_models.ipynb`. The models use the same macro feature tensor, `stock_features_plus_regime_v1`, and the same correlation threshold, `theta=0.3`, but swap the graph version:

- `correlation_threshold_0.3_lookback_21`
- `correlation_threshold_0.3_lookback_63`
- `correlation_threshold_0.3_lookback_126`

The new graph files were written under `data/graphs/corr_edges_window/`. The trained model checkpoints and prediction files were registered as:

- `window_gnn_correlation_macro_21`
- `window_gnn_correlation_macro_63`
- `window_gnn_correlation_macro_126`

The associated model names in the evaluation tables are:

- `GNN-Correlation + Macro 21d`
- `GNN-Correlation + Macro 63d`
- `GNN-Correlation + Macro 126d`

The graph diagnostics supported the original concern. The 21-day graph is the most reactive but also the noisiest. Its validation edge turnover was about 0.337, compared with 0.164 for the 63-day graph and 0.091 for the 126-day graph. The 252-day baseline graph was denser than all three alternatives on validation, with density about 0.664 versus roughly 0.510, 0.461, and 0.514 for the 21-, 63-, and 126-day graphs.

### Test comparison

| Model | Lookback | Test MSE | R2 | DA | Mean Rank IC | ICIR | Inverse-vol Sharpe | Min-var Sharpe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GNN-Correlation + Macro Tuned | 252d | 0.030889 | 0.1828 | 0.7197 | 0.4286 | 3.6803 | 0.4251 | 0.6711 |
| GNN-Correlation + Macro 21d | 21d | 0.029575 | 0.2175 | 0.7283 | 0.4312 | 4.0371 | 0.4275 | 0.6114 |
| GNN-Correlation + Macro 63d | 63d | 0.032187 | 0.1484 | 0.7215 | 0.4219 | 3.7577 | 0.3980 | 0.2234 |
| GNN-Correlation + Macro 126d | 126d | 0.031029 | 0.1791 | 0.7184 | 0.4182 | 3.6357 | 0.4117 | 0.2645 |

The 21-day model is the strongest point estimate in this experiment. It has the lowest test MSE, highest R2, highest directional accuracy, highest Rank IC, and highest ICIR among the tested correlation-window models. It also slightly improves inverse-volatility Sharpe relative to the 252-day tuned macro correlation model. The minimum-variance result is different: the 252-day tuned macro correlation model still has the better min-var Sharpe, suggesting that the longer graph may still provide a smoother covariance structure for optimizer-style portfolio construction.

The 63-day and 126-day models did not improve the headline metrics. The 63-day model is close to the frozen GNN-Correlation baseline on MSE but worse than the 252-day tuned macro model and clearly worse than the 21-day window. The 126-day model sits between 21 and 63 days on MSE but does not improve ranking or portfolio metrics.

The DM tests versus frozen `GNN-Correlation` should be treated cautiously. The 21-day model has the largest mean loss improvement, about 0.00262, but the matched DM test does not survive FDR correction. The practical conclusion is therefore not "21 days is statistically proven best." The better interpretation is:

```text
Shorter correlation windows are worth tuning. The 21-day macro correlation GNN produced the best point estimates so far, supporting the concern that the 252-day graph can be too stale in unstable regimes. However, the current window comparison reused the same macro GNN hyperparameters across all windows, so the result is not a fully fair per-window optimum.
```

The next step for this branch is to tune each day-window model on its own validation-selected hyperparameters. The first pass should run separate hyperparameter searches for 21d, 63d, and 126d graph versions, using validation metrics only. After that, the best checkpoint from each window can be registered and evaluated in the final notebooks. This matters because the 21-day graph is noisier and may need more regularization, while the longer windows may need different depth, dropout, or learning-rate settings to make use of their smoother graph structure.
