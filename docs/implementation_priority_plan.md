# Implementation Priority Plan

Generated: 2026-04-28

This plan orders the remaining project work by implementation dependency and research importance. The goal is to move the project from a strong exploratory codebase into a defensible ML/finance research paper. Complete items in order unless a later item is explicitly marked as independent.

The organizing principle is:

1. Fix research-validity gaps first.
2. Make results reproducible and auditable.
3. Add diagnostics that explain current model behavior.
4. Improve the feature set and retrain fairly.
5. Improve objectives and portfolio construction.
6. Add more complex GNN architecture only after the simpler issues are measured.

## Scope Boundary

Use this plan as a ranked backlog, not as a requirement that every item must go into the first paper.

**Current paper scope:** Phases 0-5. These are the work items most directly tied to making the current project credible, reproducible, and publishable. If time is limited, prioritize Phases 0-2 first, then add Phases 3-5 only as far as results and timeline allow. Within Phases 3-5, the highest-value additions are market-regime features, mixed rank-MSE loss, ex-sector robustness, and sector-neutral portfolio construction.

**Future work:** Phases 6-7, except for publication figures if a manuscript is being written. These items are valuable, but they expand the project into a larger architecture and benchmarking study. They should be described in the paper's limitations/future-work section unless the simpler paper-scope changes fail and a stronger intervention becomes necessary.

## Implementation Order

Complete the work in this order. This is the practical execution sequence for the current paper.

Notebook writing rule for every applicable item: create or update notebooks so the reviewed outputs are visible after opening the `.ipynb` file. Run notebooks through the project environment with `uv run jupyter lab` for interactive work, or execute/save outputs with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb` before treating an item as complete. Keep notebook cells ordered as: load saved artifacts, compute or refresh the item-specific outputs, display compact tables/figures, then write reusable artifacts to `data/results/` or `data/results/figures/`. Do not leave final evidence only in terminal logs.

- [x] **1. Freeze the current baseline**

  Snapshot the current metrics, prediction files, checkpoints, and model roster before changing features or retraining. This gives every later experiment a clean comparison point.
  Notebook output: add a short frozen-baseline summary section to `notebooks/05_evaluate.ipynb` for ML/ranking metrics and to `notebooks/06_portfolio.ipynb` for portfolio metrics. The visible notebook output should include the frozen metric tables, snapshot timestamp or manifest path, and model roster. Execute and save both notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/05_evaluate.ipynb` and the same command for `notebooks/06_portfolio.ipynb`; saved snapshot artifacts remain under `data/results/`.

- [x] **2. Create the experiment registry**

  Add a machine-readable registry before producing more results. Every current and future model should have a row with checkpoint path, feature version, graph version, loss type, hyperparameters, and metric artifact paths.
  Registry behavior: the current registry is schema-forward-compatible but intentionally explicit. New trained models are not auto-discovered from checkpoint files; each training step that creates a new model must add registry rows for the new experiment IDs before the step is complete.
  Notebook output: display the registry in `notebooks/05_evaluate.ipynb` with enough columns visible to audit provenance, and add a small registry-reference cell to model-training notebooks after new runs. Execute/save the affected notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb` so the registry preview is visible; saved registry should be `data/results/experiment_registry.csv` or `data/results/experiment_registry.json`.

- [x] **3. Fix reproducibility inconsistencies**

  Resolve the GNN hyperparameter/config mismatch, fix the LSTM hidden-dimension default, lock the current-constituent universe language, and standardize sector labels. These are small changes that prevent confusing or invalid comparisons later.
  Notebook output: no need to make any notebook changes.

- [x] **4. Implement statistical testing infrastructure**

  Implement `src/significance.py`, generate per-week model error series, and add block-bootstrap Sharpe confidence intervals. This can run before or alongside diagnostics, but it must be complete before any final performance claims.
  Notebook output: write and display DM, bootstrap, and significance summary tables in `notebooks/07_significance.ipynb`. The notebook should show the main tables with sorted model comparisons, confidence intervals, and FDR-adjusted significance flags. Execute/save with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/07_significance.ipynb`; saved artifacts should be `data/results/dm_test_results.csv`, `data/results/bootstrap_sharpe_ci.csv`, and `data/results/significance_summary.csv`.

- [x] **5. Add core diagnostics on the existing baseline**

  Add calibration diagnostics, prediction-spread diagnostics, graph-density time series, regime breakdowns, oversmoothing audit, and within-sector Rank IC. These explain why the current GNNs help in some places and fail in others.
  Notebook output: add ML diagnostics to `notebooks/05_evaluate.ipynb`, graph-density and oversmoothing diagnostics to `notebooks/03_graphs.ipynb` or `notebooks/04_gnn_models.ipynb` as appropriate, and save diagnostic tables/figures under `data/results/` and `data/results/figures/`. Each diagnostic notebook should display the key table plus at least one rendered figure where useful; execute/save each affected notebook with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`.

- [x] **6. Add market-regime features**

  Add macro/regime data, build regime features with train-only normalization, extend the feature tensor, version the feature set, and retrain LSTM plus all GNN variants fairly.
  Notebook output: show macro/regime feature coverage and normalization checks in `notebooks/02_features.ipynb`; show retraining results in `notebooks/04_baseline_models.ipynb` and `notebooks/04_gnn_models.ipynb`; save new feature artifacts under `data/features/` and new model artifacts under `data/results/` and `checkpoints/`. Execute/save notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`; for long training cells, save checkpoints and metrics from the script/notebook run, then keep the notebook output focused on the final training summary and artifact paths.

- [x] **7. Evaluate macro-feature models**

  Compare macro-feature models against the frozen baseline using MSE, Rank IC, calibration, portfolio metrics, and significance tests. Continue only if the results clarify or improve the paper story.
  Notebook output: display macro-vs-baseline ML metrics in `notebooks/05_evaluate.ipynb`, portfolio metrics in `notebooks/06_portfolio.ipynb`, and significance results in `notebooks/07_significance.ipynb`; saved comparison tables should live under `data/results/`. Execute/save all three notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb` so the comparison tables and figures are visible when opened.

- [ ] **8. Implement mixed rank-MSE loss**

  Run an alpha sweep on GNN-Correlation first, select alpha using validation only, then train all three graph variants with the selected mixed loss. Evaluate point accuracy, ranking, calibration, and portfolio behavior together.
  Notebook output: run/display the alpha sweep and selected-alpha training summary in `notebooks/04c_rank_loss_models.ipynb`; display downstream evaluation in `notebooks/05_evaluate.ipynb`, `notebooks/06_portfolio.ipynb`, and `notebooks/07_significance.ipynb`; save sweep, prediction, checkpoint, and metrics artifacts under `data/results/` and `checkpoints/`. Execute/save the notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`; if training is too long for full notebook execution, run training through `uv run` scripts and keep the notebook as a reproducible artifact reader that displays the saved sweep and final metrics.

- [ ] **9. Add portfolio robustness checks**

  Implement sector-neutral portfolios, ex-IT/Communication Services portfolios, and optionally long-only low-volatility sleeves. These test whether portfolio results are genuine stock-level signal or sector/regime artifacts.
  Notebook output: display robustness return curves, metrics tables, and concentration diagnostics in `notebooks/06_portfolio.ipynb`; save robustness returns and metrics under `data/results/`. Execute/save with `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/06_portfolio.ipynb` so the return curves and robustness tables are visible without rerunning the portfolio code manually.

- [ ] **10. Run final significance tests and update result tables**

  Once the model set is final, regenerate all metrics, confidence intervals, FDR-corrected tests, and summary tables. These are the numbers that should feed the paper.
  Notebook output: regenerate and display final ML tables in `notebooks/05_evaluate.ipynb`, final portfolio tables in `notebooks/06_portfolio.ipynb`, and final significance tables in `notebooks/07_significance.ipynb`; saved final tables should live under `data/results/`. Execute/save each final-results notebook with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`, then open the notebooks to confirm the saved outputs match the final CSV/parquet artifacts.

- [ ] **11. Prepare paper figures and final narrative**

  Build the figures and frame the paper around conditional graph value, regime sensitivity, objective alignment, and finance-specific evaluation. Avoid claiming unconditional GNN superiority unless the final tests support it.
  Notebook output: generate publication figures from `notebooks/07_significance.ipynb` or a dedicated final-figures section that reads only saved artifacts; write all PNGs to `data/results/figures/` and keep final narrative notes in `docs/research_journey.md` or the manuscript draft. Execute/save the figure notebook with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`, and make sure every figure used in the paper is both rendered in the notebook and saved as a standalone PNG.

- [ ] **12. Defer future-work architecture**

  DropEdge, residual/Jumping Knowledge GNNs, edge-weighted graphs, multi-graph fusion, regime-gated GNNs, stronger baseline suites, multi-seed runs, and rolling-origin evaluation should remain future work unless the paper-scope items fail to produce a credible result.
  Notebook output: not applicable unless one of these is promoted into current scope; if promoted, add its training outputs to the relevant `notebooks/04*.ipynb` notebook and its evaluation outputs to `notebooks/05_evaluate.ipynb`, `notebooks/06_portfolio.ipynb`, and `notebooks/07_significance.ipynb`. Execute/save any promoted-work notebooks with `uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`, and keep future-work placeholders out of final notebooks unless they contain actual results.

---

## Phase 0: Freeze The Current Baseline

These tasks preserve the current state before making changes. This matters because the next phases will rebuild features, retrain models, and change result tables.

- [x] **Create a current-results snapshot**

  Save a copy or manifest of the current result files before modifying the pipeline:

  - `data/results/ml_metrics_table.csv`
  - `data/results/rank_ic_table.csv`
  - `data/results/portfolio_metrics_table.csv`
  - `data/results/portfolio_ls_metrics_table.csv`
  - `data/results/portfolio_vt_metrics_table.csv`
  - `data/results/portfolio_mv_metrics_table.csv`
  - `data/results/gnn_hparam_search_results.json`
  - `data/results/corr_threshold_ablation.json`

  Context: the current baseline is useful even though it is imperfect. It shows that GNN-Ensemble has the best test MSE, LSTM has the best Rank IC, long-only inverse-vol portfolios trail equal-weight, and minimum-variance portfolios are led by HAR pooled. These facts are the control condition for all future changes.

- [x] **Write down the official current model roster**

  Record each model currently included in the experiment:

  - HAR per-stock
  - HAR pooled
  - LSTM
  - GNN-Correlation
  - GNN-Sector
  - GNN-Granger
  - GNN-Ensemble
  - Rank-loss GNN-Correlation
  - Rank-loss GNN-Sector
  - Rank-loss GNN-Granger

  Context: later additions should not silently replace these. New models should be added as new rows with clear names such as `GNN-Correlation + Macro`, `GNN-Correlation MixedLoss`, or `GNN-Fusion`.

---

## Phase 1: Research Validity And Reproducibility

This is the highest-priority phase. Do not use new model results in a paper until this phase is complete.

- [x] **Implement statistical significance testing in `src/significance.py`**

  Replace the current `NotImplementedError` stubs with working implementations:

  - Diebold-Mariano test with Harvey-Leybourne-Newbold small-sample correction.
  - Benjamini-Hochberg FDR correction.
  - Full pairwise DM test orchestration for model-vs-baseline comparisons.
  - Circular block bootstrap for Sharpe ratio and Sharpe differences.

  Required outputs:

  - `data/results/dm_test_results.csv`
  - `data/results/bootstrap_sharpe_ci.csv`
  - `data/results/significance_summary.csv`

  Context: current MSE differences are small enough that point estimates alone are not publishable. The GNN ensemble beats LSTM by about `0.00041` MSE and HAR per-stock by about `0.00084` MSE. These improvements may or may not survive formal tests over 103 test weeks.

- [x] **Add per-week model error series**

  Produce one weekly error series per model:

  ```text
  weekly_mse[t] = mean across stocks of (actual_rv[t] - predicted_rv[t])^2
  ```

  Save these in a reusable artifact such as:

  - `data/results/weekly_model_errors.parquet`

  Context: the DM test should operate on per-week loss differentials, not on one pooled scalar MSE. This preserves the time-series structure of forecast errors.

- [x] **Add Sharpe bootstrap inputs**

  Ensure portfolio return files contain clean weekly net return series by model and strategy:

  - long-only inverse-vol
  - long-short
  - volatility-targeted
  - minimum variance

  Context: bootstrap Sharpe intervals require the actual weekly return series, not only summary tables. The block bootstrap should preserve serial dependence by resampling contiguous blocks.

- [x] **Create an experiment registry**

  Add a machine-readable registry, for example:

  - `data/results/experiment_registry.csv`
  - or `data/results/experiment_registry.json`

  Minimum fields:

  ```text
  experiment_id
  model_name
  model_family
  graph_type
  loss_type
  feature_version
  graph_version
  checkpoint_path
  train_split
  val_split
  test_split
  hyperparameters
  validation_metrics_path
  test_metrics_path
  portfolio_metrics_path
  notes
  ```

  Context: right now, result provenance is spread across docs, JSON files, CSV files, checkpoints, and notebooks. A registry prevents confusion when rerunning models with macro features or mixed loss.

- [x] **Resolve the tuned GNN hyperparameter inconsistency**

  Decide which tuned GNN-Correlation model is official.

  Current inconsistency:

  - `config.py` and docs describe `hidden_dim=128`, `lr=3e-4`.
  - `data/results/gnn_hparam_search_results.json` reports best config as `hidden_dim=256`, `lr=1e-3`, `dropout=0.3`, `batch_norm=False`, `num_layers=3`.

  Required work:

  - Update `config.py` to match the official checkpoint.
  - Update `docs/project_outline.md`.
  - Update `docs/research_journey.md`.
  - Add the final config to the experiment registry.

  Context: a paper cannot report one architecture while the code evaluates another.

- [x] **Fix LSTM default hidden dimension**

  In `src/models.py`, `LSTMModel` should default to `config.LSTM_HIDDEN_DIM`, not `config.HIDDEN_DIM`.

  Context: `config.HIDDEN_DIM` is now a GNN setting. If the LSTM notebook instantiates `LSTMModel(input_size=n_feats)` without passing `hidden_dim`, future training may use the wrong hidden size. This could make baseline results non-reproducible.

- [x] **Lock the universe construction language to the existing method**

  Use the existing universe construction method and document it consistently.

  Official universe definition:

  - scrape the current Wikipedia S&P 500 constituent list
  - convert symbols to yfinance format
  - download historical adjusted close and volume data
  - retain tickers with at least `config.MIN_COVERAGE` historical price coverage

  Required work:

  - keep methodology language aligned with the existing current-constituent universe
  - add a reproducibility table with candidate count, final count, and coverage threshold

  Context: the project is intentionally sticking with the existing current-constituent full-history universe.

- [x] **Standardize sector taxonomy**

  Add a canonical sector mapping so code and docs use the same sector labels.

  Current issue:

  - Docs refer to GICS names like `Information Technology`.
  - `src/data.py` uses yfinance names like `Technology`, `Financial Services`, and `Consumer Cyclical`.

  Required work:

  - Create a mapping from yfinance sector labels to canonical GICS-style labels.
  - Rebuild `sector_history.json`.
  - Rebuild sector graphs.
  - Recompute sector-level metrics.

  Context: this affects sector graph construction, sector-neutral portfolios, ex-sector robustness checks, and paper claims about sector behavior.

---

## Phase 2: Diagnostics For Current Model Behavior

This phase explains why existing results look the way they do. Complete it before adding major new architectures.

- [x] **Add calibration diagnostics for all prediction models**

  For each model, compute:

  - prediction mean
  - prediction standard deviation
  - prediction min/max
  - cross-sectional prediction spread by week
  - calibration slope and intercept from `actual_rv ~ predicted_rv`
  - Pearson correlation between predictions and actuals
  - Spearman Rank IC
  - predicted-RV decile vs realized-RV average

  Suggested outputs:

  - `data/results/calibration_summary.csv`
  - `data/results/calibration_bins.csv`
  - calibration plots in `data/results/figures/`

  Context: rank-loss models improved ranking metrics but destroyed absolute calibration. Portfolio methods like inverse-volatility and minimum variance need calibrated levels, not only ranks.

- [x] **Add prediction-spread diagnostics**

  Compute weekly cross-sectional spread:

  ```text
  spread[t] = percentile_90(preds[t]) - percentile_10(preds[t])
  ```

  Also compute prediction range and standard deviation by week.

  Context: the long-only inverse-volatility portfolios collapsed toward equal-weight because model predictions were not differentiated enough across 465 stocks. This diagnostic measures that problem directly.

- [x] **Add graph-density time series**

  For each correlation graph week, compute:

  - number of edges
  - density
  - mean degree
  - max degree
  - average absolute correlation

  Suggested output:

  - `data/results/correlation_graph_density.csv`

  Context: GNN-Correlation may work differently in sparse calm regimes versus dense crisis regimes. Dense graphs may cause oversmoothing or stale neighbor propagation.

- [x] **Run regime breakdowns of existing results**

  Split test-period results by regimes:

  - 2024 vs 2025.
  - high vs low VIX.
  - high vs low market return.
  - high vs low graph density.
  - high vs low average pairwise correlation.

  For each regime, report:

  - MSE
  - MAE
  - Rank IC
  - directional accuracy
  - portfolio return and Sharpe where applicable

  Context: the major diagnosis is regime instability. A model can be useful in one market environment and harmful in another. The paper should show this directly.

  Implementation note: current Step 5 artifacts include 2024/2025, market-return, graph-density, and average-absolute-correlation regimes from existing cached data. The high/low VIX split remains tied to the Phase 3 macro data cache because no VIX artifact exists in the current repository state.

- [x] **Audit GNN oversmoothing**

  Add a diagnostic that captures node embeddings after each GraphSAGE layer and computes cross-sectional dispersion.

  Metrics:

  - mean pairwise embedding distance
  - mean absolute deviation across nodes
  - layer-3 dispersion divided by layer-1 dispersion

  Test weeks:

  - calm 2017 week
  - COVID 2020 week
  - representative 2023 validation week
  - representative 2024 test week
  - representative 2025 test week

  Context: a 3-layer GNN on dense correlation graphs may make node embeddings too similar. If oversmoothing is present, residual connections, DropEdge, or Jumping Knowledge should be tested before deeper models.

- [x] **Add within-sector Rank IC**

  Compute Spearman Rank IC separately within each sector at each week, then average.

  Suggested outputs:

  - `data/results/within_sector_rank_ic_table.csv`
  - `data/results/within_sector_rank_ic_by_sector.csv`

  Context: full-universe Rank IC may be driven by easy sector-level differences. Within-sector IC tests whether the models can rank stocks inside each sector.

---

## Phase 3: Market-Regime Feature Upgrade

This is the first major feature improvement. It should be implemented after Phase 1 and ideally after Phase 2 diagnostics are available.

- [x] **Define the macro/regime feature set**

  Start with a compact set:

  - VIX level.
  - VIX 1-week change.
  - SPY 21-day realized volatility.
  - SPY 1-week return.
  - SPY 1-month return.
  - 10Y-2Y Treasury spread.
  - investment-grade credit spread.
  - average pairwise stock correlation.
  - correlation graph density.

  Context: the goal is to tell the model whether the market is calm, stressed, momentum-led, rates-sensitive, credit-stressed, or highly correlated. Do not add dozens of macro variables at first. A small, interpretable set is easier to defend.

- [x] **Add macro data download/caching**

  Add a data function that downloads and caches macro series.

  Likely sources:

  - FRED for Treasury spread and credit spread.
  - Yahoo/yfinance or existing price data for SPY returns and SPY RV.
  - FRED or yfinance for VIX.

  Suggested raw output:

  - `data/raw/macro_series.parquet`

  Context: all macro data must be available before the prediction week. Use values through Friday of week `T` to predict week `T+1`.

- [x] **Add regime feature engineering in `src/features.py`**

  Build a weekly macro/regime feature frame aligned to the existing weekly feature index.

  Required lookahead rule:

  ```text
  feature row T may use data through Friday of week T only
  target row T is RV in week T+1
  ```

  Suggested output:

  - `data/features/regime_features.parquet`

  Context: these features are global per week, so they will be duplicated across all stocks when building the node feature tensor.

- [x] **Normalize macro/regime features carefully**

  Do not cross-sectionally z-score macro features after duplicating them across nodes. If every stock has the same VIX value in a given week, cross-sectional z-scoring would divide by zero or erase the feature.

  Use time-series normalization based on training data only:

  ```text
  normalized_value[t] = (value[t] - train_mean) / train_std
  ```

  Then apply the same train mean/std to validation and test.

  Context: this is different from the current stock-level features, which are cross-sectionally normalized each week.

- [x] **Update feature tensor construction**

  Extend the feature tensor from:

  ```text
  shape = (num_weeks, num_stocks, stock_features)
  ```

  to:

  ```text
  shape = (num_weeks, num_stocks, stock_features + regime_features)
  ```

  Every stock receives the same regime feature values for a given week.

  Context: this is the simplest correct implementation. It does not require a new GNN architecture.

- [x] **Version the feature set**

  Save metadata that records whether features are baseline-only or baseline-plus-regime.

  Suggested names:

  - `features_base.parquet`
  - `features_macro.parquet`
  - or include `feature_version` in `features_meta.json`

  Context: once macro features are added, old checkpoints are not compatible with the new input dimension. The registry must make this explicit.

- [x] **Retrain all neural models fairly**

  Retrain:

  - LSTM with macro features.
  - GNN-Correlation with macro features.
  - GNN-Sector with macro features.
  - GNN-Granger with macro features.
  - GNN-Ensemble from the three macro-feature GNNs.

  Registry requirement: add new rows for every macro-feature neural model and ensemble with explicit `feature_version`, `graph_version`, checkpoint paths, validation metrics, test prediction paths, and notes tying them to the macro feature artifact.

  Keep HAR baselines unchanged as pure HAR baselines unless adding a separate macro-HAR model.

  Context: if only the GNNs receive macro features, the comparison against LSTM becomes unfair. LSTM should get the same expanded node feature sequence.

- [x] **Evaluate macro-feature models against the frozen baseline**

  Compare old vs new:

  - MSE
  - MAE
  - R2
  - directional accuracy
  - Rank IC
  - ICIR
  - top-quartile hit rate
  - calibration slope
  - portfolio Sharpe
  - significance tests

  Context: the key question is not just whether macro features improve the GNN. It is whether they improve the GNN more than they improve LSTM, and whether the improvement survives significance testing.

---

## Phase 4: Objective Alignment

This phase addresses the mismatch between point-forecast MSE and portfolio/ranking objectives.

Scope status: **current paper scope, high importance.** This phase corresponds to the mixed rank-MSE loss work described in `docs/sp500_gnn_improvement_plan.md`. It should not be treated as future work because it directly addresses one of the central problems already observed: pure MSE preserves calibration but does not optimize ranking, while pure rank loss improves Rank IC but damages volatility-scale calibration.

- [ ] **Implement mixed rank-MSE loss**

  Add a loss function:

  ```text
  loss = (1 - alpha) * masked_MSE + alpha * BPR_rank_loss
  ```

  Start with:

  ```text
  alpha in {0.05, 0.10, 0.20, 0.30, 0.50}
  ```

  Context: pure rank loss improved Rank IC and hit rate but broke calibration. Mixed loss should test whether the ranking improvement can be retained while keeping predictions on the correct volatility scale.

- [ ] **Run alpha sweep on GNN-Correlation first**

  For each alpha, record:

  - validation MSE
  - validation Rank IC
  - validation ICIR
  - prediction spread
  - calibration slope

  Suggested output:

  - `data/results/mixed_loss_alpha_sweep.csv`

  Registry requirement: register each alpha-sweep run with its validation-only config, checkpoint path, validation metrics path, and prediction artifact path before comparing downstream results.

  Context: do not train all graph variants until the alpha tradeoff is understood on the strongest GNN variant.

- [ ] **Select the mixed-loss alpha using validation only**

  Choose alpha before touching test results.

  Selection rule should balance:

  - materially better Rank IC or ICIR than pure MSE
  - materially better calibration than pure rank loss
  - validation MSE not catastrophically worse than pure MSE

  Context: selecting alpha based on test results would invalidate the final test evaluation.

- [ ] **Train all three GNN graph variants with selected mixed loss**

  Train:

  - GNN-Correlation MixedLoss
  - GNN-Sector MixedLoss
  - GNN-Granger MixedLoss
  - GNN-Ensemble MixedLoss

  Registry requirement: add new rows for all selected-alpha mixed-loss models and the ensemble, including the selected alpha, validation-selection artifact, checkpoint paths, prediction paths, and metric artifact paths.

  Context: this tests whether objective alignment benefits all graph types or only the correlation graph.

- [ ] **Evaluate mixed-loss models across all metrics**

  Include:

  - point forecast metrics
  - ranking metrics
  - calibration metrics
  - long-only portfolios
  - sector-neutral portfolios
  - minimum-variance portfolios
  - significance tests

  Context: mixed loss is only useful if it improves ranking without destroying the downstream portfolio methods that need calibrated volatility.

---

## Phase 5: Portfolio Robustness And Finance Evaluation

This phase makes the project more credible to finance reviewers.

Scope status: **current paper scope, high importance.** The ex-sector robustness check and sector-neutral portfolio construction from `docs/sp500_gnn_improvement_plan.md` belong here. They are not future-work items because they test whether the current results are being driven by 2024-2025 IT/Communication Services sector behavior rather than generalizable stock-level volatility prediction.

- [ ] **Implement sector-neutral portfolio construction**

  Scope status: current paper scope. This is the stronger robustness test because it removes sector-timing effects by construction and forces the model to prove within-sector ranking value.

  At each week:

  - group stocks by sector
  - compute inverse-vol weights within each sector
  - assign equal gross weight to each active sector
  - normalize total weights to sum to 1

  Add a sector-neutral equal-weight benchmark.

  Suggested outputs:

  - `data/results/portfolio_sn_returns.parquet`
  - `data/results/portfolio_sn_metrics_table.csv`

  Context: this removes sector-timing effects and tests whether the model has within-sector stock-selection value.

- [ ] **Implement ex-sector robustness checks**

  Scope status: current paper scope. This is a targeted diagnostic for the 2024-2025 AI-led market regime and should be implemented before treating portfolio results as final.

  Run portfolios excluding:

  - Information Technology
  - Communication Services

  Suggested outputs:

  - `data/results/portfolio_exsector_returns.parquet`
  - `data/results/portfolio_exsector_metrics_table.csv`

  Context: the 2024-2025 test period was heavily affected by AI-driven tech leadership. This check tests whether that regime distorted the portfolio conclusions.

- [ ] **Add long-only low-volatility sleeve portfolios**

  Instead of dollar-neutral long-short, build long-only portfolios using:

  - lowest predicted-volatility decile
  - lowest predicted-volatility quartile
  - bottom half by predicted volatility

  Context: the long-short strategy lost money because it shorted high-vol/high-return winners. A long-only low-volatility sleeve may better isolate defensive volatility-forecasting value without taking an explicit short position against momentum names.

- [ ] **Add transaction cost sensitivity**

  Recompute portfolio metrics at:

  ```text
  transaction_cost_bps in {0, 5, 10, 25, 50}
  ```

  Context: some model portfolios have much higher turnover than others. Turnover sensitivity is important for economic interpretation.

- [ ] **Add portfolio concentration diagnostics**

  For every portfolio construction, report:

  - max single-stock weight
  - effective number of holdings
  - Herfindahl-Hirschman Index
  - sector weights over time
  - average weekly turnover

  Context: minimum-variance portfolios hit the `MAX_WEIGHT=0.05` cap. Concentration diagnostics help explain whether performance comes from prediction quality or optimizer concentration.

---

## Phase 6: Future Work - Graph Model Improvements

This phase is **future work for the first paper** unless Phases 1-5 fail to produce a credible result. Only start this phase after the simpler diagnostics show a specific graph failure. The goal is to fix measured graph problems, not add complexity blindly.

- [ ] **Add DropEdge regularization**

  Future-work status: likely future work unless the oversmoothing audit shows severe embedding collapse.

  During GNN training, randomly drop a fraction of graph edges, then use the full graph at inference.

  Suggested initial values:

  ```text
  edge_dropout_p in {0.05, 0.10, 0.20}
  ```

  Context: this may reduce over-reliance on stale neighbors and reduce oversmoothing during dense graph weeks.

- [ ] **Add residual connections to `GNNModelV2`**

  Future-work status: likely future work unless the oversmoothing audit indicates that deeper GraphSAGE layers are washing out stock-specific information.

  Each layer should preserve node-specific information:

  ```text
  h_next = conv(h_prev, edge_index) + projection(h_prev)
  ```

  Context: residuals help prevent graph aggregation from washing out individual stock signals.

- [ ] **Add Jumping Knowledge aggregation**

  Future-work status: likely future work unless layer-depth diagnostics show that different stocks benefit from different receptive-field depths.

  Concatenate or aggregate embeddings from all GNN layers before the output head.

  Context: different stocks may need different neighborhood depths. Some may benefit from direct self features, while others benefit from 1-hop or 2-hop neighbors.

- [ ] **Add edge-weighted correlation graph**

  Future-work status: good second-paper or extended-ablation item. It is more invasive than macro features or mixed loss because graph storage, model forward passes, and training loops all need to handle edge attributes.

  Extend correlation graph construction to save edge weights:

  ```text
  edge_attr = correlation_value
  ```

  Then use a GNN layer that can consume edge weights or edge attributes.

  Context: the current binary graph treats `corr=0.31` and `corr=0.90` as equally strong edges. That discards important information.

- [ ] **Build a multi-graph fusion model**

  Future-work status: defer unless the separate graph variants show complementary errors after the diagnostics and mixed-loss work. This is a real architectural contribution, but it is too much to combine with every paper-scope fix unless time is abundant.

  Use separate message-passing branches for:

  - correlation graph
  - sector graph
  - Granger graph

  Then combine embeddings through concatenation, learned weights, or attention.

  Context: the current ensemble averages outputs after training. A fusion model can learn how to combine graph types inside the model.

- [ ] **Build a regime-gated GNN**

  Future-work status: strongest future architecture idea. It should only be attempted after simple macro/regime feature concatenation proves useful.

  Use macro/regime features to decide how much the model should trust graph neighbors:

  ```text
  output_embedding = gate * neighbor_embedding + (1 - gate) * self_embedding
  ```

  Context: this is the architecture that most directly uses the market-regime feature upgrade. It should come after simple macro-feature concatenation proves useful.

---

## Phase 7: Future Work And Optional Publication Extensions

These tasks strengthen the project but are mostly **future work** for the first paper. Treat them as optional extensions unless a reviewer, advisor, or result gap makes one of them necessary.

- [ ] **Tune the LSTM baseline**

  Scope status: optional for the first paper, but important if LSTM remains a primary baseline and appears competitive with or better than the GNNs.

  Run a small validation-only grid:

  - hidden dim: 32, 64, 128
  - sequence length: 4, 8, 13
  - dropout: 0.1, 0.3
  - learning rate: 3e-4, 1e-3

  Registry requirement: if any tuned LSTM model is trained and evaluated, add it as a new registry row with a distinct experiment ID rather than replacing the frozen LSTM baseline row.

  Context: GNN-Correlation has received hyperparameter search. LSTM should not remain under-tuned if it is a primary baseline.

- [ ] **Add stronger volatility baselines**

  Future-work status: likely future work. Add only one stronger baseline for the first paper if the current baseline set is challenged or if time permits.

  Consider:

  - HAR with macro features.
  - HAR with sector fixed effects.
  - Elastic Net pooled volatility model.
  - XGBoost or Random Forest tabular model.
  - GARCH-family baseline on a smaller representative universe if full-scale is too expensive.

  Registry requirement: any added baseline must be registered as a new experiment with explicit feature version, split definition, hyperparameters, prediction paths, and metric artifact paths.

  Context: finance reviewers will compare against classical volatility models, not only LSTM.

- [ ] **Run multiple random seeds**

  Scope status: optional but valuable. If compute allows, this improves paper credibility; if not, report single-seed results as a limitation.

  For neural models, train at least 3 seeds if compute allows.

  Registry requirement: record each seed as its own registry row or add a seed-level registry artifact that maps every seed to its checkpoint, config, predictions, and metrics before aggregating across seeds.

  Context: a single seed is weak for publication. Seed variance can be larger than the MSE differences currently observed.

- [ ] **Add rolling-origin evaluation**

  Future-work status: future work. This is a much larger evaluation design and should not block the first paper.

  Instead of one fixed train/val/test split, add rolling-origin backtests if time permits.

  Example:

  ```text
  train 2015-2019, validate 2020, test 2021
  train 2015-2020, validate 2021, test 2022
  train 2015-2021, validate 2022, test 2023
  train 2015-2022, validate 2023, test 2024-2025
  ```

  Context: this makes regime robustness much more convincing, but it is compute-heavy.

- [ ] **Prepare publication figures**

  Scope status: current paper scope if writing begins. This is not future work; it is manuscript preparation.

  Required figure set:

  - model test MSE with confidence intervals
  - Rank IC by model
  - calibration curves
  - graph density over time
  - portfolio cumulative returns
  - sector-neutral portfolio results
  - regime breakdown heatmap
  - significance test summary

  Context: the final paper should not rely on raw CSV tables alone.

---

## Recommended Immediate Sequence

If work starts now, the next ten concrete tasks should be:

- [x] Implement `src/significance.py`.
- [x] Generate weekly model error series.
- [x] Generate bootstrap Sharpe confidence intervals.
- [x] Create the experiment registry.
- [x] Resolve GNN hyperparameter/config mismatch.
- [x] Fix LSTM hidden-dim default.
- [x] Lock universe construction docs to current S&P 500 constituents filtered by historical coverage.
- [x] Standardize sector taxonomy.
- [x] Add calibration and prediction-spread diagnostics.
- [x] Add graph-density and regime-breakdown diagnostics.

After those are done, the next model-improvement task should be the market-regime feature upgrade, followed by mixed rank-MSE loss.

Future-work items should not be started until the paper-scope checklist is stable unless a specific diagnostic makes one necessary.

---

## Research Framing To Preserve

The current evidence does not support a simple claim that GNNs dominate all baselines. The stronger paper is:

```text
Graph structure contains useful cross-sectional volatility information, but its value depends on regime stability, objective alignment, calibration, and portfolio construction. We show this through controlled comparisons of correlation, sector, and Granger graphs, then improve the setup using regime-aware features and ranking-aware training objectives.
```

This framing is honest, defensible, and leaves room for both positive and negative results.
