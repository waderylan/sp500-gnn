# Results Handoff for Report Writing Agent

Generated from `notebooks/08_final_results.ipynb` and saved artifacts in `data/results/`.

This document is a writing handoff, not a new analysis. Use it to draft the report while preserving the project's core framing:

> Graph structure contains useful cross-sectional volatility information, but its value depends on regime stability, objective alignment, calibration, and portfolio construction.

Do not claim that GNNs universally dominate all baselines. The final evidence supports a more conditional claim: graph models, especially with regime/macro features, improve several forecasting and ranking metrics, but the improvements are not uniformly significant and do not automatically translate into superior portfolio performance across all constructions.

## Primary Sources

Use these artifacts as the paper-facing source of truth:

- Notebook: `notebooks/08_final_results.ipynb`
- ML metrics: `data/results/final_ml_metrics_table.csv`
- Rank metrics: `data/results/final_rank_ic_table.csv`
- Calibration: `data/results/final_calibration_summary.csv`
- Portfolio metrics:
  - `data/results/final_portfolio_inverse_vol_metrics.csv`
  - `data/results/final_portfolio_long_short_metrics.csv`
  - `data/results/final_portfolio_vol_target_metrics.csv`
  - `data/results/final_portfolio_minvar_metrics.csv`
- Significance:
  - `data/results/final_dm_test_summary.csv`
  - `data/results/final_bootstrap_sharpe_summary.csv`
  - `data/results/final_significance_summary.csv`
- Macro deltas: `data/results/final_macro_vs_baseline_deltas.csv`
- Figures:
  - `data/results/figures/final_test_mse_by_model.png`
  - `data/results/figures/final_rank_ic_by_model.png`
  - `data/results/figures/final_calibration_by_predicted_rv.png`
  - `data/results/figures/final_portfolio_cumulative_returns.png`
  - `data/results/figures/final_macro_vs_baseline_deltas.png`
  - `data/results/figures/final_significance_summary.png`

The final results notebook is an artifact reader. It reads saved CSV/parquet artifacts, creates paper-facing CSVs, and writes publication figures. It does not retrain models or recompute predictions.

## Final Model Roster

The headline final tables include:

- HAR per-stock
- HAR pooled
- LSTM
- GNN-Correlation
- GNN-Sector
- GNN-Granger
- GNN-Ensemble
- GNN-Correlation + Macro Tuned
- GNN-Sector + Macro
- GNN-Granger + Macro
- GNN-Ensemble + Macro

The notebook intentionally excludes exploratory variants such as untuned `GNN-Correlation + Macro` and `LSTM + Macro` from the main headline table, while retaining them in appendix and macro-delta sections for auditability.

## Headline Forecasting Results

The best point-forecast model is `GNN-Correlation + Macro Tuned`:

- MSE: `0.030889`
- MAE: `0.106992`
- R2: `0.182801`
- Directional accuracy: `0.719720`

Important comparisons:

- `GNN-Correlation + Macro Tuned` improves MSE versus `HAR per-stock` by about `0.001969` absolute MSE, roughly a `6.0%` reduction.
- It improves MSE versus `LSTM` by about `0.001535`, roughly a `4.7%` reduction.
- It improves MSE versus the baseline `GNN-Correlation` by about `0.001302`, roughly a `4.0%` reduction.
- Among non-macro models, `GNN-Ensemble` has the best MSE at `0.032012`, followed by `GNN-Correlation` at `0.032191` and `LSTM` at `0.032424`.
- The sector and Granger GNNs without macro features underperform the main LSTM and HAR baselines on MSE.

Suggested wording:

> The strongest point-forecast result comes from adding regime-aware macro features to the correlation graph GNN. This model achieves the lowest test MSE and highest directional accuracy in the final roster, suggesting that graph information is most useful when conditioned on market regime features.

Avoid wording:

> GNNs dominate all baselines.

That is not supported because some graph variants underperform, and significance is limited.

## Ranking Results

The best Rank IC model is `GNN-Ensemble + Macro`:

- Mean Rank IC: `0.437834`
- ICIR: `3.934901`
- Top-quartile hit rate: `0.501255`
- Pairwise accuracy: `0.652820`

Other notable ranking results:

- `LSTM` remains very competitive with mean Rank IC `0.428819` and the highest ICIR in the final table at `4.362584`.
- `GNN-Granger + Macro` has mean Rank IC `0.429478`.
- `GNN-Correlation + Macro Tuned` has mean Rank IC `0.428585`, nearly matching LSTM and improving top-quartile hit rate to `0.502260`.
- Baseline sector and Granger GNNs have weaker ranking metrics than LSTM and correlation/ensemble graph models.

Suggested interpretation:

> Macro features improve the cross-sectional ranking behavior of graph models, with the macro ensemble producing the highest average Rank IC. However, the LSTM remains a strong ranking baseline, especially by ICIR, so the ranking evidence favors a nuanced conclusion rather than unconditional GNN superiority.

## Calibration Results

Calibration is mixed and matters for portfolio construction:

- `GNN-Correlation + Macro Tuned` has the highest Pearson correlation with realized volatility at `0.449488`, but its calibration slope is `0.767841`, indicating compressed realized response relative to predictions.
- `GNN-Sector + Macro` is closest to unit calibration slope among macro GNNs with slope `1.025958`.
- `GNN-Granger + Macro` and `GNN-Ensemble + Macro` have steep slopes, `1.478146` and `1.440601`, suggesting possible under-dispersion or scale mismatch in predictions.
- Baseline `GNN-Correlation` has slope `1.182748`; baseline `LSTM` has slope `0.929696`.

Suggested interpretation:

> The models that perform best on MSE or Rank IC are not always the cleanest calibrated models. This supports the paper's emphasis on calibration and objective alignment: volatility forecasts must be evaluated not only by pooled error but also by whether their predicted levels are usable for portfolio construction.

## Portfolio Results

### Long-Only Inverse-Volatility

Equal-weight remains the strongest long-only benchmark:

- Equal-weight Sharpe: `0.513179`
- Best model Sharpe: `GNN-Sector + Macro` at `0.468077`
- `GNN-Ensemble + Macro`: `0.464722`
- `GNN-Granger + Macro`: `0.456100`
- `GNN-Correlation + Macro Tuned`: `0.425094`

Interpretation:

> Macro GNNs improve long-only inverse-volatility portfolio Sharpe relative to their non-macro counterparts, but none beats equal-weight in the final table. This weakens any claim that better volatility forecasts automatically create superior long-only allocations.

### Long-Short

All long-short portfolios have negative Sharpe ratios:

- Best, least negative: `GNN-Ensemble + Macro` at `-1.137323`
- `GNN-Correlation + Macro Tuned`: `-1.201610`
- `LSTM`: `-1.251771`
- Baseline `GNN-Correlation`: `-1.778586`

Interpretation:

> The long-short construction is economically unattractive over this test period. This is consistent with a regime where high-volatility stocks may also have carried strong returns, making shorting high predicted volatility costly.

### Volatility-Targeted

The volatility-targeted portfolios also have negative Sharpe ratios across the final roster:

- Baseline `LSTM` is least negative at `-0.113978`.
- `GNN-Granger + Macro` is `-0.136091`.
- `GNN-Ensemble + Macro` is `-0.142323`.
- `GNN-Correlation + Macro Tuned` is weaker at `-0.358619`.

Interpretation:

> Volatility targeting does not validate the models economically in this test window. This should be presented as a limitation or as evidence that forecast accuracy alone is not sufficient for a robust trading rule.

### Minimum-Variance

Minimum-variance portfolios show the strongest economic result for macro GNNs:

- `GNN-Sector + Macro` Sharpe: `0.983839`
- `GNN-Granger + Macro` Sharpe: `0.972609`
- `GNN-Ensemble + Macro` Sharpe: `0.914225`
- `HAR pooled` Sharpe: `0.728926`
- `GNN-Correlation + Macro Tuned` Sharpe: `0.671057`

Interpretation:

> The clearest portfolio benefit appears in minimum-variance construction, where macro graph models substantially outperform HAR and baseline neural alternatives by Sharpe. This is the strongest evidence that graph/regime information may be economically useful when the portfolio objective is aligned with volatility control.

## Statistical Significance

The DM tests use weekly MSE series over `103` test weeks, averaging squared errors across stocks each week. This is the correct unit for testing forecast loss differences because it preserves the time-series structure of errors.

Key DM-test facts:

- `data/results/final_significance_summary.csv` reports `2` FDR-significant model-vs-baseline DM comparisons at FDR `0.05`.
- The significant comparisons in `data/results/dm_test_results.csv` are:
  - `GNN-Correlation + Macro Tuned` vs `HAR per-stock`: BH-adjusted p-value `0.011158`, mean loss improvement `0.001969`.
  - `GNN-Correlation + Macro Tuned` vs `HAR pooled`: BH-adjusted p-value `0.003266`, mean loss improvement `0.002216`.
- `GNN-Correlation + Macro Tuned` vs `LSTM` is not FDR-significant in the final summary table.
- Matched macro-vs-baseline DM tests are not FDR-significant; the significance summary reports `0` significant matched macro-vs-baseline pairs and a minimum BH-adjusted p-value of `0.199861`.

Bootstrap Sharpe summary:

- The compact final significance summary reports `0` positive Sharpe-difference confidence intervals in the broad baseline/final comparison set.
- For matched macro comparisons, it reports `7` positive Sharpe-difference intervals out of `24`, indicating that macro features help some portfolio configurations but not uniformly.
- Strong matched macro Sharpe improvements include minimum-variance `GNN-Ensemble + Macro`, `GNN-Granger + Macro`, and `GNN-Sector + Macro`.

Suggested interpretation:

> Formal tests temper the point-estimate story. The tuned macro correlation GNN significantly improves over HAR baselines, but improvements over LSTM and matched macro upgrades generally do not survive FDR correction. Portfolio bootstrap intervals show selective economic gains, especially for minimum-variance macro GNNs, rather than broad dominance.

## Macro Feature Takeaways

Macro/regime features are useful but uneven:

- They improve final MSE for:
  - `GNN-Correlation + Macro Tuned` vs `GNN-Correlation`: `-0.001302`
  - `GNN-Sector + Macro` vs `GNN-Sector`: `-0.002123`
  - `GNN-Granger + Macro` vs `GNN-Granger`: `-0.002263`
  - `GNN-Ensemble + Macro` vs `GNN-Ensemble`: `-0.000414`
- The exploratory `LSTM + Macro` worsens MSE versus `LSTM` by `0.002421`, so macro features should not be described as universally beneficial to all neural models.
- Portfolio Sharpe deltas from macro features are strongest in minimum-variance portfolios:
  - `GNN-Granger + Macro`: `+0.560745`
  - `GNN-Sector + Macro`: `+0.492023`
  - `GNN-Ensemble + Macro`: `+0.416518`

Suggested interpretation:

> Macro features appear most useful when combined with graph structure. They improve several GNN variants on point forecasts and portfolio Sharpe, especially in minimum-variance allocations, but they do not uniformly improve every model or every portfolio construction.

## Recommended Report Narrative

Use a structure like this:

1. Establish the forecasting benchmark: HAR and LSTM are strong baselines; raw GNN variants are mixed.
2. Show that graph structure helps conditionally: correlation and ensemble GNNs beat classical baselines on some point metrics, but sector and Granger graphs alone are weaker.
3. Introduce regime-aware features: macro features improve the best graph models, with `GNN-Correlation + Macro Tuned` producing the best MSE and directional accuracy.
4. Separate ranking from calibration: `GNN-Ensemble + Macro` gives the best Rank IC, but calibration varies across models.
5. Evaluate economic usefulness: equal-weight remains hard to beat in inverse-volatility portfolios; long-short and volatility-targeted strategies are weak; minimum-variance portfolios show the strongest macro-GNN benefit.
6. Close with significance: only a subset of forecast improvements are statistically significant after FDR correction, so conclusions should be conditional and not overstated.

## Claims Supported by the Current Evidence

- The tuned macro correlation GNN is the best final point-forecast model by MSE, MAE, R2, and directional accuracy.
- The macro GNN ensemble is the best final ranking model by mean Rank IC and pairwise accuracy.
- Macro features improve several graph models, particularly sector and Granger GNNs, relative to their non-macro versions.
- Minimum-variance portfolios benefit most from macro GNN forecasts.
- Equal-weight remains stronger than model-based long-only inverse-volatility portfolios in the final table.
- Long-short and volatility-targeted strategies do not provide convincing economic validation in the test window.
- Statistical tests support improvement versus HAR baselines, but not broad dominance over LSTM or every matched baseline.

## Claims to Avoid

- Avoid saying GNNs categorically dominate HAR and LSTM.
- Avoid saying macro features always improve neural models.
- Avoid treating Rank IC gains as equivalent to calibrated volatility forecasts.
- Avoid claiming the portfolio evidence is uniformly positive.
- Avoid selecting or emphasizing models based only on test-period point estimates without acknowledging significance and portfolio robustness limits.

## Open Limitations for the Report

- The test window has `103` weekly observations, so statistical power is limited.
- Current evidence is mostly single-split and appears to be single-seed for neural models unless separately documented.
- Matched macro-vs-baseline forecast improvements do not survive FDR correction.
- Sector-neutral, ex-sector, and long-only low-volatility sleeve robustness checks remain next-priority items in the implementation plan.
- Mixed rank-MSE loss is the next unchecked methodology task and may address the observed tension between ranking and calibration.

## Suggested Next Work From the Implementation Plan

The next unchecked task in `docs/implementation_priority_plan.md` is:

> Implement mixed rank-MSE loss.

For report writing, this should be framed as either ongoing work or future/next-step work unless it is completed before the manuscript is finalized.
