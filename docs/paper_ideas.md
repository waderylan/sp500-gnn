# Paper Ideas

## Working Title

**Do Better Volatility Forecasts Lead to Better Portfolios? Evidence from Graph Neural Networks**

## Why This Title Fits

This title matches the current evidence without overstating the results. The final results show that the tuned macro-enhanced correlation GNN has the best point-estimate test MSE and significantly improves weekly MSE relative to HAR baselines after FDR correction. At the same time, the evidence does not support a broad claim that GNNs dominate every baseline or that better prediction metrics automatically produce better portfolio outcomes.

The title puts the central research tension first: whether stronger volatility forecasts translate into investable portfolio gains. It also keeps the GNN contribution visible without making the paper sound like a pure architecture paper.

## Core Paper Claim

The strongest defensible claim is:

> Graph-based volatility models are not uniformly superior, but correlation-graph structure combined with regime-aware features can improve point forecasting relative to classical HAR baselines. The economic value of that signal depends on calibration, ranking quality, and portfolio construction.

This preserves the project framing:

> Graph structure contains useful cross-sectional volatility information, but its value depends on regime stability, objective alignment, calibration, and portfolio construction.

## Supported Results

- `GNN-Correlation + Macro Tuned` has the best point-estimate test MSE in the final model roster.
- Weekly Diebold-Mariano tests with FDR correction support significant MSE improvement for `GNN-Correlation + Macro Tuned` versus HAR baselines.
- Macro-enhanced graph models improve several point-estimate metrics, especially for sector and Granger graph variants.
- `GNN-Ensemble + Macro` has the strongest final Rank IC point estimate, suggesting useful cross-sectional ranking signal.
- Portfolio results are more mixed than forecast metrics, which supports the prediction-to-portfolio theme.

## Claim Boundaries

The paper should not claim:

- GNNs dominate all baselines.
- Macro features are statistically significant across all model families.
- The tuned macro correlation GNN significantly beats LSTM after multiple-testing correction.
- Stronger Rank IC automatically implies better portfolio performance.
- Portfolio Sharpe improvements are broadly significant unless the relevant bootstrap intervals support that claim.

## Recommended Framing

A good paper narrative is:

1. Classical HAR and LSTM baselines are strong and hard to beat conclusively.
2. Simple graph structure alone is not enough; graph choice matters.
3. Correlation graphs are the most useful graph structure in the current setup.
4. Regime-aware macro features improve the best correlation-GNN point forecast and help some weaker graph variants.
5. Forecast accuracy, rank quality, calibration, and portfolio performance move together imperfectly.
6. The contribution is the controlled evaluation of this full chain, not a claim of unconditional GNN superiority.

## Alternative Titles Considered

**Do Better Models Lead to Better Portfolios? Evidence from Volatility Prediction with Graph Neural Networks**

This was the closest alternative. It is academic and clear, but the chosen title is sharper because it names volatility forecasts directly.

**Cross-Sectional Volatility Prediction with Graph Neural Networks: Signal Strength vs. Portfolio Impact**

This is a strong technical title. It may be better for an ML-heavy venue, but it is less memorable and less question-driven.

**On the Disconnect Between Predictive Accuracy and Portfolio Performance in Volatility Forecasting**

This is clean and publishable, but it hides the GNN contribution and sounds more like a general empirical finance paper.

**From Prediction to Portfolio: Why Volatility Models with Strong Signal Fail to Deliver Returns**

This is memorable but too negative for the current results. The evidence is mixed, not a simple failure.

**When Good Predictions Fail: Bridging the Gap Between Machine Learning Signals and Portfolio Performance**

This is catchy, but it is too broad and slightly dramatic relative to the current evidence.
