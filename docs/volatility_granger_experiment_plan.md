# Volatility-Granger Graph Experiment Plan

## Goal

Test whether the Granger graph should be based on volatility spillovers rather than return predictability.

The frozen baseline remains unchanged:

- `data/graphs/granger_pvalues.parquet`
- `data/graphs/granger_edges.parquet`
- `data/results/checkpoints/gnn_granger_best.pt`

This branch adds a separate graph version:

- `data/graphs/granger_vol_pvalues.parquet`
- `data/graphs/granger_vol_edges.parquet`
- `data/graphs/granger_vol_meta.json`
- graph version: `granger_vol_weekly_rv_lag_5_<correction>`

## Definition

For each stock, construct weekly realized volatility from daily log returns:

```text
weekly_rv[week T] = sqrt(sum daily_return^2 during week T)
```

Run ordered-pair Granger tests over training data only. An edge `A -> B` means:

```text
A's past weekly realized volatility helps predict B's weekly realized volatility.
```

This differs from the frozen Granger graph, where `A -> B` means past returns of `A` help predict returns of `B`.

## Workflow

1. Build the graph artifacts:

```bash
uv run python scripts/build_volatility_granger_graph.py
```

For CPU-only runs:

```bash
uv run python scripts/build_volatility_granger_graph.py --use-gpu=false --n-workers 8
```

If p-values already exist and only the multiple-testing correction should be rebuilt:

```bash
uv run python scripts/build_volatility_granger_graph.py --skip-tests
```

2. Train an experimental model using `src.train.train_gnn_volatility_granger`.

Suggested model name and registry row:

```text
experiment_id: gnn_granger_vol
model_name: GNN-Granger-Volatility
model_family: GNN
graph_type: granger_volatility
loss_type: mse
feature_version: stock_features_v1 or stock_features_plus_regime_v1
graph_version: granger_vol_weekly_rv_lag_5_<correction>
checkpoint_path: data/results/checkpoints/gnn_granger_vol_best.pt
```

3. Evaluate against matched baselines.

Primary comparisons:

- `GNN-Granger-Volatility` vs `GNN-Granger`
- `GNN-Granger-Volatility + Macro` vs `GNN-Granger + Macro`

Use the existing evaluation, portfolio, and significance notebooks after prediction artifacts are generated.

## Research Interpretation

If the volatility-Granger model improves over return-Granger, the paper should describe the original result as a return-predictability graph and the new result as evidence from volatility-spillover graph structure.

If it does not improve, the result is still useful: it shows that direct volatility-spillover edges are not automatically better than return-information-flow edges under this model and split.
