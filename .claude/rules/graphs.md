# Rules: src/graphs.py

Loaded when working on graph construction or the make_pyg_data() helper.

---

## Three Graph Types — Quick Reference

| Graph | Directed? | Dynamic? | Key Config |
|---|---|---|---|
| Correlation | No | Yes (weekly) | CORR_THRESHOLD, CORR_LOOKBACK_DAYS=252 |
| Sector | No | Annually (point-in-time) | sector_history.json, updated each calendar year |
| Granger | Yes | No (static, train period only) | GRANGER_LAG=5, Bonferroni or BH correction |

---

## Correlation Graph

```python
def build_correlation_graph(log_returns: pd.DataFrame, date: pd.Timestamp,
                             lookback: int, threshold: float) -> torch.LongTensor:
    """
    Build undirected correlation graph for a given date.
    
    Lookahead safety: window ends AT `date` (inclusive). The function is called
    with `date` = last trading day before the prediction week starts, so no
    future data enters.
    
    Returns edge_index of shape (2, num_edges) as LongTensor.
    Both directions included (undirected: if A->B then B->A).
    Self-loops excluded.
    """
```

The threshold θ is tuned on validation. Three values are tested: {0.3, 0.5, 0.7}.
Results saved to data/results/corr_threshold_ablation.json.
The chosen threshold is stored in config.CORR_THRESHOLD after ablation.

---

## Sector Graph

```python
def build_sector_graph(tickers: list, year: int,
                        sector_history: dict) -> torch.LongTensor:
    """
    Build undirected sector graph for a given calendar year.
    
    Connects all pairs of stocks sharing the same GICS sector in `year`.
    Uses sector_history[ticker][str(year)] for assignment.
    Both directions included. Self-loops excluded.
    
    Returns edge_index of shape (2, num_edges) as LongTensor.
    """
```

Pre-build graphs for all years 2015–2025 and cache as a dict `{year: edge_index}`.
Saves to data/graphs/sector_edges_by_year.parquet.

---

## Granger Graph — Critical Notes

The Granger graph is directed. Edge A→B means "A's past returns Granger-cause B's returns."

**Multiple comparison correction:**
- Primary: Bonferroni at α=0.05 / num_pairs
- Fallback if resulting graph has <500 edges: Benjamini-Hochberg FDR
- Whichever is used, store the method name in config.GRANGER_CORRECTION and document it in the paper

**SAGEConv directionality — VERIFY BEFORE TRAINING:**
SAGEConv must be initialized with `flow=config.SAGE_FLOW` ("source_to_target").
Before any GNN-Granger training run, confirm that reversing the edge_index produces different outputs.

```python
# Verification test (run once in 03_graphs.ipynb):
out_forward = model(x, edge_index)
out_reversed = model(x, edge_index[[1, 0]])  # flip src and dst
assert not torch.allclose(out_forward, out_reversed), \
    "SAGEConv is symmetrizing edges — directionality not working"
```

If the assertion fails, do not proceed with GNN-Granger until the issue is resolved.

---

## make_pyg_data() Helper

```python
def make_pyg_data(features_t: torch.Tensor, edge_index: torch.LongTensor,
                   target_t: torch.Tensor) -> Data:
    """
    Construct a PyTorch Geometric Data object for one time step.
    
    features_t: shape (num_stocks, num_features)
    edge_index:  shape (2, num_edges) — LongTensor
    target_t:    shape (num_stocks,)
    
    Returns Data(x=features_t, edge_index=edge_index, y=target_t)
    Asserts: x.shape[0] == y.shape[0], edge_index.max() < x.shape[0]
    """
```

Always validate that edge indices don't reference out-of-bounds nodes.

---

## Graph Density Expectations

After construction, log these stats and confirm they're in expected range:

| Graph | Expected Edge Count | Notes |
|---|---|---|
| Correlation (θ=0.5, calm period) | 2,000–8,000 | Rises during high-vol regimes |
| Correlation (θ=0.5, COVID week) | 15,000–40,000 | Near-full connectivity expected |
| Sector (any year) | ~3,000–5,000 | Depends on sector size distribution |
| Granger (Bonferroni) | 200–2,000 | Very sparse; check before proceeding |

If Granger produces 0 edges after Bonferroni, switch to BH immediately.
