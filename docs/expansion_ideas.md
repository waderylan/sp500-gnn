# GNN Expansion Ideas

Generated after completing Phase 5.2 (portfolio backtest). All ideas below are ranked by expected impact on portfolio performance and paper contribution.

---

## Diagnosis

The 0.999+ return correlation between all models is the core problem, and it is structural. With 465 stocks and `MAX_WEIGHT=0.05`, inverse-volatility weighting collapses to near-equal-weight no matter what the predictions are. The max single-stock weight for most models is 0.002-0.005 — essentially the same as equal-weight's 0.002. The GNN would need radically differentiated cross-sectional predictions to move the needle here. Better MSE alone will not fix this.

There are two independent problems: (1) model predictions are not differentiated enough cross-sectionally, and (2) the portfolio construction cannot amplify whatever cross-sectional signal does exist.

---

## Tier 1 — Highest expected impact

### 1. Long-short portfolio construction

**What:** Instead of long-only inverse-vol, go long the bottom quartile of predicted RV (low-vol stocks) and short the top quartile (high-vol stocks). Dollar-neutral.

**Why it works:** This construction *isolates* the cross-sectional signal. The GNN's edge over baselines becomes visible because you are directly exploiting ranking accuracy rather than diluting it across 465 long-only positions. A long-only inverse-vol strategy on a 465-stock universe is near-identical to equal-weight regardless of prediction quality; a long-short strategy is not.

**Effort:** No retraining. Add a second portfolio variant in `src/portfolio.py` and a new column in the metrics table.

**Paper angle:** Shows the existing GNN signal is real and economically significant. Gives the paper a strong portfolio result even if the long-only Sharpe is modest.

---

### 2. Rank-based training loss

**What:** Replace MSE with a differentiable rank correlation loss (ListMLE or soft Spearman) in `train_gnn()`.

**Why it works:** For portfolio construction, only the cross-sectional *ranking* of predicted RV matters. Whether you predict 0.18 or 0.22 for a stock is irrelevant as long as the ordering is correct. MSE penalizes absolute errors; a rank loss penalizes ordering errors. This directly aligns the training objective with what the portfolio rewards.

**Effort:** One new loss function in `src/train.py`. Retrain all three GNN variants.

**Paper angle:** Demonstrates task-aligned training. Compare rank IC (Spearman correlation between predicted and actual RV) before and after the loss change.

---

### 3. Multi-graph fusion

**What:** Combine all three graph types into one model. Run separate SAGEConv message passing on each edge set (correlation, sector, Granger), concatenate or sum the resulting node representations, then pass to the output head.

**Why it works:** Each graph encodes a different type of inter-stock relationship. Correlation captures co-movement. Sector captures fundamental similarity. Granger captures directional lead-lag. Training three separate models and picking the best discards the complementary information in the other two.

**Effort:** New model class in `src/models.py`. Forward signature becomes `forward(x, ei_corr, ei_sector, ei_granger)`. Retrain once.

**Paper angle:** The primary architectural contribution. The multi-relational design is the novel part of the paper.

---

## Tier 2 — Meaningful gains, moderate effort

### 4. GNN ensemble

**What:** Average test predictions from GNN-Correlation, GNN-Sector, and GNN-Granger weighted by `1/val_MSE`.

**Why it works:** Ensemble averaging reduces variance. Each graph type makes different errors; their average is more stable than any individual prediction.

**Effort:** No retraining. Compute in the evaluation notebook and add a "GNN-Ensemble" row to the metrics table.

**Paper angle:** Free performance gain. Serves as an upper bound on what the multi-graph fusion model should beat.

---

### 5. Graph Attention (GATConv) instead of SAGEConv

**What:** Replace `SAGEConv` with `GATConv` in `GNNModel`. GATConv learns attention weights over neighbors rather than averaging them uniformly.

**Why it works:** Some neighboring stocks are more informative than others. In the correlation graph, a stock with 0.9 correlation should have more influence than one at 0.31 (just above threshold). SAGEConv cannot represent this distinction; GATConv can.

**Effort:** Near drop-in replacement in `GNNModel.__init__`. Attention heads are a new hyperparameter to tune. Retrain.

**Paper angle:** Provides interpretability: the learned attention weights reveal which neighbor relationships the model considers most predictive.

---

### 6. Temporal GNN (node-level GRU)

**What:** After graph convolution, pass each node's embedding through a GRU conditioned on the previous week's hidden state before the output head.

**Why it works:** The current GNN sees each week in complete isolation. Volatility is highly persistent (high-vol weeks cluster). A temporal component lets the model condition predictions on recent history at the node level, which is complementary to the cross-sectional message passing.

**Effort:** Moderate. Requires managing per-node hidden states across time steps in the training loop. New model class in `src/models.py`. Changes to `train_gnn()` to pass hidden states forward chronologically.

**Paper angle:** Positions the model as a spatio-temporal GNN, which is a more compelling contribution than a purely cross-sectional one.

---

### 7. Rank IC as the primary evaluation metric

**What:** Add Spearman rank correlation between predicted and actual RV, computed cross-sectionally at each test week, then averaged. Report mean IC, IC t-stat, and IR (IC / std(IC)).

**Why it works:** MSE measures absolute prediction error. Rank IC measures whether the model correctly identifies which stocks will be relatively more or less volatile. IC is the standard metric in quantitative finance for evaluating cross-sectional predictors. It will likely show clearer GNN advantage over baselines than MSE does.

**Effort:** Compute in `notebooks/05_evaluate.ipynb`. No new training. Add to the significance tests.

**Paper angle:** Bridges the gap between ML evaluation (MSE) and finance evaluation (IC). Reviewers from a finance background will expect this metric.

---

## Tier 3 — Solid paper contributions, longer horizon

### 8. Rolling Granger graph

**What:** Instead of estimating the Granger graph once on all training data, recompute it annually on a rolling 5-year window (e.g., for predictions in 2022, use data from 2017-2021).

**Why it works:** Causality structure changes across market regimes. The 2022 rate-hike regime has different lead-lag relationships than the 2017-2019 low-vol expansion. A static graph estimated on 2015-2022 data averages over these regimes and may not reflect the structure at any given point in time.

**Effort:** Significant. Need to rerun Granger tests for each rolling window, store multiple edge sets, and look up the correct graph at inference time. Computationally expensive.

**Paper angle:** Addresses the most obvious weakness of the current Granger design. Worth noting in the paper even if only the static version is implemented, as a known limitation.

---

### 9. Edge features in correlation graph

**What:** Instead of a binary edge (|corr| > theta), pass the actual correlation value as an edge attribute. Use a conv operator that supports edge features (e.g., `NNConv`, `GATConv` with edge attributes, or a custom MessagePassing layer).

**Why it works:** A correlation of 0.9 carries much more information than 0.31. The binary threshold discards this. Edge features let the model scale each neighbor's contribution by the strength of the relationship.

**Effort:** Moderate. Extend `build_correlation_graph()` in `src/graphs.py` to return edge weights alongside `edge_index`. Update model forward pass and training loop.

**Paper angle:** A natural extension of the correlation graph design. Easy to motivate.

---

### 10. Macro features as global node features

**What:** Add VIX level, 10Y-2Y Treasury yield spread, and IG credit spread (OAS) as three additional features at each week. Every stock receives the same macro values at each time step.

**Why it works:** These features signal market-wide volatility regimes. The GNN currently has no way to distinguish a calm 2017 week from a stressed 2022 week other than through the node features. Macro features provide this context without any lookahead risk (all three are available before the start of each prediction week).

**Effort:** Low. Add three new columns to `features.parquet` in `src/features.py`. FRED has all three series (VIX: VIXCLS, 10Y-2Y: T10Y2Y, IG OAS: BAMLC0A0CM).

**Paper angle:** Incorporates macro conditioning at no architectural cost. Adds to the story that the model adapts to market regimes.

---

---

## GNN Architecture and Training Improvements

These ideas are specific to the GNN models and are independent of portfolio construction. They target prediction quality directly — better cross-sectional ranking of RV, lower MSE, and better generalization across market regimes.

---

### G1. Jumping Knowledge (JK-Net) aggregation

**What:** Instead of using only the final layer's node embeddings, concatenate (or take the max of) embeddings from all GNN layers before the output head. This gives each node access to both local (layer 1) and wider-neighborhood (layer 2-3) information simultaneously.

**Why it works:** Different stocks may benefit from different receptive field depths. A stock with few but highly correlated neighbors may be best predicted by layer-1 embeddings; a stock at the center of a dense sector cluster may benefit from layer-3 aggregation. JK-Net lets the model choose. It also mitigates oversmoothing — with 3 SAGEConv layers and the full correlation graph, node representations start converging and losing individuality.

**Effort:** Low. Add a final `torch.cat([h1, h2, h3], dim=-1)` before the output linear layer and update `in_channels` of the head accordingly. No changes to training loop.

**Where:** `src/models.py` in `GNNModelV2`.

---

### G2. DropEdge regularization

**What:** During training, randomly drop a fraction of edges at each forward pass (e.g., drop 10-20% of edges). At inference time, use the full graph.

**Why it works:** Acts as a graph-specific regularizer. Forces the model to not over-rely on any single neighbor relationship, which improves generalization and reduces oversmoothing in dense graphs. Particularly relevant for the correlation graph during COVID weeks when the graph is near-fully connected — without DropEdge, every stock receives almost identical aggregated messages.

**Effort:** Low. Wrap `edge_index` with `dropout_adj` from `torch_geometric.utils` before each forward call in the training loop. One new config constant `EDGE_DROPOUT_P`.

**Where:** `src/train.py` in the training step, `config.py` for the rate.

---

### G3. Residual connections between GNN layers

**What:** Add a skip connection from each layer's input to its output: `h = conv(h_prev) + h_prev` (after a linear projection if dimensions differ).

**Why it works:** Prevents the gradient vanishing that makes deep GNNs hard to train. With 3 layers and a moderately large hidden dim (128), the gradients reaching the first layer can be weak. Residual connections also preserve each node's individual features alongside the aggregated neighborhood signal, which is important when some stocks have highly distinctive RV patterns not well-explained by their neighbors.

**Effort:** Low. Add `self.skip = nn.Linear(in_channels, hidden_dim)` and modify the forward pass. Already partially structured for this in `GNNModelV2`.

**Where:** `src/models.py` in `GNNModelV2`.

---

### G4. GraphNorm instead of BatchNorm

**What:** Replace `BatchNorm1d` (which normalizes across the node dimension) with `GraphNorm` from PyG, which normalizes within each graph while preserving inter-graph variance.

**Why it works:** `BatchNorm1d` applied to node features treats all 465 stocks as a batch, normalizing away cross-sectional variance — exactly the signal you need to preserve for cross-sectional ranking. `GraphNorm` normalizes within a single time step's graph while keeping the relative structure of node embeddings intact.

**Effort:** Low. Swap `nn.BatchNorm1d` for `torch_geometric.nn.norm.GraphNorm` in `GNNModelV2`. The `batch` argument to GraphNorm is `None` when processing one graph at a time, which is the current setup.

**Where:** `src/models.py` in `GNNModelV2`.

---

### G5. Node positional encodings

**What:** Augment input node features with static stock-level encodings: sector one-hot (11 dims), log market cap rank (1 dim), and index beta (1 dim). These are concatenated to `x` before the first GNN layer.

**Why it works:** The current GNN has no way to distinguish node identity beyond the feature vector. Two stocks with identical RV history look identical to the model even if one is a mega-cap tech stock and the other is a small-cap utility. Positional encodings give the model structural anchors — it can learn, for example, that tech stocks tend to exhibit volatility clustering differently than utilities.

**Effort:** Moderate. Compute the encodings once and store in a lookup table indexed by ticker position. Concatenate at the start of each forward pass. Market cap data is available from yfinance; beta can be estimated from the training returns.

**Where:** `src/features.py` for computation, `src/models.py` for integration.

---

### G6. Heterogeneous GNN (separate weights per graph type)

**What:** In the multi-graph fusion model (idea 3 above), use separate SAGEConv weight matrices for each edge type rather than sharing weights. This is the standard heterogeneous GNN design (`HeteroConv` in PyG).

**Why it works:** Correlation, sector, and Granger edges represent fundamentally different relationships. The message a stock should receive from a Granger-causally-linked neighbor is not the same function as the message from a same-sector neighbor. Sharing weights forces the model to use one aggregation function for all three; separate weights let each graph type learn its own message function.

**Effort:** Moderate. Refactor the fusion model to use `torch_geometric.nn.HeteroConv` with a dict of edge types. Requires converting the graph to `HeteroData` format.

**Where:** `src/models.py`, new model class. `src/graphs.py` for data format conversion.

---

### G7. Oversmoothing audit

**What:** Before adding more layers or complexity, measure how much oversmoothing is occurring by computing mean absolute difference (MAD) between node embeddings at each layer. If MAD collapses toward zero by layer 3, the model is oversmoothing.

**Why it matters:** With 3 SAGEConv layers and a dense correlation graph (15,000-40,000 edges during high-vol regimes), every node receives messages from almost every other node. After 3 hops, all embeddings converge. This would explain why GNN-Correlation underperforms the other GNNs despite having the richest graph — the dense graph may be collapsing the cross-sectional signal.

**Effort:** Zero additional training. Add a diagnostic cell in `notebooks/04_gnn_models.ipynb` that hooks into each layer and computes `MAD = mean(|h_i - h_j|)` over a sample week. If MAD at layer 3 is less than 10% of MAD at layer 1, oversmoothing is severe and the fixes are: DropEdge (G2), residual connections (G3), or reducing to 2 layers.

**Where:** `notebooks/04_gnn_models.ipynb` as a diagnostic cell.

---

### G8. Asymmetric loss (penalize underestimation more)

**What:** Use a weighted MSE or Huber loss that penalizes underestimating RV more than overestimating it.

**Why it works:** In a portfolio context, underestimating a stock's volatility causes you to overweight it, concentrating risk. Overestimating causes you to underweight it, leaving return on the table — a milder consequence. An asymmetric loss that applies, say, 2x penalty when `pred < actual` would push the model toward slightly conservative predictions and reduce the worst-case portfolio blowups.

**Effort:** Low. One new loss function in `src/train.py`. One new config constant for the asymmetry ratio.

**Where:** `src/train.py`, `config.py`.

---

### G9. Contrastive pre-training on sector membership

**What:** Pre-train the GNN with a contrastive objective: node embeddings from stocks in the same sector should be closer in embedding space than embeddings from different sectors. Then fine-tune on the RV prediction task.

**Why it works:** The sector graph encodes industry structure that is informative for RV prediction but hard to learn from a supervised MSE signal alone. Pre-training on sector membership gives the GNN a head start: it learns to produce sector-coherent embeddings before it ever sees a target RV value. Fine-tuning then adjusts these embeddings toward RV prediction.

**Effort:** Significant. Requires a separate pre-training loop with contrastive loss (e.g., NT-Xent or triplet loss), followed by fine-tuning. Worth attempting only after the simpler architectural improvements are exhausted.

**Where:** `src/train.py`, new `pretrain_gnn_contrastive()` function. `src/models.py` for the projection head used during pre-training.

---

### G10. Spectral baseline (ChebConv)

**What:** Train a version of the GNN using `ChebConv` (Chebyshev spectral convolution) instead of SAGEConv. ChebConv approximates spectral graph convolution using Chebyshev polynomials of the graph Laplacian.

**Why it works:** SAGEConv is a spatial method — it aggregates neighbor features directly. ChebConv is a spectral method — it operates on the frequency decomposition of the graph signal. For the correlation graph, low-frequency graph signals correspond to market-wide or sector-wide volatility patterns; high-frequency signals correspond to idiosyncratic stock behavior. ChebConv lets the model separately weight these frequency components, which SAGEConv cannot.

**Effort:** Low. Drop-in replacement for SAGEConv. `ChebConv(in_channels, hidden_dim, K=3)` where K is the polynomial order (controls the receptive field).

**Where:** `src/models.py`, new ablation variant.

---

## Recommended sequence

If implementing all of the above, the following order minimizes wasted effort:

1. **Long-short portfolio** (idea 1) — no retraining, validates existing signal
2. **Rank IC metric** (idea 7) — no retraining, gives a cleaner evaluation picture before retraining
3. **GNN ensemble** (idea 4) — no retraining, sets a ceiling for fusion model to beat
4. **Macro features** (idea 10) — low effort, improves all subsequent models
5. **Rank-based loss** (idea 2) — retrain all three GNNs with improved features + new loss
6. **Multi-graph fusion** (idea 3) — the main architectural contribution; build on top of rank loss + macro features
7. **GATConv** (idea 5) — swap into the fusion model as an ablation
8. **Temporal GNN** (idea 6) — highest complexity, implement last once the static design is stable
9. **Edge features** (idea 9) — natural extension once GATConv is in place
10. **Rolling Granger** (idea 8) — most expensive; implement if time allows before submission
