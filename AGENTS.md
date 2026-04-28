# AGENTS.md

## Project Context

This repository is an ML/finance research codebase for forecasting S&P 500 realized volatility using classical baselines, sequence models, and graph neural networks. The project currently has a strong exploratory implementation, but the next goal is to make the results credible, reproducible, auditable, and suitable for a research paper.

The main planning document is:

```text
docs/implementation_priority_plan.md
```

Always read that file before making implementation decisions. It defines the ranked backlog, current paper scope, future-work boundary, required notebook outputs, and artifact expectations.

The highest-level research framing to preserve is:

```text
Graph structure contains useful cross-sectional volatility information, but its value depends on regime stability, objective alignment, calibration, and portfolio construction. We show this through controlled comparisons of correlation, sector, and Granger graphs, then improve the setup using regime-aware features and ranking-aware training objectives.
```

Do not overstate that GNNs dominate all baselines unless the final statistical tests support that claim.

---

## Current Priority Order

Follow `docs/implementation_priority_plan.md` exactly unless the user explicitly changes the order.

Immediate priorities:

1. Freeze the current baseline.
2. Create the experiment registry.
3. Fix reproducibility inconsistencies.
4. Implement statistical testing infrastructure.
5. Add diagnostics on the existing baseline.
6. Add market-regime features.
7. Evaluate macro-feature models.
8. Implement mixed rank-MSE loss.
9. Add portfolio robustness checks.
10. Run final significance tests.
11. Prepare paper figures and final narrative.
12. Defer future-work architecture unless diagnostics justify it.

Current paper scope is Phases 0-5. Phases 6-7 are future work unless promoted by the user or justified by diagnostics.

---

## Repository Conventions

Use the existing repository structure. 
This project uses uv.
Do not silently overwrite important baseline results. If an implementation step changes outputs, preserve the old version through a snapshot, manifest, timestamped copy, or clearly named frozen-baseline artifact.

---

## Notebook Rules

For every applicable task, notebooks must show visible reviewed outputs after opening the `.ipynb` file.

Notebook cells should generally be ordered as:

1. Load saved artifacts.
2. Compute or refresh item-specific outputs.
3. Display compact tables and figures.
4. Write reusable artifacts to `data/results/` or `data/results/figures/`.

Do not leave final evidence only in terminal logs.

Use the project environment:

```bash
uv run jupyter lab
```

For executable notebook updates, run:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

If training is too long for full notebook execution, run training through `uv run` scripts and keep the notebook as a reproducible artifact reader that displays saved checkpoints, metrics, tables, and artifact paths.

---

## Implementation Rules

Before coding:

1. Read `docs/implementation_priority_plan.md`.
2. Identify the single next unchecked task.
3. Inspect the relevant existing files before editing.
4. Make the smallest coherent change that completes the current step.
5. Preserve compatibility with existing artifacts unless the task requires a versioned change.

When implementing:

- Prefer small, auditable commits.
- Keep functions reusable and testable.
- Save machine-readable outputs under `data/results/`.
- Save figures under `data/results/figures/`.
- Update notebooks so outputs are visible.
- Update docs only when the implementation changes methodology, assumptions, or reported results.
- Do not start future-work architecture unless explicitly requested.

When changing model behavior:

- Do not compare models unfairly.
- Do not give macro/regime features only to GNNs if LSTM is a comparable neural baseline.
- Do not tune using test results.
- Make feature versions explicit.
- Make graph versions explicit.
- Register checkpoints, configs, metrics, and artifact paths in the experiment registry.

---

## Research Validity Requirements

The project is moving toward a defensible paper. Treat these as hard requirements:

- Baselines must be frozen before major feature/model changes.
- Every result must have provenance.
- Statistical testing must support final performance claims.
- DM tests should use per-week model error series, not only pooled scalar MSE.
- Sharpe intervals should use block bootstrap on actual weekly return series.
- Macro/regime features must obey the lookahead rule.
- Validation data may be used for model selection.
- Test data must not be used for selecting hyperparameters, loss weights, or feature variants.

Lookahead rule for regime features:

```text
feature row T may use data through Friday of week T only
target row T is RV in week T+1
```

Macro/regime features should use train-only time-series normalization. Do not cross-sectionally z-score global macro features after duplicating them across stocks.

---

## Known Issues To Fix Early

The implementation plan calls out these existing inconsistencies:

1. GNN hyperparameter/config mismatch:
   - Some config/docs describe `hidden_dim=128`, `lr=3e-4`.
   - `data/results/gnn_hparam_search_results.json` reports best config as `hidden_dim=256`, `lr=1e-3`, `dropout=0.3`, `batch_norm=False`, `num_layers=3`.
   - Decide which checkpoint/config is official and align code, docs, and registry.

2. LSTM default hidden dimension:
   - `src/models.py` should default `LSTMModel` to `config.LSTM_HIDDEN_DIM`, not `config.HIDDEN_DIM`.

3. Universe construction language:
   - Use the existing current-constituent S&P 500 method and document it consistently.

4. Sector taxonomy:
   - Standardize yfinance sector names into canonical GICS-style labels before relying on sector graphs, sector-neutral portfolios, or ex-sector robustness checks.

---

## Expected Agent Behavior

When given a task, implement only that task unless a small prerequisite is necessary. If a prerequisite is required, explain it briefly and keep the scope tight.

For each completed step, report:

1. Files changed.
2. Artifacts created or updated.
3. Commands run.
4. Tests or notebook executions completed.
5. Any assumptions or unresolved issues.
6. The recommended next task from `docs/implementation_priority_plan.md`.

Do not claim completion if notebooks were not executed or artifacts were not generated. Instead, state exactly what was implemented and what remains to verify.

---

## Suggested Commit Style

Use concise commit messages tied to the implementation plan, for example:

```text
Freeze baseline result artifacts
Add experiment registry
Fix LSTM hidden dimension default
Implement significance testing utilities
Add calibration diagnostics
Add market regime feature pipeline
Add mixed rank-MSE loss sweep
Add sector-neutral portfolio evaluation
```

---

## Stop Conditions

Stop and ask for direction if:

- The next task would overwrite baseline results without a snapshot.
- Required input artifacts are missing.
- Existing code contradicts `docs/implementation_priority_plan.md`.
- A test result suggests the paper framing needs to change.
- The task belongs to future work and was not explicitly requested.

Otherwise, proceed with the next scoped implementation step.
