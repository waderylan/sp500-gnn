from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "08_final_results.ipynb"


def main() -> None:
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(text.strip()))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md(
        """
# Final Results Tables and Figures

This notebook is an artifact reader for the final paper-facing results. It reads saved CSV/parquet files from `data/results/`, applies lightweight ordering and formatting, writes clean final CSV artifacts, and saves publication-ready figures under `data/results/figures/`.

It does not train models, recompute predictions, or overwrite frozen baseline artifacts.
"""
    )

    code(
        r'''
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
RESULTS = ROOT / "data" / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

FINAL_ROSTER = [
    "HAR per-stock",
    "HAR pooled",
    "LSTM",
    "GNN-Correlation",
    "GNN-Sector",
    "GNN-Granger",
    "GNN-Ensemble",
    "GNN-Correlation + Macro Tuned",
    "GNN-Sector + Macro",
    "GNN-Granger + Macro",
    "GNN-Ensemble + Macro",
]

EXPLORATORY_MODELS = [
    "GNN-Correlation + Macro",
    "LSTM + Macro",
]

PORTFOLIO_ROSTER = FINAL_ROSTER + ["Equal-weight"]

FOCUSED_CALIBRATION_MODELS = [
    "HAR per-stock",
    "LSTM",
    "GNN-Correlation",
    "GNN-Correlation + Macro Tuned",
    "GNN-Sector + Macro",
    "GNN-Granger + Macro",
]

FOCUSED_PORTFOLIO_MODELS = [
    "Equal-weight",
    "HAR per-stock",
    "LSTM",
    "GNN-Ensemble",
    "GNN-Correlation + Macro Tuned",
    "GNN-Ensemble + Macro",
]

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def read_csv(name):
    path = RESULTS / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def ordered(df, roster, model_col="model"):
    out = df[df[model_col].isin(roster)].copy()
    order = {name: i for i, name in enumerate(roster)}
    out["_order"] = out[model_col].map(order)
    out = out.sort_values("_order").drop(columns="_order")
    return out


def save_final(df, name):
    path = RESULTS / name
    df.to_csv(path, index=False)
    return path


def rounded(df, decimals=4):
    out = df.copy()
    numeric_cols = out.select_dtypes(include=["number"]).columns
    out[numeric_cols] = out[numeric_cols].round(decimals)
    return out


def display_table(title, df, decimals=4):
    display(Markdown(f"### {title}"))
    display(rounded(df, decimals))


def model_colors(models):
    colors = []
    for model in models:
        if "Macro" in model:
            colors.append("#2f7d59")
        elif model == "Equal-weight":
            colors.append("#6b7280")
        elif model.startswith("HAR"):
            colors.append("#4c78a8")
        elif model == "LSTM":
            colors.append("#b279a2")
        else:
            colors.append("#f58518")
    return colors


def horizontal_bar(df, x_col, y_col, title, xlabel, path, value_fmt="{:.4f}"):
    plot_df = df[[y_col, x_col]].dropna().copy()
    plot_df = plot_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.barh(
        plot_df[y_col],
        plot_df[x_col],
        color=model_colors(plot_df[y_col]),
        alpha=0.92,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    for bar, value in zip(bars, plot_df[x_col]):
        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            "  " + value_fmt.format(value),
            va="center",
            fontsize=8,
        )
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    return path


print(f"Reading artifacts from: {RESULTS}")
print(f"Saving figures to: {FIGURES}")
'''
    )

    md(
        """
## 1. ML Metrics

The headline ML table is restricted to the final model roster. Exploratory variants such as untuned `GNN-Correlation + Macro` and `LSTM + Macro` are not included in this main table.
"""
    )
    code(
        r'''
ml_metrics = read_csv("ml_metrics_table.csv")
final_ml = ordered(ml_metrics, FINAL_ROSTER)[["model", "mse", "mae", "r2", "da"]]
ml_path = save_final(final_ml, "final_ml_metrics_table.csv")
display_table("Final ML Metrics", final_ml)
print(f"Saved: {ml_path.relative_to(ROOT)}")
'''
    )
    md(
        """
Point-estimate test MSE is lowest for `GNN-Correlation + Macro Tuned` in the saved artifacts. This table should be interpreted with the formal weekly DM tests below rather than as standalone evidence that one model family dominates all baselines.
"""
    )
    code(
        r'''
mse_fig = horizontal_bar(
    final_ml.sort_values("mse", ascending=True),
    x_col="mse",
    y_col="model",
    title="Test MSE by Model",
    xlabel="Test MSE",
    path=FIGURES / "final_test_mse_by_model.png",
    value_fmt="{:.5f}",
)
print(f"Saved: {mse_fig.relative_to(ROOT)}")
'''
    )

    md(
        """
## 2. Rank IC / ICIR

Rank metrics evaluate whether models correctly order stocks by next-week realized volatility. The table includes hit-rate and pairwise-accuracy columns when present in the saved artifact.
"""
    )
    code(
        r'''
rank_ic = read_csv("rank_ic_table.csv")
rank_cols = ["model", "mean_ic", "ic_ir"]
for optional in ["pct_positive", "top_quartile_hit_rate", "pairwise_accuracy"]:
    if optional in rank_ic.columns:
        rank_cols.append(optional)
final_rank = ordered(rank_ic, FINAL_ROSTER)[rank_cols]
rank_path = save_final(final_rank, "final_rank_ic_table.csv")
display_table("Final Rank IC Metrics", final_rank)
print(f"Saved: {rank_path.relative_to(ROOT)}")
'''
    )
    md(
        """
Rank IC and ICIR measure a different objective than MSE. Stronger ranking does not automatically imply better calibrated volatility levels or better portfolio behavior.
"""
    )
    code(
        r'''
rank_fig = horizontal_bar(
    final_rank.sort_values("mean_ic", ascending=True),
    x_col="mean_ic",
    y_col="model",
    title="Mean Rank IC by Model",
    xlabel="Mean weekly Spearman Rank IC",
    path=FIGURES / "final_rank_ic_by_model.png",
    value_fmt="{:.3f}",
)
print(f"Saved: {rank_fig.relative_to(ROOT)}")
'''
    )

    md(
        """
## 3. Calibration

Calibration summaries and decile plots show whether predicted volatility levels line up with realized volatility levels. This matters for inverse-volatility and minimum-variance portfolio construction.
"""
    )
    code(
        r'''
calibration = read_csv("calibration_summary.csv")
calibration_bins = read_csv("calibration_bins.csv")
cal_cols = [
    "model",
    "calibration_slope",
    "calibration_intercept",
    "pearson_corr",
    "prediction_mean",
    "prediction_std",
    "prediction_min",
    "prediction_max",
    "avg_weekly_prediction_spread_p90_p10",
]
final_calibration = ordered(calibration, FINAL_ROSTER)[cal_cols]
cal_path = save_final(final_calibration, "final_calibration_summary.csv")
display_table("Final Calibration Summary", final_calibration)
print(f"Saved: {cal_path.relative_to(ROOT)}")
'''
    )
    md(
        """
A well-ranked model can still be poorly calibrated if its slope, intercept, or prediction spread are distorted. The final narrative should keep ranking and level calibration separate.
"""
    )
    code(
        r'''
plot_bins = calibration_bins[
    calibration_bins["model"].isin(FOCUSED_CALIBRATION_MODELS)
].copy()
fig, ax = plt.subplots(figsize=(7.2, 5.2))
for model in FOCUSED_CALIBRATION_MODELS:
    model_bins = plot_bins[plot_bins["model"] == model].sort_values("decile")
    if model_bins.empty:
        continue
    ax.plot(
        model_bins["predicted_rv_mean"],
        model_bins["actual_rv_mean"],
        marker="o",
        linewidth=1.8,
        label=model,
    )
lims = [
    min(plot_bins["predicted_rv_mean"].min(), plot_bins["actual_rv_mean"].min()),
    max(plot_bins["predicted_rv_mean"].max(), plot_bins["actual_rv_mean"].max()),
]
pad = (lims[1] - lims[0]) * 0.05
lims = [lims[0] - pad, lims[1] + pad]
ax.plot(
    lims,
    lims,
    color="#333333",
    linestyle="--",
    linewidth=1,
    label="45-degree reference",
)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_title("Calibration by Predicted RV Decile")
ax.set_xlabel("Mean predicted RV")
ax.set_ylabel("Mean actual RV")
ax.legend(frameon=False, fontsize=8, loc="best")
fig.tight_layout()
cal_fig = FIGURES / "final_calibration_by_predicted_rv.png"
fig.savefig(cal_fig, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved: {cal_fig.relative_to(ROOT)}")
'''
    )

    md(
        """
## 4. Portfolio Metrics

These tables summarize the saved portfolio evaluations by strategy. Equal-weight is retained where it exists as a strategy benchmark.
"""
    )
    code(
        r'''
portfolio_specs = [
    (
        "Inverse-Volatility",
        "portfolio_metrics_table.csv",
        "final_portfolio_inverse_vol_metrics.csv",
        PORTFOLIO_ROSTER,
    ),
    (
        "Long-Short",
        "portfolio_ls_metrics_table.csv",
        "final_portfolio_long_short_metrics.csv",
        FINAL_ROSTER,
    ),
    (
        "Volatility-Targeted",
        "portfolio_vt_metrics_table.csv",
        "final_portfolio_vol_target_metrics.csv",
        FINAL_ROSTER,
    ),
    (
        "Minimum-Variance",
        "portfolio_mv_metrics_table.csv",
        "final_portfolio_minvar_metrics.csv",
        FINAL_ROSTER,
    ),
]
final_portfolio_tables = {}
for title, source, output, roster in portfolio_specs:
    df = read_csv(source)
    cols = [
        c
        for c in [
            "model",
            "ann_return",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "avg_turnover",
            "max_single_stock_weight",
            "max_long_weight",
            "avg_equity_weight",
        ]
        if c in df.columns
    ]
    final_df = ordered(df, roster)[cols]
    final_portfolio_tables[title] = final_df
    out_path = save_final(final_df, output)
    display_table(f"Final Portfolio Metrics: {title}", final_df)
    print(f"Saved: {out_path.relative_to(ROOT)}")
'''
    )
    md(
        """
Portfolio outcomes depend on construction choices as well as forecast quality. The final paper should avoid treating a point-forecast win as equivalent to an investable performance win.
"""
    )
    code(
        r'''
returns = pd.read_parquet(RESULTS / "portfolio_returns.parquet")
returns["week"] = pd.to_datetime(returns["week"])
plot_returns = returns[returns["model"].isin(FOCUSED_PORTFOLIO_MODELS)].copy()
plot_returns = plot_returns.sort_values(["model", "week"])
plot_returns["cum_net_return"] = plot_returns.groupby("model")[
    "net_return"
].transform(lambda s: (1 + s).cumprod() - 1)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
for model in FOCUSED_PORTFOLIO_MODELS:
    model_returns = plot_returns[plot_returns["model"] == model]
    if model_returns.empty:
        continue
    ax.plot(
        model_returns["week"],
        model_returns["cum_net_return"],
        linewidth=1.8,
        label=model,
    )
ax.set_title("Inverse-Volatility Portfolio Cumulative Net Returns")
ax.set_xlabel("")
ax.set_ylabel("Cumulative net return")
ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
ax.legend(frameon=False, fontsize=8, loc="best")
fig.autofmt_xdate()
fig.tight_layout()
portfolio_fig = FIGURES / "final_portfolio_cumulative_returns.png"
fig.savefig(portfolio_fig, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved: {portfolio_fig.relative_to(ROOT)}")
'''
    )

    md(
        """
## 5. DM Tests

The Diebold-Mariano tests use weekly MSE series, where each weekly loss is averaged across stocks. This preserves the time-series structure of the forecast errors better than comparing only pooled scalar MSE values.
"""
    )
    code(
        r'''
dm = read_csv("dm_test_results.csv")
macro_dm = read_csv("macro_dm_test_results.csv")

main_dm = dm[
    dm["model"].isin(FINAL_ROSTER)
    & dm["baseline"].isin(
        [
            "HAR per-stock",
            "LSTM",
            "GNN-Correlation",
            "GNN-Sector",
            "GNN-Granger",
            "GNN-Ensemble",
        ]
    )
].copy()
macro_dm_final = macro_dm[
    macro_dm["model"].isin(FINAL_ROSTER)
    | macro_dm["model"].isin(EXPLORATORY_MODELS)
].copy()
main_dm["test_group"] = "final_vs_baseline"
macro_dm_final["test_group"] = "macro_vs_matched_baseline"
final_dm = pd.concat([main_dm, macro_dm_final], ignore_index=True, sort=False)
final_dm = final_dm[
    [
        "test_group",
        "model",
        "baseline",
        "dm_stat",
        "p_value",
        "p_value_bh",
        "rejected_bh",
        "mean_loss_diff",
        "n_weeks",
    ]
]
final_dm = final_dm.sort_values(
    ["test_group", "p_value_bh", "model", "baseline"]
)
dm_path = save_final(final_dm, "final_dm_test_summary.csv")
display_table("Final DM Test Summary", final_dm, decimals=5)
print(f"Saved: {dm_path.relative_to(ROOT)}")
'''
    )
    md(
        """
The statistical-test section should govern final performance claims. A macro point-estimate improvement should not be described as statistically significant unless the adjusted tests support that statement.
"""
    )

    md(
        """
## 6. Bootstrap Sharpe Intervals

Sharpe intervals and Sharpe-difference intervals are read from saved block-bootstrap outputs. These use weekly portfolio return series rather than summary metrics alone.
"""
    )
    code(
        r'''
bootstrap = read_csv("bootstrap_sharpe_ci.csv")
macro_bootstrap = read_csv("macro_bootstrap_sharpe_ci.csv")

boot_main = bootstrap[bootstrap["model"].isin(PORTFOLIO_ROSTER)].copy()
macro_boot = macro_bootstrap[
    macro_bootstrap["model"].isin(FINAL_ROSTER)
    | macro_bootstrap["model"].isin(EXPLORATORY_MODELS)
].copy()
boot_main["summary_type"] = "sharpe_interval"
macro_boot["summary_type"] = "macro_sharpe_difference"

boot_cols = [
    "summary_type",
    "strategy",
    "model",
    "benchmark",
    "baseline",
    "comparison",
    "point_estimate",
    "ci_lower",
    "ci_upper",
    "n_weeks",
    "block_size",
    "n_bootstrap",
]
for col in boot_cols:
    if col not in boot_main.columns:
        boot_main[col] = np.nan
    if col not in macro_boot.columns:
        macro_boot[col] = np.nan
final_bootstrap = pd.concat(
    [boot_main[boot_cols], macro_boot[boot_cols]], ignore_index=True
)
final_bootstrap = final_bootstrap.sort_values(
    ["summary_type", "strategy", "model", "comparison"]
)
boot_path = save_final(final_bootstrap, "final_bootstrap_sharpe_summary.csv")
display_table("Final Bootstrap Sharpe Summary", final_bootstrap, decimals=4)
print(f"Saved: {boot_path.relative_to(ROOT)}")
'''
    )
    md(
        """
Sharpe intervals are wide over the available 103-week test window. The final discussion should distinguish economically meaningful point estimates from intervals that include zero or overlap materially.
"""
    )

    md(
        """
## 7. Macro-vs-Baseline Deltas

This section is explicitly a macro comparison table. It includes the tuned macro correlation GNN and also keeps exploratory macro variants clearly separated from the main headline model tables.
"""
    )
    code(
        r'''
macro_ml = read_csv("macro_ml_metric_deltas.csv")
macro_port = read_csv("macro_portfolio_metric_deltas.csv")

macro_ml_final = macro_ml[
    macro_ml["macro_model"].isin(FINAL_ROSTER + EXPLORATORY_MODELS)
].copy()
macro_ml_final["strategy"] = "ml_metrics"
macro_ml_final["delta_sharpe"] = np.nan
macro_ml_final["delta_ann_return"] = np.nan
macro_ml_final["delta_source"] = "ML"
macro_ml_view = macro_ml_final[
    [
        "delta_source",
        "strategy",
        "baseline_model",
        "macro_model",
        "delta_mse",
        "delta_mae",
        "delta_r2",
        "delta_da",
        "delta_ann_return",
        "delta_sharpe",
    ]
]

macro_port_final = macro_port[
    macro_port["macro_model"].isin(FINAL_ROSTER + EXPLORATORY_MODELS)
].copy()
macro_port_final["delta_mse"] = np.nan
macro_port_final["delta_mae"] = np.nan
macro_port_final["delta_r2"] = np.nan
macro_port_final["delta_da"] = np.nan
macro_port_final["delta_source"] = "Portfolio"
macro_port_view = macro_port_final[
    [
        "delta_source",
        "strategy",
        "baseline_model",
        "macro_model",
        "delta_mse",
        "delta_mae",
        "delta_r2",
        "delta_da",
        "delta_ann_return",
        "delta_sharpe",
    ]
]

final_macro_deltas = pd.concat([macro_ml_view, macro_port_view], ignore_index=True)
final_macro_deltas = final_macro_deltas.sort_values(
    ["delta_source", "strategy", "baseline_model", "macro_model"]
)
macro_delta_path = save_final(
    final_macro_deltas, "final_macro_vs_baseline_deltas.csv"
)
display_table("Final Macro-vs-Baseline Deltas", final_macro_deltas, decimals=5)
print(f"Saved: {macro_delta_path.relative_to(ROOT)}")
'''
    )
    code(
        r'''
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
ml_plot = macro_ml_final.sort_values("delta_mse")
axes[0].barh(
    ml_plot["macro_model"],
    ml_plot["delta_mse"],
    color=["#2f7d59" if v < 0 else "#9ca3af" for v in ml_plot["delta_mse"]],
)
axes[0].axvline(0, color="#333333", linewidth=0.8)
axes[0].set_title("Macro Delta: MSE")
axes[0].set_xlabel("Macro minus baseline MSE")
axes[0].set_ylabel("")

port_plot = macro_port_final[macro_port_final["strategy"] == "inverse_vol"].sort_values(
    "delta_sharpe"
)
axes[1].barh(
    port_plot["macro_model"],
    port_plot["delta_sharpe"],
    color=[
        "#2f7d59" if v > 0 else "#9ca3af" for v in port_plot["delta_sharpe"]
    ],
)
axes[1].axvline(0, color="#333333", linewidth=0.8)
axes[1].set_title("Macro Delta: Inverse-Vol Sharpe")
axes[1].set_xlabel("Macro minus baseline Sharpe")
axes[1].set_ylabel("")
fig.tight_layout()
macro_delta_fig = FIGURES / "final_macro_vs_baseline_deltas.png"
fig.savefig(macro_delta_fig, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved: {macro_delta_fig.relative_to(ROOT)}")
'''
    )
    md(
        """
The tuned macro correlation GNN has the best saved point-estimate MSE, while macro Sharpe differences vary by strategy and baseline. Final claims should follow the matched DM and bootstrap summaries rather than the delta table alone.
"""
    )

    md(
        """
## 8. Significance Summary

This compact summary combines the saved baseline and macro significance summaries for paper-facing review.
"""
    )
    code(
        r'''
sig = read_csv("significance_summary.csv")
macro_sig = read_csv("macro_significance_summary.csv")
sig["source"] = "baseline_and_final"
macro_sig["source"] = "macro_comparisons"
final_sig = pd.concat([sig, macro_sig], ignore_index=True, sort=False)
final_sig = final_sig[["source", "section", "metric", "value", "details"]]
sig_path = save_final(final_sig, "final_significance_summary.csv")
display_table("Final Significance Summary", final_sig, decimals=5)
print(f"Saved: {sig_path.relative_to(ROOT)}")
'''
    )
    code(
        r'''
plot_sig = final_sig.copy()
plot_sig["label"] = plot_sig["source"] + " | " + plot_sig["metric"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.barh(plot_sig["label"].iloc[::-1], plot_sig["value"].iloc[::-1], color="#4c78a8")
ax.set_title("Significance Summary Counts and Minimum Adjusted P-Values")
ax.set_xlabel("Saved summary value")
ax.set_ylabel("")
fig.tight_layout()
sig_fig = FIGURES / "final_significance_summary.png"
fig.savefig(sig_fig, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved: {sig_fig.relative_to(ROOT)}")
'''
    )
    md(
        """
The saved significance artifacts indicate that final model comparisons and macro-vs-baseline comparisons should be discussed cautiously. The weekly DM tests and block-bootstrap intervals are the basis for any statistical language in the paper.
"""
    )

    md(
        """
## 9. Appendix: Exploratory Macro Variants

The following rows are intentionally excluded from headline tables because they are exploratory or not part of the final model roster. They are displayed here for auditability only.
"""
    )
    code(
        r'''
appendix_rows = []
for source_name, df in [
    ("ML metrics", ml_metrics),
    ("Rank IC", rank_ic),
    ("Calibration", calibration),
]:
    subset = df[df["model"].isin(EXPLORATORY_MODELS)].copy()
    if subset.empty:
        continue
    subset.insert(0, "source", source_name)
    appendix_rows.append(subset)

if appendix_rows:
    display(Markdown("### Exploratory Model Rows"))
    for subset in appendix_rows:
        display(rounded(subset, 5))
else:
    display(Markdown("No exploratory model rows found in the saved artifacts."))
'''
    )

    md(
        """
## Output Manifest

The notebook wrote paper-facing CSVs under `data/results/` and PNG figures under `data/results/figures/`.
"""
    )
    code(
        r'''
created_csvs = [
    "final_ml_metrics_table.csv",
    "final_rank_ic_table.csv",
    "final_calibration_summary.csv",
    "final_portfolio_inverse_vol_metrics.csv",
    "final_portfolio_long_short_metrics.csv",
    "final_portfolio_vol_target_metrics.csv",
    "final_portfolio_minvar_metrics.csv",
    "final_dm_test_summary.csv",
    "final_bootstrap_sharpe_summary.csv",
    "final_macro_vs_baseline_deltas.csv",
    "final_significance_summary.csv",
]
created_figures = [
    "final_test_mse_by_model.png",
    "final_rank_ic_by_model.png",
    "final_calibration_by_predicted_rv.png",
    "final_portfolio_cumulative_returns.png",
    "final_macro_vs_baseline_deltas.png",
    "final_significance_summary.png",
]
manifest = pd.DataFrame(
    {
        "artifact": created_csvs + created_figures,
        "path": [str((RESULTS / name).relative_to(ROOT)) for name in created_csvs]
        + [str((FIGURES / name).relative_to(ROOT)) for name in created_figures],
        "exists": [(RESULTS / name).exists() for name in created_csvs]
        + [(FIGURES / name).exists() for name in created_figures],
    }
)
display(manifest)
assert manifest["exists"].all(), "One or more final artifacts were not created."
'''
    )

    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH}")


if __name__ == "__main__":
    main()
