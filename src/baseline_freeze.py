"""
Freeze and read the Phase 0 baseline result artifacts.

This module creates an immutable timestamped copy of the current result files,
checkpoints, predictions, portfolio returns, and model roster before later
experiments change features, training objectives, or evaluation outputs. The
latest manifest remains at ``data/results/frozen_baseline_manifest.json`` so
notebooks can display the frozen control condition without re-freezing it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config


CURRENT_MODEL_ROSTER = [
    {
        "model_name": "HAR per-stock",
        "model_family": "HAR",
        "graph_type": "none",
        "loss_type": "squared_error",
        "notes": "Per-stock HAR volatility baseline.",
    },
    {
        "model_name": "HAR pooled",
        "model_family": "HAR",
        "graph_type": "none",
        "loss_type": "squared_error",
        "notes": "Pooled HAR volatility baseline.",
    },
    {
        "model_name": "LSTM",
        "model_family": "LSTM",
        "graph_type": "none",
        "loss_type": "mse",
        "notes": "Sequence neural baseline.",
    },
    {
        "model_name": "GNN-Correlation",
        "model_family": "GNN",
        "graph_type": "correlation",
        "loss_type": "mse",
        "notes": "GraphSAGE model using rolling correlation graphs.",
    },
    {
        "model_name": "GNN-Sector",
        "model_family": "GNN",
        "graph_type": "sector",
        "loss_type": "mse",
        "notes": "GraphSAGE model using same-sector graphs.",
    },
    {
        "model_name": "GNN-Granger",
        "model_family": "GNN",
        "graph_type": "granger",
        "loss_type": "mse",
        "notes": "GraphSAGE model using Granger-causality graphs.",
    },
    {
        "model_name": "GNN-Ensemble",
        "model_family": "GNN ensemble",
        "graph_type": "correlation+sector+granger",
        "loss_type": "mse",
        "notes": "Prediction ensemble of the three MSE-trained GNN variants.",
    },
    {
        "model_name": "Rank-loss GNN-Correlation",
        "model_family": "GNN",
        "graph_type": "correlation",
        "loss_type": "bpr_rank",
        "notes": "Correlation-graph GNN trained with rank loss.",
    },
    {
        "model_name": "Rank-loss GNN-Sector",
        "model_family": "GNN",
        "graph_type": "sector",
        "loss_type": "bpr_rank",
        "notes": "Sector-graph GNN trained with rank loss.",
    },
    {
        "model_name": "Rank-loss GNN-Granger",
        "model_family": "GNN",
        "graph_type": "granger",
        "loss_type": "bpr_rank",
        "notes": "Granger-graph GNN trained with rank loss.",
    },
]

REQUIRED_BASELINE_FILES = [
    "ml_metrics_table.csv",
    "rank_ic_table.csv",
    "portfolio_metrics_table.csv",
    "portfolio_ls_metrics_table.csv",
    "portfolio_vt_metrics_table.csv",
    "portfolio_mv_metrics_table.csv",
    "gnn_hparam_search_results.json",
    "corr_threshold_ablation.json",
]

BASELINE_GLOBS = [
    "*_val_preds.parquet",
    "test_preds_*.parquet",
    "portfolio*_returns.parquet",
    "*_val_loss.json",
    "validation_summary.json",
]


def _sha256(path: Path) -> str:
    """Return the SHA-256 checksum for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    """Return a POSIX-style path relative to the results directory."""
    return path.relative_to(root).as_posix()


def collect_baseline_files(results_dir: Path) -> list[Path]:
    """Collect current baseline metric, prediction, validation, and checkpoint files."""
    files: dict[Path, Path] = {}

    for name in REQUIRED_BASELINE_FILES:
        path = results_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Required baseline artifact is missing: {path}")
        files[path] = path

    for pattern in BASELINE_GLOBS:
        for path in results_dir.glob(pattern):
            if path.is_file():
                files[path] = path

    checkpoints_dir = results_dir / "checkpoints"
    if checkpoints_dir.exists():
        for path in checkpoints_dir.rglob("*"):
            if path.is_file():
                files[path] = path

    return sorted(files)


def create_frozen_baseline_snapshot(
    *,
    results_dir: Path | None = None,
    timestamp: str | None = None,
) -> dict:
    """
    Copy current baseline artifacts into a timestamped snapshot directory.

    The original result files are never modified. A manifest with checksums is
    written both inside the snapshot directory and at
    data/results/frozen_baseline_manifest.json as the latest pointer.
    """
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    snapshot_name = f"frozen_baseline_{timestamp}"
    snapshot_dir = results_dir / snapshot_name
    if snapshot_dir.exists():
        raise FileExistsError(f"Snapshot already exists: {snapshot_dir}")

    files = collect_baseline_files(results_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    manifest_files = []
    for source in files:
        relative_path = _relative(source, results_dir)
        destination = snapshot_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        stat = source.stat()
        manifest_files.append(
            {
                "relative_path": relative_path,
                "snapshot_path": _relative(destination, results_dir),
                "size_bytes": stat.st_size,
                "source_mtime_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
                "sha256": _sha256(destination),
            }
        )

    roster_path = snapshot_dir / "model_roster.csv"
    roster = pd.DataFrame(CURRENT_MODEL_ROSTER)
    roster.to_csv(roster_path, index=False)

    latest_roster_path = results_dir / "frozen_baseline_model_roster.csv"
    roster.to_csv(latest_roster_path, index=False)

    manifest = {
        "snapshot_name": snapshot_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_dir": _relative(snapshot_dir, results_dir),
        "results_dir": str(results_dir),
        "model_roster_path": _relative(latest_roster_path, results_dir),
        "snapshot_model_roster_path": _relative(roster_path, results_dir),
        "n_models": len(CURRENT_MODEL_ROSTER),
        "n_files": len(manifest_files),
        "files": manifest_files,
    }

    snapshot_manifest_path = snapshot_dir / "manifest.json"
    snapshot_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    latest_manifest_path = results_dir / "frozen_baseline_manifest.json"
    latest_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return manifest


def load_frozen_baseline_manifest(results_dir: Path | None = None) -> dict:
    """Load the latest frozen-baseline manifest from ``data/results``."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    path = results_dir / "frozen_baseline_manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Frozen baseline manifest not found at {path}. "
            "Run `uv run python -m src.baseline_freeze` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_baseline_roster(results_dir: Path | None = None) -> pd.DataFrame:
    """Load the official frozen-baseline model roster."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    manifest = load_frozen_baseline_manifest(results_dir)
    return pd.read_csv(results_dir / manifest["model_roster_path"])


def load_frozen_baseline_table(relative_path: str, results_dir: Path | None = None) -> pd.DataFrame:
    """Load a CSV table from the latest frozen-baseline snapshot."""
    results_dir = Path(results_dir or config.DATA_RESULTS_DIR)
    manifest = load_frozen_baseline_manifest(results_dir)
    return pd.read_csv(results_dir / manifest["snapshot_dir"] / relative_path)


def main() -> None:
    """Command-line entry point for creating a frozen baseline snapshot."""
    parser = argparse.ArgumentParser(description="Freeze current baseline result artifacts.")
    parser.add_argument("--timestamp", default=None, help="Optional UTC snapshot timestamp.")
    args = parser.parse_args()

    manifest = create_frozen_baseline_snapshot(timestamp=args.timestamp)
    print(f"Created snapshot: {manifest['snapshot_dir']}")
    print(f"Files copied: {manifest['n_files']}")
    print("Latest manifest: data/results/frozen_baseline_manifest.json")


if __name__ == "__main__":
    main()
