"""Versioned stock-plus-regime feature dataset construction.

The baseline stock feature artifact remains ``data/features/features.parquet``.
This module reads that artifact, appends train-normalized market-regime
features, and writes a separate macro dataset so old checkpoints stay
compatible with the original input dimension.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config


MACRO_FEATURE_VERSION = "stock_features_plus_regime_v1"
BASE_FEATURE_VERSION = "stock_features_v1"


def _features_dir() -> Path:
    return Path(config.DATA_FEATURES_DIR)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_flat_features(
    *,
    feature_path: Path | None = None,
    meta_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a flat feature parquet plus its metadata."""
    features_dir = _features_dir()
    feature_path = feature_path or features_dir / "features.parquet"
    meta_path = meta_path or features_dir / "features_meta.json"

    df = pd.read_parquet(feature_path)
    df["week"] = pd.to_datetime(df["week"])
    meta = _load_json(meta_path)
    return df, meta


def load_feature_tensor(
    *,
    feature_path: Path | None = None,
    meta_path: Path | None = None,
    target_index: pd.Index | None = None,
) -> tuple[np.ndarray, list[str], list[str], pd.DatetimeIndex]:
    """Reconstruct a 3D feature tensor from a flat feature artifact."""
    df, meta = load_flat_features(feature_path=feature_path, meta_path=meta_path)
    feature_names = list(meta["feature_names"])

    if target_index is not None:
        target_weeks = pd.DatetimeIndex(target_index)
        df = df[df["week"].isin(target_weeks)].copy()

    weeks = pd.DatetimeIndex(sorted(df["week"].unique()))
    tickers = sorted(df["ticker"].unique())
    expected_rows = len(weeks) * len(tickers)
    assert len(df) == expected_rows, (
        f"Flat feature rows are not rectangular: {len(df)} != {expected_rows}"
    )

    arrays = []
    for name in feature_names:
        pivot = df.pivot(index="week", columns="ticker", values=name).reindex(
            index=weeks,
            columns=tickers,
        )
        arrays.append(pivot.to_numpy(dtype=float))
    tensor = np.stack(arrays, axis=2)
    return tensor, feature_names, tickers, weeks


def normalize_regime_features_train_only(
    regime_features: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize global regime features with train-only time-series statistics."""
    regime = regime_features.copy()
    regime.index = pd.to_datetime(regime.index)
    split_frame = splits.copy()
    split_frame["week"] = pd.to_datetime(split_frame["week"])

    train_weeks = pd.DatetimeIndex(split_frame.loc[split_frame["split"] == "train", "week"])
    train_regime = regime.loc[regime.index.intersection(train_weeks)]
    means = train_regime.mean(skipna=True)
    stds = train_regime.std(skipna=True, ddof=1).replace(0.0, np.nan)
    if stds.isna().any():
        missing = stds[stds.isna()].index.tolist()
        raise ValueError(f"Cannot normalize regime features with zero/missing train std: {missing}")

    normalized = regime.sub(means, axis=1).div(stds, axis=1)
    stats = pd.DataFrame(
        {
            "feature": regime.columns,
            "train_mean": means.reindex(regime.columns).to_numpy(dtype=float),
            "train_std": stds.reindex(regime.columns).to_numpy(dtype=float),
            "train_missing_frac": train_regime.isna().mean().reindex(regime.columns).to_numpy(dtype=float),
            "all_missing_frac": regime.isna().mean().reindex(regime.columns).to_numpy(dtype=float),
        }
    )
    return normalized, stats


def build_macro_feature_frame(
    base_features: pd.DataFrame,
    regime_features: pd.DataFrame,
    splits: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Append normalized weekly regime features to the flat stock feature frame."""
    base = base_features.copy()
    base["week"] = pd.to_datetime(base["week"])
    regime_norm, norm_stats = normalize_regime_features_train_only(regime_features, splits)
    regime_norm = regime_norm.copy()
    regime_norm.index.name = "week"
    regime_flat = regime_norm.reset_index()

    macro = base.merge(regime_flat, on="week", how="left", validate="many_to_one")
    regime_names = list(regime_norm.columns)
    assert len(macro) == len(base), "Macro merge changed row count."
    assert set(regime_names).issubset(macro.columns), "Missing appended regime columns."
    return macro, norm_stats, regime_names


def save_macro_feature_dataset(
    macro_features: pd.DataFrame,
    feature_names: list[str],
    regime_feature_names: list[str],
    normalization_stats: pd.DataFrame,
    *,
    base_meta: dict[str, Any],
) -> tuple[Path, Path, Path]:
    """Save the macro feature parquet, metadata, and normalization table."""
    features_dir = _features_dir()
    features_dir.mkdir(parents=True, exist_ok=True)
    feature_path = features_dir / "features_macro.parquet"
    meta_path = features_dir / "features_macro_meta.json"
    stats_path = features_dir / "regime_normalization_stats.csv"

    macro_features.to_parquet(feature_path, index=False)
    normalization_stats.to_csv(stats_path, index=False)

    weeks = pd.DatetimeIndex(pd.to_datetime(macro_features["week"]).unique())
    tickers = sorted(macro_features["ticker"].unique())
    all_feature_names = list(feature_names) + list(regime_feature_names)
    meta = {
        "feature_version": MACRO_FEATURE_VERSION,
        "base_feature_version": BASE_FEATURE_VERSION,
        "base_feature_artifact": "data/features/features.parquet",
        "regime_feature_artifact": "data/features/regime_features.parquet",
        "normalization_stats_artifact": "data/features/regime_normalization_stats.csv",
        "shape": [int(len(weeks)), int(len(tickers)), int(len(all_feature_names))],
        "feature_names": all_feature_names,
        "stock_feature_names": list(feature_names),
        "regime_feature_names": list(regime_feature_names),
        "normalization": "Stock features keep existing cross-sectional normalization; regime features use train-only time-series z-score.",
        "lookahead_rule": "Feature row T uses data through Friday of week T; target row T is RV in week T+1.",
        "base_meta": base_meta,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved {feature_path} shape={macro_features.shape}")
    return feature_path, meta_path, stats_path


def build_and_save_macro_dataset(
    *,
    base_feature_path: Path | None = None,
    base_meta_path: Path | None = None,
    regime_feature_path: Path | None = None,
    splits_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create ``features_macro.parquet`` from saved base and regime artifacts."""
    features_dir = _features_dir()
    base_features, base_meta = load_flat_features(
        feature_path=base_feature_path or features_dir / "features.parquet",
        meta_path=base_meta_path or features_dir / "features_meta.json",
    )
    regime_path = regime_feature_path or features_dir / "regime_features.parquet"
    regime_features = pd.read_parquet(regime_path)
    regime_features.index = pd.to_datetime(regime_features.index)
    splits = pd.read_parquet(splits_path or features_dir / "splits.parquet")

    macro_features, norm_stats, regime_names = build_macro_feature_frame(
        base_features=base_features,
        regime_features=regime_features,
        splits=splits,
    )
    feature_path, meta_path, stats_path = save_macro_feature_dataset(
        macro_features=macro_features,
        feature_names=list(base_meta["feature_names"]),
        regime_feature_names=regime_names,
        normalization_stats=norm_stats,
        base_meta=base_meta,
    )
    meta = _load_json(meta_path)
    meta["feature_artifact"] = feature_path.as_posix()
    meta["meta_artifact"] = meta_path.as_posix()
    meta["normalization_stats_artifact_absolute"] = stats_path.as_posix()
    return macro_features, meta
