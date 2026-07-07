#!/usr/bin/env python3
"""Join SoFiA catalogue variables with Phase 2 spectral/local features.

This script writes two datasets:

- raw_joined: full merge output, including duplicated *_spectral columns for audit;
- clean: original SoFiA columns plus documented spectral/local features.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SPECTRAL_FEATURE_COLUMNS = [
    "feature_extraction_ok",
    "edge_clipped",
    "n_valid_channels",
    "spec_flux_sum_max",
    "spec_flux_sum_mean",
    "spec_flux_sum_std",
    "spec_flux_sum_argmax_rel",
    "spec_flux_sum_snr_like",
    "spec_flux_peak_max",
    "spec_flux_peak_mean",
    "area_mean",
    "area_max",
    "area_std",
    "area_n_active_channels",
    "area_fraction_active_channels",
    "centroid_dx_std",
    "centroid_dy_std",
    "centroid_drift_total",
    "centroid_drift_mean_step",
    "centroid_valid_fraction",
    "overlap_mean",
    "overlap_std",
    "overlap_min",
    "overlap_valid_pairs",
    "spectral_continuity_score",
    "local_contrast_mean",
    "local_contrast_max",
    "local_background_std",
    "local_source_mean",
    "local_source_max",
]


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_catalog_config(cfg: dict[str, Any], catalog_key: str | None) -> dict[str, Any]:
    resolved = dict(cfg)
    catalogs = cfg.get("catalogs") or {}
    if catalog_key is None:
        if catalogs:
            catalog_key = "baseline_current_full"
        else:
            resolved["catalog_key"] = "default"
            return resolved
    if catalog_key not in catalogs:
        available = ", ".join(sorted(catalogs)) or "<none>"
        raise KeyError(f"Unknown catalog_key '{catalog_key}'. Available: {available}")
    resolved.update(catalogs[catalog_key])
    resolved["catalog_key"] = catalog_key
    return resolved


def default_output_path(catalog_key: str, kind: str) -> Path:
    if kind == "raw":
        return Path(
            f"phase_02_spectral_features/02_build_extended_datasets/outputs/raw_joined/"
            f"{catalog_key}_extended_raw_joined.csv"
        )
    if kind == "clean":
        return Path(
            f"phase_02_spectral_features/02_build_extended_datasets/outputs/clean/"
            f"{catalog_key}_extended_clean.csv"
        )
    raise ValueError(f"Unknown output kind: {kind}")


def make_clean_dataset(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    duplicated_spectral_cols = [c for c in df_raw.columns if c.endswith("_spectral")]
    df_clean = df_raw.drop(columns=duplicated_spectral_cols)

    bad_cols = [c for c in df_clean.columns if c.endswith("_spectral")]
    if bad_cols:
        raise ValueError(f"Unexpected duplicated spectral columns in clean dataset: {bad_cols}")
    if "spectral_continuity_score" not in df_clean.columns:
        raise ValueError("Expected valid feature column missing: spectral_continuity_score")

    return df_clean, duplicated_spectral_cols


def format_distribution(df: pd.DataFrame, column: str) -> list[str]:
    if column not in df.columns:
        return [f"- `{column}` not found"]
    counts = df[column].value_counts(dropna=False).sort_index()
    return [f"- `{idx}`: {count}" for idx, count in counts.items()]


def feature_ok_percent(df: pd.DataFrame) -> str:
    if "feature_extraction_ok" not in df.columns:
        return "n/a"
    return f"{100.0 * df['feature_extraction_ok'].fillna(False).astype(bool).mean():.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--catalog-key", default=None)
    args = parser.parse_args()

    cfg = resolve_catalog_config(load_config(args.config), args.catalog_key)
    catalog_path = Path(cfg["candidate_catalog"])
    feature_path = Path(cfg["output_features"])
    catalog_key = cfg.get("catalog_key", "default")
    raw_output_path = Path(cfg.get("output_raw_joined", default_output_path(catalog_key, "raw")))
    clean_output_path = Path(cfg.get("output_clean", cfg.get("output_dataset", default_output_path(catalog_key, "clean"))))

    missing = [str(p) for p in [catalog_path, feature_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Required input path(s) not found: " + ", ".join(missing))

    catalog = pd.read_csv(catalog_path)
    features = pd.read_csv(feature_path)

    if "candidate_index" not in features.columns:
        raise ValueError("Feature CSV must contain candidate_index.")
    if features["candidate_index"].duplicated().any():
        raise ValueError("Feature CSV contains duplicate candidate_index values.")

    catalog = catalog.reset_index().rename(columns={"index": "candidate_index"})
    merged = catalog.merge(features, on="candidate_index", how="left", suffixes=("", "_spectral"))
    clean, removed_cols = make_clean_dataset(merged)

    spectral_cols = [c for c in SPECTRAL_FEATURE_COLUMNS if c in clean.columns]

    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    clean_output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(raw_output_path, index=False)
    clean.to_csv(clean_output_path, index=False)

    print(f"Wrote raw joined dataset: {raw_output_path}")
    print(f"Wrote clean dataset: {clean_output_path}")
    print(f"Catalog key: {catalog_key}")
    print(f"Raw shape: {merged.shape}")
    print(f"Clean shape: {clean.shape}")
    print(f"Removed duplicated spectral columns: {len(removed_cols)}")
    print(f"New spectral/local feature columns conserved: {len(spectral_cols)}")
    print(f"feature_extraction_ok: {feature_ok_percent(clean)}")
    print("clean_label distribution:")
    for line in format_distribution(clean, "clean_label"):
        print(line)


if __name__ == "__main__":
    main()
