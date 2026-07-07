#!/usr/bin/env python3
"""Audit Phase 2 spectral/local features for one or more catalogues."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
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

LEAKAGE_COLUMNS = [
    "clean_label",
    "label",
    "is_ambiguous",
    "matched_truth_id",
    "truth_row",
    "truth_x",
    "truth_y",
    "truth_z",
    "min_abs_dx",
    "min_abs_dy",
    "min_abs_dz",
    "min_dist_3d",
    "matching_mode",
    "name",
    "id",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_catalogs(cfg: dict[str, Any], catalog_key: str | None) -> dict[str, dict[str, Any]]:
    catalogs = cfg.get("catalogs") or {
        "default": {
            "candidate_catalog": cfg["candidate_catalog"],
            "output_features": cfg["output_features"],
            "output_dataset": cfg.get(
                "output_dataset",
                "phase_02_spectral_features/02_build_extended_datasets/outputs/clean/"
                "baseline_current_full_extended_clean.csv",
            ),
        }
    }
    if catalog_key:
        if catalog_key not in catalogs:
            raise KeyError(f"Unknown catalog_key '{catalog_key}'. Available: {', '.join(sorted(catalogs))}")
        return {catalog_key: catalogs[catalog_key]}
    return catalogs


def audit_one(catalog_key: str, entry: dict[str, Any]) -> Path:
    dataset_path = Path(entry["output_dataset"])
    if not dataset_path.exists():
        raise FileNotFoundError(f"Extended dataset not found for {catalog_key}: {dataset_path}")

    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in SPECTRAL_FEATURE_COLUMNS if c in df.columns]
    numeric_features = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]

    audit_path = Path(
        f"phase_02_spectral_features/03_feature_audit/outputs/reports/"
        f"spectral_feature_audit_{catalog_key}.csv"
    )
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    clean = df[df["clean_label"].isin([0, 1])].copy() if "clean_label" in df.columns else pd.DataFrame()
    rows = []
    for col in feature_cols:
        missing = int(df[col].isna().sum())
        non_na = df[col].dropna()
        nunique = int(non_na.nunique(dropna=True)) if not non_na.empty else 0
        top_freq = float(non_na.value_counts(normalize=True, dropna=False).iloc[0]) if not non_na.empty else 1.0
        fp_mean = tp_mean = diff = standardized_difference = np.nan
        if col in numeric_features and not clean.empty:
            fp_mean = clean.loc[clean["clean_label"] == 0, col].mean()
            tp_mean = clean.loc[clean["clean_label"] == 1, col].mean()
            diff = tp_mean - fp_mean
            std = clean[col].std()
            standardized_difference = abs(diff / std) if pd.notna(std) and std > 0 else np.nan
        rows.append(
            {
                "catalog_key": catalog_key,
                "dataset": str(dataset_path),
                "n_rows": len(df),
                "feature": col,
                "is_numeric": col in numeric_features,
                "missing_count": missing,
                "missing_fraction": missing / len(df) if len(df) else np.nan,
                "n_unique": nunique,
                "top_frequency": top_freq,
                "constant_or_almost_constant": nunique <= 1 or top_freq >= 0.99,
                "fp_mean": fp_mean,
                "tp_mean": tp_mean,
                "tp_minus_fp": diff,
                "standardized_difference_abs": standardized_difference,
                "excluded_for_leakage_or_id": col in LEAKAGE_COLUMNS,
            }
        )
    pd.DataFrame(rows).to_csv(audit_path, index=False)
    print(f"Wrote audit table: {audit_path}")
    return audit_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--catalog-key", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for key, entry in resolve_catalogs(cfg, args.catalog_key).items():
        audit_one(key, entry)


if __name__ == "__main__":
    main()
