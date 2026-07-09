#!/usr/bin/env python3
"""Extract frozen Phase 2 features for 40 GB extended-validation tiled catalogues."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import LARGE_CUBE, ensure_dir


BASE = VALIDATION_ROOT
MERGED_DIR = BASE / "outputs" / "merged_tile_catalogs"
OUT_DIR = BASE / "outputs" / "external_extended_features"
CATALOG_DIR = OUT_DIR / "candidate_catalogs"
EXTRACTOR = VALIDATION_ROOT.parent / "phase_02_spectral_features" / "01_extract_spectral_features" / "scripts" / "01_extract_candidate_z_features.py"

CATALOGS = {
    "baseline_current_40gb": MERGED_DIR / "baseline_current_40gb_external_merged.csv",
    "sdc2_team_sofia_like_40gb": MERGED_DIR / "sdc2_team_sofia_like_40gb_external_merged.csv",
}

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


def normalize_global_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Use global tiled coordinates as x/y/z while preserving local originals."""
    out = df.copy()
    for axis in ["x", "y", "z"]:
        global_col = f"{axis}_global"
        if axis not in out.columns and global_col not in out.columns:
            continue
        if global_col not in out.columns:
            continue

        global_values = pd.to_numeric(out[global_col], errors="coerce")
        if axis in out.columns:
            local_values = pd.to_numeric(out[axis], errors="coerce")
            out[f"{axis}_tile_local"] = out[axis]
            delta = global_values - local_values
        else:
            delta = pd.Series([0.0] * len(out), index=out.index)

        out[axis] = global_values
        for suffix in ["_min", "_max", "_peak"]:
            column = f"{axis}{suffix}"
            if column in out.columns:
                out[f"{column}_tile_local"] = out[column]
                out[column] = pd.to_numeric(out[column], errors="coerce") + delta
    return out


def prepare_candidate_catalog(catalog_key: str, path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing merged external tiled catalogue: {path}")
    df = pd.read_csv(path).reset_index(drop=True)
    df = normalize_global_positions(df)

    missing = [column for column in ["x", "y", "z"] if column not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required position columns after normalization: {missing}")

    out_path = CATALOG_DIR / f"{catalog_key}_external_candidates_for_features.csv"
    df.to_csv(out_path, index=False)
    return out_path


def write_extractor_config(candidate_paths: dict[str, Path]) -> Path:
    cfg = {
        "cube_path": str(LARGE_CUBE),
        "window_xy": 16,
        "window_z": 10,
        "source_radius_px": 4.0,
        "background_inner_radius_px": 6.0,
        "background_outer_radius_px": 12.0,
        "random_state": 42,
        "catalogs": {},
    }
    for catalog_key, candidate_path in candidate_paths.items():
        cfg["catalogs"][catalog_key] = {
            "candidate_catalog": str(candidate_path),
            "output_features": str(OUT_DIR / f"{catalog_key}_external_extended_features.csv"),
        }

    config_path = OUT_DIR / "validation_40gb_phase2_features_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return config_path


def validate_features(catalog_key: str) -> dict[str, object]:
    features_path = OUT_DIR / f"{catalog_key}_external_extended_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Feature extractor did not create expected output: {features_path}")
    features = pd.read_csv(features_path)
    missing = [column for column in SPECTRAL_FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"{features_path} missing Phase 2 spectral features: {missing}")
    return {
        "base_catalog": catalog_key,
        "features_path": str(features_path),
        "n_rows": len(features),
        "n_phase2_features": len(SPECTRAL_FEATURE_COLUMNS),
        "feature_extraction_ok": int(features["feature_extraction_ok"].sum()),
    }


def main() -> None:
    if not LARGE_CUBE.exists():
        raise FileNotFoundError(f"Missing 40GB FITS cube: {LARGE_CUBE}")
    if not EXTRACTOR.exists():
        raise FileNotFoundError(f"Missing Phase 2 feature extractor: {EXTRACTOR}")

    ensure_dir(OUT_DIR)
    ensure_dir(CATALOG_DIR)

    candidate_paths = {
        catalog_key: prepare_candidate_catalog(catalog_key, path)
        for catalog_key, path in CATALOGS.items()
    }
    config_path = write_extractor_config(candidate_paths)

    for catalog_key in CATALOGS:
        subprocess.run(
            [sys.executable, str(EXTRACTOR), "--config", str(config_path), "--catalog-key", catalog_key],
            check=True,
        )

    summary = pd.DataFrame([validate_features(catalog_key) for catalog_key in CATALOGS])
    summary_path = OUT_DIR / "external_extended_features_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
