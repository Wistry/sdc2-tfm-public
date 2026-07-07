#!/usr/bin/env python3
"""Apply frozen Phase 2 strategy to external merged 40GB tiled catalogues."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd

PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from phase03_utils import ensure_dir


BASE = PHASE3_ROOT
MERGED_DIR = BASE / "outputs" / "merged_tile_catalogs"
FEATURE_DIR = BASE / "outputs" / "external_extended_features"
OUT_DIR = BASE / "outputs" / "external_filtered_catalogs"
PREDICTION_DIR = OUT_DIR / "predictions"
DATASET_DIR = OUT_DIR / "extended_datasets"

SELECTED_STRATEGIES = (
    PHASE3_ROOT.parent
    / "phase_02_spectral_features"
    / "05_focused_winners_comparison"
    / "outputs"
    / "reports"
    / "focused_phase2_selected_strategies.json"
)
FROZEN_INTERNAL_STRATEGY = "best_extended_full_conservative_fp"
OUTPUT_STRATEGY = "SDC2_extended_full_conservative_fp"

CATALOGS = {
    "baseline_current_40gb": MERGED_DIR / "baseline_current_40gb_external_merged.csv",
    "sdc2_team_sofia_like_40gb": MERGED_DIR / "sdc2_team_sofia_like_40gb_external_merged.csv",
}


def normalize_global_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Use global tiled coordinates in model columns and keep local copies."""
    out = df.copy()
    for axis in ["x", "y", "z"]:
        global_col = f"{axis}_global"
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


def load_frozen_strategy() -> dict[str, object]:
    if not SELECTED_STRATEGIES.exists():
        raise FileNotFoundError(
            "Missing frozen Phase 2 selected strategies file: "
            f"{SELECTED_STRATEGIES}. Cannot locate {OUTPUT_STRATEGY} safely."
        )
    strategies = json.loads(SELECTED_STRATEGIES.read_text(encoding="utf-8"))
    for strategy in strategies:
        if strategy.get("strategy_name") == FROZEN_INTERNAL_STRATEGY:
            model_path = Path(str(strategy.get("model_artifact", "")))
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Frozen strategy {FROZEN_INTERNAL_STRATEGY} points to missing model: {model_path}"
                )
            return strategy
    raise ValueError(
        f"Could not find frozen strategy {FROZEN_INTERNAL_STRATEGY} in {SELECTED_STRATEGIES}. "
        f"Cannot apply {OUTPUT_STRATEGY} safely."
    )


def build_extended_dataset(catalog_key: str, catalog_path: Path) -> pd.DataFrame:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing merged external tiled catalogue: {catalog_path}")
    features_path = FEATURE_DIR / f"{catalog_key}_external_extended_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Missing external Phase 2 features: {features_path}. "
            "Run 04_tabular_models_40gb/01_extract_phase2_features_external_tiles_40gb.py first."
        )

    catalog = normalize_global_positions(pd.read_csv(catalog_path).reset_index(drop=True))
    catalog = catalog.reset_index().rename(columns={"index": "candidate_index"})
    features = pd.read_csv(features_path)
    if "candidate_index" not in features.columns:
        raise ValueError(f"Feature CSV must contain candidate_index: {features_path}")

    merged = catalog.merge(features, on="candidate_index", how="left", suffixes=("", "_spectral"))
    duplicate_columns = [column for column in merged.columns if column.endswith("_spectral")]
    return merged.drop(columns=duplicate_columns)


def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(PREDICTION_DIR)
    ensure_dir(DATASET_DIR)

    strategy = load_frozen_strategy()
    model_path = Path(str(strategy["model_artifact"]))
    artifact = joblib.load(model_path)
    pipeline = artifact["pipeline"]
    model_columns = list(artifact["columns"])
    threshold = float(strategy["threshold"])

    summary_rows = []
    for catalog_key, catalog_path in CATALOGS.items():
        df = build_extended_dataset(catalog_key, catalog_path)
        for column in model_columns:
            if column not in df.columns:
                df[column] = pd.NA

        probabilities = pipeline.predict_proba(df[model_columns])[:, 1]
        df["phase2_strategy"] = OUTPUT_STRATEGY
        df["phase2_model"] = str(strategy.get("model", ""))
        df["phase2_feature_set"] = str(strategy.get("feature_set", ""))
        df["phase2_threshold_mode"] = str(strategy.get("threshold_mode", ""))
        df["phase2_threshold"] = threshold
        df["phase2_score"] = probabilities
        df["model_probability"] = probabilities
        df["phase2_keep"] = df["phase2_score"] >= threshold

        dataset_path = DATASET_DIR / f"{catalog_key}_external_extended_clean.csv"
        prediction_path = PREDICTION_DIR / f"{catalog_key}_external_{OUTPUT_STRATEGY}_predictions.csv"
        filtered_path = OUT_DIR / f"{catalog_key}_external_{OUTPUT_STRATEGY}.csv"

        df.to_csv(dataset_path, index=False)
        df.to_csv(prediction_path, index=False)
        accepted = df[df["phase2_keep"]].copy()
        accepted.to_csv(filtered_path, index=False)

        summary_rows.append(
            {
                "base_catalog": catalog_key,
                "strategy": OUTPUT_STRATEGY,
                "model_artifact": str(model_path),
                "threshold": threshold,
                "n_input": len(df),
                "n_accepted": len(accepted),
                "n_discarded": int((~df["phase2_keep"]).sum()),
                "dataset_path": str(dataset_path),
                "prediction_path": str(prediction_path),
                "filtered_catalog_path": str(filtered_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = OUT_DIR / "external_phase2_filter_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary[["base_catalog", "strategy", "threshold", "n_input", "n_accepted", "n_discarded"]].to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
