#!/usr/bin/env python3
"""Score a 40GB ablation: SoFiA-only tabular features vs extended features."""

from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd

PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from phase03_utils import convert_to_sdc2_submission, ensure_dir, score_submission


BASE = PHASE3_ROOT
MERGED_DIR = BASE / "outputs" / "merged_tile_catalogs"
FEATURE_DIR = BASE / "outputs" / "external_extended_features"
OUT_DIR = BASE / "outputs" / "external_feature_ablation"
FILTERED_DIR = OUT_DIR / "filtered_catalogs"
PREDICTION_DIR = OUT_DIR / "predictions"
SUBMISSIONS_DIR = OUT_DIR / "submissions"
TRUTH_EXTERNAL = BASE / "outputs" / "external_truth" / "sky_ldev_truthcat_v2_external_only.txt"

MODEL_DIR = PHASE3_ROOT.parent / "phase_02_spectral_features" / "05_focused_winners_comparison" / "outputs" / "models"
THRESHOLDS_CSV = MODEL_DIR.parent / "reports" / "focused_phase2_thresholds.csv"

CATALOGS = {
    "baseline_current_40gb": MERGED_DIR / "baseline_current_40gb_external_merged.csv",
    "sdc2_team_sofia_like_40gb": MERGED_DIR / "sdc2_team_sofia_like_40gb_external_merged.csv",
}

STRATEGIES = [
    ("sofia_only_full", "RandomForest", "f0_5"),
    ("sofia_only_full", "RandomForest", "conservative_fp"),
    ("extended_full", "RandomForest", "f0_5"),
    ("extended_full", "RandomForest", "conservative_fp"),
]


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


def build_dataset(catalog_key: str, catalog_path: Path, feature_set: str) -> pd.DataFrame:
    if not catalog_path.exists():
        raise FileNotFoundError(f"Missing merged external tiled catalogue: {catalog_path}")

    catalog = normalize_global_positions(pd.read_csv(catalog_path).reset_index(drop=True))
    catalog = catalog.reset_index().rename(columns={"index": "candidate_index"})

    if feature_set == "sofia_only_full":
        return catalog

    if feature_set != "extended_full":
        raise ValueError(f"Unsupported feature set: {feature_set}")

    features_path = FEATURE_DIR / f"{catalog_key}_external_extended_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing external Phase 2 features: {features_path}")
    features = pd.read_csv(features_path)
    if "candidate_index" not in features.columns:
        raise ValueError(f"Feature CSV must contain candidate_index: {features_path}")

    merged = catalog.merge(features, on="candidate_index", how="left", suffixes=("", "_spectral"))
    duplicate_columns = [column for column in merged.columns if column.endswith("_spectral")]
    return merged.drop(columns=duplicate_columns)


def threshold_for(thresholds: pd.DataFrame, feature_set: str, model: str, mode: str) -> float:
    row = thresholds[
        (thresholds["feature_set"] == feature_set)
        & (thresholds["model"] == model)
        & (thresholds["threshold_mode"] == mode)
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one threshold for {feature_set}/{model}/{mode}, found {len(row)}")
    return float(row.iloc[0]["threshold"])


def main() -> None:
    if not TRUTH_EXTERNAL.exists():
        raise FileNotFoundError(f"Missing external truth catalogue: {TRUTH_EXTERNAL}")
    if not THRESHOLDS_CSV.exists():
        raise FileNotFoundError(f"Missing thresholds CSV: {THRESHOLDS_CSV}")

    ensure_dir(OUT_DIR)
    ensure_dir(FILTERED_DIR)
    ensure_dir(PREDICTION_DIR)
    ensure_dir(SUBMISSIONS_DIR)

    thresholds = pd.read_csv(THRESHOLDS_CSV)
    rows = []

    for catalog_key, catalog_path in CATALOGS.items():
        datasets: dict[str, pd.DataFrame] = {}
        for feature_set, model_name, mode in STRATEGIES:
            dataset = datasets.setdefault(feature_set, build_dataset(catalog_key, catalog_path, feature_set))
            model_path = MODEL_DIR / f"{model_name}_{feature_set}.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Missing model artifact: {model_path}")

            artifact = joblib.load(model_path)
            pipeline = artifact["pipeline"]
            model_columns = list(artifact["columns"])
            threshold = threshold_for(thresholds, feature_set, model_name, mode)

            df = dataset.copy()
            for column in model_columns:
                if column not in df.columns:
                    df[column] = pd.NA

            probabilities = pipeline.predict_proba(df[model_columns])[:, 1]
            strategy = f"{model_name}_{feature_set}_{mode}"
            df["ablation_strategy"] = strategy
            df["ablation_feature_set"] = feature_set
            df["ablation_model"] = model_name
            df["ablation_threshold_mode"] = mode
            df["ablation_threshold"] = threshold
            df["ablation_score"] = probabilities
            df["model_probability"] = probabilities
            df["ablation_keep"] = df["ablation_score"] >= threshold

            prediction_path = PREDICTION_DIR / f"{catalog_key}_external_{strategy}_predictions.csv"
            filtered_path = FILTERED_DIR / f"{catalog_key}_external_{strategy}.csv"
            submission_path = SUBMISSIONS_DIR / f"{catalog_key}_external_{strategy}_submission.csv"

            df.to_csv(prediction_path, index=False)
            accepted = df[df["ablation_keep"]].copy()
            accepted.to_csv(filtered_path, index=False)

            submission, diagnostics = convert_to_sdc2_submission(accepted)
            submission.to_csv(submission_path, index=False)
            score = score_submission(submission, TRUTH_EXTERNAL)

            rows.append(
                {
                    "base_catalog": catalog_key,
                    "feature_set": feature_set,
                    "model": model_name,
                    "threshold_mode": mode,
                    "threshold": threshold,
                    "n_input": len(df),
                    "n_candidates": len(accepted),
                    "matches": score.get("matches"),
                    "false": score.get("false"),
                    "score": score.get("score"),
                    "status": score.get("status"),
                    "method_used": "official_sdc2_scorer_with_filtered_external_truth",
                    "model_artifact": str(model_path),
                    "filtered_catalog_path": str(filtered_path),
                    "submission_path": str(submission_path),
                    "error": score.get("error", ""),
                    **diagnostics,
                }
            )

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "sofia_only_vs_extended_40gb_ablation_scores.csv"
    summary.to_csv(summary_path, index=False)
    print(
        summary[
            [
                "base_catalog",
                "feature_set",
                "threshold_mode",
                "threshold",
                "n_candidates",
                "matches",
                "false",
                "score",
                "status",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
