#!/usr/bin/env python3
"""Score methodologically valid frozen CNN catalogues for 40 GB extended validation."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import convert_to_sdc2_submission, ensure_dir, score_submission


BASE = VALIDATION_ROOT
FILTERED_DIR = BASE / "outputs" / "external_cnn_valid_models_filtered_catalogs"
OUT_DIR = BASE / "outputs" / "external_cnn_valid_models_scores"
SUBMISSIONS_DIR = OUT_DIR / "submissions"
TRUTH_EXTERNAL = BASE / "outputs" / "external_truth" / "sky_ldev_truthcat_v2_external_only.txt"
DATASET_REGION = "external_40gb_outside_10gb"


def main() -> None:
    if not TRUTH_EXTERNAL.exists():
        raise FileNotFoundError(f"Missing external truth catalogue: {TRUTH_EXTERNAL}")
    ensure_dir(OUT_DIR)
    ensure_dir(SUBMISSIONS_DIR)

    paths = sorted(FILTERED_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No valid CNN filtered catalogues found in {FILTERED_DIR}. Run 05_cnn_models_40gb/04_apply_valid_frozen_cnns_external_40gb.py first.")

    rows = []
    for catalog_path in paths:
        df = pd.read_csv(catalog_path)
        base_catalog = str(df["base_catalog"].iloc[0]) if "base_catalog" in df.columns and not df.empty else catalog_path.stem
        cnn_model = str(df["cnn_model"].iloc[0]) if "cnn_model" in df.columns and not df.empty else ""
        threshold_mode = str(df["threshold_mode"].iloc[0]) if "threshold_mode" in df.columns and not df.empty else ""
        threshold = float(df["cnn_threshold"].iloc[0]) if "cnn_threshold" in df.columns and not df.empty else None

        submission, diagnostics = convert_to_sdc2_submission(df)
        submission_path = SUBMISSIONS_DIR / f"{catalog_path.stem}_submission.csv"
        submission.to_csv(submission_path, index=False)
        score = score_submission(submission, TRUTH_EXTERNAL)

        rows.append(
            {
                "base_catalog": base_catalog,
                "cnn_model": cnn_model,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "dataset_region": DATASET_REGION,
                "n_candidates": len(df),
                "matches": score.get("matches"),
                "false": score.get("false"),
                "score": score.get("score"),
                "status": score.get("status"),
                "catalog_path": str(catalog_path),
                "method_used": "official_sdc2_scorer_with_filtered_external_truth",
                "error": score.get("error", ""),
                "submission_path": str(submission_path),
                **diagnostics,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "valid_frozen_cnn_external_40gb_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary[["base_catalog", "cnn_model", "threshold_mode", "n_candidates", "matches", "false", "score", "status"]].to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
