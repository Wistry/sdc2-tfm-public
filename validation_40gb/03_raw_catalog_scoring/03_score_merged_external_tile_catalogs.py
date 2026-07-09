#!/usr/bin/env python3
"""Score merged tiled catalogues against the 40 GB extended-validation truth."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import CONFIGS, convert_to_sdc2_submission, ensure_dir, score_submission


TRUTH_EXTERNAL = VALIDATION_ROOT / "outputs" / "external_truth" / "sky_ldev_truthcat_v2_external_only.txt"
MERGED_DIR = VALIDATION_ROOT / "outputs" / "merged_tile_catalogs"
OUT_DIR = VALIDATION_ROOT / "outputs" / "external_tile_scores"


def ensure_external_truth() -> None:
    if TRUTH_EXTERNAL.exists():
        return
    subprocess.run([sys.executable, str(VALIDATION_ROOT / "01_region_definition" / "03_filter_truth_external_region.py")], check=True)


def main() -> None:
    ensure_external_truth()
    ensure_dir(OUT_DIR)
    submissions = ensure_dir(OUT_DIR / "submissions")
    rows = []
    for _, meta in CONFIGS.items():
        config_name = meta["run_name"]
        catalog_path = MERGED_DIR / f"{config_name}_external_merged.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(f"Missing merged external catalogue. Run 03_raw_catalog_scoring/02_merge_tile_catalogs_external.py first: {catalog_path}")
        df = pd.read_csv(catalog_path)
        submission, diagnostics = convert_to_sdc2_submission(df)
        submission_path = submissions / f"{config_name}_external_merged_submission.csv"
        submission.to_csv(submission_path, index=False)
        score = score_submission(submission, TRUTH_EXTERNAL)
        rows.append(
            {
                "config_name": config_name,
                "dataset_region": "external_40gb_outside_10gb_tiled",
                "n_candidates": len(df),
                "matches": score.get("matches"),
                "false": score.get("false"),
                "score": score.get("score"),
                "status": score.get("status"),
                "method_used": "official_sdc2_scorer_with_filtered_truth",
                "error": score.get("error", ""),
                "catalog_path": str(catalog_path),
                "submission_path": str(submission_path),
                **diagnostics,
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "external_tile_score_summary.csv", index=False)
    print(summary[["config_name", "dataset_region", "n_candidates", "matches", "false", "score", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()
