#!/usr/bin/env python3
"""Score frozen Phase 2 filtered external 40GB tiled catalogues."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from phase03_utils import convert_to_sdc2_submission, ensure_dir, score_submission


BASE = PHASE3_ROOT
FILTERED_DIR = BASE / "outputs" / "external_filtered_catalogs"
OUT_DIR = BASE / "outputs" / "external_phase2_scores"
SUBMISSIONS_DIR = OUT_DIR / "submissions"
TRUTH_EXTERNAL = BASE / "outputs" / "external_truth" / "sky_ldev_truthcat_v2_external_only.txt"
STRATEGY = "SDC2_extended_full_conservative_fp"
DATASET_REGION = "external_40gb_outside_10gb"


def base_catalog_from_path(path: Path) -> str:
    suffix = f"_external_{STRATEGY}"
    if path.stem.endswith(suffix):
        return path.stem[: -len(suffix)]
    return path.stem


def main() -> None:
    if not TRUTH_EXTERNAL.exists():
        raise FileNotFoundError(f"Missing external truth catalogue: {TRUTH_EXTERNAL}")

    ensure_dir(OUT_DIR)
    ensure_dir(SUBMISSIONS_DIR)

    paths = sorted(FILTERED_DIR.glob(f"*_external_{STRATEGY}.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No external filtered catalogues found in {FILTERED_DIR}. "
            "Run 04_tabular_models_40gb/02_apply_frozen_phase2_models_external_tiles_40gb.py first."
        )

    rows = []
    for catalog_path in paths:
        base_catalog = base_catalog_from_path(catalog_path)
        df = pd.read_csv(catalog_path)
        submission, diagnostics = convert_to_sdc2_submission(df)
        submission_path = SUBMISSIONS_DIR / f"{catalog_path.stem}_submission.csv"
        submission.to_csv(submission_path, index=False)
        score = score_submission(submission, TRUTH_EXTERNAL)
        rows.append(
            {
                "base_catalog": base_catalog,
                "strategy": STRATEGY,
                "dataset_region": DATASET_REGION,
                "n_candidates": len(df),
                "matches": score.get("matches"),
                "false": score.get("false"),
                "score": score.get("score"),
                "method_used": "official_sdc2_scorer_with_filtered_external_truth",
                "catalog_path": str(catalog_path),
                "status": score.get("status"),
                "error": score.get("error", ""),
                "submission_path": str(submission_path),
                **diagnostics,
            }
        )

    summary = pd.DataFrame(rows)
    summary_path = OUT_DIR / "phase2_external_40gb_score_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary[["base_catalog", "strategy", "dataset_region", "n_candidates", "matches", "false", "score", "status"]].to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
