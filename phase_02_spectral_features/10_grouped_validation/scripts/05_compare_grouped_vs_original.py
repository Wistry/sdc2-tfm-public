#!/usr/bin/env python3
"""Compare grouped validation outputs with original Phase 2/CNN results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE = Path("phase_02_spectral_features/10_grouped_validation")
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"
ORIG_CNN_SCORES = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/official_scores/cnn_official_scores.csv")
ORIG_CNN_THRESHOLDS = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/reports/cnn_thresholds.csv")


def safe_read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def classify_result(name: str) -> str:
    if "greedy" in name.lower():
        return "scorer-driven / no generalizable"
    if "official_like" in name.lower():
        return "exploratorio"
    if "grouped" in name.lower():
        return "prometedor pero necesita validación externa"
    return "exploratorio"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args

    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    tab = safe_read(TABLES / "grouped_tabular_results.csv")
    grouped_cnn = safe_read(TABLES / "grouped_cnn_thresholds.csv")
    grouped_scores = safe_read(BASE / "outputs" / "official_scores" / "grouped_cnn_official_scores.csv")
    orig_scores = safe_read(ORIG_CNN_SCORES)
    orig_thresholds = safe_read(ORIG_CNN_THRESHOLDS)

    rows = []
    if not tab.empty:
        best_tab = tab.sort_values("average_precision", ascending=False).iloc[0]
        rows.append(
            {
                "result": "best_grouped_tabular_by_pr_auc",
                "metric": "average_precision",
                "value": best_tab["average_precision"],
                "model": best_tab["model"],
                "feature_set": best_tab["feature_set"],
                "classification": "prometedor pero necesita validación externa",
            }
        )
    if not grouped_cnn.empty:
        best_cnn = grouped_cnn.sort_values("average_precision", ascending=False).iloc[0]
        rows.append(
            {
                "result": "grouped_cnn_internal",
                "metric": "average_precision",
                "value": best_cnn["average_precision"],
                "model": "SmallCNN",
                "feature_set": "patch_21x32x32",
                "classification": "prometedor pero necesita validación externa",
            }
        )
    if not orig_thresholds.empty:
        rows.append(
            {
                "result": "original_cnn_internal",
                "metric": "average_precision",
                "value": orig_thresholds["average_precision"].iloc[0],
                "model": "SmallCNN",
                "feature_set": "patch_21x32x32",
                "classification": "exploratorio",
            }
        )
    if not grouped_scores.empty:
        ok = grouped_scores[grouped_scores["status"] == "OK"].copy()
        if not ok.empty:
            best = ok.sort_values("score", ascending=False).iloc[0]
            rows.append(
                {
                    "result": best["catalog_name"],
                    "metric": "official_score_development",
                    "value": best["score"],
                    "model": "SmallCNN grouped",
                    "feature_set": "patch_21x32x32",
                    "classification": classify_result(str(best["catalog_name"])),
                }
            )
    if not orig_scores.empty:
        ok = orig_scores[orig_scores["status"] == "OK"].copy()
        if not ok.empty:
            best = ok.sort_values("score", ascending=False).iloc[0]
            rows.append(
                {
                    "result": best["catalog_name"],
                    "metric": "official_score_development",
                    "value": best["score"],
                    "model": "SmallCNN original",
                    "feature_set": "patch_21x32x32",
                    "classification": "exploratorio",
                }
            )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES / "grouped_vs_original_comparison.csv", index=False)

    print(comparison.to_string(index=False) if not comparison.empty else "No comparison rows available.")


if __name__ == "__main__":
    main()
