#!/usr/bin/env python3
"""Build a structured comparison of frozen CNN results in 10 GB and 40 GB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PHASE3_ROOT = Path(__file__).resolve().parents[1]
PHASE2_ROOT = PHASE3_ROOT.parent / "phase_02_spectral_features"
OUT_DIR = PHASE3_ROOT / "outputs" / "external_cnn_valid_models_reports"
COMPARISON_PATH = OUT_DIR / "valid_cnn_external_40gb_comparison.csv"

SCORE_SUMMARY = (
    PHASE3_ROOT
    / "outputs"
    / "external_cnn_valid_models_scores"
    / "valid_frozen_cnn_external_40gb_score_summary.csv"
)

SOURCES = {
    "phase11_scores": PHASE2_ROOT / "08_cnn_candidate_classifier" / "outputs" / "official_scores" / "cnn_official_scores.csv",
    "phase14_scores": PHASE2_ROOT / "10_grouped_validation" / "outputs" / "official_scores" / "grouped_cnn_official_scores.csv",
    "phase15_scores": PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "official_scores" / "robust_grouped_cnn_official_scores.csv",
    "phase16_scores": PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "official_scores" / "final_grouped_official_scores.csv",
    "phase11_meta": PHASE2_ROOT / "08_cnn_candidate_classifier" / "outputs" / "models" / "small_cnn_metadata.json",
    "phase14_meta": PHASE2_ROOT / "10_grouped_validation" / "outputs" / "models" / "small_cnn_grouped_metadata.json",
    "phase15_meta": PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "models" / "best_robust_grouped_cnn_metadata.json",
    "phase16_meta": PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "models" / "final_grouped_cnn_metadata.json",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def score_row(path: Path, catalog_name: str) -> dict[str, Any]:
    df = pd.read_csv(path)
    row = df[df["catalog_name"] == catalog_name]
    return row.iloc[0].to_dict() if not row.empty else {}


def external_row(scores: pd.DataFrame, model: str, mode: str, base_catalog: str) -> dict[str, Any]:
    row = scores[
        (scores["cnn_model"] == model)
        & (scores["threshold_mode"] == mode)
        & (scores["base_catalog"] == base_catalog)
    ]
    return row.iloc[0].to_dict() if not row.empty else {}


def model_specs() -> list[dict[str, Any]]:
    phase11_meta = read_json(SOURCES["phase11_meta"])
    phase14_meta = read_json(SOURCES["phase14_meta"])
    phase15_meta = read_json(SOURCES["phase15_meta"])
    phase16_meta = read_json(SOURCES["phase16_meta"])

    return [
        {
            "cnn_model": "CNN_initial_phase11",
            "phase2_folder": "08_cnn_candidate_classifier",
            "validation": "stratified_random_split",
            "leakage_control": "row_split",
            "threshold_mode": "f0_5",
            "threshold": 0.97,
            "score10": score_row(SOURCES["phase11_scores"], "CNN_f0_5"),
            "external_model": None,
            "train_examples": phase11_meta.get("train_examples", ""),
        },
        {
            "cnn_model": "CNN_grouped_phase14",
            "phase2_folder": "10_grouped_validation",
            "validation": "grouped_holdout",
            "leakage_control": f"shared_groups={phase14_meta.get('shared_groups', '')}",
            "threshold_mode": "f0_5",
            "threshold": 0.70,
            "score10": score_row(SOURCES["phase14_scores"], "CNN_grouped_f0_5"),
            "external_model": "CNN_grouped_phase14",
            "train_examples": phase14_meta.get("train_examples", ""),
        },
        {
            "cnn_model": "ROBUST_CNN_grouped_phase15",
            "phase2_folder": "11_robust_grouped_cnn",
            "validation": "stratified_group_kfold",
            "leakage_control": f"shared_groups={phase15_meta.get('best', {}).get('shared_groups', '')}",
            "threshold_mode": "f0_5",
            "threshold": 0.98,
            "score10": score_row(SOURCES["phase15_scores"], "ROBUST_CNN_grouped_f0_5"),
            "external_model": "ROBUST_CNN_grouped_phase15",
            "train_examples": phase15_meta.get("examples", ""),
        },
        {
            "cnn_model": "ROBUST_CNN_grouped_phase15",
            "phase2_folder": "11_robust_grouped_cnn",
            "validation": "stratified_group_kfold",
            "leakage_control": f"shared_groups={phase15_meta.get('best', {}).get('shared_groups', '')}",
            "threshold_mode": "conservative_fp",
            "threshold": 0.98,
            "score10": score_row(SOURCES["phase15_scores"], "ROBUST_CNN_grouped_conservative_fp"),
            "external_model": "ROBUST_CNN_grouped_phase15",
            "train_examples": phase15_meta.get("examples", ""),
        },
        {
            "cnn_model": "final_grouped_cnn_phase16",
            "phase2_folder": "12_final_grouped_evaluation",
            "validation": "stratified_group_kfold_5",
            "leakage_control": f"max_shared_groups={phase16_meta.get('max_shared_groups', '')}",
            "threshold_mode": "f0_5",
            "threshold": 0.06,
            "score10": score_row(SOURCES["phase16_scores"], "final_grouped_cnn_f0_5"),
            "external_model": "final_grouped_cnn_phase16",
            "train_examples": phase16_meta.get("examples", ""),
        },
        {
            "cnn_model": "final_grouped_cnn_phase16",
            "phase2_folder": "12_final_grouped_evaluation",
            "validation": "stratified_group_kfold_5",
            "leakage_control": f"max_shared_groups={phase16_meta.get('max_shared_groups', '')}",
            "threshold_mode": "conservative_fp",
            "threshold": 0.71,
            "score10": score_row(SOURCES["phase16_scores"], "final_grouped_cnn_conservative_fp"),
            "external_model": "final_grouped_cnn_phase16",
            "train_examples": phase16_meta.get("examples", ""),
        },
    ]


def comparison_rows(scores_external: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for base_catalog in ["baseline_current_40gb", "sdc2_team_sofia_like_40gb"]:
        for spec in model_specs():
            score40 = (
                external_row(
                    scores_external,
                    str(spec["external_model"]),
                    str(spec["threshold_mode"]),
                    base_catalog,
                )
                if spec["external_model"]
                else {}
            )
            score10 = spec["score10"]
            rows.append(
                {
                    "base_catalog_40gb": base_catalog,
                    "cnn_model": spec["cnn_model"],
                    "phase2_folder": spec["phase2_folder"],
                    "input_shape": "21x32x32",
                    "validation": spec["validation"],
                    "leakage_control": spec["leakage_control"],
                    "threshold_mode": spec["threshold_mode"],
                    "threshold": spec["threshold"],
                    "train_examples": spec["train_examples"],
                    "score_10gb": score10.get("score"),
                    "matches_10gb": score10.get("matches"),
                    "false_10gb": score10.get("false"),
                    "n_candidates_40gb": score40.get("n_candidates"),
                    "score_40gb": score40.get("score"),
                    "matches_40gb": score40.get("matches"),
                    "false_40gb": score40.get("false"),
                    "status_40gb": score40.get("status"),
                }
            )
    return rows


def main() -> None:
    if not SCORE_SUMMARY.exists():
        raise FileNotFoundError(f"Missing external CNN score summary: {SCORE_SUMMARY}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_external = pd.read_csv(SCORE_SUMMARY)
    comparison = pd.DataFrame(comparison_rows(scores_external))
    comparison.to_csv(COMPARISON_PATH, index=False)
    print(comparison.to_string(index=False))
    print(f"Wrote comparison: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
