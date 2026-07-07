#!/usr/bin/env python3
"""Build final grouped evaluation summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE = Path("phase_02_spectral_features/12_final_grouped_evaluation")
TABLES = BASE / "outputs" / "tables"
SCORES = BASE / "outputs" / "official_scores"

REFERENCE_ROWS = [
    {"result": "Phase 1 official", "score": 17.524923, "matches": 137, "false": 52, "classification": "baseline de desarrollo"},
    {"result": "Phase 2 tabular grouped/manual features", "score": 18.524923, "matches": 137, "false": 51, "classification": "resultado principal"},
    {"result": "CNN original", "score": 28.333489815821608, "matches": 129, "false": 37, "classification": "exploratoria/optimista por leakage"},
    {"result": "CNN grouped previa", "score": 18.44173292591026, "matches": 133, "false": 49, "classification": "extension deep learning inicial sin leakage"},
    {"result": "official_like", "score": 25.916593, "matches": 134, "false": 42, "classification": "exploratoria/scorer-oriented"},
    {"result": "greedy", "score": 38.524923, "matches": 137, "false": 31, "classification": "cota superior/scorer-driven"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args
    TABLES.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(SCORES / "final_grouped_official_scores.csv")
    ok_scores = scores[scores["status"] == "OK"].copy()
    comparison_rows = REFERENCE_ROWS.copy()
    for _, row in ok_scores.iterrows():
        comparison_rows.append(
            {
                "result": row["catalog_name"],
                "score": float(row["score"]),
                "matches": int(row["matches"]),
                "false": int(row["false"]),
                "classification": "resultado final grouped evaluado con scorer oficial",
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(TABLES / "final_grouped_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
