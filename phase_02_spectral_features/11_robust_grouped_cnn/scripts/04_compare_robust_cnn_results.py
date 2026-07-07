#!/usr/bin/env python3
"""Compare robust grouped CNN results against Phase 2 and previous CNN runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


BASE = Path("phase_02_spectral_features/11_robust_grouped_cnn")
TABLES = BASE / "outputs" / "tables"
SCORES = BASE / "outputs" / "official_scores"

REF_ROWS = [
    {"result": "Phase 2 tabular principal", "score": 18.524923, "matches": 137, "false": 51, "classification": "resultado principal general"},
    {"result": "CNN original", "score": 28.333489815821608, "matches": 129, "false": 37, "classification": "exploratoria, optimista por leakage"},
    {"result": "CNN grouped previa", "score": 18.44173292591026, "matches": 133, "false": 49, "classification": "valida como prueba inicial sin leakage"},
    {"result": "official_like", "score": 25.916593, "matches": 134, "false": 42, "classification": "exploratoria"},
    {"result": "greedy pruning", "score": 38.524923, "matches": 137, "false": 31, "classification": "scorer-driven / cota superior"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args

    TABLES.mkdir(parents=True, exist_ok=True)
    official_path = SCORES / "robust_grouped_cnn_official_scores.csv"
    if not official_path.exists():
        raise FileNotFoundError("Run official scoring before comparison.")
    official = pd.read_csv(official_path)
    ok = official[official["status"] == "OK"].copy()
    best_official = ok.sort_values(["score", "matches", "false"], ascending=[False, False, True]).iloc[0] if not ok.empty else None

    rows = REF_ROWS.copy()
    if best_official is not None:
        rows.append(
            {
                "result": best_official["catalog_name"],
                "score": float(best_official["score"]),
                "matches": int(best_official["matches"]),
                "false": int(best_official["false"]),
                "classification": "resultado CNN principal si mejora a Fase 2 tabular",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(TABLES / "robust_cnn_comparison.csv", index=False)

    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
