#!/usr/bin/env python3
"""Run official SDC2 scoring for robust grouped CNN catalogues."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path("phase_02_spectral_features/11_robust_grouped_cnn")
FILTERED = BASE / "outputs" / "filtered_catalogs"
SUBMISSIONS = BASE / "outputs" / "submissions"
SCORES = BASE / "outputs" / "official_scores"
REPORTS = BASE / "outputs" / "reports"
SUBMISSION_COLUMNS = ["id", "ra", "dec", "hi_size", "line_flux_integral", "central_freq", "pa", "i", "w20"]
REFS = {
    "phase1_best": {"score": 17.524923, "matches": 137, "false": 52},
    "phase2_principal": {"score": 18.524923, "matches": 137, "false": 51},
    "cnn_original": {"score": 28.333489815821608, "matches": 129, "false": 37, "status": "exploratory_leakage_detected"},
    "cnn_grouped_previous": {"score": 18.44173292591026, "matches": 133, "false": 49},
    "official_like": {"score": 25.916593, "matches": 134, "false": 42, "status": "exploratory"},
    "greedy_pruning": {"score": 38.524923, "matches": 137, "false": 31, "status": "scorer_driven_upper_bound"},
}


def numeric(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), dtype="float64")


def convert_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ra"] = numeric(df, ["ra"])
    out["dec"] = numeric(df, ["dec"])
    out["central_freq"] = numeric(df, ["freq", "central_freq"])
    out["line_flux_integral"] = numeric(df, ["f_sum", "line_flux_integral"])
    out["hi_size"] = numeric(df, ["ell_maj", "hi_size"]) * (4.0 if "ell_maj" in df.columns else 1.0)
    out["pa"] = numeric(df, ["kin_pa", "pa"])
    out["i"] = numeric(df, ["ell_min", "i"])
    out["w20"] = numeric(df, ["w20"])
    out = out.dropna(subset=["ra", "dec", "central_freq"]).fillna(0.0).copy()
    out.insert(0, "id", range(1, len(out) + 1))
    return out[SUBMISSION_COLUMNS]


def import_scorer():
    try:
        from ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer
    except Exception:
        return None
    return Sdc2Scorer


def scalar_attrs(obj: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if not callable(value) and isinstance(value, (str, int, float, bool, type(None))):
            attrs[attr] = value
    return attrs


def score_value_from(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    for attr in ["value", "score", "score_value"]:
        if hasattr(score, attr):
            value = getattr(score, attr)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--truth-file", type=Path, default=Path("data/sky_dev_truthcat_v2.txt"))
    args = parser.parse_args()
    del args.config

    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    SCORES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    catalog_paths = sorted(FILTERED.glob("ROBUST_CNN_grouped_*.csv"))
    Sdc2Scorer = import_scorer()
    truth = pd.read_csv(args.truth_file, sep=r"\s+", comment="#", engine="python") if args.truth_file.exists() and Sdc2Scorer else None
    rows = []
    for path in catalog_paths:
        df = pd.read_csv(path)
        sub = convert_to_submission(df)
        sub_path = SUBMISSIONS / f"{path.stem}_submission.csv"
        sub.to_csv(sub_path, index=False)
        row: dict[str, Any] = {"catalog_name": path.stem, "status": "NOT_RUN", "n_input": len(df), "n_submission": len(sub), "score": None, "matches": None, "false": None, "error": ""}
        if Sdc2Scorer is None:
            row["status"] = "SKIPPED"
            row["error"] = "Sdc2Scorer not importable"
        elif truth is None:
            row["status"] = "SKIPPED"
            row["error"] = f"Truth file missing: {args.truth_file}"
        else:
            try:
                score = Sdc2Scorer(sub, truth).run()
                attrs = scalar_attrs(score)
                value = score_value_from(score)
                row.update({"status": "OK", "score": value, "matches": attrs.get("n_match"), "false": attrs.get("n_false"), "score_attrs_json": json.dumps(attrs, sort_keys=True)})
            except Exception as exc:  # noqa: BLE001
                row["status"] = "ERROR"
                row["error"] = f"{exc}\n{traceback.format_exc()}"
        rows.append(row)
    scores = pd.DataFrame(rows)
    if not scores.empty:
        for name, ref in REFS.items():
            scores[f"delta_vs_{name}"] = pd.to_numeric(scores["score"], errors="coerce") - float(ref["score"])
    scores.to_csv(SCORES / "robust_grouped_cnn_official_scores.csv", index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
