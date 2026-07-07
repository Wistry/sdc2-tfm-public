#!/usr/bin/env python3
"""Convert CNN filtered catalogues to SDC2 submissions and score them."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier")
FILTERED_DIR = BASE / "outputs" / "filtered_catalogs"
SUBMISSIONS_DIR = BASE / "outputs" / "submissions"
SCORES_DIR = BASE / "outputs" / "official_scores"
SUBMISSION_COLUMNS = ["id", "ra", "dec", "hi_size", "line_flux_integral", "central_freq", "pa", "i", "w20"]
PHASE1 = 17.524923
PHASE2 = 18.524923
OFFICIAL_LIKE = 25.916593


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

    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    catalog_paths = sorted(FILTERED_DIR.glob("CNN_*.csv"))
    if not catalog_paths:
        rows = [{"catalog_name": "", "status": "SKIPPED", "error": "No CNN filtered catalogues found."}]
        scores = pd.DataFrame(rows)
    else:
        Sdc2Scorer = import_scorer()
        truth = pd.read_csv(args.truth_file, sep=r"\s+", comment="#", engine="python") if args.truth_file.exists() and Sdc2Scorer else None
        rows = []
        for path in catalog_paths:
            df = pd.read_csv(path)
            sub = convert_to_submission(df)
            sub_path = SUBMISSIONS_DIR / f"{path.stem}_submission.csv"
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

    scores["delta_vs_phase1_best"] = pd.to_numeric(scores.get("score"), errors="coerce") - PHASE1
    scores["delta_vs_phase2_principal"] = pd.to_numeric(scores.get("score"), errors="coerce") - PHASE2
    scores["delta_vs_official_like"] = pd.to_numeric(scores.get("score"), errors="coerce") - OFFICIAL_LIKE
    csv_path = SCORES_DIR / "cnn_official_scores.csv"
    scores.to_csv(csv_path, index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
