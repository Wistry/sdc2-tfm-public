#!/usr/bin/env python3
"""Build Phase 2 SDC2 submissions and run the official scorer when available."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


BEST_PHASE1_SCORE = 17.52492347223125
BEST_PHASE1_MATCHES = 137
BEST_PHASE1_FALSE = 52
SUBMISSION_COLUMNS = ["id", "ra", "dec", "hi_size", "line_flux_integral", "central_freq", "pa", "i", "w20"]
KEY_SOURCE_COLUMNS = ["f_sum", "ell_maj", "ell_min", "kin_pa", "w20", "ra", "dec", "freq"]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def numeric(series_or_value: pd.Series | float | int | None, length: int) -> pd.Series:
    if isinstance(series_or_value, pd.Series):
        return pd.to_numeric(series_or_value, errors="coerce")
    value = 0.0 if series_or_value is None else series_or_value
    return pd.Series([value] * length, dtype="float64")


def first_existing(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
    for column in candidates:
        if column in df.columns:
            return df[column]
    return None


def convert_to_sdc2_submission(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_input = len(df)
    missing_key = [column for column in KEY_SOURCE_COLUMNS if column not in df.columns]
    out = pd.DataFrame(index=df.index)
    out["ra"] = numeric(first_existing(df, ["ra"]), n_input)
    out["dec"] = numeric(first_existing(df, ["dec"]), n_input)
    out["central_freq"] = numeric(first_existing(df, ["freq", "central_freq"]), n_input)
    out["line_flux_integral"] = numeric(first_existing(df, ["f_sum", "line_flux_integral"]), n_input)
    if "ell_maj" in df.columns:
        out["hi_size"] = pd.to_numeric(df["ell_maj"], errors="coerce") * 4.0
    else:
        out["hi_size"] = numeric(first_existing(df, ["hi_size"]), n_input)
    out["pa"] = numeric(first_existing(df, ["kin_pa", "pa"]), n_input)
    out["i"] = numeric(first_existing(df, ["ell_min", "i"]), n_input)
    out["w20"] = numeric(first_existing(df, ["w20"]), n_input)
    before_required = len(out)
    out = out.dropna(subset=["ra", "dec", "central_freq"]).copy()
    out = out.fillna(0.0)
    out.insert(0, "id", range(1, len(out) + 1))
    out = out[SUBMISSION_COLUMNS]
    return out, {
        "n_input_rows": int(n_input),
        "n_submission_rows": int(len(out)),
        "n_dropped_rows": int(before_required - len(out)),
        "missing_key_columns": ",".join(missing_key),
    }


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
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, type(None))):
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

    load_yaml(args.config)
    filtered_dir = Path("phase_02_spectral_features/06_apply_to_conservative_catalog/outputs/filtered_catalogs")
    submissions_dir = Path("phase_02_spectral_features/07_official_scoring/outputs/submissions")
    scores_dir = Path("phase_02_spectral_features/07_official_scoring/outputs/official_scores")
    submissions_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)

    catalog_paths = sorted(filtered_dir.glob("SDC2_extended_*.csv"))
    if not catalog_paths:
        raise FileNotFoundError(f"No Phase 2 filtered catalogues found in {filtered_dir}")

    Sdc2Scorer = import_scorer()
    truth = None
    if Sdc2Scorer is not None and args.truth_file.exists():
        truth = pd.read_csv(args.truth_file, sep=r"\s+", comment="#", engine="python")

    rows: list[dict[str, Any]] = []
    for catalog_path in catalog_paths:
        catalog_name = catalog_path.stem
        df = pd.read_csv(catalog_path)
        submission, diagnostics = convert_to_sdc2_submission(df)
        submission_path = submissions_dir / f"{catalog_name}_submission.csv"
        submission.to_csv(submission_path, index=False)
        row: dict[str, Any] = {
            "catalog_name": catalog_name,
            "status": "NOT_RUN",
            "n_rows": int(len(df)),
            "n_submission": int(len(submission)),
            "n_dropped_rows": diagnostics["n_dropped_rows"],
            "score_value": None,
            "n_match": None,
            "n_false": None,
            "improvement_vs_phase1": None,
            "catalog_path": str(catalog_path),
            "submission_path": str(submission_path),
            "error": "",
        }
        if Sdc2Scorer is None:
            row["error"] = "ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer not importable"
        elif truth is None:
            row["error"] = f"truth file missing: {args.truth_file}"
        else:
            try:
                score = Sdc2Scorer(submission, truth).run()
                attrs = scalar_attrs(score)
                score_value = score_value_from(score)
                row.update(
                    {
                        "status": "OK",
                        "score_value": score_value,
                        "n_match": attrs.get("n_match"),
                        "n_false": attrs.get("n_false"),
                        "improvement_vs_phase1": score_value - BEST_PHASE1_SCORE if score_value is not None else None,
                        "score_attrs_json": json.dumps(attrs, ensure_ascii=True, sort_keys=True),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - continue scoring remaining catalogues.
                row["status"] = "ERROR"
                row["error"] = f"{exc}\n{traceback.format_exc()}"
        rows.append(row)

    scores = pd.DataFrame(rows)
    csv_path = scores_dir / "phase2_official_scores.csv"
    scores.to_csv(csv_path, index=False)
    print(f"Scores: {csv_path}")
    print(scores[["catalog_name", "status", "n_submission", "score_value", "n_match", "n_false", "improvement_vs_phase1"]].to_string(index=False))


if __name__ == "__main__":
    main()
