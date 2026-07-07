"""Utilities for Phase 3 external validation on the 40GB development cube."""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

import pandas as pd


BASE = Path(__file__).resolve().parent
REPO_ROOT = BASE.parent
DATA_ROOT = Path(os.environ.get("SDC2_DATA_ROOT", REPO_ROOT / "data")).expanduser()
LARGE_CUBE = Path(
    os.environ.get("SDC2_40GB_CUBE", DATA_ROOT / "sky_ldev_v2.fits")
).expanduser()
LARGE_TRUTH = Path(
    os.environ.get("SDC2_TRUTH", DATA_ROOT / "sky_ldev_truthcat_v2.txt")
).expanduser()
CONFIGS = {
    "baseline_current": {
        "source_config": REPO_ROOT / "phase_01_sofia_ml_pipeline" / "03_candidate_dataset" / "configs" / "baseline_current_full.par",
        "run_name": "baseline_current_40gb",
    },
    "sdc2_team_sofia_like": {
        "source_config": REPO_ROOT / "phase_01_sofia_ml_pipeline" / "03_candidate_dataset" / "configs" / "sdc2_team_sofia_like_full.par",
        "run_name": "sdc2_team_sofia_like_40gb",
    },
}
EXCLUDE_X_MIN = 301
EXCLUDE_X_MAX = 983
EXCLUDE_Y_MIN = 301
EXCLUDE_Y_MAX = 983
CENTRAL_X_MIN = 321
CENTRAL_X_MAX = 963
CENTRAL_Y_MIN = 321
CENTRAL_Y_MAX = 963
CUBE_X_MIN = 0
CUBE_X_MAX = 1285
CUBE_Y_MIN = 0
CUBE_Y_MAX = 1285
CUBE_Z_MIN = 0
CUBE_Z_MAX = 6667
TILE_MARGIN = 20
EXTERNAL_TILE_NAMES = [
    "left_bottom",
    "left_center",
    "left_top",
    "center_bottom",
    "center_top",
    "right_bottom",
    "right_center",
    "right_top",
]
SUBMISSION_COLUMNS = ["id", "ra", "dec", "hi_size", "line_flux_integral", "central_freq", "pa", "i", "w20"]
KEY_SOURCE_COLUMNS = ["f_sum", "ell_maj", "ell_min", "kin_pa", "w20", "ra", "dec", "freq"]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(config: str) -> Path:
    return BASE / "sofia_runs" / CONFIGS[config]["run_name"]


def output_dir(config: str) -> Path:
    return run_dir(config) / "outputs"


def find_sofia_catalog(config: str) -> Path:
    out = output_dir(config)
    patterns = [
        f"{CONFIGS[config]['run_name']}_cat.txt",
        "*_cat.txt",
        "candidates_sofia_only.csv",
        "*.csv",
    ]
    for pattern in patterns:
        paths = sorted(out.glob(pattern))
        if paths:
            return paths[0]
    raise FileNotFoundError(
        f"No SoFiA catalogue found for {config} in {out}. "
        f"Run bash {run_dir(config) / 'run_sofia.sh'} first."
    )


def sofia_txt_columns(path: Path) -> list[str]:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("#") and re.search(r"\bname\b", stripped) and re.search(r"\bx\b", stripped) and re.search(r"\bf_sum\b", stripped):
            return stripped.lstrip("#").split()
    raise ValueError(f"Could not find SoFiA column header in {path}")


def read_sofia_catalog(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Catalogue not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    columns = sofia_txt_columns(path)
    return pd.read_csv(path, sep=r"\s+", comment="#", names=columns, quotechar='"', engine="python")


def numeric(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), dtype="float64")


def convert_to_sdc2_submission(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_input = len(df)
    missing_key = [column for column in KEY_SOURCE_COLUMNS if column not in df.columns]
    out = pd.DataFrame(index=df.index)
    out["ra"] = numeric(df, ["ra"])
    out["dec"] = numeric(df, ["dec"])
    out["central_freq"] = numeric(df, ["freq", "central_freq"])
    out["line_flux_integral"] = numeric(df, ["f_sum", "line_flux_integral"])
    if "ell_maj" in df.columns:
        out["hi_size"] = pd.to_numeric(df["ell_maj"], errors="coerce") * 4.0
    else:
        out["hi_size"] = numeric(df, ["hi_size"])
    out["pa"] = numeric(df, ["kin_pa", "pa"])
    out["i"] = numeric(df, ["ell_min", "i"])
    out["w20"] = numeric(df, ["w20"])
    before = len(out)
    out = out.dropna(subset=["ra", "dec", "central_freq"]).fillna(0.0).copy()
    out.insert(0, "id", range(1, len(out) + 1))
    return out[SUBMISSION_COLUMNS], {
        "n_input_rows": int(n_input),
        "n_submission_rows": int(len(out)),
        "n_dropped_rows": int(before - len(out)),
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


def score_submission(submission: pd.DataFrame, truth_path: Path) -> dict[str, Any]:
    Sdc2Scorer = import_scorer()
    if Sdc2Scorer is None:
        return {"status": "SKIPPED", "error": "ska_sdc.sdc2.sdc2_scorer.Sdc2Scorer not importable"}
    if not truth_path.exists():
        return {"status": "SKIPPED", "error": f"truth file missing: {truth_path}"}
    truth = pd.read_csv(truth_path, sep=r"\s+", comment="#", engine="python")
    try:
        score = Sdc2Scorer(submission, truth).run()
        attrs = scalar_attrs(score)
        return {
            "status": "OK",
            "score": score_value_from(score),
            "matches": attrs.get("n_match"),
            "false": attrs.get("n_false"),
            "score_attrs_json": json.dumps(attrs, sort_keys=True),
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {"status": "ERROR", "score": None, "matches": None, "false": None, "error": f"{exc}\n{traceback.format_exc()}"}


def outside_10gb_region(df: pd.DataFrame) -> pd.Series:
    if "x" not in df.columns or "y" not in df.columns:
        raise ValueError("Catalogue must contain x and y columns for external-region filtering.")
    x = pd.to_numeric(df["x"], errors="coerce")
    y = pd.to_numeric(df["y"], errors="coerce")
    return (x < EXCLUDE_X_MIN) | (x > EXCLUDE_X_MAX) | (y < EXCLUDE_Y_MIN) | (y > EXCLUDE_Y_MAX)


def inside_central_10gb_region(df: pd.DataFrame, x_col: str = "x", y_col: str = "y") -> pd.Series:
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Catalogue must contain {x_col}/{y_col} columns for central-region filtering.")
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    return (x >= CENTRAL_X_MIN) & (x <= CENTRAL_X_MAX) & (y >= CENTRAL_Y_MIN) & (y <= CENTRAL_Y_MAX)


def tile_run_dir(config_run_name: str, tile_name: str) -> Path:
    return BASE / "sofia_tile_runs" / config_run_name / tile_name


def find_tile_catalog(config_run_name: str, tile_name: str) -> Path:
    out = tile_run_dir(config_run_name, tile_name) / "outputs"
    patterns = [
        f"{config_run_name}_{tile_name}_cat.txt",
        "*_cat.txt",
        "*.csv",
    ]
    for pattern in patterns:
        paths = sorted(out.glob(pattern))
        if paths:
            return paths[0]
    raise FileNotFoundError(
        f"No tiled SoFiA catalogue found for {config_run_name}/{tile_name} in {out}. "
        "Run the corresponding tile runner first."
    )
