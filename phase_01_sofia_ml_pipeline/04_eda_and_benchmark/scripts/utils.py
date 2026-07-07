from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.yaml"

EXPLICIT_LEAKAGE_COLUMNS = {
    "clean_label",
    "label",
    "is_tp",
    "is_fp",
    "is_ambiguous",
    "match_status",
    "matched_truth_id",
    "truth_id",
    "source_id_truth",
    "truth_row",
    "truth_x",
    "truth_y",
    "truth_z",
    "min_abs_dx",
    "min_abs_dy",
    "min_abs_dz",
    "min_dist_3d",
    "dist_3d",
    "dx",
    "dy",
    "dz",
    "score",
    "sdc2_score",
    "matching_mode",
    "name",
    "id",
}

LEAKAGE_PATTERNS = ("truth", "match", "dist", "score", "label")
REVIEW_ONLY_COLUMNS = {"snr", "snr_max", "f_sum", "rms", "w20", "w50", "rel", "flag"}


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    section: str | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            config[section] = {}
            continue
        if section and ":" in line:
            key, value = line.strip().split(":", 1)
            config[section][key.strip()] = _coerce_scalar(value)
    return config


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Config no encontrada: {path}")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded or {}
    except ImportError:
        return _simple_yaml(path)


def resolve_path(path_value: str | Path, base_dir: Path = BASE_DIR) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    output_cfg = config.get("outputs", {})
    paths = {
        "eda_dir": resolve_path(output_cfg.get("eda_dir", "outputs/eda")),
        "benchmark_dir": resolve_path(output_cfg.get("benchmark_dir", "outputs/benchmark")),
        "summary_dir": resolve_path(output_cfg.get("summary_dir", "outputs/summary")),
        "reports_dir": resolve_path(output_cfg.get("reports_dir", "reports")),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_candidates(config: dict[str, Any], candidates_path: str | Path | None = None) -> pd.DataFrame:
    data_cfg = config.get("data", {})
    path = resolve_path(candidates_path or data_cfg.get("candidates_path"))
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el dataset de candidatos: {path}. "
            "Genera primero el CSV tabular desde phase_01_sofia_ml_pipeline/03_candidate_dataset."
        )
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def detect_label_column(df: pd.DataFrame, preferred: str | None = "clean_label") -> str:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in ("clean_label", "label", "target", "y"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No se encontro columna de etiqueta. Esperada: clean_label.")


def split_clean_ambiguous(
    df: pd.DataFrame,
    label_column: str,
    positive_label: int = 1,
    negative_label: int = 0,
    ambiguous_label: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = df[df[label_column].isin([positive_label, negative_label])].copy()
    ambiguous = df[df[label_column] == ambiguous_label].copy()
    return clean, ambiguous


def detect_leakage_columns(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    rows = []
    for column in df.columns:
        lower = column.lower()
        explicit = lower in EXPLICIT_LEAKAGE_COLUMNS or column == label_column
        matched_patterns = [pattern for pattern in LEAKAGE_PATTERNS if pattern in lower]
        review_only = lower in REVIEW_ONLY_COLUMNS
        exclude = explicit or (bool(matched_patterns) and not review_only)
        reason = []
        if explicit:
            reason.append("explicit")
        if matched_patterns:
            reason.append("pattern:" + ",".join(matched_patterns))
        if review_only:
            reason.append("review_only_physical")
        rows.append({
            "column": column,
            "exclude": bool(exclude),
            "reason": ";".join(reason) if reason else "",
        })
    return pd.DataFrame(rows)


def get_numeric_feature_columns(df: pd.DataFrame, label_column: str, leakage_df: pd.DataFrame) -> list[str]:
    excluded = set(leakage_df.loc[leakage_df["exclude"], "column"])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [column for column in numeric_cols if column not in excluded and column != label_column]


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_json(payload: dict[str, Any] | list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def plot_and_save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    fig.clf()


def safe_model_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")

