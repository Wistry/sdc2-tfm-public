from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = BASE_DIR / "config.yaml"


def _coerce_scalar(value: str) -> Any:
    value = value.strip().strip('"').strip("'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _simple_yaml(path: Path) -> dict[str, Any]:
    config: dict[str, Any] = {}
    current_section: str | None = None
    current_list_key: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            config[current_section] = {}
            current_list_key = None
            continue
        if current_section is None:
            continue
        text = line.strip()
        if text.startswith("- ") and current_list_key:
            config[current_section][current_list_key].append(_coerce_scalar(text[2:]))
            continue
        key, _, value = text.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            config[current_section][key] = _coerce_scalar(value)
            current_list_key = None
        else:
            config[current_section][key] = []
            current_list_key = key
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    try:
        import yaml  # type: ignore

        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return _simple_yaml(config_path)


def resolve_path(value: str | Path, base_dir: Path = BASE_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    out = config["outputs"]
    paths = {
        "predictions_dir": resolve_path(out["predictions_dir"]),
        "accepted_dir": resolve_path(out["accepted_dir"]),
        "scores_dir": resolve_path(out["scores_dir"]),
        "reports_dir": resolve_path(out["reports_dir"]),
        "selected_for_scoring_dir": resolve_path(out.get("selected_for_scoring_dir", "outputs/selected_for_scoring")),
        "sdc2_postfilter_dir": resolve_path(out.get("sdc2_postfilter_dir", "outputs/sdc2_postfilter")),
        "logs_dir": resolve_path(out.get("logs_dir", "outputs/logs")),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_candidates(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["data"]["candidates_path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset de candidatos no encontrado: {path}")
    return pd.read_csv(path).replace([np.inf, -np.inf], np.nan)


def load_scoring_summary(config: dict[str, Any]) -> pd.DataFrame:
    path = resolve_path(config["data"]["scoring_full_cube_path"])
    if not path.exists():
        warnings.warn(f"No existe scoring local full cube: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def infer_feature_set_from_model_key(model_key: str) -> str:
    if model_key.endswith("_no_position"):
        return "no_position"
    if model_key.endswith("_full"):
        return "full"
    return "unknown"


def infer_model_name_from_model_key(model_key: str) -> str:
    if model_key.endswith("_no_position"):
        return model_key.removesuffix("_no_position")
    if model_key.endswith("_full"):
        return model_key.removesuffix("_full")
    return model_key


def resolve_threshold(metadata: dict, threshold_mode: str, fallback: float = 0.5) -> float:
    if threshold_mode == "selected":
        return float(metadata.get("selected_threshold", fallback))

    thresholds = metadata.get("thresholds", {})
    entry = thresholds.get(threshold_mode)
    if isinstance(entry, dict) and "threshold" in entry:
        return float(entry["threshold"])
    if isinstance(entry, (int, float)):
        return float(entry)
    return fallback


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def resolve_model_files(final_models_dir: str | Path, model_key: str) -> dict[str, Path]:
    base = resolve_path(final_models_dir)
    return {
        "model": base / f"{model_key}.pkl",
        "features": base / f"{model_key}_features.json",
        "metadata": base / f"{model_key}_metadata.json",
    }


def _extract_features(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        if "features" in payload:
            return list(payload["features"])
        if "feature_names" in payload:
            return list(payload["feature_names"])
    if isinstance(payload, list):
        return list(payload)
    raise ValueError("Formato de features JSON no soportado")


def load_final_model_bundle(final_models_dir: str | Path, model_key: str, fallback_threshold: float = 0.5, use_metadata_threshold: bool = True) -> dict[str, Any] | None:
    files = resolve_model_files(final_models_dir, model_key)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        warnings.warn(f"Saltando {model_key}: faltan archivos {missing}")
        return None
    with files["model"].open("rb") as handle:
        model = pickle.load(handle)
    features = _extract_features(load_json(files["features"]))
    metadata = load_json(files["metadata"])
    threshold = fallback_threshold
    if use_metadata_threshold:
        threshold = metadata.get("selected_threshold", fallback_threshold)
    return {
        "model_key": model_key,
        "model": model,
        "features": features,
        "metadata": metadata,
        "threshold": float(threshold),
    }


def get_positive_scores(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X).astype(float)

