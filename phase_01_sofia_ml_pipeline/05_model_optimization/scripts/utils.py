from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


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
    stack: list[tuple[int, Any]] = [(-1, config)]
    current_list_key: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if text.startswith("- "):
            if current_list_key is None:
                continue
            parent[current_list_key].append(_coerce_scalar(text[2:]))
            continue
        key, _, value = text.partition(":")
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _coerce_scalar(value)
            current_list_key = None
        else:
            next_obj: dict[str, Any] | list[Any] = {}
            parent[key] = next_obj
            stack.append((indent, next_obj))
            current_list_key = key
            parent[key] = []
            stack[-1] = (indent, parent)
    return config


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ImportError:
        return _simple_yaml(path)


def resolve_path(value: str | Path, base_dir: Path = BASE_DIR) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def ensure_dirs(config: dict[str, Any]) -> dict[str, Path]:
    out = config["outputs"]
    paths = {
        "optuna_dir": resolve_path(out["optuna_dir"]),
        "validation_dir": resolve_path(out["validation_dir"]),
        "final_models_dir": resolve_path(out["final_models_dir"]),
        "report_figures_dir": resolve_path(out["report_figures_dir"]),
        "reports_dir": resolve_path(out["reports_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, int]]:
    data_cfg = config["data"]
    path = resolve_path(data_cfg["candidates_path"])
    if not path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {path}")
    df = pd.read_csv(path).replace([np.inf, -np.inf], np.nan)
    label = data_cfg["label_column"]
    clean = df[df[label].isin([data_cfg["positive_label"], data_cfg["negative_label"]])].copy()
    y = clean[label].astype(int)
    counts = {
        "n_total": int(len(df)),
        "n_clean": int(len(clean)),
        "n_tp": int((df[label] == data_cfg["positive_label"]).sum()),
        "n_fp": int((df[label] == data_cfg["negative_label"]).sum()),
        "n_ambiguous": int((df[label] == data_cfg["ambiguous_label"]).sum()),
    }
    return df, clean, y, counts


def feature_columns_path(config: dict[str, Any], feature_set: str) -> Path:
    key = f"{feature_set}_feature_columns_path"
    return resolve_path(config["benchmark"][key])


def load_feature_columns(config: dict[str, Any], feature_set: str) -> list[str]:
    path = feature_columns_path(config, feature_set)
    if not path.exists():
        raise FileNotFoundError(f"Features de benchmark no encontradas: {path}")
    df = pd.read_csv(path)
    column = "feature" if "feature" in df.columns else df.columns[0]
    return df[column].dropna().astype(str).tolist()


def save_json(payload: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_pickle(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle)


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_model(model_name: str, params: dict[str, Any], random_state: int) -> Pipeline:
    if model_name == "RandomForest":
        model = RandomForestClassifier(random_state=random_state, n_jobs=1, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_name == "ExtraTrees":
        model = ExtraTreesClassifier(random_state=random_state, n_jobs=1, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_name == "GradientBoosting":
        model = GradientBoostingClassifier(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_name == "HistGradientBoosting":
        model = HistGradientBoostingClassifier(random_state=random_state, **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    if model_name == "XGBoost":
        if XGBClassifier is None:
            raise ImportError("xgboost no esta instalado")
        model = XGBClassifier(random_state=random_state, eval_metric="logloss", **params)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Modelo no soportado: {model_name}")


def positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X).astype(float)


def metric_dict(y_true: pd.Series | np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) == 2 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f0_5": float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(y_true: pd.Series, y_score: np.ndarray, grid_min: float, grid_max: float, grid_step: float, min_recall: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    thresholds = np.round(np.arange(grid_min, grid_max + grid_step / 2, grid_step), 6)
    rows = [metric_dict(y_true, y_score, threshold) for threshold in thresholds]
    sweep = pd.DataFrame(rows)
    best = {
        "f1": sweep.sort_values(["f1", "average_precision"], ascending=False).iloc[0].to_dict(),
        "f0_5": sweep.sort_values(["f0_5", "precision", "recall"], ascending=False).iloc[0].to_dict(),
        "f2": sweep.sort_values(["f2", "recall"], ascending=False).iloc[0].to_dict(),
        "balanced_accuracy": sweep.sort_values(["balanced_accuracy", "f1"], ascending=False).iloc[0].to_dict(),
    }
    conservative_pool = sweep[sweep["recall"] >= min_recall]
    if conservative_pool.empty:
        conservative_pool = sweep
    best["conservative_fp"] = conservative_pool.sort_values(["fp", "precision", "recall"], ascending=[True, False, False]).iloc[0].to_dict()
    return sweep, best
