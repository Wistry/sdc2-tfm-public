#!/usr/bin/env python3
"""Grouped validation for Phase 2 tabular finalist models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]


BASE = Path("phase_02_spectral_features/10_grouped_validation")
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"
NO_POSITION_COLUMNS = {
    "x",
    "y",
    "z",
    "ra",
    "dec",
    "freq",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "x_peak",
    "y_peak",
    "z_peak",
    "ra_peak",
    "dec_peak",
    "freq_peak",
    "z_w20",
    "z_w50",
    "z_wm50",
    "err_x",
    "err_y",
    "err_z",
}
LEAKAGE_COLUMNS = {
    "candidate_index",
    "name",
    "id",
    "clean_label",
    "label",
    "is_ambiguous",
    "matched_truth_id",
    "truth_row",
    "truth_x",
    "truth_y",
    "truth_z",
    "min_abs_dx",
    "min_abs_dy",
    "min_abs_dz",
    "min_dist_3d",
    "matching_mode",
}
SPECTRAL_PREFIXES = ("spec_", "area_", "centroid_", "overlap_", "spectral_", "local_")
SPECTRAL_EXACT = {"feature_extraction_ok", "edge_clipped", "n_valid_channels"}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def truth_group(row: pd.Series) -> str:
    label = int(row["clean_label"])
    truth = row.get("matched_truth_id")
    if label == 1 and pd.notna(truth) and str(truth).strip() not in {"", "nan", "-1"}:
        return f"truth_{truth}"
    return f"candidate_{row.get('candidate_index', row.name)}"


def is_spectral_col(col: str) -> bool:
    return col in SPECTRAL_EXACT or col.startswith(SPECTRAL_PREFIXES)


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            cols.append(col)
    return cols


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    all_cols = numeric_feature_columns(df)
    spectral = [c for c in all_cols if is_spectral_col(c)]
    sofia = [c for c in all_cols if c not in spectral]
    return {
        "sofia_only_full": sofia,
        "extended_full": all_cols,
        "sofia_only_no_position": [c for c in sofia if c not in NO_POSITION_COLUMNS],
        "extended_no_position": [c for c in all_cols if c not in NO_POSITION_COLUMNS],
        "spectral_only": spectral,
    }


def make_models(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=2, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=random_state, n_jobs=2, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=2,
        )
    return models


def wrap_model(model: Any) -> Any:
    if isinstance(model, Pipeline):
        return model
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def score_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    pred = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "f0_5": fbeta_score(y_true, pred, beta=0.5, zero_division=0),
        "f1": fbeta_score(y_true, pred, beta=1.0, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def split_iterator(y: np.ndarray, groups: np.ndarray, random_state: int):
    n_splits = 5
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return splitter.split(np.zeros_like(y), y, groups), "StratifiedGroupKFold", n_splits
    splitter = GroupKFold(n_splits=n_splits)
    return splitter.split(np.zeros_like(y), y, groups), "GroupKFold", n_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    random_state = int(cfg.get("random_state", 42))
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    dataset_path = Path(cfg["catalogs"]["baseline_current_full"]["output_clean"])
    df_all = pd.read_csv(dataset_path)
    df = df_all[df_all["clean_label"].isin([0, 1])].copy().reset_index(drop=True)
    df["group_id"] = df.apply(truth_group, axis=1)
    y = df["clean_label"].astype(int).to_numpy()
    groups = df["group_id"].astype(str).to_numpy()
    features = feature_sets(df)
    models = make_models(random_state)

    rows: list[dict[str, Any]] = []
    max_shared = 0
    split_name = ""
    folds_used = 0
    for feature_set, cols in features.items():
        if not cols:
            continue
        x = df[cols].copy()
        for model_name, base_model in models.items():
            scores = np.full(len(df), np.nan, dtype=float)
            fold_checks = []
            iterator, split_name, folds_used = split_iterator(y, groups, random_state)
            for fold, (train_idx, test_idx) in enumerate(iterator, start=1):
                train_groups = set(groups[train_idx])
                test_groups = set(groups[test_idx])
                shared = train_groups.intersection(test_groups)
                max_shared = max(max_shared, len(shared))
                fold_checks.append({"fold": fold, "shared_groups": len(shared), "train": len(train_idx), "test": len(test_idx)})
                model = wrap_model(base_model)
                model.fit(x.iloc[train_idx], y[train_idx])
                if hasattr(model, "predict_proba"):
                    pred_scores = model.predict_proba(x.iloc[test_idx])[:, 1]
                else:
                    pred_scores = model.decision_function(x.iloc[test_idx])
                scores[test_idx] = pred_scores
            metrics = score_metrics(y[~np.isnan(scores)], scores[~np.isnan(scores)])
            rows.append(
                {
                    "feature_set": feature_set,
                    "model": model_name,
                    "n_features": len(cols),
                    "splitter": split_name,
                    "n_folds": folds_used,
                    "max_shared_groups": max(check["shared_groups"] for check in fold_checks),
                    "fold_checks_json": json.dumps(fold_checks),
                    **metrics,
                }
            )

    results = pd.DataFrame(rows).sort_values(["average_precision", "f0_5"], ascending=False)
    results.to_csv(TABLES / "grouped_tabular_results.csv", index=False)
    metrics = ["average_precision", "roc_auc", "f0_5", "f1", "f2", "precision", "recall", "balanced_accuracy"]
    best_rows = []
    for metric in metrics:
        best = results.sort_values(metric, ascending=False).iloc[0].copy()
        best["selected_by_metric"] = metric
        best_rows.append(best)
    best_by_metric = pd.DataFrame(best_rows)
    best_by_metric.to_csv(TABLES / "grouped_tabular_best_by_metric.csv", index=False)

    print(results.head(10)[["feature_set", "model", "average_precision", "roc_auc", "f0_5", "f1", "f2", "precision", "recall", "balanced_accuracy", "max_shared_groups"]].to_string(index=False))


if __name__ == "__main__":
    main()
