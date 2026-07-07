#!/usr/bin/env python3
"""Final tabular evaluation with StratifiedGroupKFold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception as exc:  # pragma: no cover
    raise RuntimeError("StratifiedGroupKFold is required for final grouped evaluation. Upgrade scikit-learn.") from exc

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None  # type: ignore[assignment]


BASE = Path("phase_02_spectral_features/12_final_grouped_evaluation")
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"
MODELS = BASE / "outputs" / "models"
DATASET = Path("phase_02_spectral_features/02_build_extended_datasets/outputs/clean/baseline_current_full_extended_clean.csv")
RANDOM_STATE = 42
N_SPLITS = 5

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
    "split_group",
}
POSITION_COLUMNS = {
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
SPECTRAL_EXACT = {"feature_extraction_ok", "edge_clipped", "n_valid_channels"}
SPECTRAL_PREFIXES = ("spec_", "area_", "centroid_", "overlap_", "spectral_", "local_")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def valid_truth_id(value: Any) -> bool:
    return pd.notna(value) and str(value).strip() not in {"", "nan", "None", "-1", "-1.0"}


def split_group(row: pd.Series) -> str:
    if int(row["clean_label"]) == 1 and valid_truth_id(row.get("matched_truth_id")):
        return f"truth_{row['matched_truth_id']}"
    return f"candidate_{row.get('candidate_index', row.name)}"


def is_spectral_feature(col: str) -> bool:
    return col in SPECTRAL_EXACT or col.startswith(SPECTRAL_PREFIXES)


def feature_columns(df: pd.DataFrame, feature_set: str) -> list[str]:
    numeric_cols = [c for c in df.columns if c not in LEAKAGE_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if feature_set == "sofia_only_full":
        return [c for c in numeric_cols if not is_spectral_feature(c)]
    if feature_set == "extended_full":
        return numeric_cols
    if feature_set == "sofia_only_no_position":
        return [c for c in numeric_cols if not is_spectral_feature(c) and c not in POSITION_COLUMNS]
    if feature_set == "extended_no_position":
        return [c for c in numeric_cols if c not in POSITION_COLUMNS]
    if feature_set == "spectral_only":
        return [c for c in numeric_cols if is_spectral_feature(c)]
    raise ValueError(f"Unknown feature set: {feature_set}")


def make_models() -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
            ]
        ),
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=3,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=RANDOM_STATE,
                        n_jobs=2,
                    ),
                ),
            ]
        )
    return models


def score_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "f0_5": float(fbeta_score(y_true, pred, beta=0.5, zero_division=0)),
        "f1": float(fbeta_score(y_true, pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def derive_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    sweep = []
    for threshold in np.linspace(0.01, 0.99, 99):
        row = score_row(y_true, scores, float(threshold))
        row["threshold"] = float(threshold)
        sweep.append(row)
    sweep_df = pd.DataFrame(sweep)
    rows = []
    for mode in ["f0_5", "f1", "f2"]:
        best = sweep_df.sort_values([mode, "threshold"], ascending=[False, False]).iloc[0].to_dict()
        best["threshold_mode"] = mode
        rows.append(best)
    viable = sweep_df[sweep_df["recall"] >= 0.7].copy()
    if viable.empty:
        best = sweep_df.sort_values(["fp", "recall", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    else:
        best = viable.sort_values(["fp", "precision", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    best["threshold_mode"] = "conservative_fp"
    rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    load_yaml(args.config)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATASET)
    clean = df[df["clean_label"].isin([0, 1])].copy().reset_index(drop=True)
    clean["split_group"] = clean.apply(split_group, axis=1)
    y = clean["clean_label"].astype(int).to_numpy()
    groups = clean["split_group"].astype(str).to_numpy()
    group_sizes = clean["split_group"].value_counts()
    duplicated_groups = group_sizes[group_sizes > 1]
    models = make_models()
    feature_sets = ["sofia_only_full", "extended_full", "sofia_only_no_position", "extended_no_position", "spectral_only"]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    oof_scores: dict[tuple[str, str], np.ndarray] = {}
    for feature_set in feature_sets:
        cols = feature_columns(clean, feature_set)
        if not cols:
            continue
        x = clean[cols]
        for model_name, model_template in models.items():
            scores_all = np.full(len(clean), np.nan, dtype="float64")
            for fold, (train_idx, test_idx) in enumerate(splitter.split(x, y, groups), start=1):
                shared = set(groups[train_idx]).intersection(set(groups[test_idx]))
                if shared:
                    raise RuntimeError(f"Fold {fold} leaked {len(shared)} groups.")
                model = clone(model_template)
                model.fit(x.iloc[train_idx], y[train_idx])
                scores = model.predict_proba(x.iloc[test_idx])[:, 1]
                scores_all[test_idx] = scores
                row = score_row(y[test_idx], scores, threshold=0.5)
                row.update(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_train": int(len(train_idx)),
                        "n_test": int(len(test_idx)),
                        "train_groups": int(len(set(groups[train_idx]))),
                        "test_groups": int(len(set(groups[test_idx]))),
                        "shared_groups": int(len(shared)),
                        "n_features": int(len(cols)),
                    }
                )
                rows.append(row)
            oof_scores[(feature_set, model_name)] = scores_all

    by_fold = pd.DataFrame(rows)
    by_fold.to_csv(TABLES / "final_grouped_tabular_results_by_fold.csv", index=False)
    metric_cols = ["average_precision", "roc_auc", "f0_5", "f1", "f2", "precision", "recall", "balanced_accuracy", "tn", "fp", "fn", "tp", "shared_groups", "n_features"]
    summary = (
        by_fold.groupby(["feature_set", "model"], as_index=False)[metric_cols]
        .agg({**{m: "mean" for m in metric_cols if m not in {"shared_groups", "n_features"}}, "shared_groups": "max", "n_features": "first"})
        .sort_values(["average_precision", "f0_5"], ascending=[False, False])
    )
    summary.to_csv(TABLES / "final_grouped_tabular_summary.csv", index=False)

    best_pr = summary.iloc[0]
    best_f05 = summary.sort_values(["f0_5", "average_precision"], ascending=[False, False]).iloc[0]
    best_key = (str(best_pr["feature_set"]), str(best_pr["model"]))
    best_cols = feature_columns(clean, best_key[0])
    best_thresholds = derive_thresholds(y, oof_scores[best_key])
    best_thresholds.insert(0, "feature_set", best_key[0])
    best_thresholds.insert(1, "model", best_key[1])
    best_thresholds.to_csv(TABLES / "final_grouped_tabular_thresholds.csv", index=False)
    final_model = clone(models[best_key[1]])
    final_model.fit(clean[best_cols], y)
    joblib.dump(final_model, MODELS / "final_grouped_tabular_model.joblib")
    metadata = {
        "selection_rule": "best mean average_precision across StratifiedGroupKFold folds",
        "selected_feature_set": best_key[0],
        "selected_model": best_key[1],
        "feature_columns": best_cols,
        "thresholds": best_thresholds.to_dict(orient="records"),
        "examples": int(len(clean)),
        "class_distribution": {str(k): int(v) for k, v in clean["clean_label"].value_counts().to_dict().items()},
        "groups": int(group_sizes.size),
        "duplicated_groups": int(len(duplicated_groups)),
        "examples_in_duplicated_groups": int(duplicated_groups.sum()) if not duplicated_groups.empty else 0,
        "max_group_size": int(duplicated_groups.max()) if not duplicated_groups.empty else 1,
        "n_splits": N_SPLITS,
        "max_shared_groups": int(by_fold["shared_groups"].max()),
        "best_by_pr_auc": best_pr.to_dict(),
        "best_by_f0_5": best_f05.to_dict(),
    }
    (MODELS / "final_grouped_tabular_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
