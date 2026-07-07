#!/usr/bin/env python3
"""Benchmark SoFiA-only vs SoFiA+spectral/local feature sets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SPECTRAL_FEATURE_COLUMNS = [
    "spec_flux_sum_max",
    "spec_flux_sum_mean",
    "spec_flux_sum_std",
    "spec_flux_sum_argmax_rel",
    "spec_flux_sum_snr_like",
    "spec_flux_peak_max",
    "spec_flux_peak_mean",
    "area_mean",
    "area_max",
    "area_std",
    "area_n_active_channels",
    "area_fraction_active_channels",
    "centroid_dx_std",
    "centroid_dy_std",
    "centroid_drift_total",
    "centroid_drift_mean_step",
    "centroid_valid_fraction",
    "overlap_mean",
    "overlap_std",
    "overlap_min",
    "overlap_valid_pairs",
    "spectral_continuity_score",
    "local_contrast_mean",
    "local_contrast_max",
    "local_background_std",
    "local_source_mean",
    "local_source_max",
]

LEAKAGE_OR_ID_COLUMNS = {
    "candidate_index",
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
    "name",
    "id",
}

POSITION_COLUMNS = {
    "x",
    "y",
    "z",
    "x_min",
    "x_max",
    "y_min",
    "y_max",
    "z_min",
    "z_max",
    "x_peak",
    "y_peak",
    "z_peak",
    "ra",
    "dec",
    "freq",
    "ra_peak",
    "dec_peak",
    "freq_peak",
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_dataset(cfg: dict[str, Any], catalog_key: str) -> Path:
    catalogs = cfg.get("catalogs") or {}
    if catalog_key not in catalogs:
        raise KeyError(f"Unknown catalog_key '{catalog_key}'. Available: {', '.join(sorted(catalogs))}")
    return Path(catalogs[catalog_key]["output_dataset"])


def model_specs(random_state: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=random_state, class_weight="balanced", n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=random_state),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=1,
        )
    except Exception:
        pass
    return models


def make_pipeline(model_name: str, model: Any, feature_names: list[str]) -> Pipeline:
    if model_name == "LogisticRegression":
        preprocessor = ColumnTransformer(
            [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), feature_names)],
            remainder="drop",
        )
    else:
        preprocessor = ColumnTransformer(
            [("num", SimpleImputer(strategy="median"), feature_names)],
            remainder="drop",
        )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def predict_scores(pipe: Pipeline, x_test: pd.DataFrame) -> np.ndarray:
    model = pipe.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipe.predict_proba(x_test)[:, 1]
    if hasattr(model, "decision_function"):
        values = pipe.decision_function(x_test)
        return 1.0 / (1.0 + np.exp(-values))
    return pipe.predict(x_test)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "f0_5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--catalog-key", default="baseline_current_full")
    args = parser.parse_args()

    cfg = load_config(args.config)
    random_state = int(cfg.get("random_state", 42))
    dataset_path = resolve_dataset(cfg, args.catalog_key)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Extended dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    clean = df[df["clean_label"].isin([0, 1])].copy()
    if clean.empty:
        raise ValueError("No clean_label values in {0,1} available for benchmark.")

    spectral_cols = [c for c in SPECTRAL_FEATURE_COLUMNS if c in clean.columns]
    numeric_cols = [c for c in clean.columns if pd.api.types.is_numeric_dtype(clean[c])]
    sofia_full = [
        c
        for c in numeric_cols
        if c not in LEAKAGE_OR_ID_COLUMNS
        and c not in spectral_cols
        and not c.endswith("_spectral")
    ]
    sofia_no_position = [c for c in sofia_full if c not in POSITION_COLUMNS]

    feature_sets = {
        "sofia_only_full": sofia_full,
        "sofia_only_no_position": sofia_no_position,
        "extended_full": sofia_full + spectral_cols,
        "extended_no_position": sofia_no_position + spectral_cols,
        "spectral_only": spectral_cols,
    }

    y = clean["clean_label"].astype(int).to_numpy()
    train_idx, test_idx = train_test_split(
        clean.index,
        test_size=0.20,
        stratify=y,
        random_state=random_state,
    )
    train = clean.loc[train_idx]
    test = clean.loc[test_idx]
    y_train = train["clean_label"].astype(int).to_numpy()
    y_test = test["clean_label"].astype(int).to_numpy()

    rows: list[dict[str, Any]] = []
    models = model_specs(random_state)
    for feature_set, columns in feature_sets.items():
        if not columns:
            continue
        x_train = train[columns]
        x_test = test[columns]
        for model_name, model in models.items():
            pipe = make_pipeline(model_name, model, columns)
            pipe.fit(x_train, y_train)
            y_pred = pipe.predict(x_test)
            y_score = predict_scores(pipe, x_test)
            row = {
                "catalog_key": args.catalog_key,
                "feature_set": feature_set,
                "model": model_name,
                "n_features": len(columns),
                "n_train": len(train),
                "n_test": len(test),
            }
            row.update(metric_row(y_test, y_pred, y_score))
            rows.append(row)

    results = pd.DataFrame(rows).sort_values(["average_precision", "f0_5"], ascending=False)
    reports_dir = Path("phase_02_spectral_features/04_extended_benchmark/outputs/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = reports_dir / "extended_benchmark_results.csv"
    comparison_path = reports_dir / "extended_feature_set_comparison.csv"

    results.to_csv(results_path, index=False)
    comparison = (
        results.sort_values(["feature_set", "average_precision"], ascending=[True, False])
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("average_precision", ascending=False)
    )
    comparison.to_csv(comparison_path, index=False)

    print(f"Wrote benchmark results: {results_path}")
    print(f"Wrote feature set comparison: {comparison_path}")
    best = results.iloc[0]
    print(f"Best feature set: {best['feature_set']} / {best['model']} / PR-AUC={best['average_precision']:.6f}")


if __name__ == "__main__":
    main()
