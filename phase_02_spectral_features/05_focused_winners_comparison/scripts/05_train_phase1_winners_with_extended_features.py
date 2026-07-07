#!/usr/bin/env python3
"""Focused Phase 2 comparison using Phase 1 finalist algorithms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
PHASE1_MODEL_DIR = Path("phase_01_sofia_ml_pipeline/05_model_optimization/outputs/final_models")
BASELINE_REFERENCE = {
    "A_baseline_current_full_raw": {"n_candidates": 1169, "TP_clean": 317, "FP_clean": 126, "ambiguous": 726},
    "ML_XGBoost_full_f0_5": {"n_candidates": 528, "TP_clean": 291, "FP_clean": 0, "ambiguous": 237},
    "ML_XGBoost_full_f1": {"n_candidates": 629, "TP_clean": 306, "FP_clean": 0, "ambiguous": 323},
    "ML_ExtraTrees_full_f2": {"n_candidates": 903, "TP_clean": 317, "FP_clean": 6, "ambiguous": 580},
}

LEAKAGE_COLUMNS = {
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
    "candidate_index",
}

SPECTRAL_FEATURE_COLUMNS = [
    "feature_extraction_ok",
    "edge_clipped",
    "n_valid_channels",
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


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_metadata(model_key: str) -> dict[str, Any] | None:
    path = PHASE1_MODEL_DIR / f"{model_key}_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def config_for_catalog(cfg: dict[str, Any], catalog_key: str) -> dict[str, Any]:
    catalogs = cfg.get("catalogs") or {}
    if catalog_key not in catalogs:
        raise KeyError(f"Missing catalog key in config: {catalog_key}")
    out = dict(cfg)
    out.update(catalogs[catalog_key])
    return out


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [c for c in columns if c in df.columns and not c.endswith("_spectral")]


def phase1_features(df: pd.DataFrame, feature_set: str) -> list[str]:
    if feature_set == "full":
        for model_key in ["XGBoost_full", "RandomForest_full", "ExtraTrees_full"]:
            meta = load_metadata(model_key)
            if meta and meta.get("features"):
                return available_columns(df, list(meta["features"]))
    if feature_set == "no_position":
        for model_key in ["XGBoost_no_position", "GradientBoosting_no_position", "RandomForest_no_position"]:
            meta = load_metadata(model_key)
            if meta and meta.get("features"):
                return available_columns(df, list(meta["features"]))
    numeric = [
        c
        for c in df.select_dtypes(include=["number", "bool"]).columns
        if c not in LEAKAGE_COLUMNS and c not in SPECTRAL_FEATURE_COLUMNS and not c.endswith("_spectral")
    ]
    if feature_set == "no_position":
        blocked = {"x", "y", "z", "ra", "dec", "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak"}
        numeric = [c for c in numeric if c not in blocked and not c.endswith(("_min", "_max"))]
    return numeric


def make_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    spectral = available_columns(df, SPECTRAL_FEATURE_COLUMNS)
    full = phase1_features(df, "full")
    no_position = phase1_features(df, "no_position")
    if len(spectral) != 30:
        raise ValueError(f"Expected 30 spectral/local features, found {len(spectral)}")
    return {
        "sofia_only_full": full,
        "extended_full": full + spectral,
        "sofia_only_no_position": no_position,
        "extended_no_position": no_position + spectral,
        "spectral_only": spectral,
    }


def xgboost_model(params: dict[str, Any]) -> Any | None:
    try:
        from xgboost import XGBClassifier
    except Exception:
        return None
    clean_params = dict(params)
    clean_params.setdefault("random_state", RANDOM_STATE)
    clean_params.setdefault("eval_metric", "logloss")
    clean_params.setdefault("n_jobs", 1)
    return XGBClassifier(**clean_params)


def model_specs(feature_set: str) -> dict[str, tuple[Any, str]]:
    use_no_position = "no_position" in feature_set
    suffix = "no_position" if use_no_position else "full"
    specs: dict[str, tuple[Any, str]] = {}

    rf_meta = load_metadata(f"RandomForest_{suffix}") or load_metadata("RandomForest_full")
    if rf_meta:
        specs["RandomForest"] = (
            RandomForestClassifier(**rf_meta["params"], random_state=RANDOM_STATE, n_jobs=-1),
            f"phase1:{rf_meta['model_key']}",
        )

    et_meta = load_metadata("ExtraTrees_full")
    if et_meta:
        specs["ExtraTrees"] = (
            ExtraTreesClassifier(**et_meta["params"], random_state=RANDOM_STATE, n_jobs=-1),
            f"phase1:{et_meta['model_key']}",
        )

    xgb_meta = load_metadata(f"XGBoost_{suffix}") or load_metadata("XGBoost_full")
    if xgb_meta:
        model = xgboost_model(xgb_meta["params"])
        if model is not None:
            specs["XGBoost"] = (model, f"phase1:{xgb_meta['model_key']}")

    gb_meta = load_metadata("GradientBoosting_no_position")
    if gb_meta:
        specs["GradientBoosting"] = (
            GradientBoostingClassifier(**gb_meta["params"], random_state=RANDOM_STATE),
            f"phase1:{gb_meta['model_key']}",
        )

    specs["HistGradientBoosting"] = (
        HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=RANDOM_STATE),
        "reasonable_default:not_phase1_optimized",
    )
    return specs


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])


def predict_scores(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    if hasattr(pipe, "decision_function"):
        values = pipe.decision_function(x)
        return (values - values.min()) / (values.max() - values.min() + 1e-12)
    return pipe.predict(x).astype(float)


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score),
        "f0_5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "f1": fbeta_score(y_true, y_pred, beta=1.0, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_policies(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, dict[str, Any]]:
    thresholds = np.unique(np.r_[np.linspace(0.0, 1.0, 101), y_score])
    rows = [metrics_at_threshold(y_true, y_score, float(t)) for t in thresholds]
    policies = {
        "f0_5": max(rows, key=lambda r: (r["f0_5"], r["precision"], r["recall"], -r["threshold"])),
        "f1": max(rows, key=lambda r: (r["f1"], r["precision"], r["recall"], -r["threshold"])),
        "f2": max(rows, key=lambda r: (r["f2"], r["recall"], r["precision"], -r["threshold"])),
        "balanced_accuracy": max(rows, key=lambda r: (r["balanced_accuracy"], r["recall"], r["precision"], -r["threshold"])),
        "conservative_fp": min(rows, key=lambda r: (r["fp"], -r["recall"], -r["precision"], r["threshold"])),
    }
    return {name: dict(values, threshold_mode=name) for name, values in policies.items()}


def local_metrics(df: pd.DataFrame, accepted: pd.Series, total_tp: int) -> dict[str, Any]:
    selected = df.loc[accepted].copy()
    tp = int((selected["clean_label"] == 1).sum())
    fp = int((selected["clean_label"] == 0).sum())
    amb = int((selected["clean_label"] == -1).sum())
    reliability = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_tp if total_tp else 0.0
    precision = reliability
    return {
        "n_candidates": int(len(selected)),
        "TP_clean": tp,
        "FP_clean": fp,
        "ambiguous": amb,
        "reliability_clean": reliability,
        "recall_clean": recall,
        "f0_5_clean": fbeta_from_pr(precision, recall, beta=0.5),
        "f1_clean": fbeta_from_pr(precision, recall, beta=1.0),
        "f2_clean": fbeta_from_pr(precision, recall, beta=2.0),
        "ambiguous_rate": amb / len(selected) if len(selected) else 0.0,
    }


def fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / ((beta2 * precision) + recall)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    entry = config_for_catalog(cfg, "baseline_current_full")
    dataset_path = Path(entry["output_dataset"])
    out_root = Path("phase_02_spectral_features/05_focused_winners_comparison/outputs")
    models_dir = out_root / "models"
    reports_dir = out_root / "reports"
    filtered_dir = out_root / "local_filtered_catalogs"
    for path in [models_dir, reports_dir, filtered_dir]:
        path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_path)
    bad_cols = [c for c in df.columns if c.endswith("_spectral")]
    if bad_cols:
        raise ValueError(f"Unexpected duplicated spectral columns: {bad_cols}")

    clean = df[df["clean_label"].isin([0, 1])].copy()
    y = clean["clean_label"].astype(int).to_numpy()
    train_idx, test_idx = train_test_split(clean.index, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    train = clean.loc[train_idx].copy()
    test = clean.loc[test_idx].copy()
    y_train = train["clean_label"].astype(int).to_numpy()
    y_test = test["clean_label"].astype(int).to_numpy()
    feature_sets = make_feature_sets(df)

    result_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    model_sources: dict[str, str] = {}
    artifacts: dict[tuple[str, str], dict[str, Any]] = {}

    for feature_set, columns in feature_sets.items():
        for model_name, (model, source) in model_specs(feature_set).items():
            pipe = make_pipeline(model)
            pipe.fit(train[columns], y_train)
            y_score = predict_scores(pipe, test[columns])
            default_metrics = metrics_at_threshold(y_test, y_score, 0.5)
            row = {
                "feature_set": feature_set,
                "model": model_name,
                "n_features": len(columns),
                "n_train": len(train),
                "n_test": len(test),
                "model_source": source,
            }
            row.update(default_metrics)
            result_rows.append(row)
            policies = threshold_policies(y_test, y_score)
            for mode, values in policies.items():
                threshold_rows.append({"feature_set": feature_set, "model": model_name, **values})
            artifact_path = models_dir / f"{model_name}_{feature_set}.joblib"
            joblib.dump({"pipeline": pipe, "feature_set": feature_set, "model": model_name, "columns": columns, "model_source": source}, artifact_path)
            artifacts[(feature_set, model_name)] = {"path": artifact_path, "columns": columns, "pipeline": pipe}
            model_sources[f"{model_name}_{feature_set}"] = source

    results = pd.DataFrame(result_rows).sort_values(["average_precision", "f0_5"], ascending=False)
    thresholds = pd.DataFrame(threshold_rows)
    best_models = (
        results.sort_values(["feature_set", "average_precision", "f0_5"], ascending=[True, False, False])
        .groupby("feature_set", as_index=False)
        .first()
        .sort_values("average_precision", ascending=False)
    )

    results_path = reports_dir / "focused_phase2_results.csv"
    thresholds_path = reports_dir / "focused_phase2_thresholds.csv"
    best_path = reports_dir / "focused_phase2_best_models.csv"
    results.to_csv(results_path, index=False)
    thresholds.to_csv(thresholds_path, index=False)
    best_models.to_csv(best_path, index=False)

    total_tp = int((df["clean_label"] == 1).sum())
    local_rows: list[dict[str, Any]] = []
    for name, values in BASELINE_REFERENCE.items():
        precision = values["TP_clean"] / (values["TP_clean"] + values["FP_clean"]) if values["TP_clean"] + values["FP_clean"] else 0.0
        recall = values["TP_clean"] / total_tp if total_tp else 0.0
        local_rows.append(
            {
                "strategy": name,
                "source": "phase1_reference",
                **values,
                "reliability_clean": precision,
                "recall_clean": recall,
                "f0_5_clean": fbeta_from_pr(precision, recall, 0.5),
                "f1_clean": fbeta_from_pr(precision, recall, 1.0),
                "f2_clean": fbeta_from_pr(precision, recall, 2.0),
                "ambiguous_rate": values["ambiguous"] / values["n_candidates"],
            }
        )

    requested = [
        ("best_extended_full_f0_5", "extended_full", "f0_5"),
        ("best_extended_full_f1", "extended_full", "f1"),
        ("best_extended_full_f2", "extended_full", "f2"),
        ("best_extended_full_conservative_fp", "extended_full", "conservative_fp"),
        ("best_extended_no_position_f0_5", "extended_no_position", "f0_5"),
        ("best_extended_no_position_conservative_fp", "extended_no_position", "conservative_fp"),
    ]
    selected_strategies: list[dict[str, Any]] = []
    for strategy_name, feature_set, threshold_mode in requested:
        best = best_models.loc[best_models["feature_set"] == feature_set].iloc[0]
        model_name = str(best["model"])
        artifact = artifacts[(feature_set, model_name)]
        trow = thresholds[
            (thresholds["feature_set"] == feature_set)
            & (thresholds["model"] == model_name)
            & (thresholds["threshold_mode"] == threshold_mode)
        ].iloc[0]
        score = predict_scores(artifact["pipeline"], df[artifact["columns"]])
        accepted = pd.Series(score >= float(trow["threshold"]), index=df.index)
        selected = df.loc[accepted].copy()
        selected["phase2_score"] = score[accepted.to_numpy()]
        selected["phase2_strategy"] = strategy_name
        selected_path = filtered_dir / f"{strategy_name}.csv"
        selected.to_csv(selected_path, index=False)
        metrics = local_metrics(df, accepted, total_tp)
        local_rows.append(
            {
                "strategy": strategy_name,
                "source": "phase2_extended",
                "feature_set": feature_set,
                "model": model_name,
                "threshold_mode": threshold_mode,
                "threshold": float(trow["threshold"]),
                "catalog_path": str(selected_path),
                **metrics,
            }
        )
        selected_strategies.append(
            {
                "strategy_name": strategy_name,
                "feature_set": feature_set,
                "model": model_name,
                "threshold_mode": threshold_mode,
                "threshold": float(trow["threshold"]),
                "model_artifact": str(artifact["path"]),
                "catalog_path": str(selected_path),
            }
        )

    local_comparison = pd.DataFrame(local_rows)
    local_csv = reports_dir / "focused_phase2_local_catalog_comparison.csv"
    local_comparison.to_csv(local_csv, index=False)

    selected_path = reports_dir / "focused_phase2_selected_strategies.json"
    selected_path.write_text(json.dumps(selected_strategies, indent=2, ensure_ascii=True), encoding="utf-8")

    print(f"Results: {results_path}")
    print(f"Thresholds: {thresholds_path}")
    print(f"Best models: {best_path}")
    print(f"Local comparison: {local_csv}")
    print(best_models[["feature_set", "model", "average_precision", "f0_5", "f1", "f2", "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
