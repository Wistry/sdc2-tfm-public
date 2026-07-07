from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from utils import (
    detect_label_column,
    detect_leakage_columns,
    ensure_dirs,
    get_numeric_feature_columns,
    load_candidates,
    load_config,
    plot_and_save,
    safe_model_name,
    save_dataframe,
    save_json,
    split_clean_ambiguous,
)

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


NO_POSITION_COLUMNS = [
    "x", "y", "z",
    "ra", "dec", "freq",
    "x_min", "x_max",
    "y_min", "y_max",
    "z_min", "z_max",
    "x_peak", "y_peak", "z_peak",
    "ra_peak", "dec_peak", "freq_peak",
    "z_w20", "z_w50", "z_wm50",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark de clasificadores para candidatos SoFiA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--candidates-path", type=Path, default=None)
    parser.add_argument("--feature-set", choices=["full", "no_position"], default="full")
    return parser.parse_args()


def make_pipeline(model, scaled: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def build_models(random_state: int) -> dict[str, Pipeline]:
    models = {
        "DummyMostFrequent": make_pipeline(DummyClassifier(strategy="most_frequent"), scaled=False),
        "DummyStratified": make_pipeline(DummyClassifier(strategy="stratified", random_state=random_state), scaled=False),
        "LogisticRegression": make_pipeline(
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
            scaled=True,
        ),
        "DecisionTree": make_pipeline(
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            scaled=False,
        ),
        "RandomForest": make_pipeline(
            RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=1),
            scaled=False,
        ),
        "ExtraTrees": make_pipeline(
            ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=random_state, n_jobs=1),
            scaled=False,
        ),
        "GradientBoosting": make_pipeline(GradientBoostingClassifier(random_state=random_state), scaled=False),
        "HistGradientBoosting": make_pipeline(HistGradientBoostingClassifier(random_state=random_state), scaled=False),
        "SVC": make_pipeline(SVC(class_weight="balanced", probability=True, random_state=random_state), scaled=True),
        "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=7), scaled=True),
        "MLPClassifier": make_pipeline(
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, early_stopping=True, random_state=random_state),
            scaled=True,
        ),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = make_pipeline(
            XGBClassifier(
                n_estimators=250,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=random_state,
            ),
            scaled=False,
        )
    return models


def positive_scores(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-raw))
    return model.predict(X).astype(float)


def test_metrics(y_true: pd.Series, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "average_precision": average_precision_score(y_true, y_score),
        "roc_auc": roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f0_5": fbeta_score(y_true, y_pred, beta=0.5, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def plot_model_outputs(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, y_score: np.ndarray, model_name: str, benchmark_dir: Path) -> None:
    safe = safe_model_name(model_name)
    y_pred = (y_score >= 0.5).astype(int)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, labels=[0, 1], ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {model_name}")
    plot_and_save(fig, benchmark_dir / "figures" / f"confusion_matrix_{safe}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title(f"ROC - {model_name}")
    plot_and_save(fig, benchmark_dir / "figures" / f"roc_curve_{safe}.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title(f"Precision-Recall - {model_name}")
    plot_and_save(fig, benchmark_dir / "figures" / f"pr_curve_{safe}.png")
    plt.close(fig)


def select_feature_columns(
    clean_df: pd.DataFrame,
    label_column: str,
    leakage_df: pd.DataFrame,
    feature_set: str,
) -> tuple[list[str], list[str], list[str]]:
    initial_features = get_numeric_feature_columns(clean_df, label_column, leakage_df)
    no_position_excluded = []
    if feature_set == "no_position":
        no_position_excluded = [column for column in NO_POSITION_COLUMNS if column in initial_features]
    final_features = [column for column in initial_features if column not in no_position_excluded]
    return initial_features, final_features, no_position_excluded


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    feature_set = args.feature_set
    benchmark_dir = paths["benchmark_dir"] / feature_set
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    exp_cfg = config.get("experiment", {})
    data_cfg = config.get("data", {})
    random_state = int(exp_cfg.get("random_state", 42))
    test_size = float(exp_cfg.get("test_size", 0.2))
    n_splits = int(exp_cfg.get("n_splits", 5))
    n_repeats = int(exp_cfg.get("n_repeats", 5))

    df = load_candidates(config, args.candidates_path)
    label_column = detect_label_column(df, data_cfg.get("label_column", "clean_label"))
    clean_df, ambiguous_df = split_clean_ambiguous(
        df,
        label_column,
        int(data_cfg.get("positive_label", 1)),
        int(data_cfg.get("negative_label", 0)),
        int(data_cfg.get("ambiguous_label", -1)),
    )
    if clean_df[label_column].nunique() < 2:
        raise SystemExit("Se necesitan TP y FP limpios para entrenar.")

    leakage_df = detect_leakage_columns(clean_df, label_column)
    initial_feature_columns, feature_columns, no_position_excluded = select_feature_columns(
        clean_df,
        label_column,
        leakage_df,
        feature_set,
    )
    if not feature_columns:
        raise SystemExit("No quedan features numericas tras excluir leakage.")

    X = clean_df[feature_columns].copy()
    y = clean_df[label_column].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scoring = {
        "average_precision": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "f0_5": make_scorer(fbeta_score, beta=0.5, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2.0, zero_division=0),
        "precision": "precision",
        "recall": "recall",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
    }
    min_class = int(y_train.value_counts().min())
    cv_splits = max(2, min(n_splits, min_class))
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=n_repeats, random_state=random_state)

    cv_rows = []
    test_rows = []
    all_predictions = []
    reports = {}
    models = build_models(random_state)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for model_name, pipeline in models.items():
        print(f"Entrenando {model_name}...")
        safe = safe_model_name(model_name)
        model = clone(pipeline)
        cv_scores = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score=np.nan)
        cv_row = {"model": model_name}
        for metric in scoring:
            values = cv_scores[f"test_{metric}"]
            cv_row[f"cv_{metric}_mean"] = float(np.nanmean(values))
            cv_row[f"cv_{metric}_std"] = float(np.nanstd(values))
        cv_rows.append(cv_row)

        model.fit(X_train, y_train)
        y_score = positive_scores(model, X_test)
        y_pred = (y_score >= 0.5).astype(int)
        row = {"model": model_name}
        row.update(test_metrics(y_test, y_score))
        test_rows.append(row)

        predictions = pd.DataFrame({
            "model": model_name,
            "row_index": X_test.index,
            "y_true": y_test.to_numpy(),
            "y_score": y_score,
            "y_pred": y_pred,
        })
        save_dataframe(predictions, benchmark_dir / f"predictions_{safe}.csv")
        all_predictions.append(predictions)

        reports[model_name] = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        plot_model_outputs(model, X_test, y_test, y_score, model_name, benchmark_dir)

    cv_df = pd.DataFrame(cv_rows).sort_values(["cv_average_precision_mean", "cv_f0_5_mean"], ascending=False)
    test_df = pd.DataFrame(test_rows).sort_values(["average_precision", "f0_5"], ascending=False)
    ranking_df = test_df[
        ["model", "average_precision", "f0_5", "f1", "f2", "precision", "recall", "balanced_accuracy", "roc_auc"]
    ].copy()
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))
    f0_5_ranking_df = test_df[
        ["model", "f0_5", "average_precision", "precision", "recall", "f1", "f2", "balanced_accuracy", "roc_auc"]
    ].sort_values(["f0_5", "average_precision"], ascending=False).copy()
    f0_5_ranking_df.insert(0, "f0_5_rank", range(1, len(f0_5_ranking_df) + 1))

    save_dataframe(cv_df, benchmark_dir / "benchmark_cv_results.csv")
    save_dataframe(test_df, benchmark_dir / "benchmark_test_results.csv")
    save_dataframe(ranking_df, benchmark_dir / "model_rankings.csv")
    save_dataframe(f0_5_ranking_df, benchmark_dir / "model_rankings_f0_5.csv")
    save_dataframe(pd.DataFrame({"feature": feature_columns}), benchmark_dir / "feature_columns_used.csv")
    save_dataframe(leakage_df[leakage_df["exclude"]], benchmark_dir / "excluded_columns.csv")
    if feature_set == "no_position":
        save_dataframe(pd.DataFrame({"column": no_position_excluded}), benchmark_dir / "excluded_no_position_columns.csv")
    save_dataframe(pd.concat(all_predictions, ignore_index=True), benchmark_dir / "test_predictions_all_models.csv")
    save_json(reports, benchmark_dir / "classification_reports.json")
    save_json({
        "feature_set": feature_set,
        "n_total_rows": int(len(df)),
        "n_clean_rows": int(len(clean_df)),
        "n_ambiguous_rows": int(len(ambiguous_df)),
        "n_features_initial_non_leakage": int(len(initial_feature_columns)),
        "n_features_used": int(len(feature_columns)),
        "n_no_position_excluded": int(len(no_position_excluded)),
        "no_position_excluded_columns": no_position_excluded,
        "test_size": test_size,
        "n_splits": cv_splits,
        "n_repeats": n_repeats,
        "models": list(models.keys()),
    }, benchmark_dir / "benchmark_metadata.json")

    print(f"Benchmark guardado en {benchmark_dir}")


if __name__ == "__main__":
    main()
