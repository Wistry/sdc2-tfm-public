from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from utils import (
    ensure_dirs,
    feature_columns_path,
    load_config,
    load_dataset,
    load_feature_columns,
    make_model,
    metric_dict,
    positive_scores,
    save_dataframe,
    save_json,
    threshold_sweep,
)

try:
    import optuna
except ImportError:
    optuna = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimiza modelos seleccionados del benchmark con Optuna.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def suggest_params(trial, model_name: str) -> dict:
    if model_name in {"RandomForest", "ExtraTrees"}:
        depth_choice = trial.suggest_categorical("max_depth_choice", ["none", "int"])
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": None if depth_choice == "none" else trial.suggest_int("max_depth", 2, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
        return params
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if model_name == "GradientBoosting":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 5),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        }
    if model_name == "HistGradientBoosting":
        return {
            "max_iter": trial.suggest_int("max_iter", 50, 400),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 7, 63),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-8, 10.0, log=True),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 30),
        }
    raise ValueError(f"Modelo no soportado: {model_name}")


def feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.ravel(estimator.coef_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": feature_columns, "importance": values}).sort_values("importance", key=lambda s: s.abs(), ascending=False)


def main() -> None:
    args = parse_args()
    if optuna is None:
        raise SystemExit("Optuna no esta instalado. Instala optuna o ejecuta esta fase en el entorno correcto.")

    config = load_config(args.config)
    paths = ensure_dirs(config)
    _, clean_df, y, _ = load_dataset(config)
    opt_cfg = config["optimization"]
    thr_cfg = config["thresholds"]
    random_state = int(opt_cfg["random_state"])
    selected_models = opt_cfg["selected_models"]
    feature_sets = opt_cfg["feature_sets"]
    summary_rows = []

    for feature_set in feature_sets:
        feature_columns = load_feature_columns(config, feature_set)
        missing = [col for col in feature_columns if col not in clean_df.columns]
        if missing:
            raise SystemExit(f"Faltan columnas para {feature_set}: {missing}")
        X = clean_df[feature_columns].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=float(opt_cfg["test_size"]),
            random_state=random_state,
            stratify=y,
        )
        cv = StratifiedKFold(n_splits=int(opt_cfg["cv_splits"]), shuffle=True, random_state=random_state)

        for model_name in selected_models:
            if model_name == "XGBoost":
                try:
                    make_model("XGBoost", {}, random_state)
                except ImportError:
                    print("Saltando XGBoost: no esta instalado.")
                    continue
            print(f"Optimizando {feature_set} / {model_name}...")
            out_dir = paths["optuna_dir"] / feature_set / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            def objective(trial) -> float:
                params = suggest_params(trial, model_name)
                model = make_model(model_name, params, random_state)
                scores = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring="average_precision",
                    n_jobs=1,
                )
                return float(np.nanmean(scores))

            study = optuna.create_study(direction="maximize")
            study.optimize(
                objective,
                n_trials=int(opt_cfg["n_trials"]),
                timeout=int(opt_cfg["timeout_minutes_per_model"]) * 60,
                show_progress_bar=False,
            )
            best_params = study.best_params
            best_params.pop("max_depth_choice", None)
            save_json(best_params, out_dir / "best_params.json")
            save_dataframe(study.trials_dataframe(), out_dir / "optuna_trials.csv")

            model = make_model(model_name, best_params, random_state)
            model.fit(X_train, y_train)
            y_score = positive_scores(model, X_test)
            default_metrics = metric_dict(y_test, y_score, threshold=0.5)
            save_json(default_metrics, out_dir / "test_metrics_default_threshold.json")

            sweep, best_thresholds = threshold_sweep(
                y_test,
                y_score,
                float(thr_cfg["grid_min"]),
                float(thr_cfg["grid_max"]),
                float(thr_cfg["grid_step"]),
                float(thr_cfg.get("conservative_min_recall", 0.70)),
            )
            save_dataframe(sweep, out_dir / "threshold_sweep.csv")
            save_json(best_thresholds, out_dir / "best_thresholds.json")
            save_dataframe(feature_importance(model, feature_columns), out_dir / "feature_importance.csv")
            save_dataframe(pd.DataFrame({"y_true": y_test.to_numpy(), "y_score": y_score}, index=X_test.index).reset_index(names="row_index"), out_dir / "predictions_test.csv")

            f1_best = best_thresholds["f1"]
            f0_5_best = best_thresholds["f0_5"]
            f2_best = best_thresholds["f2"]
            ba_best = best_thresholds["balanced_accuracy"]
            conservative = best_thresholds["conservative_fp"]
            summary_rows.append({
                "feature_set": feature_set,
                "model": model_name,
                "n_features": len(feature_columns),
                "best_cv_average_precision": study.best_value,
                "test_average_precision": default_metrics["average_precision"],
                "test_roc_auc": default_metrics["roc_auc"],
                "best_f1_threshold": f1_best["threshold"],
                "best_f1": f1_best["f1"],
                "best_f0_5_threshold": f0_5_best["threshold"],
                "best_f0_5": f0_5_best["f0_5"],
                "best_f2_threshold": f2_best["threshold"],
                "best_f2": f2_best["f2"],
                "best_balanced_accuracy_threshold": ba_best["threshold"],
                "best_balanced_accuracy": ba_best["balanced_accuracy"],
                "conservative_threshold": conservative["threshold"],
                "conservative_precision": conservative["precision"],
                "conservative_recall": conservative["recall"],
                "conservative_fp": conservative["fp"],
                "conservative_tp": conservative["tp"],
            })

    summary = pd.DataFrame(summary_rows).sort_values(["test_average_precision", "best_f2"], ascending=False)
    save_dataframe(summary, paths["optuna_dir"] / "optimization_summary.csv")
    print(f"Resumen Optuna guardado en {paths['optuna_dir'] / 'optimization_summary.csv'}")


if __name__ == "__main__":
    main()
