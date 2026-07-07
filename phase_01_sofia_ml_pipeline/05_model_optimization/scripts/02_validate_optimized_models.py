from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from utils import (
    ensure_dirs,
    load_config,
    load_dataset,
    load_feature_columns,
    load_json,
    make_model,
    metric_dict,
    positive_scores,
    save_dataframe,
)


DEFAULT_MODELS = [
    "RandomForest_full",
    "ExtraTrees_full",
    "XGBoost_full",
    "RandomForest_no_position",
    "GradientBoosting_no_position",
    "XGBoost_no_position",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida modelos optimizados con repeated CV y permutation test."
    )
    parser.add_argument("--config", type=Path, default=None)

    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help=(
            "Modelos a validar en formato Model_feature_set. "
            "Ejemplo: RandomForest_full GradientBoosting_no_position"
        ),
    )

    parser.add_argument(
        "--threshold-mode",
        choices=["default", "f1", "f0_5", "f2", "balanced_accuracy", "conservative"],
        default="f2",
        help=(
            "Threshold usado para métricas dependientes de umbral. "
            "'default' usa 0.5. El resto lee best_thresholds.json."
        ),
    )

    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=5)

    return parser.parse_args()


def parse_model_key(model_key: str) -> tuple[str, str]:
    if model_key.endswith("_no_position"):
        return model_key.removesuffix("_no_position"), "no_position"
    if model_key.endswith("_full"):
        return model_key.removesuffix("_full"), "full"
    raise ValueError(
        f"Nombre de modelo no reconocido: {model_key}. "
        "Usa formato Model_full o Model_no_position."
    )


def load_threshold(paths: dict, feature_set: str, model_name: str, threshold_mode: str) -> float:
    if threshold_mode == "default":
        return 0.5

    thresholds_path = paths["optuna_dir"] / feature_set / model_name / "best_thresholds.json"

    if not thresholds_path.exists():
        warnings.warn(
            f"No existe {thresholds_path}. Usando threshold=0.5 para {feature_set}/{model_name}."
        )
        return 0.5

    thresholds = load_json(thresholds_path)
    lookup_mode = "conservative_fp" if threshold_mode == "conservative" else threshold_mode
    entry = thresholds.get(lookup_mode)
    if isinstance(entry, dict) and "threshold" in entry:
        return float(entry["threshold"])
    if isinstance(entry, (int, float)):
        return float(entry)

    possible_keys = {
        "f1": ["best_f1_threshold", "f1_threshold", "threshold_f1"],
        "f0_5": ["best_f0_5_threshold", "f0_5_threshold", "threshold_f0_5"],
        "f2": ["best_f2_threshold", "f2_threshold", "threshold_f2"],
        "balanced_accuracy": [
            "best_balanced_accuracy_threshold",
            "balanced_accuracy_threshold",
            "threshold_balanced_accuracy",
        ],
        "conservative": [
            "conservative_threshold",
            "threshold_conservative",
        ],
    }

    for key in possible_keys[threshold_mode]:
        if key in thresholds:
            return float(thresholds[key])

    warnings.warn(
        f"No encuentro threshold '{threshold_mode}' en {thresholds_path}. "
        f"Claves disponibles: {list(thresholds.keys())}. Usando 0.5."
    )
    return 0.5


def repeated_cv(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    threshold: float,
    cv_splits: int,
    cv_repeats: int,
) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(
        n_splits=cv_splits,
        n_repeats=cv_repeats,
        random_state=random_state,
    )

    rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        model = make_model(model_name, params, random_state)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_score = positive_scores(model, X_test)

        row = {
            "fold": fold,
            "threshold": threshold,
        }
        row.update(metric_dict(y_test, y_score, threshold=threshold))
        rows.append(row)

    return pd.DataFrame(rows)


def permutation_test(
    model_name: str,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int,
    threshold: float,
    repeats: int,
    cv_splits: int,
) -> pd.DataFrame:
    """
    Permutation test reducido.

    Para no hacerlo carísimo, usa 1 repeat de CV por permutación.
    Es suficiente como sanity check: con etiquetas aleatorias deberían caer
    ROC-AUC y balanced accuracy cerca de 0.5.
    """
    rng = np.random.default_rng(random_state)
    rows = []

    for repeat in range(1, repeats + 1):
        shuffled = pd.Series(rng.permutation(y.to_numpy()), index=y.index)

        cv_df = repeated_cv(
            model_name=model_name,
            params=params,
            X=X,
            y=shuffled,
            random_state=random_state + repeat,
            threshold=threshold,
            cv_splits=cv_splits,
            cv_repeats=1,
        )

        rows.append({
            "repeat": repeat,
            "average_precision_mean": cv_df["average_precision"].mean(),
            "roc_auc_mean": cv_df["roc_auc"].mean(),
            "balanced_accuracy_mean": cv_df["balanced_accuracy"].mean(),
            "f2_mean": cv_df["f2"].mean(),
        })

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)

    _, clean_df, y, _ = load_dataset(config)

    opt_summary_path = paths["optuna_dir"] / "optimization_summary.csv"
    if not opt_summary_path.exists():
        raise SystemExit(f"Falta {opt_summary_path}. Ejecuta primero 01_optimize_models.py")

    opt_summary = pd.read_csv(opt_summary_path)

    random_state = int(config["optimization"]["random_state"])
    validation_rows = []

    for model_key in args.models:
        try:
            model_name, feature_set = parse_model_key(model_key)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue

        params_path = paths["optuna_dir"] / feature_set / model_name / "best_params.json"

        if not params_path.exists():
            warnings.warn(f"No existe {params_path}. Saltando {model_key}.")
            continue

        feature_columns = load_feature_columns(config, feature_set)
        X = clean_df[feature_columns].copy()

        params = load_json(params_path)
        threshold = load_threshold(paths, feature_set, model_name, args.threshold_mode)

        out_dir = paths["validation_dir"] / feature_set / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(
            f"Validando {feature_set} / {model_name} "
            f"con threshold_mode={args.threshold_mode}, threshold={threshold:.4f}..."
        )

        metrics = repeated_cv(
            model_name=model_name,
            params=params,
            X=X,
            y=y,
            random_state=random_state,
            threshold=threshold,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )

        permutation = permutation_test(
            model_name=model_name,
            params=params,
            X=X,
            y=y,
            random_state=random_state,
            threshold=threshold,
            repeats=args.permutation_repeats,
            cv_splits=args.cv_splits,
        )

        save_dataframe(metrics, out_dir / "repeated_cv_metrics.csv")
        save_dataframe(permutation, out_dir / "permutation_test_metrics.csv")

        validation_rows.append({
            "model_key": model_key,
            "feature_set": feature_set,
            "model": model_name,
            "threshold_mode": args.threshold_mode,
            "threshold": threshold,
            "average_precision_mean": metrics["average_precision"].mean(),
            "average_precision_std": metrics["average_precision"].std(),
            "roc_auc_mean": metrics["roc_auc"].mean(),
            "roc_auc_std": metrics["roc_auc"].std(),
            "f1_mean": metrics["f1"].mean(),
            "f2_mean": metrics["f2"].mean(),
            "precision_mean": metrics["precision"].mean(),
            "recall_mean": metrics["recall"].mean(),
            "balanced_accuracy_mean": metrics["balanced_accuracy"].mean(),
            "tn_sum": int(metrics["tn"].sum()),
            "fp_sum": int(metrics["fp"].sum()),
            "fn_sum": int(metrics["fn"].sum()),
            "tp_sum": int(metrics["tp"].sum()),
            "permutation_average_precision_mean": permutation["average_precision_mean"].mean(),
            "permutation_roc_auc_mean": permutation["roc_auc_mean"].mean(),
            "permutation_balanced_accuracy_mean": permutation["balanced_accuracy_mean"].mean(),
            "permutation_f2_mean": permutation["f2_mean"].mean(),
        })

    if not validation_rows:
        raise SystemExit("No se validó ningún modelo. Revisa --models y outputs/optuna.")

    validation_summary = (
        pd.DataFrame(validation_rows)
        .sort_values(["average_precision_mean", "f2_mean"], ascending=False)
    )

    save_dataframe(validation_summary, paths["validation_dir"] / "validation_summary.csv")

    print(f"Resumen de validación guardado en {paths['validation_dir'] / 'validation_summary.csv'}")


if __name__ == "__main__":
    main()
