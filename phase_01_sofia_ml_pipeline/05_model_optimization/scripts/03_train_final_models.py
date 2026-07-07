from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from utils import (
    ensure_dirs,
    load_config,
    load_dataset,
    load_feature_columns,
    load_json,
    make_model,
    resolve_path,
    save_json,
    save_pickle,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena y guarda modelos finales optimizados.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "RandomForest_full",
            "ExtraTrees_full",
            "XGBoost_full",
            "RandomForest_no_position",
            "GradientBoosting_no_position",
            "XGBoost_no_position",
        ],
    )
    parser.add_argument(
        "--threshold-mode",
        choices=["f1", "f0_5", "f2", "balanced_accuracy", "conservative"],
        default="f2",
    )
    return parser.parse_args()


def parse_model_key(model_key: str) -> tuple[str, str]:
    if model_key.endswith("_no_position"):
        return model_key.removesuffix("_no_position"), "no_position"
    if model_key.endswith("_full"):
        return model_key.removesuffix("_full"), "full"
    raise ValueError(f"Nombre de modelo no reconocido: {model_key}")


def get_selected_threshold(thresholds: dict, threshold_mode: str) -> float:
    if threshold_mode == "conservative":
        if "conservative_fp" in thresholds:
            return float(thresholds["conservative_fp"]["threshold"])
        if "conservative" in thresholds:
            return float(thresholds["conservative"]["threshold"])
    if threshold_mode in thresholds and isinstance(thresholds[threshold_mode], dict):
        return float(thresholds[threshold_mode]["threshold"])
    if threshold_mode in thresholds:
        return float(thresholds[threshold_mode])
    return 0.5


def selection_role(model_key: str, feature_set: str, threshold_mode: str) -> str:
    if threshold_mode == "conservative":
        return "conservative_final"
    if feature_set == "full":
        return "optimized_full"
    if feature_set == "no_position":
        return "optimized_no_position"
    return "manual_final"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    df, clean_df, y, counts = load_dataset(config)
    opt_summary_path = paths["optuna_dir"] / "optimization_summary.csv"
    if not opt_summary_path.exists():
        raise SystemExit(f"Falta {opt_summary_path}. Ejecuta primero 01_optimize_models.py")
    random_state = int(config["optimization"]["random_state"])
    source_dataset = str(resolve_path(config["data"]["candidates_path"]))
    saved_rows = []

    for model_key in args.models:
        try:
            model_name, feature_set = parse_model_key(model_key)
        except ValueError as exc:
            warnings.warn(str(exc))
            continue
        role = selection_role(model_key, feature_set, args.threshold_mode)
        feature_columns = load_feature_columns(config, feature_set)
        X = clean_df[feature_columns].copy()
        params_path = paths["optuna_dir"] / feature_set / model_name / "best_params.json"
        thresholds_path = paths["optuna_dir"] / feature_set / model_name / "best_thresholds.json"
        if not params_path.exists() or not thresholds_path.exists():
            warnings.warn(f"Saltando {model_key}: faltan {params_path} o {thresholds_path}")
            continue
        params = load_json(params_path)
        thresholds = load_json(thresholds_path)
        selected_threshold = get_selected_threshold(thresholds, args.threshold_mode)

        model = make_model(model_name, params, random_state)
        model.fit(X, y)
        model_path = paths["final_models_dir"] / f"{model_key}.pkl"
        features_path = paths["final_models_dir"] / f"{model_key}_features.json"
        metadata_path = paths["final_models_dir"] / f"{model_key}_metadata.json"
        save_pickle(model, model_path)
        save_json({"feature_set": feature_set, "model": model_name, "features": feature_columns}, features_path)
        metadata = {
            "selection_role": role,
            "model_key": model_key,
            "feature_set": feature_set,
            "model": model_name,
            "threshold_mode": args.threshold_mode,
            "params": params,
            "thresholds": thresholds,
            "selected_threshold": selected_threshold,
            "n_features": len(feature_columns),
            "features": feature_columns,
            "n_clean": counts["n_clean"],
            "n_tp": counts["n_tp"],
            "n_fp": counts["n_fp"],
            "n_ambiguous": counts["n_ambiguous"],
            "source_dataset": source_dataset,
        }
        save_json(metadata, metadata_path)
        saved_rows.append({
            "selection_role": role,
            "model_key": model_key,
            "feature_set": feature_set,
            "model": model_name,
            "model_path": str(model_path),
            "features_path": str(features_path),
            "metadata_path": str(metadata_path),
            "selected_threshold": selected_threshold,
        })
        print(f"Modelo final guardado: {model_path}")

    pd.DataFrame(saved_rows).to_csv(paths["final_models_dir"] / "final_models_manifest.csv", index=False)


if __name__ == "__main__":
    main()
