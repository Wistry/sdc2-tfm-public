from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    ensure_dirs,
    get_positive_scores,
    infer_feature_set_from_model_key,
    infer_model_name_from_model_key,
    load_candidates,
    load_config,
    load_final_model_bundle,
    resolve_threshold,
    safe_name,
    save_dataframe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica modelos finales al catalogo de candidatos.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    candidates = load_candidates(config)
    model_cfg = config["models"]
    final_models_dir = model_cfg["final_models_dir"]
    fallback_threshold = float(model_cfg.get("fallback_threshold", 0.5))
    use_metadata_threshold = bool(model_cfg.get("use_threshold_from_metadata", True))
    threshold_modes = model_cfg.get("threshold_modes", ["selected"])
    rows = []

    label_column = config.get("data", {}).get("label_column", "clean_label")
    if label_column in candidates.columns:
        n_tp_clean = int((candidates[label_column] == 1).sum())
        n_fp_clean = int((candidates[label_column] == 0).sum())
        n_ambiguous = int((candidates[label_column] == -1).sum())
    else:
        n_tp_clean = 0
        n_fp_clean = 0
        n_ambiguous = 0
    print(
        "Catalogo cargado: "
        f"n_total={len(candidates)}, "
        f"n_tp_clean={n_tp_clean}, "
        f"n_fp_clean={n_fp_clean}, "
        f"n_ambiguous={n_ambiguous}"
    )
    if len(candidates) == n_tp_clean + n_fp_clean and n_ambiguous == 0:
        print("WARNING: el catalogo no contiene ambiguos; revisa que no se haya filtrado antes de aplicar modelos.")

    for model_key in model_cfg.get("selected_models", []):
        bundle = load_final_model_bundle(final_models_dir, model_key, fallback_threshold, use_metadata_threshold)
        if bundle is None:
            continue
        missing_features = [feature for feature in bundle["features"] if feature not in candidates.columns]
        if missing_features:
            print(f"WARNING: saltando {model_key}; faltan features: {missing_features}")
            continue

        X = candidates[bundle["features"]].copy()
        scores = get_positive_scores(bundle["model"], X)
        metadata = bundle["metadata"]
        model_name = metadata.get("model") or infer_model_name_from_model_key(model_key)
        feature_set = metadata.get("feature_set") or infer_feature_set_from_model_key(model_key)

        for threshold_mode in threshold_modes:
            threshold = (
                resolve_threshold(metadata, threshold_mode, fallback_threshold)
                if use_metadata_threshold
                else fallback_threshold
            )
            pred_label = (scores >= threshold).astype(int)
            predictions = candidates.copy()
            predictions["pred_proba_tp"] = scores
            predictions["pred_label"] = pred_label
            predictions["threshold"] = threshold
            predictions["threshold_mode"] = threshold_mode
            predictions["model_key"] = model_key
            predictions["model_name"] = model_name
            predictions["feature_set"] = feature_set

            safe = safe_name(f"{model_key}_{threshold_mode}")
            pred_path = paths["predictions_dir"] / f"predictions_{safe}.csv"
            accepted_path = paths["accepted_dir"] / f"accepted_{safe}.csv"
            accepted = predictions[predictions["pred_label"] == 1].copy()
            save_dataframe(predictions, pred_path)
            save_dataframe(accepted, accepted_path)

            n_total = len(predictions)
            n_accepted = int((predictions["pred_label"] == 1).sum())
            if label_column in accepted.columns:
                accepted_tp_clean = int((accepted[label_column] == 1).sum())
                accepted_fp_clean = int((accepted[label_column] == 0).sum())
                accepted_ambiguous = int((accepted[label_column] == -1).sum())
            else:
                accepted_tp_clean = 0
                accepted_fp_clean = 0
                accepted_ambiguous = 0
            rows.append({
                "model_key": model_key,
                "model_name": model_name,
                "feature_set": feature_set,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "n_total": n_total,
                "n_accepted": n_accepted,
                "n_rejected": n_total - n_accepted,
                "accepted_rate": n_accepted / n_total if n_total else 0.0,
                "accepted_tp_clean": accepted_tp_clean,
                "accepted_fp_clean": accepted_fp_clean,
                "accepted_ambiguous": accepted_ambiguous,
            })
            print(f"Predicciones guardadas: {pred_path}")

    summary = pd.DataFrame(rows)
    save_dataframe(summary, paths["predictions_dir"] / "prediction_summary.csv")
    print(f"Resumen de predicciones: {paths['predictions_dir'] / 'prediction_summary.csv'}")


if __name__ == "__main__":
    main()
