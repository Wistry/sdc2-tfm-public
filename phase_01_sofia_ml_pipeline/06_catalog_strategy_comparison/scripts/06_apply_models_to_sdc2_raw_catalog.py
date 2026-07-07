from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import (
    ensure_dirs,
    get_positive_scores,
    infer_feature_set_from_model_key,
    infer_model_name_from_model_key,
    load_config,
    load_final_model_bundle,
    resolve_path,
    resolve_threshold,
    safe_name,
    save_dataframe,
)


SOFIA_COLUMNS = [
    "name", "id", "x", "y", "z", "x_min", "x_max", "y_min", "y_max",
    "z_min", "z_max", "n_pix", "f_min", "f_max", "f_sum", "rel", "flag",
    "rms", "w20", "w50", "wm50", "z_w20", "z_w50", "z_wm50", "ell_maj",
    "ell_min", "ell_pa", "ell3s_maj", "ell3s_min", "ell3s_pa", "kin_pa",
    "err_x", "err_y", "err_z", "err_f_sum", "snr", "snr_max", "ra", "dec",
    "freq", "x_peak", "y_peak", "z_peak", "ra_peak", "dec_peak", "freq_peak",
]

STRATEGIES = [
    ("XGBoost_full", "f1"),
    ("XGBoost_full", "f0_5"),
    ("XGBoost_full", "conservative_fp"),
    ("ExtraTrees_full", "f0_5"),
    ("RandomForest_full", "f0_5"),
    ("GradientBoosting_no_position", "f0_5"),
    ("GradientBoosting_no_position", "conservative_fp"),
    ("XGBoost_no_position", "f0_5"),
    ("ExtraTrees_full", "balanced_accuracy"),
    ("RandomForest_full", "conservative_fp"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aplica modelos finales a sdc2_team_sofia_like_full raw.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def sdc2_catalog_path(config: dict) -> Path:
    for item in config.get("application_catalogs", []):
        if item.get("name") == "sdc2_team_sofia_like_full":
            return resolve_path(item["path"])
    return resolve_path("../03_candidate_dataset/outputs/sdc2_team_sofia_like_full/sdc2_team_sofia_like_full_cat.txt")


def read_sofia_txt(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        comment="#",
        names=SOFIA_COLUMNS,
        quotechar='"',
        engine="python",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    model_cfg = config["models"]
    fallback_threshold = float(model_cfg.get("fallback_threshold", 0.5))
    use_metadata_threshold = bool(model_cfg.get("use_threshold_from_metadata", True))
    final_models_dir = model_cfg["final_models_dir"]

    catalog_path = sdc2_catalog_path(config)
    if not catalog_path.exists():
        raise SystemExit(f"No existe catalogo SoFiA SDC2 conservador: {catalog_path}")
    candidates = read_sofia_txt(catalog_path)
    candidates["source_catalog"] = "sdc2_team_sofia_like_full"

    out_root = paths["sdc2_postfilter_dir"]
    predictions_dir = out_root / "predictions"
    accepted_dir = out_root / "accepted"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    accepted_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for model_key, threshold_mode in STRATEGIES:
        bundle = load_final_model_bundle(final_models_dir, model_key, fallback_threshold, use_metadata_threshold)
        if bundle is None:
            continue
        missing_features = [feature for feature in bundle["features"] if feature not in candidates.columns]
        if missing_features:
            print(f"WARNING: saltando SDC2_{model_key}_{threshold_mode}; faltan features: {missing_features}")
            continue

        X = candidates[bundle["features"]].copy()
        scores = get_positive_scores(bundle["model"], X)
        metadata = bundle["metadata"]
        threshold = resolve_threshold(metadata, threshold_mode, fallback_threshold) if use_metadata_threshold else fallback_threshold
        model_name = metadata.get("model") or infer_model_name_from_model_key(model_key)
        feature_set = metadata.get("feature_set") or infer_feature_set_from_model_key(model_key)
        strategy_name = f"SDC2_{model_key}_{threshold_mode}"

        pred = candidates.copy()
        pred["pred_proba_tp"] = scores
        pred["pred_label"] = (scores >= threshold).astype(int)
        pred["threshold"] = threshold
        pred["threshold_mode"] = threshold_mode
        pred["model_key"] = model_key
        pred["model_name"] = model_name
        pred["feature_set"] = feature_set
        pred["source_catalog"] = "sdc2_team_sofia_like_full"

        safe = safe_name(strategy_name)
        pred_path = predictions_dir / f"predictions_{safe}.csv"
        accepted_path = accepted_dir / f"accepted_{safe}.csv"
        accepted = pred[pred["pred_label"] == 1].copy()
        save_dataframe(pred, pred_path)
        save_dataframe(accepted, accepted_path)

        rows.append(
            {
                "strategy_name": strategy_name,
                "source_catalog": "sdc2_team_sofia_like_full",
                "model_key": model_key,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "n_total": int(len(pred)),
                "n_accepted": int(len(accepted)),
                "accepted_file": str(accepted_path),
            }
        )
        print(f"Guardado post-filtro SDC2: {accepted_path}")

    manifest = pd.DataFrame(rows)
    manifest_path = out_root / "sdc2_postfilter_manifest.csv"
    save_dataframe(manifest, manifest_path)
    print(f"Manifest post-filtro SDC2: {manifest_path}")


if __name__ == "__main__":
    main()
