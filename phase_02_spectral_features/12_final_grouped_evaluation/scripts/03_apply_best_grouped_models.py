#!/usr/bin/env python3
"""Apply selected final grouped tabular and CNN models to the conservative catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BASE = Path("phase_02_spectral_features/12_final_grouped_evaluation")
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"
FILTERED = BASE / "outputs" / "filtered_catalogs"
REPORTS = BASE / "outputs" / "reports"
CONSERVATIVE_DATASET = Path("phase_02_spectral_features/02_build_extended_datasets/outputs/clean/sdc2_team_sofia_like_full_extended_clean.csv")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")


def prepare_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = np.nan
    return out[columns]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args
    FILTERED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    catalog = pd.read_csv(CONSERVATIVE_DATASET)
    rows = []

    tabular_model_path = MODELS / "final_grouped_tabular_model.joblib"
    tabular_meta_path = MODELS / "final_grouped_tabular_metadata.json"
    tabular_thresholds_path = TABLES / "final_grouped_tabular_thresholds.csv"
    if not tabular_model_path.exists() or not tabular_meta_path.exists() or not tabular_thresholds_path.exists():
        raise FileNotFoundError("Run 01_final_grouped_tabular_evaluation.py before applying tabular models.")
    tabular_model = joblib.load(tabular_model_path)
    tabular_meta = json.loads(tabular_meta_path.read_text(encoding="utf-8"))
    tabular_thresholds = pd.read_csv(tabular_thresholds_path)
    feature_columns = list(tabular_meta["feature_columns"])
    tabular_scores = tabular_model.predict_proba(prepare_features(catalog, feature_columns))[:, 1]
    catalog_tabular = catalog.copy()
    catalog_tabular["final_grouped_tabular_score"] = tabular_scores
    catalog_tabular["final_grouped_tabular_model"] = tabular_meta["selected_model"]
    catalog_tabular["final_grouped_tabular_feature_set"] = tabular_meta["selected_feature_set"]
    for mode, out_name in [("f0_5", "best_tabular_grouped_f0_5"), ("conservative_fp", "best_tabular_grouped_conservative_fp")]:
        threshold = float(tabular_thresholds.loc[tabular_thresholds["threshold_mode"] == mode, "threshold"].iloc[0])
        out = catalog_tabular[catalog_tabular["final_grouped_tabular_score"] >= threshold].copy()
        out["final_grouped_tabular_threshold_mode"] = mode
        out["final_grouped_tabular_threshold"] = threshold
        path = FILTERED / f"{out_name}.csv"
        out.to_csv(path, index=False)
        rows.append({"catalog_name": out_name, "model_family": "tabular", "threshold_mode": mode, "threshold": threshold, "accepted": len(out), "input": len(catalog), "path": path.as_posix()})

    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch is required to apply final grouped CNN: {exc}") from exc

    class FinalGroupedCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(21, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.classifier = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1))

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.classifier(self.features(x)).squeeze(1)

    cnn_path = MODELS / "final_grouped_cnn.pt"
    cnn_thresholds_path = TABLES / "final_grouped_cnn_thresholds.csv"
    if not cnn_path.exists() or not cnn_thresholds_path.exists():
        raise FileNotFoundError("Run 02_final_grouped_cnn_evaluation.py before applying CNN models.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(cnn_path, map_location=device)
    cnn = FinalGroupedCNN().to(device)
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()
    mean = np.asarray(checkpoint["channel_mean"], dtype="float32")
    std = np.asarray(checkpoint["channel_std"], dtype="float32")
    std[std < 1e-6] = 1.0
    patches = np.load(PATCH_BASE / "sdc2_conservative_patches.npy").astype("float32")
    meta = pd.read_csv(PATCH_BASE / "sdc2_conservative_metadata.csv")
    patches = ((patches - mean) / std).astype("float32")
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            batch = torch.tensor(patches[start : start + 64], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(cnn(batch)).detach().cpu().numpy())
    meta["final_grouped_cnn_score"] = np.concatenate(scores) if scores else []
    meta["final_grouped_cnn_architecture"] = checkpoint["architecture"]
    cnn_thresholds = pd.read_csv(cnn_thresholds_path)
    for mode, out_name in [("f0_5", "final_grouped_cnn_f0_5"), ("conservative_fp", "final_grouped_cnn_conservative_fp")]:
        threshold = float(cnn_thresholds.loc[cnn_thresholds["threshold_mode"] == mode, "threshold"].iloc[0])
        out = meta[meta["final_grouped_cnn_score"] >= threshold].copy()
        out["final_grouped_cnn_threshold_mode"] = mode
        out["final_grouped_cnn_threshold"] = threshold
        path = FILTERED / f"{out_name}.csv"
        out.to_csv(path, index=False)
        rows.append({"catalog_name": out_name, "model_family": "cnn", "threshold_mode": mode, "threshold": threshold, "accepted": len(out), "input": len(meta), "path": path.as_posix()})

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "final_grouped_filter_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
