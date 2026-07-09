#!/usr/bin/env python3
"""Apply all methodologically valid frozen CNNs to 40 GB extended-validation patches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

VALIDATION_ROOT = Path(__file__).resolve().parents[1]
if str(VALIDATION_ROOT) not in sys.path:
    sys.path.insert(0, str(VALIDATION_ROOT))

from validation40gb_utils import ensure_dir


BASE = VALIDATION_ROOT
PHASE2_ROOT = VALIDATION_ROOT.parent / "phase_02_spectral_features"
PATCH_DIR = BASE / "outputs" / "external_cnn_patches"
PRED_DIR = BASE / "outputs" / "external_cnn_valid_models_predictions"
FILTERED_DIR = BASE / "outputs" / "external_cnn_valid_models_filtered_catalogs"

CATALOGS = ["baseline_current_40gb", "sdc2_team_sofia_like_40gb"]
THRESHOLD_MODES = ["f0_5", "conservative_fp"]


@dataclass(frozen=True)
class FrozenCNN:
    model_name: str
    phase: str
    model_path: Path
    thresholds_path: Path
    metadata_path: Path
    architecture_hint: str
    validation: str
    leakage: str
    input_description: str = "cube patch 21x32x32, z/frequency as channel axis, y/x as spatial axes"


def valid_models() -> list[FrozenCNN]:
    return [
        FrozenCNN(
            model_name="CNN_grouped_phase14",
            phase="10_grouped_validation",
            model_path=PHASE2_ROOT / "10_grouped_validation" / "outputs" / "models" / "small_cnn_grouped.pt",
            thresholds_path=PHASE2_ROOT / "10_grouped_validation" / "outputs" / "tables" / "grouped_cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "10_grouped_validation" / "outputs" / "models" / "small_cnn_grouped_metadata.json",
            architecture_hint="small_grouped_2_5d",
            validation="StratifiedGroupKFold / grouped holdout",
            leakage="shared_groups=0",
        ),
        FrozenCNN(
            model_name="ROBUST_CNN_grouped_phase15",
            phase="11_robust_grouped_cnn",
            model_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "models" / "best_robust_grouped_cnn.pt",
            thresholds_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "tables" / "best_robust_grouped_cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "models" / "best_robust_grouped_cnn_metadata.json",
            architecture_hint="checkpoint_architecture",
            validation="StratifiedGroupKFold, multi-seed/model search",
            leakage="shared_groups=0",
        ),
        FrozenCNN(
            model_name="final_grouped_cnn_phase16",
            phase="12_final_grouped_evaluation",
            model_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "models" / "final_grouped_cnn.pt",
            thresholds_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "tables" / "final_grouped_cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "models" / "final_grouped_cnn_metadata.json",
            architecture_hint="checkpoint_architecture",
            validation="5-fold StratifiedGroupKFold",
            leakage="max_shared_groups=0",
        ),
    ]


def load_thresholds(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if not {"threshold_mode", "threshold"}.issubset(df.columns):
        raise ValueError(f"Threshold table must contain threshold_mode and threshold: {path}")
    values = {str(row["threshold_mode"]): float(row["threshold"]) for _, row in df.iterrows()}
    missing = [mode for mode in THRESHOLD_MODES if mode not in values]
    if missing:
        raise ValueError(f"Missing frozen threshold(s) {missing} in {path}")
    return {mode: values[mode] for mode in THRESHOLD_MODES}


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_external_patches(catalog_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    patch_path = PATCH_DIR / f"{catalog_key}_external_patches.npz"
    metadata_path = PATCH_DIR / f"{catalog_key}_external_patches_metadata.csv"
    if not patch_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing external CNN patches for {catalog_key}. Run 05_cnn_models_40gb/01_extract_cnn_patches_external_40gb.py first.")
    with np.load(patch_path) as data:
        patches = data["patches"].astype("float32")
    metadata = pd.read_csv(metadata_path)
    if len(patches) != len(metadata):
        raise ValueError(f"Patch count and metadata rows differ for {catalog_key}: {len(patches)} != {len(metadata)}")
    return patches, metadata


def channel_stats(checkpoint: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
    if "channel_mean" not in checkpoint or "channel_std" not in checkpoint:
        return None, None
    mean = np.asarray(checkpoint["channel_mean"], dtype="float32")
    std = np.asarray(checkpoint["channel_std"], dtype="float32")
    if mean.size == 0 or std.size == 0:
        return None, None
    if mean.ndim == 1:
        mean = mean.reshape(1, mean.shape[0], 1, 1)
    if std.ndim == 1:
        std = std.reshape(1, std.shape[0], 1, 1)
    std = std.copy()
    std[std < 1e-6] = 1.0
    return mean.astype("float32"), std.astype("float32")


def main() -> None:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch is required to apply frozen CNNs: {exc}") from exc

    class SmallCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(21, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x).squeeze(1)

    class RobustCNN25D(nn.Module):
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

    class Lightweight3DCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d((1, 2, 2)),
                nn.Conv3d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool3d((2, 2, 2)),
                nn.AdaptiveAvgPool3d((1, 1, 1)),
                nn.Flatten(),
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 1),
            )

        def forward(self, x):  # type: ignore[no-untyped-def]
            return self.net(x.unsqueeze(1)).squeeze(1)

    model_classes = {
        "small_grouped_2_5d": SmallCNN,
        "baseline_grouped": SmallCNN,
        "robust_2_5d": RobustCNN25D,
        "final_grouped_cnn": RobustCNN25D,
        "final_grouped_cnn_2_5d": RobustCNN25D,
        "lightweight_3d": Lightweight3DCNN,
    }

    ensure_dir(PRED_DIR)
    ensure_dir(FILTERED_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_rows: list[dict[str, Any]] = []

    for spec in valid_models():
        if not spec.model_path.exists() or not spec.thresholds_path.exists() or not spec.metadata_path.exists():
            summary_rows.append(
                {
                    "cnn_model": spec.model_name,
                    "phase": spec.phase,
                    "status": "SKIPPED",
                    "reason": "missing checkpoint, thresholds, or metadata",
                    "model_path": str(spec.model_path),
                }
            )
            continue

        checkpoint = torch.load(spec.model_path, map_location=device)
        architecture = str(checkpoint.get("architecture") or spec.architecture_hint)
        if architecture not in model_classes:
            raise ValueError(f"Unsupported architecture '{architecture}' in {spec.model_path}")
        model = model_classes[architecture]().to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        mean, std = channel_stats(checkpoint)
        thresholds = load_thresholds(spec.thresholds_path)
        metadata = load_metadata(spec.metadata_path)

        for catalog_key in CATALOGS:
            patches, meta = load_external_patches(catalog_key)
            if mean is not None and std is not None:
                patches = ((patches - mean) / std).astype("float32")
                normalization = "per-patch robust normalization plus frozen channel_mean/channel_std"
            else:
                normalization = "per-patch robust normalization only"

            scores: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(patches), 64):
                    batch = torch.tensor(patches[start : start + 64], dtype=torch.float32, device=device)
                    scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
            probabilities = np.concatenate(scores) if scores else np.empty((0,), dtype="float32")

            pred = meta.copy()
            pred["cnn_probability"] = probabilities
            pred["cnn_model"] = spec.model_name
            pred["cnn_phase"] = spec.phase
            pred["cnn_architecture"] = architecture
            pred["cnn_validation"] = spec.validation
            pred["cnn_leakage"] = spec.leakage
            pred["cnn_input"] = spec.input_description
            pred["cnn_normalization"] = normalization
            pred["cnn_model_path"] = str(spec.model_path)
            pred["cnn_thresholds_path"] = str(spec.thresholds_path)
            pred["cnn_metadata_path"] = str(spec.metadata_path)
            pred["cnn_training_examples"] = metadata.get("examples", metadata.get("train_examples", ""))
            pred["cnn_training_class_distribution"] = json.dumps(metadata.get("class_distribution", {}), sort_keys=True)

            pred_path = PRED_DIR / f"{catalog_key}_external_{spec.model_name}_predictions.csv"
            pred.to_csv(pred_path, index=False)

            for mode, threshold in thresholds.items():
                out = pred.copy()
                out["threshold_mode"] = mode
                out["cnn_threshold"] = threshold
                out["cnn_keep"] = out["cnn_probability"] >= threshold
                filtered = out[out["cnn_keep"]].copy()
                filtered_path = FILTERED_DIR / f"{catalog_key}_external_{spec.model_name}_{mode}.csv"
                filtered.to_csv(filtered_path, index=False)
                summary_rows.append(
                    {
                        "base_catalog": catalog_key,
                        "cnn_model": spec.model_name,
                        "phase": spec.phase,
                        "architecture": architecture,
                        "threshold_mode": mode,
                        "threshold": threshold,
                        "n_input": len(out),
                        "n_accepted": len(filtered),
                        "validation": spec.validation,
                        "leakage": spec.leakage,
                        "normalization": normalization,
                        "status": "OK",
                        "reason": "",
                        "prediction_path": str(pred_path),
                        "filtered_catalog_path": str(filtered_path),
                        "model_path": str(spec.model_path),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    summary_path = PRED_DIR / "valid_frozen_cnn_external_prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
