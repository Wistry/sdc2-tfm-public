#!/usr/bin/env python3
"""Apply frozen CNN models to external 40GB candidate patches."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PHASE3_ROOT = Path(__file__).resolve().parents[1]
if str(PHASE3_ROOT) not in sys.path:
    sys.path.insert(0, str(PHASE3_ROOT))

from phase03_utils import ensure_dir


BASE = PHASE3_ROOT
PHASE2_ROOT = PHASE3_ROOT.parent / "phase_02_spectral_features"
PATCH_DIR = BASE / "outputs" / "external_cnn_patches"
PRED_DIR = BASE / "outputs" / "external_cnn_predictions"
FILTERED_DIR = BASE / "outputs" / "external_cnn_filtered_catalogs"

THRESHOLD_MODES = ["f0_5", "conservative_fp"]
CATALOGS = ["baseline_current_40gb", "sdc2_team_sofia_like_40gb"]


@dataclass(frozen=True)
class FrozenCNN:
    model_name: str
    model_path: Path
    thresholds_path: Path
    metadata_path: Path
    priority: int
    exploratory_leakage_warning: bool = False


def candidate_models() -> list[FrozenCNN]:
    return [
        FrozenCNN(
            model_name="final_grouped_cnn",
            model_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "models" / "final_grouped_cnn.pt",
            thresholds_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "tables" / "final_grouped_cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "12_final_grouped_evaluation" / "outputs" / "models" / "final_grouped_cnn_metadata.json",
            priority=1,
        ),
        FrozenCNN(
            model_name="robust_grouped_cnn",
            model_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "models" / "best_robust_grouped_cnn.pt",
            thresholds_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "tables" / "best_robust_grouped_cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "11_robust_grouped_cnn" / "outputs" / "models" / "best_robust_grouped_cnn_metadata.json",
            priority=2,
        ),
        FrozenCNN(
            model_name="original_small_cnn_leakage_reference",
            model_path=PHASE2_ROOT / "08_cnn_candidate_classifier" / "outputs" / "models" / "small_cnn.pt",
            thresholds_path=PHASE2_ROOT / "08_cnn_candidate_classifier" / "outputs" / "reports" / "cnn_thresholds.csv",
            metadata_path=PHASE2_ROOT / "08_cnn_candidate_classifier" / "outputs" / "models" / "small_cnn_metadata.json",
            priority=3,
            exploratory_leakage_warning=True,
        ),
    ]


def available_models() -> list[FrozenCNN]:
    models = [
        model
        for model in candidate_models()
        if model.model_path.exists() and model.thresholds_path.exists() and model.metadata_path.exists()
    ]
    return sorted(models, key=lambda item: item.priority)


def load_thresholds(path: Path) -> dict[str, float]:
    df = pd.read_csv(path)
    if "threshold_mode" not in df.columns or "threshold" not in df.columns:
        raise ValueError(f"Threshold table must contain threshold_mode and threshold: {path}")
    mapping = {str(row["threshold_mode"]): float(row["threshold"]) for _, row in df.iterrows()}
    missing = [mode for mode in THRESHOLD_MODES if mode not in mapping]
    if missing:
        raise ValueError(f"Missing frozen CNN threshold(s) {missing} in {path}. Refusing to invent thresholds.")
    return {mode: mapping[mode] for mode in THRESHOLD_MODES}


def channel_stats(checkpoint: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(checkpoint.get("channel_mean"), dtype="float32")
    std = np.asarray(checkpoint.get("channel_std"), dtype="float32")
    if mean.size == 0 or std.size == 0:
        raise ValueError("Checkpoint does not contain channel_mean/channel_std.")
    if mean.ndim == 1:
        mean = mean.reshape(1, mean.shape[0], 1, 1)
    if std.ndim == 1:
        std = std.reshape(1, std.shape[0], 1, 1)
    std = std.copy()
    std[std < 1e-6] = 1.0
    return mean.astype("float32"), std.astype("float32")


def load_external_patches(catalog_key: str) -> tuple[np.ndarray, pd.DataFrame]:
    patch_path = PATCH_DIR / f"{catalog_key}_external_patches.npz"
    metadata_path = PATCH_DIR / f"{catalog_key}_external_patches_metadata.csv"
    if not patch_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing patches or metadata for {catalog_key}. "
            "Run 05_cnn_models_40gb/01_extract_cnn_patches_external_40gb.py first."
        )
    with np.load(patch_path) as data:
        patches = data["patches"].astype("float32")
    metadata = pd.read_csv(metadata_path)
    if len(patches) != len(metadata):
        raise ValueError(f"Patch count and metadata rows differ for {catalog_key}: {len(patches)} != {len(metadata)}")
    return patches, metadata


def main() -> None:
    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch is required to apply frozen CNN models: {exc}") from exc

    class BaselineCNN(nn.Module):
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
        "baseline_grouped": BaselineCNN,
        "robust_2_5d": RobustCNN25D,
        "lightweight_3d": Lightweight3DCNN,
        "final_grouped_cnn": RobustCNN25D,
        "final_grouped_cnn_2_5d": RobustCNN25D,
    }

    models = available_models()
    if not models:
        searched = "\n".join(str(model.model_path) for model in candidate_models())
        raise FileNotFoundError(f"No frozen CNN model located safely. Searched:\n{searched}")
    selected = models[0]
    if selected.exploratory_leakage_warning:
        raise RuntimeError(
            "Only the original CNN with possible leakage was found. "
            "It is exploratory reference only and is not used by default."
        )

    thresholds = load_thresholds(selected.thresholds_path)
    metadata = json.loads(selected.metadata_path.read_text(encoding="utf-8"))
    ensure_dir(PRED_DIR)
    ensure_dir(FILTERED_DIR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(selected.model_path, map_location=device)
    architecture = str(checkpoint.get("architecture", selected.model_name))
    if architecture not in model_classes:
        raise ValueError(f"Unsupported CNN architecture '{architecture}' in {selected.model_path}")
    model = model_classes[architecture]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    mean, std = channel_stats(checkpoint)

    summary_rows = []
    for catalog_key in CATALOGS:
        patches, meta = load_external_patches(catalog_key)
        patches = ((patches - mean) / std).astype("float32")
        scores: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(patches), 64):
                batch = torch.tensor(patches[start : start + 64], dtype=torch.float32, device=device)
                scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())

        probabilities = np.concatenate(scores) if scores else np.empty((0,), dtype="float32")
        pred = meta.copy()
        pred["cnn_probability"] = probabilities
        pred["cnn_model"] = selected.model_name
        pred["cnn_architecture"] = architecture
        pred["cnn_model_path"] = str(selected.model_path)
        pred["cnn_selection_rule"] = str(metadata.get("selection_rule", ""))

        pred_path = PRED_DIR / f"{catalog_key}_external_{selected.model_name}_predictions.csv"
        pred.to_csv(pred_path, index=False)

        for mode, threshold in thresholds.items():
            out = pred.copy()
            out["threshold_mode"] = mode
            out["cnn_threshold"] = threshold
            out["cnn_keep"] = out["cnn_probability"] >= threshold
            filtered = out[out["cnn_keep"]].copy()
            filtered_path = FILTERED_DIR / f"{catalog_key}_external_{selected.model_name}_{mode}.csv"
            filtered.to_csv(filtered_path, index=False)
            summary_rows.append(
                {
                    "base_catalog": catalog_key,
                    "cnn_model": selected.model_name,
                    "architecture": architecture,
                    "threshold_mode": mode,
                    "threshold": threshold,
                    "n_input": len(out),
                    "n_accepted": len(filtered),
                    "prediction_path": str(pred_path),
                    "filtered_catalog_path": str(filtered_path),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary_path = PRED_DIR / "external_cnn_prediction_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))
    print(f"Default CNN model: {selected.model_name}")
    print(f"Thresholds: {thresholds}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
