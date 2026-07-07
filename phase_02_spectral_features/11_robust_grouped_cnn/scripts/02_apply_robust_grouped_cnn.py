#!/usr/bin/env python3
"""Apply the selected robust grouped CNN to the conservative catalogue."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("phase_02_spectral_features/11_robust_grouped_cnn")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"
FILTERED = BASE / "outputs" / "filtered_catalogs"
REPORTS = BASE / "outputs" / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args

    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch_not_available: {exc}") from exc

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
    }
    model_path = MODELS / "best_robust_grouped_cnn.pt"
    thresholds_path = TABLES / "best_robust_grouped_cnn_thresholds.csv"
    if not model_path.exists() or not thresholds_path.exists():
        raise FileNotFoundError("Run 01_train_robust_grouped_cnn.py before applying the model.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    architecture = checkpoint["architecture"]
    model = model_classes[architecture]().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
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
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    meta["robust_grouped_cnn_score"] = np.concatenate(scores) if scores else []
    meta["robust_grouped_cnn_architecture"] = architecture

    thresholds = pd.read_csv(thresholds_path)
    threshold_map = dict(zip(thresholds["threshold_mode"], thresholds["threshold"]))
    FILTERED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ["f0_5", "f1", "f2", "conservative_fp"]:
        threshold = float(threshold_map.get(mode, 0.5))
        out = meta[meta["robust_grouped_cnn_score"] >= threshold].copy()
        out["robust_grouped_cnn_threshold_mode"] = mode
        out["robust_grouped_cnn_threshold"] = threshold
        out_path = FILTERED / f"ROBUST_CNN_grouped_{mode}.csv"
        out.to_csv(out_path, index=False)
        rows.append({"threshold_mode": mode, "threshold": threshold, "accepted": len(out), "input": len(meta), "path": out_path.as_posix()})
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "robust_grouped_cnn_filter_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
