#!/usr/bin/env python3
"""Apply the grouped CNN to the conservative catalogue patches."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("phase_02_spectral_features/10_grouped_validation")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"
FILTERED = BASE / "outputs" / "filtered_catalogs"
REPORTS = BASE / "outputs" / "reports"


def write_skip(reason: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    print(f"Skipped: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    del args

    try:
        import torch
        from torch import nn
    except Exception as exc:  # noqa: BLE001
        write_skip(f"torch_not_available: {exc}")
        return

    model_path = MODELS / "small_cnn_grouped.pt"
    threshold_path = TABLES / "grouped_cnn_thresholds.csv"
    if not model_path.exists() or not threshold_path.exists():
        write_skip("grouped CNN model or thresholds missing.")
        return

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = SmallCNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    patches = np.load(PATCH_BASE / "sdc2_conservative_patches.npy").astype("float32")
    meta = pd.read_csv(PATCH_BASE / "sdc2_conservative_metadata.csv")
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            batch = torch.tensor(patches[start : start + 64], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    meta["grouped_cnn_score"] = np.concatenate(scores) if scores else []
    thresholds = pd.read_csv(threshold_path)
    threshold_map = dict(zip(thresholds["threshold_mode"], thresholds["threshold"]))

    FILTERED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ["f0_5", "f1", "f2", "conservative_fp"]:
        threshold = float(threshold_map.get(mode, 0.5))
        out = meta[meta["grouped_cnn_score"] >= threshold].copy()
        out["grouped_cnn_threshold_mode"] = mode
        out["grouped_cnn_threshold"] = threshold
        out_path = FILTERED / f"CNN_grouped_{mode}.csv"
        out.to_csv(out_path, index=False)
        rows.append({"threshold_mode": mode, "threshold": threshold, "accepted": len(out), "input": len(meta), "path": out_path.as_posix()})
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "grouped_cnn_conservative_filter_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
