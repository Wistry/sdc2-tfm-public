#!/usr/bin/env python3
"""Apply the trained small CNN to the conservative SDC2-like catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier")
PATCH_DIR = BASE / "outputs" / "patches"
MODEL_DIR = BASE / "outputs" / "models"
FILTERED_DIR = BASE / "outputs" / "filtered_catalogs"
REPORT_DIR = BASE / "outputs" / "reports"


def write_skip(reason: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "SKIPPED", "reason": reason}]).to_csv(
        REPORT_DIR / "cnn_conservative_filter_summary.csv",
        index=False,
    )
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

    model_path = MODEL_DIR / "small_cnn.pt"
    metadata_path = MODEL_DIR / "small_cnn_metadata.json"
    if not model_path.exists() or not metadata_path.exists():
        write_skip("small_cnn.pt or metadata missing; train the CNN first.")
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

    patches = np.load(PATCH_DIR / "sdc2_conservative_patches.npy")
    meta = pd.read_csv(PATCH_DIR / "sdc2_conservative_metadata.csv")
    scores = []
    with torch.no_grad():
        for start in range(0, len(patches), 64):
            batch = torch.tensor(patches[start : start + 64], dtype=torch.float32, device=device)
            scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
    meta["cnn_score"] = np.concatenate(scores) if scores else []

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    thresholds = {row["threshold_mode"]: row["threshold"] for row in metadata.get("thresholds", [])}
    rows = []
    FILTERED_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ["f0_5", "f1", "conservative_fp"]:
        threshold = float(thresholds.get(mode, 0.5))
        out = meta[meta["cnn_score"] >= threshold].copy()
        out["cnn_threshold_mode"] = mode
        out["cnn_threshold"] = threshold
        out.to_csv(FILTERED_DIR / f"CNN_{mode}.csv", index=False)
        rows.append({"threshold_mode": mode, "threshold": threshold, "accepted": len(out), "input": len(meta)})

    summary = pd.DataFrame(rows)
    summary.to_csv(REPORT_DIR / "cnn_conservative_filter_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
