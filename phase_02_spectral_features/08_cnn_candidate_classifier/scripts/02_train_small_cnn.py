#!/usr/bin/env python3
"""Train a minimal 2.5D CNN on candidate patches when PyTorch is available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier")
PATCH_DIR = BASE / "outputs" / "patches"
MODEL_DIR = BASE / "outputs" / "models"
REPORT_DIR = BASE / "outputs" / "reports"
PHASE2_PRINCIPAL = {"score": 18.524923, "matches": 137, "false": 51}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float, name: str) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    row: dict[str, Any] = {
        "threshold_mode": name,
        "threshold": float(threshold),
        "average_precision": average_precision_score(y_true, scores),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "f0_5": fbeta_score(y_true, pred, beta=0.5, zero_division=0),
        "f1": fbeta_score(y_true, pred, beta=1.0, zero_division=0),
        "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    row.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return row


def find_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    grid = np.linspace(0.01, 0.99, 99)
    scored = []
    for t in grid:
        pred = (scores >= t).astype(int)
        scored.append(
            {
                "threshold": float(t),
                "f0_5": fbeta_score(y_true, pred, beta=0.5, zero_division=0),
                "f1": fbeta_score(y_true, pred, beta=1.0, zero_division=0),
                "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "fp": int(((pred == 1) & (y_true == 0)).sum()),
            }
        )
    sweep = pd.DataFrame(scored)
    for mode in ["f0_5", "f1", "f2"]:
        best = sweep.sort_values([mode, "threshold"], ascending=[False, False]).iloc[0]
        rows.append(metric_row(y_true, scores, float(best["threshold"]), mode))
    viable = sweep[sweep["recall"] >= 0.7].copy()
    if viable.empty:
        best = sweep.sort_values(["fp", "recall", "threshold"], ascending=[True, False, False]).iloc[0]
    else:
        best = viable.sort_values(["fp", "precision", "threshold"], ascending=[True, False, False]).iloc[0]
    rows.append(metric_row(y_true, scores, float(best["threshold"]), "conservative_fp"))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    random_state = int(cfg.get("random_state", 42))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # noqa: BLE001
        message = f"PyTorch is not available. Training skipped. Import error: `{exc}`"
        pd.DataFrame([{"status": "SKIPPED", "reason": "torch_not_available"}]).to_csv(
            REPORT_DIR / "cnn_training_results.csv",
            index=False,
        )
        pd.DataFrame().to_csv(REPORT_DIR / "cnn_thresholds.csv", index=False)
        print(message)
        return

    patches = np.load(PATCH_DIR / "baseline_patches.npy")
    labels = np.load(PATCH_DIR / "baseline_labels.npy").astype("int64")
    x_train, x_test, y_train, y_test = train_test_split(
        patches,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=random_state,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = SmallCNN().to(device)
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_ap = -np.inf
    best_state = None
    patience = 5
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(test_tensor)).detach().cpu().numpy()
        ap = average_precision_score(y_test, scores)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "test_average_precision": float(ap)})
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(test_tensor)).detach().cpu().numpy()
    thresholds = find_thresholds(y_test, scores)
    thresholds.to_csv(REPORT_DIR / "cnn_thresholds.csv", index=False)
    pd.DataFrame(history).to_csv(REPORT_DIR / "cnn_training_results.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "patch_shape": [21, 32, 32]}, MODEL_DIR / "small_cnn.pt")
    metadata = {
        "random_state": random_state,
        "device": str(device),
        "epochs_run": len(history),
        "phase2_principal_reference": PHASE2_PRINCIPAL,
        "thresholds": thresholds.to_dict(orient="records"),
    }
    (MODEL_DIR / "small_cnn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
