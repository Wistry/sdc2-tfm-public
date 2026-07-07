#!/usr/bin/env python3
"""Train the small CNN with a grouped train/test split."""

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

try:
    from sklearn.model_selection import StratifiedGroupKFold
except Exception:  # pragma: no cover
    StratifiedGroupKFold = None  # type: ignore[assignment]
from sklearn.model_selection import GroupKFold


BASE = Path("phase_02_spectral_features/10_grouped_validation")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"
ORIGINAL_CNN = {
    "average_precision": 0.920106,
    "roc_auc": 0.786875,
    "official_score": 28.333489815821608,
    "matches": 129,
    "false": 37,
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def truth_group(row: pd.Series) -> str:
    label = int(row["clean_label"])
    truth = row.get("matched_truth_id")
    if label == 1 and pd.notna(truth) and str(truth).strip() not in {"", "nan", "-1"}:
        return f"truth_{truth}"
    return f"candidate_{row.get('candidate_index', row.name)}"


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float, name: str) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
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
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def find_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    sweep_rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (scores >= threshold).astype(int)
        sweep_rows.append(
            {
                "threshold": float(threshold),
                "f0_5": fbeta_score(y_true, pred, beta=0.5, zero_division=0),
                "f1": fbeta_score(y_true, pred, beta=1.0, zero_division=0),
                "f2": fbeta_score(y_true, pred, beta=2.0, zero_division=0),
                "precision": precision_score(y_true, pred, zero_division=0),
                "recall": recall_score(y_true, pred, zero_division=0),
                "fp": int(((pred == 1) & (y_true == 0)).sum()),
            }
        )
    sweep = pd.DataFrame(sweep_rows)
    rows = []
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


def grouped_split(y: np.ndarray, groups: np.ndarray, random_state: int):
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
        split_name = "StratifiedGroupKFold"
    else:
        splitter = GroupKFold(n_splits=5)
        split_name = "GroupKFold"
    train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
    return train_idx, test_idx, split_name


def write_skip(reason: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"status": "SKIPPED", "reason": reason}]).to_csv(TABLES / "grouped_cnn_training_results.csv", index=False)
    pd.DataFrame().to_csv(TABLES / "grouped_cnn_thresholds.csv", index=False)
    print(f"Skipped: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    random_state = int(cfg.get("random_state", 42))
    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # noqa: BLE001
        write_skip(f"torch_not_available: {exc}")
        return

    patches = np.load(PATCH_BASE / "baseline_patches.npy").astype("float32")
    labels = np.load(PATCH_BASE / "baseline_labels.npy").astype("int64")
    meta = pd.read_csv(PATCH_BASE / "baseline_metadata.csv")
    meta["group_id"] = meta.apply(truth_group, axis=1)
    groups = meta["group_id"].astype(str).to_numpy()
    train_idx, test_idx, split_name = grouped_split(labels, groups, random_state)
    shared = set(groups[train_idx]).intersection(set(groups[test_idx]))

    torch.manual_seed(random_state)
    np.random.seed(random_state)
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

    x_train, x_test = patches[train_idx], patches[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    model = SmallCNN().to(device)
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)

    best_ap = -np.inf
    best_state = None
    stale = 0
    patience = 5
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
    pd.DataFrame(history).to_csv(TABLES / "grouped_cnn_training_results.csv", index=False)
    thresholds.to_csv(TABLES / "grouped_cnn_thresholds.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "patch_shape": [21, 32, 32]}, MODELS / "small_cnn_grouped.pt")
    metadata = {
        "random_state": random_state,
        "device": str(device),
        "splitter": split_name,
        "train_examples": int(len(train_idx)),
        "test_examples": int(len(test_idx)),
        "train_class_counts": {str(k): int(v) for k, v in pd.Series(y_train).value_counts().to_dict().items()},
        "test_class_counts": {str(k): int(v) for k, v in pd.Series(y_test).value_counts().to_dict().items()},
        "train_groups": int(len(set(groups[train_idx]))),
        "test_groups": int(len(set(groups[test_idx]))),
        "shared_groups": int(len(shared)),
        "epochs_run": len(history),
        "thresholds": thresholds.to_dict(orient="records"),
        "original_cnn_reference": ORIGINAL_CNN,
    }
    (MODELS / "small_cnn_grouped_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(thresholds.to_string(index=False))


if __name__ == "__main__":
    main()
