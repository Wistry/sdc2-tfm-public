#!/usr/bin/env python3
"""Final CNN evaluation with StratifiedGroupKFold."""

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
except Exception as exc:  # pragma: no cover
    raise RuntimeError("StratifiedGroupKFold is required for final grouped CNN evaluation. Upgrade scikit-learn.") from exc


BASE = Path("phase_02_spectral_features/12_final_grouped_evaluation")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"
MODELS = BASE / "outputs" / "models"
RANDOM_STATE = 42
N_SPLITS = 5


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def valid_truth_id(value: Any) -> bool:
    return pd.notna(value) and str(value).strip() not in {"", "nan", "None", "-1", "-1.0"}


def split_group(row: pd.Series) -> str:
    if int(row["clean_label"]) == 1 and valid_truth_id(row.get("matched_truth_id")):
        return f"truth_{row['matched_truth_id']}"
    return f"candidate_{row.get('candidate_index', row.name)}"


def channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True).astype("float32")
    std = x_train.std(axis=(0, 2, 3), keepdims=True).astype("float32")
    std[std < 1e-6] = 1.0
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype("float32")


def score_row(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else np.nan,
        "f0_5": float(fbeta_score(y_true, pred, beta=0.5, zero_division=0)),
        "f1": float(fbeta_score(y_true, pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2.0, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def derive_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    sweep = []
    for threshold in np.linspace(0.01, 0.99, 99):
        row = score_row(y_true, scores, float(threshold))
        row["threshold"] = float(threshold)
        sweep.append(row)
    sweep_df = pd.DataFrame(sweep)
    rows = []
    for mode in ["f0_5", "f1", "f2"]:
        best = sweep_df.sort_values([mode, "threshold"], ascending=[False, False]).iloc[0].to_dict()
        best["threshold_mode"] = mode
        rows.append(best)
    viable = sweep_df[sweep_df["recall"] >= 0.7].copy()
    if viable.empty:
        best = sweep_df.sort_values(["fp", "recall", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    else:
        best = viable.sort_values(["fp", "precision", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    best["threshold_mode"] = "conservative_fp"
    rows.append(best)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    load_yaml(args.config)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch is required for final grouped CNN evaluation: {exc}") from exc

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

    patches_all = np.load(PATCH_BASE / "baseline_patches.npy").astype("float32")
    labels_all = np.load(PATCH_BASE / "baseline_labels.npy").astype("int64")
    meta_all = pd.read_csv(PATCH_BASE / "baseline_metadata.csv")
    clean_mask = meta_all["clean_label"].isin([0, 1]).to_numpy()
    patches = patches_all[clean_mask]
    labels = labels_all[clean_mask].astype("int64")
    meta = meta_all.loc[clean_mask].reset_index(drop=True).copy()
    meta["split_group"] = meta.apply(split_group, axis=1)
    groups = meta["split_group"].astype(str).to_numpy()
    group_sizes = meta["split_group"].value_counts()
    duplicated_groups = group_sizes[group_sizes > 1]
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    thresholds_all = []
    oof_scores = np.full(len(labels), np.nan, dtype="float64")
    best_key: tuple[float, float, float] | None = None
    best_payload: dict[str, Any] | None = None
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros_like(labels), labels, groups), start=1):
        shared = set(groups[train_idx]).intersection(set(groups[test_idx]))
        if shared:
            raise RuntimeError(f"CNN fold {fold} leaked {len(shared)} groups.")
        mean, std = channel_stats(patches[train_idx])
        x_train = normalize(patches[train_idx], mean, std)
        x_test = normalize(patches[test_idx], mean, std)
        y_train, y_test = labels[train_idx], labels[test_idx]
        torch.manual_seed(RANDOM_STATE + fold)
        np.random.seed(RANDOM_STATE + fold)
        generator = torch.Generator()
        generator.manual_seed(RANDOM_STATE + fold)
        train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
        model = FinalGroupedCNN().to(device)
        pos = max(1, int((y_train == 1).sum()))
        neg = max(1, int((y_train == 0).sum()))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
        best_ap = -np.inf
        best_state = None
        stale = 0
        epochs_run = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                scores_epoch = torch.sigmoid(model(test_tensor)).detach().cpu().numpy()
            ap = average_precision_score(y_test, scores_epoch)
            epochs_run = epoch
            if ap > best_ap:
                best_ap = ap
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            scores = torch.sigmoid(model(test_tensor)).detach().cpu().numpy()
        oof_scores[test_idx] = scores
        thresholds = derive_thresholds(y_test, scores)
        thresholds.insert(0, "fold", fold)
        thresholds_all.append(thresholds)
        f05_row = thresholds[thresholds["threshold_mode"] == "f0_5"].iloc[0].to_dict()
        row = {
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_groups": int(len(set(groups[train_idx]))),
            "test_groups": int(len(set(groups[test_idx]))),
            "shared_groups": int(len(shared)),
            "epochs_run": int(epochs_run),
            **{k: f05_row[k] for k in ["average_precision", "roc_auc", "f0_5", "f1", "f2", "precision", "recall", "balanced_accuracy", "tn", "fp", "fn", "tp"]},
            "threshold_f0_5": float(thresholds[thresholds["threshold_mode"] == "f0_5"]["threshold"].iloc[0]),
            "threshold_f1": float(thresholds[thresholds["threshold_mode"] == "f1"]["threshold"].iloc[0]),
            "threshold_f2": float(thresholds[thresholds["threshold_mode"] == "f2"]["threshold"].iloc[0]),
            "threshold_conservative_fp": float(thresholds[thresholds["threshold_mode"] == "conservative_fp"]["threshold"].iloc[0]),
        }
        rows.append(row)
        key = (float(row["f0_5"]), float(row["average_precision"]), float(row["precision"]))
        if best_key is None or key > best_key:
            best_key = key
            best_payload = {"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "fold": fold, "mean": mean, "std": std, "row": row}
        print(f"fold={fold} AP={row['average_precision']:.6f} F0.5={row['f0_5']:.6f} shared_groups=0")

    by_fold = pd.DataFrame(rows)
    thresholds_by_fold = pd.concat(thresholds_all, ignore_index=True)
    oof_thresholds = derive_thresholds(labels, oof_scores)
    by_fold.to_csv(TABLES / "final_grouped_cnn_results_by_fold.csv", index=False)
    by_fold.mean(numeric_only=True).to_frame().T.to_csv(TABLES / "final_grouped_cnn_summary.csv", index=False)
    thresholds_by_fold.to_csv(TABLES / "final_grouped_cnn_thresholds_by_fold.csv", index=False)
    oof_thresholds.to_csv(TABLES / "final_grouped_cnn_thresholds.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No CNN checkpoint was selected.")
    torch.save(
        {
            "model_state_dict": best_payload["state"],
            "architecture": "final_grouped_cnn_2_5d",
            "patch_shape": [21, 32, 32],
            "channel_mean": best_payload["mean"].tolist(),
            "channel_std": best_payload["std"].tolist(),
            "selected_fold": int(best_payload["fold"]),
        },
        MODELS / "final_grouped_cnn.pt",
    )
    metadata = {
        "selection_rule": "best fold by grouped F0.5, then average_precision and precision",
        "selected_fold": int(best_payload["fold"]),
        "selected_fold_metrics": best_payload["row"],
        "examples": int(len(labels)),
        "class_distribution": {str(k): int(v) for k, v in pd.Series(labels).value_counts().to_dict().items()},
        "groups": int(group_sizes.size),
        "duplicated_groups": int(len(duplicated_groups)),
        "examples_in_duplicated_groups": int(duplicated_groups.sum()) if not duplicated_groups.empty else 0,
        "max_group_size": int(duplicated_groups.max()) if not duplicated_groups.empty else 1,
        "n_splits": N_SPLITS,
        "max_shared_groups": int(by_fold["shared_groups"].max()),
        "device": str(device),
        "thresholds_oof": oof_thresholds.to_dict(orient="records"),
    }
    (MODELS / "final_grouped_cnn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(by_fold.to_string(index=False))


if __name__ == "__main__":
    main()
