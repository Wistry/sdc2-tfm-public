#!/usr/bin/env python3
"""Train robust CNN variants with grouped validation by matched_truth_id."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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
from sklearn.model_selection import GroupShuffleSplit


BASE = Path("phase_02_spectral_features/11_robust_grouped_cnn")
PATCH_BASE = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches")
MODELS = BASE / "outputs" / "models"
TABLES = BASE / "outputs" / "tables"
REPORTS = BASE / "outputs" / "reports"

SEEDS = [42, 7, 13, 21, 123]
LEARNING_RATES = [1e-3, 3e-4]

REFERENCES = {
    "phase1_best": {"score": 17.524923, "matches": 137, "false": 52},
    "phase2_principal": {"score": 18.524923, "matches": 137, "false": 51},
    "cnn_original": {"score": 28.333489815821608, "matches": 129, "false": 37, "average_precision": 0.920106, "roc_auc": 0.786875},
    "cnn_grouped_previous": {"score": 18.44173292591026, "matches": 133, "false": 49, "average_precision": 0.941195, "roc_auc": 0.831349},
    "official_like": {"score": 25.916593, "matches": 134, "false": 42},
    "greedy_pruning": {"score": 38.524923, "matches": 137, "false": 31},
}


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def valid_truth_id(value: Any) -> bool:
    return pd.notna(value) and str(value).strip() not in {"", "nan", "None", "-1", "-1.0"}


def truth_group(row: pd.Series) -> str:
    label = int(row["clean_label"])
    truth = row.get("matched_truth_id")
    if label == 1 and valid_truth_id(truth):
        return f"truth_{truth}"
    return f"candidate_{row.get('candidate_index', row.name)}"


def grouped_split(y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, str]:
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
        return train_idx, test_idx, "StratifiedGroupKFold"
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups))
    return train_idx, test_idx, "GroupShuffleSplit"


def channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=(0, 2, 3), keepdims=True).astype("float32")
    std = x_train.std(axis=(0, 2, 3), keepdims=True).astype("float32")
    std[std < 1e-6] = 1.0
    return mean, std


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype("float32")


def metric_row(y_true: np.ndarray, scores: np.ndarray, threshold: float, mode: str) -> dict[str, Any]:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold_mode": mode,
        "threshold": float(threshold),
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


@dataclass
class TrainResult:
    architecture: str
    seed: int
    learning_rate: float
    splitter: str
    train_examples: int
    test_examples: int
    train_groups: int
    test_groups: int
    duplicated_groups: int
    examples_in_duplicated_groups: int
    shared_groups: int
    epochs_run: int
    average_precision: float
    roc_auc: float
    f0_5: float
    f1: float
    f2: float
    precision: float
    recall: float
    balanced_accuracy: float
    tn: int
    fp: int
    fn: int
    tp: int
    threshold_f0_5: float
    threshold_f1: float
    threshold_f2: float
    threshold_conservative_fp: float


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--include-3d", action="store_true", help="Also train the optional lightweight 3D CNN.")
    args = parser.parse_args()
    cfg = load_yaml(args.config)
    del cfg

    MODELS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # noqa: BLE001
        reason = f"torch_not_available: {exc}"
        pd.DataFrame([{"status": "SKIPPED", "reason": reason}]).to_csv(TABLES / "robust_grouped_cnn_internal_results.csv", index=False)
        print(reason)
        return

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
            self.classifier = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
            )

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

    architectures: dict[str, type[nn.Module]] = {
        "baseline_grouped": BaselineCNN,
        "robust_2_5d": RobustCNN25D,
    }
    if args.include_3d:
        architectures["lightweight_3d"] = Lightweight3DCNN

    patches_all = np.load(PATCH_BASE / "baseline_patches.npy").astype("float32")
    labels_all = np.load(PATCH_BASE / "baseline_labels.npy").astype("int64")
    meta_all = pd.read_csv(PATCH_BASE / "baseline_metadata.csv")
    clean_mask = meta_all["clean_label"].isin([0, 1]).to_numpy()
    patches = patches_all[clean_mask]
    labels = labels_all[clean_mask].astype("int64")
    meta = meta_all.loc[clean_mask].reset_index(drop=True).copy()
    meta["group_id"] = meta.apply(truth_group, axis=1)
    groups = meta["group_id"].astype(str).to_numpy()
    group_sizes = pd.Series(groups).value_counts()
    duplicated_groups = group_sizes[group_sizes > 1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, Any]] = []
    threshold_frames: list[pd.DataFrame] = []
    best_key: tuple[float, float, float, float] | None = None
    best_payload: dict[str, Any] | None = None

    for architecture, model_cls in architectures.items():
        for seed in SEEDS:
            train_idx, test_idx, split_name = grouped_split(labels, groups, seed)
            shared = set(groups[train_idx]).intersection(set(groups[test_idx]))
            if shared:
                raise RuntimeError(f"Grouped split leaked {len(shared)} groups for seed {seed}: {sorted(shared)[:5]}")
            x_train_raw, x_test_raw = patches[train_idx], patches[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            mean, std = channel_stats(x_train_raw)
            x_train = normalize(x_train_raw, mean, std)
            x_test = normalize(x_test_raw, mean, std)
            for lr in LEARNING_RATES:
                torch.manual_seed(seed)
                np.random.seed(seed)
                train_ds = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
                generator = torch.Generator()
                generator.manual_seed(seed)
                loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator)
                model = model_cls().to(device)
                pos = max(1, int((y_train == 1).sum()))
                neg = max(1, int((y_train == 0).sum()))
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
                opt = torch.optim.Adam(model.parameters(), lr=lr)
                test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
                best_ap = -np.inf
                best_state = None
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
                    history.append({"epoch": epoch, "loss": float(np.mean(losses)), "average_precision": float(ap)})
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
                thresholds = find_thresholds(y_test, scores)
                threshold_wide = thresholds.set_index("threshold_mode")["threshold"].to_dict()
                selected = thresholds[thresholds["threshold_mode"] == "f0_5"].iloc[0].to_dict()
                row = TrainResult(
                    architecture=architecture,
                    seed=seed,
                    learning_rate=lr,
                    splitter=split_name,
                    train_examples=int(len(train_idx)),
                    test_examples=int(len(test_idx)),
                    train_groups=int(len(set(groups[train_idx]))),
                    test_groups=int(len(set(groups[test_idx]))),
                    duplicated_groups=int(len(duplicated_groups)),
                    examples_in_duplicated_groups=int(duplicated_groups.sum()) if not duplicated_groups.empty else 0,
                    shared_groups=0,
                    epochs_run=len(history),
                    average_precision=float(selected["average_precision"]),
                    roc_auc=float(selected["roc_auc"]),
                    f0_5=float(selected["f0_5"]),
                    f1=float(selected["f1"]),
                    f2=float(selected["f2"]),
                    precision=float(selected["precision"]),
                    recall=float(selected["recall"]),
                    balanced_accuracy=float(selected["balanced_accuracy"]),
                    tn=int(selected["tn"]),
                    fp=int(selected["fp"]),
                    fn=int(selected["fn"]),
                    tp=int(selected["tp"]),
                    threshold_f0_5=float(threshold_wide["f0_5"]),
                    threshold_f1=float(threshold_wide["f1"]),
                    threshold_f2=float(threshold_wide["f2"]),
                    threshold_conservative_fp=float(threshold_wide["conservative_fp"]),
                ).__dict__
                results.append(row)
                thresholds.insert(0, "architecture", architecture)
                thresholds.insert(1, "seed", seed)
                thresholds.insert(2, "learning_rate", lr)
                threshold_frames.append(thresholds)
                key = (row["f0_5"], row["average_precision"], row["precision"], row["recall"])
                if best_key is None or key > best_key:
                    best_key = key
                    best_payload = {
                        "architecture": architecture,
                        "seed": seed,
                        "learning_rate": lr,
                        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "mean": mean,
                        "std": std,
                        "thresholds": thresholds,
                        "row": row,
                        "history": history,
                        "model_class_name": model_cls.__name__,
                    }
                print(
                    f"{architecture} seed={seed} lr={lr:g} "
                    f"AP={row['average_precision']:.6f} f0.5={row['f0_5']:.6f} "
                    f"precision={row['precision']:.3f} recall={row['recall']:.3f}"
                )

    results_df = pd.DataFrame(results).sort_values(["f0_5", "average_precision", "precision"], ascending=[False, False, False])
    thresholds_df = pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame()
    results_df.to_csv(TABLES / "robust_grouped_cnn_internal_results.csv", index=False)
    thresholds_df.to_csv(TABLES / "robust_grouped_cnn_thresholds.csv", index=False)
    if best_payload is None:
        raise RuntimeError("No CNN model was trained.")

    torch.save(
        {
            "model_state_dict": best_payload["model_state_dict"],
            "architecture": best_payload["architecture"],
            "model_class_name": best_payload["model_class_name"],
            "patch_shape": [21, 32, 32],
            "channel_mean": best_payload["mean"].tolist(),
            "channel_std": best_payload["std"].tolist(),
        },
        MODELS / "best_robust_grouped_cnn.pt",
    )
    best_thresholds = best_payload["thresholds"].copy()
    best_thresholds.to_csv(TABLES / "best_robust_grouped_cnn_thresholds.csv", index=False)
    metadata = {
        "selection_rule": "max f0_5, then average_precision, precision, recall on grouped validation",
        "best": {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v) for k, v in best_payload["row"].items()},
        "references": REFERENCES,
        "examples": int(len(labels)),
        "class_distribution": {str(k): int(v) for k, v in pd.Series(labels).value_counts().to_dict().items()},
        "groups": int(group_sizes.size),
        "duplicated_groups": int(len(duplicated_groups)),
        "examples_in_duplicated_groups": int(duplicated_groups.sum()) if not duplicated_groups.empty else 0,
        "max_group_size": int(duplicated_groups.max()) if not duplicated_groups.empty else 1,
        "architectures_trained": list(architectures.keys()),
        "seeds": SEEDS,
        "learning_rates": LEARNING_RATES,
        "device": str(device),
    }
    (MODELS / "best_robust_grouped_cnn_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    best = results_df.iloc[0]
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
