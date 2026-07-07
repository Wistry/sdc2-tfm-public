#!/usr/bin/env python3
"""Controlled comparison between Phase 2 CNN pipelines 15 and 16."""

from __future__ import annotations

import argparse
import json
import traceback
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
from sklearn.model_selection import StratifiedGroupKFold


BASE = Path("phase_02_spectral_features/13_cnn_pipeline_comparison")
OUT_REPORTS = BASE / "outputs" / "reports"
OUT_FILTERED = BASE / "outputs" / "filtered_catalogs"
OUT_SCORES = BASE / "outputs" / "official_scores"
OUT_MODELS = BASE / "outputs" / "models"

PATCH_CANDIDATES = [
    Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches"),
]
CNN15_MODEL = Path("phase_02_spectral_features/11_robust_grouped_cnn/outputs/models/best_robust_grouped_cnn.pt")
CNN15_META = Path("phase_02_spectral_features/11_robust_grouped_cnn/outputs/models/best_robust_grouped_cnn_metadata.json")
CNN15_THRESHOLDS = Path("phase_02_spectral_features/11_robust_grouped_cnn/outputs/tables/best_robust_grouped_cnn_thresholds.csv")
CNN16_MODEL = Path("phase_02_spectral_features/12_final_grouped_evaluation/outputs/models/final_grouped_cnn.pt")
CNN16_META = Path("phase_02_spectral_features/12_final_grouped_evaluation/outputs/models/final_grouped_cnn_metadata.json")
CNN16_THRESHOLDS = Path("phase_02_spectral_features/12_final_grouped_evaluation/outputs/tables/final_grouped_cnn_thresholds.csv")
CNN15_OFFICIAL = Path("phase_02_spectral_features/11_robust_grouped_cnn/outputs/official_scores/robust_grouped_cnn_official_scores.csv")
CNN16_OFFICIAL = Path("phase_02_spectral_features/12_final_grouped_evaluation/outputs/official_scores/final_grouped_official_scores.csv")

TRUTH = Path("data/sky_dev_truthcat_v2.txt")
COMMON_THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
SUBMISSION_COLUMNS = ["id", "ra", "dec", "hi_size", "line_flux_integral", "central_freq", "pa", "i", "w20"]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def patch_base() -> Path:
    for path in PATCH_CANDIDATES:
        if (path / "baseline_patches.npy").exists() and (path / "sdc2_conservative_patches.npy").exists():
            return path
    raise FileNotFoundError(f"No usable CNN patch directory found in: {PATCH_CANDIDATES}")


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


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


def derive_thresholds(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        row = metric_row(y_true, scores, float(threshold), "sweep")
        rows.append(row)
    sweep = pd.DataFrame(rows)
    selected = []
    for mode in ["f0_5", "f1", "f2"]:
        best = sweep.sort_values([mode, "threshold"], ascending=[False, False]).iloc[0].to_dict()
        best["threshold_mode"] = mode
        selected.append(best)
    viable = sweep[sweep["recall"] >= 0.7].copy()
    if viable.empty:
        best = sweep.sort_values(["fp", "recall", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    else:
        best = viable.sort_values(["fp", "precision", "threshold"], ascending=[True, False, False]).iloc[0].to_dict()
    best["threshold_mode"] = "conservative_fp"
    selected.append(best)
    return pd.DataFrame(selected)


def numeric(df: pd.DataFrame, names: list[str], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series([default] * len(df), dtype="float64")


def convert_to_submission(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ra"] = numeric(df, ["ra"])
    out["dec"] = numeric(df, ["dec"])
    out["central_freq"] = numeric(df, ["freq", "central_freq"])
    out["line_flux_integral"] = numeric(df, ["f_sum", "line_flux_integral"])
    out["hi_size"] = numeric(df, ["ell_maj", "hi_size"]) * (4.0 if "ell_maj" in df.columns else 1.0)
    out["pa"] = numeric(df, ["kin_pa", "pa"])
    out["i"] = numeric(df, ["ell_min", "i"])
    out["w20"] = numeric(df, ["w20"])
    out = out.dropna(subset=["ra", "dec", "central_freq"]).fillna(0.0).copy()
    out.insert(0, "id", range(1, len(out) + 1))
    return out[SUBMISSION_COLUMNS]


def import_scorer():
    try:
        from ska_sdc.sdc2.sdc2_scorer import Sdc2Scorer
    except Exception:
        return None
    return Sdc2Scorer


def scalar_attrs(obj: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr in dir(obj):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(obj, attr)
        except Exception:
            continue
        if not callable(value) and isinstance(value, (str, int, float, bool, type(None))):
            attrs[attr] = value
    return attrs


def score_value_from(score: Any) -> float | None:
    if isinstance(score, (int, float)):
        return float(score)
    for attr in ["value", "score", "score_value"]:
        if hasattr(score, attr):
            value = getattr(score, attr)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def score_catalog(catalog_name: str, df: pd.DataFrame, scorer: Any, truth: pd.DataFrame | None) -> dict[str, Any]:
    submission = convert_to_submission(df)
    submission_path = OUT_SCORES / f"{catalog_name}_submission.csv"
    submission.to_csv(submission_path, index=False)
    row: dict[str, Any] = {
        "catalog_name": catalog_name,
        "status": "NOT_RUN",
        "n_input": int(len(df)),
        "n_submission": int(len(submission)),
        "score": None,
        "matches": None,
        "false": None,
        "error": "",
        "submission_path": str(submission_path),
    }
    if scorer is None:
        row["status"] = "SKIPPED"
        row["error"] = "Sdc2Scorer not importable"
        return row
    if truth is None:
        row["status"] = "SKIPPED"
        row["error"] = f"Truth file missing: {TRUTH}"
        return row
    try:
        score = scorer(submission, truth).run()
        attrs = scalar_attrs(score)
        row.update(
            {
                "status": "OK",
                "score": score_value_from(score),
                "matches": attrs.get("n_match"),
                "false": attrs.get("n_false"),
                "score_attrs_json": json.dumps(attrs, sort_keys=True),
            }
        )
    except Exception as exc:  # noqa: BLE001
        row["status"] = "ERROR"
        row["error"] = f"{exc}\n{traceback.format_exc()}"
    return row


def architecture_summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "pipeline": "11_robust_grouped_cnn",
                "class_name": "BaselineCNN",
                "architecture": "2.5D Conv2d over 21 spectral channels",
                "conv_layers": 2,
                "batch_norm": False,
                "pooling": "MaxPool2d x2",
                "head": "Flatten -> Linear(2048,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)",
                "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "seed": 123,
                "selection": "max grouped F0.5 among architectures/seeds/LR",
            },
            {
                "pipeline": "12_final_grouped_evaluation",
                "class_name": "FinalGroupedCNN",
                "architecture": "2.5D Conv2d over 21 spectral channels",
                "conv_layers": 3,
                "batch_norm": True,
                "pooling": "MaxPool2d x2 + AdaptiveAvgPool2d(1,1)",
                "head": "Linear(128,64) -> ReLU -> Dropout(0.3) -> Linear(64,1)",
                "loss": "BCEWithLogitsLoss(pos_weight=neg/pos)",
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "seed": "42 + fold",
                "selection": "best fold by grouped F0.5; thresholds from OOF scores",
            },
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    load_yaml(args.config)

    for path in [OUT_REPORTS, OUT_FILTERED, OUT_SCORES, OUT_MODELS]:
        path.mkdir(parents=True, exist_ok=True)
    for path in [CNN15_MODEL, CNN15_META, CNN15_THRESHOLDS, CNN16_MODEL, CNN16_META, CNN16_THRESHOLDS]:
        require(path)

    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"torch is required for this comparison: {exc}") from exc

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

    pbase = patch_base()
    patches_all = np.load(pbase / "baseline_patches.npy").astype("float32")
    labels_all = np.load(pbase / "baseline_labels.npy").astype("int64")
    meta_all = pd.read_csv(pbase / "baseline_metadata.csv")
    clean_mask = meta_all["clean_label"].isin([0, 1]).to_numpy()
    patches = patches_all[clean_mask]
    labels = labels_all[clean_mask].astype("int64")
    meta = meta_all.loc[clean_mask].reset_index(drop=True).copy()
    meta["split_group"] = meta.apply(split_group, axis=1)
    groups = meta["split_group"].astype(str).to_numpy()
    group_sizes = pd.Series(groups).value_counts()
    duplicated_groups = group_sizes[group_sizes > 1]

    conservative_patches_raw = np.load(pbase / "sdc2_conservative_patches.npy").astype("float32")
    conservative_meta = pd.read_csv(pbase / "sdc2_conservative_metadata.csv")

    scorer = import_scorer()
    truth = pd.read_csv(TRUTH, sep=r"\s+", comment="#", engine="python") if TRUTH.exists() and scorer else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_summary = pd.DataFrame(
        [
            {
                "patch_base": str(pbase),
                "training_examples": len(labels),
                "tp": int((labels == 1).sum()),
                "fp": int((labels == 0).sum()),
                "groups": int(group_sizes.size),
                "duplicated_groups": int(len(duplicated_groups)),
                "shared_groups_expected": 0,
                "patch_shape": "21x32x32",
                "conservative_examples": len(conservative_meta),
                "conservative_clean_label_counts": str(conservative_meta["clean_label"].value_counts(dropna=False).to_dict()),
            }
        ]
    )
    data_summary.to_csv(OUT_REPORTS / "data_and_group_summary.csv", index=False)
    architecture_summary().to_csv(OUT_REPORTS / "architecture_comparison.csv", index=False)

    # CNN15 checkpoint, applied with the same inference/scoring path as this comparison.
    cnn15_ckpt = torch.load(CNN15_MODEL, map_location=device)
    cnn15 = BaselineCNN().to(device)
    cnn15.load_state_dict(cnn15_ckpt["model_state_dict"])
    cnn15.eval()
    cnn15_mean = np.asarray(cnn15_ckpt["channel_mean"], dtype="float32")
    cnn15_std = np.asarray(cnn15_ckpt["channel_std"], dtype="float32")
    cnn15_std[cnn15_std < 1e-6] = 1.0
    cnn15_patches = normalize(conservative_patches_raw, cnn15_mean, cnn15_std)

    def predict(model: nn.Module, arr: np.ndarray) -> np.ndarray:
        scores = []
        with torch.no_grad():
            for start in range(0, len(arr), 64):
                batch = torch.tensor(arr[start : start + 64], dtype=torch.float32, device=device)
                scores.append(torch.sigmoid(model(batch)).detach().cpu().numpy())
        return np.concatenate(scores) if scores else np.asarray([], dtype="float32")

    cnn15_scores = predict(cnn15, cnn15_patches)

    # CNN16 original checkpoint.
    cnn16_ckpt = torch.load(CNN16_MODEL, map_location=device)
    cnn16 = FinalGroupedCNN().to(device)
    cnn16.load_state_dict(cnn16_ckpt["model_state_dict"])
    cnn16.eval()
    cnn16_mean = np.asarray(cnn16_ckpt["channel_mean"], dtype="float32")
    cnn16_std = np.asarray(cnn16_ckpt["channel_std"], dtype="float32")
    cnn16_std[cnn16_std < 1e-6] = 1.0
    cnn16_scores = predict(cnn16, normalize(conservative_patches_raw, cnn16_mean, cnn16_std))

    # CNN16 architecture retrained with CNN15 settings on the same grouped split seed.
    seed = 123
    lr = 0.001
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros_like(labels), labels, groups))
    shared = set(groups[train_idx]).intersection(set(groups[test_idx]))
    if shared:
        raise RuntimeError(f"Unexpected shared groups in controlled split: {len(shared)}")
    mean, std = channel_stats(patches[train_idx])
    x_train = normalize(patches[train_idx], mean, std)
    x_test = normalize(patches[test_idx], mean, std)
    y_train, y_test = labels[train_idx], labels[test_idx]

    torch.manual_seed(seed)
    np.random.seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    model16_settings = FinalGroupedCNN().to(device)
    pos = max(1, int((y_train == 1).sum()))
    neg = max(1, int((y_train == 0).sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model16_settings.parameters(), lr=lr)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_tensor = torch.tensor(x_test, dtype=torch.float32, device=device)
    best_ap = -np.inf
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model16_settings.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model16_settings(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model16_settings.eval()
        with torch.no_grad():
            fold_scores = torch.sigmoid(model16_settings(test_tensor)).detach().cpu().numpy()
        ap = average_precision_score(y_test, fold_scores)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "average_precision": float(ap)})
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.detach().cpu() for k, v in model16_settings.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model16_settings.load_state_dict(best_state)
    model16_settings.eval()
    controlled_fold_scores = predict(model16_settings, x_test)
    controlled_thresholds = derive_thresholds(y_test, controlled_fold_scores)
    controlled_thresholds.insert(0, "model", "CNN16_with_cnn15_settings")
    controlled_thresholds.to_csv(OUT_REPORTS / "cnn16_with_cnn15_settings_metrics.csv", index=False)
    pd.DataFrame(history).to_csv(OUT_REPORTS / "cnn16_with_cnn15_settings_training_history.csv", index=False)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model16_settings.state_dict().items()},
            "architecture": "final_grouped_cnn_2_5d",
            "patch_shape": [21, 32, 32],
            "channel_mean": mean.tolist(),
            "channel_std": std.tolist(),
            "seed": seed,
            "learning_rate": lr,
        },
        OUT_MODELS / "cnn16_with_cnn15_settings.pt",
    )
    cnn16_settings_scores = predict(model16_settings, normalize(conservative_patches_raw, mean, std))

    # Fixed catalogues requested by the task.
    score_rows = []
    score_map = {
        "CNN15_in_pipeline16": (cnn15_scores, 0.98, "cnn15_in_pipeline16"),
        "CNN16_with_cnn15_settings": (cnn16_settings_scores, 0.98, "cnn16_with_cnn15_settings"),
    }
    for catalog_name, (scores, threshold, score_col) in score_map.items():
        out = conservative_meta.copy()
        out[score_col] = scores
        out[f"{score_col}_threshold"] = threshold
        out = out[out[score_col] >= threshold].copy()
        out_path = OUT_FILTERED / f"{catalog_name}.csv"
        out.to_csv(out_path, index=False)
        row = score_catalog(catalog_name, out, scorer, truth)
        row["threshold"] = threshold
        row["catalog_path"] = str(out_path)
        score_rows.append(row)
        pd.DataFrame([row]).to_csv(OUT_SCORES / f"{catalog_name}_score.csv", index=False)

    # CNN15 internal metric row from the stored validation fold.
    cnn15_thresholds = pd.read_csv(CNN15_THRESHOLDS)
    cnn15_thresholds.to_csv(OUT_REPORTS / "cnn15_in_pipeline16_metrics.csv", index=False)

    # Common threshold sweep.
    sweep_rows = []
    sweep_models = {
        "CNN15_checkpoint": cnn15_scores,
        "CNN16_final_checkpoint": cnn16_scores,
        "CNN16_with_cnn15_settings": cnn16_settings_scores,
    }
    for model_name, scores in sweep_models.items():
        for threshold in COMMON_THRESHOLDS:
            out = conservative_meta.copy()
            out["comparison_score"] = scores
            out["comparison_model"] = model_name
            out["comparison_threshold"] = threshold
            out = out[out["comparison_score"] >= threshold].copy()
            threshold_label = f"{threshold:.2f}".replace(".", "p")
            out_path = OUT_FILTERED / f"{model_name}_threshold_{threshold_label}.csv"
            out.to_csv(out_path, index=False)
            score_row_out = score_catalog(f"{model_name}_threshold_{threshold_label}", out, scorer, truth)
            sweep_rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "accepted": int(len(out)),
                    "official_score": score_row_out["score"],
                    "matches": score_row_out["matches"],
                    "false": score_row_out["false"],
                    "status": score_row_out["status"],
                    "catalog_path": str(out_path),
                }
            )
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_REPORTS / "common_threshold_sweep.csv", index=False)

    probability_summary = []
    for name, scores in sweep_models.items():
        probability_summary.append(
            {
                "model": name,
                "min": float(np.min(scores)),
                "p05": float(np.quantile(scores, 0.05)),
                "p25": float(np.quantile(scores, 0.25)),
                "median": float(np.median(scores)),
                "p75": float(np.quantile(scores, 0.75)),
                "p95": float(np.quantile(scores, 0.95)),
                "max": float(np.max(scores)),
                "mean": float(np.mean(scores)),
            }
        )
    pd.DataFrame(probability_summary).to_csv(OUT_REPORTS / "probability_summary.csv", index=False)

    existing_rows = []
    if CNN15_OFFICIAL.exists():
        df15 = pd.read_csv(CNN15_OFFICIAL)
        for name in ["ROBUST_CNN_grouped_f0_5", "ROBUST_CNN_grouped_conservative_fp"]:
            match = df15[df15["catalog_name"] == name]
            if not match.empty:
                row = match.iloc[0]
                existing_rows.append(
                    {
                        "Modelo": name,
                        "Arquitectura": "BaselineCNN / baseline_grouped",
                        "Seed": 123,
                        "LR": 0.001,
                        "Threshold": float(pd.read_csv(CNN15_THRESHOLDS).set_index("threshold_mode").loc["f0_5", "threshold"]),
                        "Accepted": int(row["n_submission"]),
                        "Score": float(row["score"]),
                        "Matches": int(row["matches"]),
                        "False": int(row["false"]),
                        "Comentario": "Resultado original de 15.",
                    }
                )
    if CNN16_OFFICIAL.exists():
        df16 = pd.read_csv(CNN16_OFFICIAL)
        thresholds16 = pd.read_csv(CNN16_THRESHOLDS).set_index("threshold_mode")["threshold"].to_dict()
        for name, mode in [("final_grouped_cnn_f0_5", "f0_5"), ("final_grouped_cnn_conservative_fp", "conservative_fp")]:
            match = df16[df16["catalog_name"] == name]
            if not match.empty:
                row = match.iloc[0]
                existing_rows.append(
                    {
                        "Modelo": name,
                        "Arquitectura": "FinalGroupedCNN",
                        "Seed": "42+fold",
                        "LR": 0.001,
                        "Threshold": float(thresholds16[mode]),
                        "Accepted": int(row["n_submission"]),
                        "Score": float(row["score"]),
                        "Matches": int(row["matches"]),
                        "False": int(row["false"]),
                        "Comentario": "Resultado original de 16.",
                    }
                )
    for row in score_rows:
        existing_rows.append(
            {
                "Modelo": row["catalog_name"],
                "Arquitectura": "BaselineCNN" if row["catalog_name"] == "CNN15_in_pipeline16" else "FinalGroupedCNN",
                "Seed": 123,
                "LR": 0.001,
                "Threshold": 0.98,
                "Accepted": int(row["n_submission"]),
                "Score": row["score"],
                "Matches": row["matches"],
                "False": row["false"],
                "Comentario": "Comparacion controlada generada en 17.",
            }
        )
    if not sweep.empty:
        ok_sweep = sweep[sweep["status"] == "OK"].copy()
        if not ok_sweep.empty:
            for model_name in ["CNN15_checkpoint", "CNN16_final_checkpoint", "CNN16_with_cnn15_settings"]:
                subset = ok_sweep[ok_sweep["model"] == model_name]
                if subset.empty:
                    continue
                best = subset.sort_values("official_score", ascending=False).iloc[0]
                existing_rows.append(
                    {
                        "Modelo": f"best_common_sweep_{model_name}",
                        "Arquitectura": model_name,
                        "Seed": 123 if "CNN16_final" not in model_name else "42+fold",
                        "LR": 0.001,
                        "Threshold": float(best["threshold"]),
                        "Accepted": int(best["accepted"]),
                        "Score": float(best["official_score"]),
                        "Matches": int(best["matches"]),
                        "False": int(best["false"]),
                        "Comentario": "Mejor threshold dentro del sweep comun.",
                    }
                )
    final_table = pd.DataFrame(existing_rows)
    final_table.to_csv(OUT_REPORTS / "final_comparison_table.csv", index=False)

    pd.DataFrame(score_rows).to_csv(OUT_SCORES / "controlled_experiment_scores.csv", index=False)
    print(final_table.to_string(index=False))


if __name__ == "__main__":
    main()
