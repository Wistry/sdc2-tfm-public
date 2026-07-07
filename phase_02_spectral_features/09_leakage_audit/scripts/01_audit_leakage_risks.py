#!/usr/bin/env python3
"""Audit possible leakage/overfit risks without changing experiment outputs."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(".")
BASE = Path("phase_02_spectral_features/09_leakage_audit")
TABLES = BASE / "outputs" / "tables"

SCAN_ROOTS = [
    Path("phase_01_sofia_ml_pipeline"),
    Path("phase_02_spectral_features"),
]
SPLIT_TERMS = [
    "train_test_split",
    "StratifiedKFold",
    "KFold",
    "RepeatedStratifiedKFold",
    "cross_val_score",
    "cross_validate",
    "Optuna",
    "optuna",
    "StratifiedGroupKFold",
    "GroupKFold",
    "GroupShuffleSplit",
]
GROUP_TERMS = ["GroupKFold", "StratifiedGroupKFold", "GroupShuffleSplit", "groups="]
DATASETS = [
    Path("phase_01_sofia_ml_pipeline/03_candidate_dataset/outputs/baseline_current_full/candidates_sofia_only.csv"),
    Path("phase_01_sofia_ml_pipeline/03_candidate_dataset/outputs/sdc2_team_sofia_like_full/candidates_sofia_only.csv"),
    Path("phase_02_spectral_features/02_build_extended_datasets/outputs/clean/baseline_current_full_extended_clean.csv"),
    Path("phase_02_spectral_features/02_build_extended_datasets/outputs/clean/sdc2_team_sofia_like_full_extended_clean.csv"),
    Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches/baseline_metadata.csv"),
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)


def iter_python_scripts() -> list[Path]:
    paths: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            rel = path.as_posix()
            if "/__pycache__/" in rel or "/outputs/" in rel:
                continue
            if rel.startswith("phase_02_spectral_features/09_leakage_audit/"):
                continue
            paths.append(path)
    return sorted(paths)


def split_types_from_text(text: str) -> list[str]:
    return [term for term in SPLIT_TERMS if re.search(rf"\b{re.escape(term)}\b", text)]


def audit_splits() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in iter_python_scripts():
        text = path.read_text(encoding="utf-8", errors="replace")
        split_types = split_types_from_text(text)
        if not split_types:
            continue
        uses_groups = any(term in text for term in GROUP_TERMS)
        uses_stratify = "stratify=" in text or "Stratified" in text
        uses_matched_truth_id = "matched_truth_id" in text
        if uses_groups:
            risk = "LOW"
            notes = "Uses group-aware split or group argument."
        elif uses_matched_truth_id:
            risk = "MEDIUM"
            notes = "References matched_truth_id but no group-aware split was detected."
        else:
            risk = "MEDIUM"
            notes = "Appears to split/CV by row or candidate; candidates from the same truth source could be split across folds if duplicated."
        if "Optuna" in split_types or "optuna" in split_types:
            notes += " Optuna usage found; check whether optimization objective uses row-wise CV."
        rows.append(
            {
                "script_path": path.as_posix(),
                "split_type": ",".join(split_types),
                "uses_stratify": uses_stratify,
                "uses_groups": uses_groups,
                "uses_matched_truth_id": uses_matched_truth_id,
                "leakage_risk": risk,
                "notes": notes,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "split_audit.csv", index=False)
    return df


def normalize_truth_id(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isin(["", "nan", "NaN", "None", "<NA>", "-1"]))


def label_counts(df: pd.DataFrame) -> str:
    if "clean_label" not in df.columns:
        return "{}"
    counts = df["clean_label"].value_counts(dropna=False).to_dict()
    return json.dumps({str(k): int(v) for k, v in counts.items()}, sort_keys=True)


def resolve_dataset(path: Path) -> Path | None:
    if path.exists():
        return path
    matches = sorted(ROOT.rglob(path.name))
    return matches[0] if matches else None


def audit_duplicate_groups() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for requested in DATASETS:
        path = resolve_dataset(requested)
        if path is None:
            rows.append(
                {
                    "dataset_path": requested.as_posix(),
                    "exists": False,
                    "n_rows": 0,
                    "clean_label_counts": "{}",
                    "n_with_matched_truth_id": 0,
                    "n_truth_groups": 0,
                    "n_duplicate_truth_groups": 0,
                    "max_candidates_per_truth_id": 0,
                    "mean_candidates_per_truth_id": np.nan,
                    "median_candidates_per_truth_id": np.nan,
                    "pct_tp_candidates_in_duplicate_groups": np.nan,
                }
            )
            continue
        df = pd.read_csv(path)
        if "matched_truth_id" not in df.columns:
            rows.append(
                {
                    "dataset_path": path.as_posix(),
                    "exists": True,
                    "n_rows": len(df),
                    "clean_label_counts": label_counts(df),
                    "n_with_matched_truth_id": 0,
                    "n_truth_groups": 0,
                    "n_duplicate_truth_groups": 0,
                    "max_candidates_per_truth_id": 0,
                    "mean_candidates_per_truth_id": np.nan,
                    "median_candidates_per_truth_id": np.nan,
                    "pct_tp_candidates_in_duplicate_groups": np.nan,
                }
            )
            continue
        truth_id = normalize_truth_id(df["matched_truth_id"])
        valid = df[truth_id.notna()].copy()
        valid["_truth_id_norm"] = truth_id[truth_id.notna()]
        group_sizes = valid["_truth_id_norm"].value_counts()
        dup_groups = group_sizes[group_sizes > 1]
        tp = valid[valid.get("clean_label", pd.Series(index=valid.index, dtype=float)) == 1].copy()
        if not tp.empty:
            tp_dup = tp["_truth_id_norm"].isin(set(dup_groups.index)).sum()
            pct_tp_dup = 100.0 * tp_dup / len(tp)
        else:
            pct_tp_dup = np.nan
        row = {
            "dataset_path": path.as_posix(),
            "exists": True,
            "n_rows": len(df),
            "clean_label_counts": label_counts(df),
            "n_with_matched_truth_id": int(len(valid)),
            "n_truth_groups": int(len(group_sizes)),
            "n_duplicate_truth_groups": int(len(dup_groups)),
            "max_candidates_per_truth_id": int(group_sizes.max()) if not group_sizes.empty else 0,
            "mean_candidates_per_truth_id": float(group_sizes.mean()) if not group_sizes.empty else np.nan,
            "median_candidates_per_truth_id": float(group_sizes.median()) if not group_sizes.empty else np.nan,
            "pct_tp_candidates_in_duplicate_groups": pct_tp_dup,
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "duplicate_truth_groups.csv", index=False)
    return out


def load_phase2_random_state() -> int:
    cfg_path = Path("phase_02_spectral_features/configs/phase2_features.yaml")
    if not cfg_path.exists():
        return 42
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return int(cfg.get("random_state", 42))


def audit_cnn_overlap() -> pd.DataFrame:
    meta_path = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches/baseline_metadata.csv")
    labels_path = Path("phase_02_spectral_features/08_cnn_candidate_classifier/outputs/patches/baseline_labels.npy")
    if not meta_path.exists() or not labels_path.exists():
        df = pd.DataFrame(
            [
                {
                    "matched_truth_id": "",
                    "n_train": 0,
                    "n_test": 0,
                    "n_total": 0,
                    "example_train_indices": "",
                    "example_test_indices": "",
                    "status": "MISSING_METADATA_OR_LABELS",
                }
            ]
        )
        df.to_csv(TABLES / "cnn_train_test_group_overlap.csv", index=False)
        return df
    meta = pd.read_csv(meta_path)
    labels = np.load(labels_path).astype(int)
    indices = np.arange(len(labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=load_phase2_random_state(),
    )
    meta["_split"] = "unused"
    meta.loc[train_idx, "_split"] = "train"
    meta.loc[test_idx, "_split"] = "test"
    if "matched_truth_id" not in meta.columns:
        overlap = pd.DataFrame(columns=["matched_truth_id", "n_train", "n_test", "n_total", "example_train_indices", "example_test_indices", "status"])
    else:
        truth_id = normalize_truth_id(meta["matched_truth_id"])
        valid = meta[truth_id.notna()].copy()
        valid["_truth_id_norm"] = truth_id[truth_id.notna()]
        pivot = valid.pivot_table(index="_truth_id_norm", columns="_split", values="candidate_index" if "candidate_index" in valid.columns else "source_row_index", aggfunc="count", fill_value=0)
        for col in ["train", "test"]:
            if col not in pivot.columns:
                pivot[col] = 0
        shared_ids = pivot[(pivot["train"] > 0) & (pivot["test"] > 0)].index
        rows = []
        for truth in shared_ids:
            sub = valid[valid["_truth_id_norm"] == truth]
            train_examples = sub[sub["_split"] == "train"].head(5)
            test_examples = sub[sub["_split"] == "test"].head(5)
            id_col = "candidate_index" if "candidate_index" in sub.columns else "source_row_index"
            rows.append(
                {
                    "matched_truth_id": truth,
                    "n_train": int((sub["_split"] == "train").sum()),
                    "n_test": int((sub["_split"] == "test").sum()),
                    "n_total": int(len(sub)),
                    "example_train_indices": ",".join(map(str, train_examples[id_col].tolist())),
                    "example_test_indices": ",".join(map(str, test_examples[id_col].tolist())),
                    "status": "OVERLAP",
                }
            )
        overlap = pd.DataFrame(rows)
    overlap.to_csv(TABLES / "cnn_train_test_group_overlap.csv", index=False)
    return overlap


def audit_scorer_usage() -> pd.DataFrame:
    roots = [
        Path("phase_02_spectral_features/08_official_like_labels"),
        Path("phase_02_spectral_features/09_score_fusion_and_pruning"),
        Path("phase_02_spectral_features/archive/10_reference_catalog_comparison"),
    ]
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            if "/__pycache__/" in path.as_posix():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            terms = [term for term in ["Sdc2Scorer", "truth", "threshold", "greedy", "official_like", "score", "best"] if term in text]
            if "09_score_fusion_and_pruning" in path.as_posix():
                classification = "scorer-driven / possible overfit to development set"
            elif "08_official_like_labels" in path.as_posix():
                classification = "exploratory / possible overfit to development set"
            elif "10_reference_catalog_comparison" in path.as_posix():
                classification = "exploratory"
            else:
                classification = "core"
            rows.append(
                {
                    "script_path": path.as_posix(),
                    "terms_found": ",".join(terms),
                    "classification": classification,
                    "notes": "Uses scorer/truth/threshold search signals; interpret as development-set analysis unless validated externally.",
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "scorer_usage_audit.csv", index=False)
    return df


def main() -> None:
    ensure_dirs()
    split_df = audit_splits()
    dup_df = audit_duplicate_groups()
    cnn_overlap = audit_cnn_overlap()
    scorer_df = audit_scorer_usage()
    print(f"Split scripts inspected: {len(split_df)}")
    print(f"Datasets audited: {len(dup_df)}")
    print(f"Duplicate truth groups total: {int(dup_df['n_duplicate_truth_groups'].fillna(0).sum()) if not dup_df.empty else 0}")
    print(f"CNN overlap groups: {len(cnn_overlap)}")
    print(f"Scorer usage rows: {len(scorer_df)}")


if __name__ == "__main__":
    main()
