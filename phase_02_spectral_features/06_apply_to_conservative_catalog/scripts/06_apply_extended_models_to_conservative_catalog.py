#!/usr/bin/env python3
"""Apply focused Phase 2 extended models to the conservative SoFiA catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml


TOTAL_TP_BASELINE_CURRENT = 317


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def config_for_catalog(cfg: dict[str, Any], catalog_key: str) -> dict[str, Any]:
    catalogs = cfg.get("catalogs") or {}
    if catalog_key not in catalogs:
        raise KeyError(f"Missing catalog key in config: {catalog_key}")
    out = dict(cfg)
    out.update(catalogs[catalog_key])
    return out


def fbeta_from_pr(precision: float, recall: float, beta: float) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * precision * recall / ((beta2 * precision) + recall)


def local_metrics(df: pd.DataFrame, accepted: pd.Series) -> dict[str, Any]:
    selected = df.loc[accepted]
    if "clean_label" not in selected.columns:
        return {"n_candidates": int(len(selected))}
    tp = int((selected["clean_label"] == 1).sum())
    fp = int((selected["clean_label"] == 0).sum())
    amb = int((selected["clean_label"] == -1).sum())
    reliability = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / TOTAL_TP_BASELINE_CURRENT
    return {
        "n_candidates": int(len(selected)),
        "TP_clean": tp,
        "FP_clean": fp,
        "ambiguous": amb,
        "reliability_clean": reliability,
        "recall_clean": recall,
        "f0_5_clean": fbeta_from_pr(reliability, recall, 0.5),
        "f1_clean": fbeta_from_pr(reliability, recall, 1.0),
        "f2_clean": fbeta_from_pr(reliability, recall, 2.0),
        "ambiguous_rate": amb / len(selected) if len(selected) else 0.0,
    }


def predict_scores(artifact: dict[str, Any], df: pd.DataFrame) -> pd.Series:
    pipe = artifact["pipeline"]
    columns = artifact["columns"]
    scores = pipe.predict_proba(df[columns])[:, 1]
    return pd.Series(scores, index=df.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    entry = config_for_catalog(cfg, "sdc2_team_sofia_like_full")
    dataset_path = Path(entry["output_dataset"])
    selected_path = Path(
        "phase_02_spectral_features/05_focused_winners_comparison/outputs/reports/"
        "focused_phase2_selected_strategies.json"
    )
    out_dir = Path("phase_02_spectral_features/06_apply_to_conservative_catalog/outputs/filtered_catalogs")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected strategies from step 05: {selected_path}")

    df = pd.read_csv(dataset_path)
    selected_strategies = json.loads(selected_path.read_text(encoding="utf-8"))
    wanted = {
        "best_extended_full_f0_5": "SDC2_extended_full_f0_5",
        "best_extended_full_conservative_fp": "SDC2_extended_full_conservative_fp",
        "best_extended_no_position_f0_5": "SDC2_extended_no_position_f0_5",
    }
    rows: list[dict[str, Any]] = []

    for strategy in selected_strategies:
        original_name = strategy["strategy_name"]
        if original_name not in wanted:
            continue
        output_name = wanted[original_name]
        artifact = joblib.load(strategy["model_artifact"])
        scores = predict_scores(artifact, df)
        accepted = scores >= float(strategy["threshold"])
        filtered = df.loc[accepted].copy()
        filtered["phase2_score"] = scores.loc[accepted].to_numpy()
        filtered["phase2_strategy"] = output_name
        filtered_path = out_dir / f"{output_name}.csv"
        filtered.to_csv(filtered_path, index=False)
        rows.append(
            {
                "strategy": output_name,
                "source_strategy": original_name,
                "feature_set": strategy["feature_set"],
                "model": strategy["model"],
                "threshold_mode": strategy["threshold_mode"],
                "threshold": float(strategy["threshold"]),
                "catalog_path": str(filtered_path),
                **local_metrics(df, accepted),
            }
        )

    summary = pd.DataFrame(rows)
    csv_path = out_dir / "conservative_catalog_local_summary.csv"
    summary.to_csv(csv_path, index=False)

    print(f"Summary: {csv_path}")
    if not summary.empty:
        print(summary[["strategy", "model", "threshold_mode", "n_candidates", "TP_clean", "FP_clean", "ambiguous"]].to_string(index=False))


if __name__ == "__main__":
    main()
