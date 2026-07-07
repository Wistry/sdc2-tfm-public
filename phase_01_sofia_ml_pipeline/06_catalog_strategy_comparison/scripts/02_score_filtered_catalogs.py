from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_candidates, load_config, load_scoring_summary, save_dataframe


EXPECTED_APPROX = {
    "baseline_current_full": {"n_candidates": 1169, "tp": 317, "fp": 126, "ambiguous": 726},
    "sdc2_team_sofia_like_full": {"n_candidates": 191, "tp": 181, "fp": 2, "ambiguous": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calcula metricas locales de catalogos filtrados.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def f_beta(reliability: float, recall: float, beta: float) -> float:
    if reliability <= 0 and recall <= 0:
        return 0.0
    beta2 = beta * beta
    return (1 + beta2) * reliability * recall / (beta2 * reliability + recall)


def score_counts(
    strategy_name: str,
    strategy_type: str,
    model_key: str,
    model_name: str,
    feature_set: str,
    threshold_mode: str,
    threshold,
    df: pd.DataFrame,
    total_tp: int,
    total_candidates: int,
) -> dict:
    tp = int((df["clean_label"] == 1).sum()) if "clean_label" in df.columns else 0
    fp = int((df["clean_label"] == 0).sum()) if "clean_label" in df.columns else 0
    ambiguous = int((df["clean_label"] == -1).sum()) if "clean_label" in df.columns else 0
    denom = tp + fp
    reliability_clean = tp / denom if denom else 0.0
    recall = tp / total_tp if total_tp else 0.0
    n_candidates = int(len(df))
    return {
        "strategy_name": strategy_name,
        "strategy_type": strategy_type,
        "model_key": model_key,
        "model_name": model_name,
        "feature_set": feature_set,
        "threshold": threshold,
        "threshold_mode": threshold_mode,
        "n_candidates": n_candidates,
        "tp_clean": tp,
        "fp_clean": fp,
        "ambiguous": ambiguous,
        "reliability_clean": reliability_clean,
        "supervised_recall_clean": recall,
        "f0_5_clean": f_beta(reliability_clean, recall, 0.5),
        "f1_clean": f_beta(reliability_clean, recall, 1.0),
        "f2_clean": f_beta(reliability_clean, recall, 2.0),
        "fp_per_tp": fp / tp if tp else float("nan"),
        "ambiguous_rate": ambiguous / n_candidates if n_candidates else 0.0,
        "accepted_rate": n_candidates / total_candidates if total_candidates else 0.0,
    }


def raw_rows(scoring: pd.DataFrame) -> list[dict]:
    rows = []
    warnings = []
    mapping = {
        "baseline_current_full": "A_baseline_current_full_raw",
        "sdc2_team_sofia_like_full": "B_sdc2_team_sofia_like_full_raw",
    }
    for config_name, strategy in mapping.items():
        match = scoring[scoring["config"] == config_name] if "config" in scoring.columns else pd.DataFrame()
        if match.empty:
            warnings.append(f"No aparece {config_name} en scoring_full_cube.csv")
            continue
        item = match.iloc[0]
        expected = EXPECTED_APPROX.get(config_name, {})
        for col, expected_value in expected.items():
            if col in item and abs(float(item[col]) - expected_value) > max(5, expected_value * 0.25):
                warnings.append(f"Valor sospechoso para {config_name}.{col}: {item[col]} (esperado aprox {expected_value})")
        tp = int(item.get("tp", 0))
        fp = int(item.get("fp", 0))
        ambiguous = int(item.get("ambiguous", 0))
        n = int(item.get("n_candidates", tp + fp + ambiguous))
        reliability = tp / (tp + fp) if (tp + fp) else 0.0
        recall = float(item.get("completeness", 0.0))
        rows.append({
            "strategy_name": strategy,
            "strategy_type": "raw",
            "model_key": config_name,
            "model_name": config_name,
            "feature_set": "raw",
            "threshold": "",
            "threshold_mode": "raw",
            "n_candidates": n,
            "tp_clean": tp,
            "fp_clean": fp,
            "ambiguous": ambiguous,
            "reliability_clean": reliability,
            "supervised_recall_clean": recall,
            "f0_5_clean": f_beta(reliability, recall, 0.5),
            "f1_clean": f_beta(reliability, recall, 1.0),
            "f2_clean": f_beta(reliability, recall, 2.0),
            "fp_per_tp": fp / tp if tp else float("nan"),
            "ambiguous_rate": ambiguous / n if n else 0.0,
            "accepted_rate": 1.0,
        })
    for warning in warnings:
        print(f"WARNING: {warning}")
    return rows


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    candidates = load_candidates(config)
    scoring = load_scoring_summary(config)
    total_tp = int((candidates["clean_label"] == 1).sum()) if "clean_label" in candidates.columns else 0
    total_candidates = int(len(candidates))
    rows = raw_rows(scoring) if not scoring.empty else []

    for pred_path in sorted(paths["predictions_dir"].glob("predictions_*.csv")):
        if pred_path.name == "prediction_summary.csv":
            continue
        pred = pd.read_csv(pred_path)
        accepted = pred[pred["pred_label"] == 1].copy()
        model_key = str(pred["model_key"].iloc[0]) if "model_key" in pred.columns and not pred.empty else pred_path.stem.replace("predictions_", "")
        model_name = str(pred["model_name"].iloc[0]) if "model_name" in pred.columns and not pred.empty else model_key
        feature_set = str(pred["feature_set"].iloc[0]) if "feature_set" in pred.columns and not pred.empty else ""
        threshold_mode = str(pred["threshold_mode"].iloc[0]) if "threshold_mode" in pred.columns and not pred.empty else ""
        threshold = float(pred["threshold"].iloc[0]) if "threshold" in pred.columns and not pred.empty else 0.5
        row = score_counts(
            f"ML_{model_key}_{threshold_mode}",
            "ml",
            model_key,
            model_name,
            feature_set,
            threshold_mode,
            threshold,
            accepted,
            total_tp,
            total_candidates,
        )
        rows.append(row)

    scores = pd.DataFrame(rows)
    save_dataframe(scores, paths["scores_dir"] / "catalog_strategy_scores.csv")
    print(f"Scores guardados en {paths['scores_dir'] / 'catalog_strategy_scores.csv'}")


if __name__ == "__main__":
    main()
