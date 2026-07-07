from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config, save_dataframe


SELECTED = [
    {
        "strategy_name": "A_baseline_current_full_raw",
        "source_type": "raw",
        "selection_role": "baseline_permissive",
        "optional": False,
    },
    {
        "strategy_name": "B_sdc2_team_sofia_like_full_raw",
        "source_type": "raw",
        "selection_role": "baseline_conservative",
        "optional": False,
    },
    {
        "strategy_name": "ML_ExtraTrees_full_f2",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_ExtraTrees_full_f2.csv",
        "selection_role": "max_local_recall",
        "optional": False,
    },
    {
        "strategy_name": "ML_XGBoost_full_f1",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_XGBoost_full_f1.csv",
        "selection_role": "balanced_full",
        "optional": False,
    },
    {
        "strategy_name": "ML_XGBoost_full_f0_5",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_XGBoost_full_f0_5.csv",
        "selection_role": "primary_conservative_precision",
        "optional": False,
    },
    {
        "strategy_name": "ML_XGBoost_full_conservative_fp",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_XGBoost_full_conservative_fp.csv",
        "selection_role": "legacy_conservative_fp",
        "optional": False,
    },
    {
        "strategy_name": "ML_GradientBoosting_no_position_conservative_fp",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_GradientBoosting_no_position_conservative_fp.csv",
        "selection_role": "conservative_no_position",
        "optional": False,
    },
    {
        "strategy_name": "ML_XGBoost_no_position_conservative_fp",
        "source_type": "accepted",
        "accepted_file": "../06_catalog_strategy_comparison/outputs/accepted/accepted_XGBoost_no_position_conservative_fp.csv",
        "selection_role": "optional_no_position",
        "optional": True,
    },
]

OUTPUT_COLUMNS = [
    "strategy_name",
    "strategy_type",
    "model_key",
    "model_name",
    "feature_set",
    "threshold_mode",
    "threshold",
    "n_candidates",
    "tp_clean",
    "fp_clean",
    "ambiguous",
    "reliability_clean",
    "supervised_recall_clean",
    "f2_clean",
    "ambiguous_rate",
    "accepted_rate",
    "selection_role",
    "selected_for_08",
    "optional",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta estrategias candidatas de 06 para scoring oficial en 08.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def yaml_quote(value: str) -> str:
    return '"' + value.replace('"', '\\"') + '"'


def write_selected_yaml(rows: list[dict], path: Path) -> None:
    lines = ["selected_strategies:"]
    for row in rows:
        lines.append(f"  - strategy_name: {row['strategy_name']}")
        lines.append(f"    source_type: {row['source_type']}")
        if row.get("accepted_file"):
            lines.append(f"    accepted_file: {yaml_quote(row['accepted_file'])}")
        lines.append(f"    selection_role: {row['selection_role']}")
        lines.append(f"    optional: {bool_text(bool(row['optional']))}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    scores_path = paths["scores_dir"] / "catalog_strategy_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Falta {scores_path}. Ejecuta primero 02_score_filtered_catalogs.py")
    scores = pd.read_csv(scores_path)

    selection_df = pd.DataFrame(SELECTED)
    selected = selection_df.merge(scores, on="strategy_name", how="left")
    missing = selected[selected["strategy_type"].isna()]["strategy_name"].tolist()
    for strategy_name in missing:
        print(f"WARNING: estrategia seleccionada no aparece en catalog_strategy_scores.csv: {strategy_name}")
    selected["selected_for_08"] = True
    selected["strategy_type"] = selected["strategy_type"].fillna(selected["source_type"])
    for column in OUTPUT_COLUMNS:
        if column not in selected.columns:
            selected[column] = pd.NA
    selected = selected[OUTPUT_COLUMNS]

    out_dir = paths["selected_for_scoring_dir"]
    csv_path = out_dir / "selected_strategies.csv"
    yaml_path = out_dir / "selected_for_08.yaml"
    save_dataframe(selected, csv_path)
    write_selected_yaml(SELECTED, yaml_path)

    print(f"Seleccion CSV: {csv_path}")
    print(f"Seleccion YAML para 08: {yaml_path}")


if __name__ == "__main__":
    main()
