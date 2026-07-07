from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dirs, load_config, save_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara estrategias raw vs ML.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def ranked(scores: pd.DataFrame, ranking: str, columns: list[str], ascending: list[bool]) -> pd.DataFrame:
    result = scores.sort_values(columns, ascending=ascending).copy()
    result.insert(0, "rank", range(1, len(result) + 1))
    result.insert(0, "ranking", ranking)
    return result


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    scores_path = paths["scores_dir"] / "catalog_strategy_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Falta {scores_path}. Ejecuta primero 02_score_filtered_catalogs.py")
    scores = pd.read_csv(scores_path)
    rankings = pd.concat(
        [
            ranked(scores, "recall", ["supervised_recall_clean", "reliability_clean"], [False, False]),
            ranked(scores, "reliability", ["reliability_clean", "fp_clean"], [False, True]),
            ranked(scores, "f2", ["f2_clean", "reliability_clean"], [False, False]),
            ranked(scores, "conservative", ["fp_clean", "ambiguous_rate", "reliability_clean"], [True, True, False]),
        ],
        ignore_index=True,
    )
    output_path = paths["scores_dir"] / "catalog_strategy_rankings.csv"
    save_dataframe(rankings, output_path)
    print(f"Rankings guardados en {output_path}")


if __name__ == "__main__":
    main()
