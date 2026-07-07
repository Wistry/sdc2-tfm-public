from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dirs, load_config, resolve_path, short_strategy_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera figuras del scoring oficial SDC2.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def load_scores(paths: dict[str, Path]) -> pd.DataFrame:
    scores_path = paths["official_scores_dir"] / "official_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Falta {scores_path}. Ejecuta primero 02_score_with_official_sdc2.py")
    return pd.read_csv(scores_path)


def local_strategy_name(catalog_name: str) -> str:
    if catalog_name.startswith("ML_"):
        return catalog_name
    if catalog_name.startswith("ENS_sdc2_plus_"):
        return "ML_" + catalog_name.removeprefix("ENS_sdc2_plus_")
    return catalog_name


def plot_score_by_catalog(ok: pd.DataFrame, figures_dir: Path) -> None:
    if ok.empty:
        return
    data = ok.sort_values("score_value", ascending=False).copy()
    labels = [short_strategy_name(name) for name in data["catalog_name"]]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(data)), data["score_value"], color="#2b8cbe")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Official score")
    ax.set_title("Official SDC2 score by catalog")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_official_score_by_catalog.png", dpi=160)
    plt.close(fig)


def plot_match_vs_false(ok: pd.DataFrame, figures_dir: Path) -> None:
    if ok.empty or not {"n_false", "n_match"}.issubset(ok.columns):
        return
    marker_map = {"raw": "o", "accepted": "s", "ensemble": "^"}
    color_map = {"raw": "#636363", "accepted": "#2b8cbe", "ensemble": "#31a354"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for source_type, group in ok.groupby("source_type"):
        ax.scatter(
            group["n_false"],
            group["n_match"],
            marker=marker_map.get(source_type, "o"),
            color=color_map.get(source_type, "#969696"),
            s=90,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
            label=source_type,
        )
    for _, row in ok.sort_values("score_value", ascending=False).head(5).iterrows():
        ax.annotate(
            short_strategy_name(row["catalog_name"]),
            (row["n_false"], row["n_match"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("n_false")
    ax.set_ylabel("n_match")
    ax.set_title("Matches vs false detections")
    ax.grid(alpha=0.25)
    ax.legend(title="source_type")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_match_vs_false.png", dpi=160)
    plt.close(fig)


def plot_local_vs_official(scores: pd.DataFrame, local_scores: pd.DataFrame, figures_dir: Path) -> None:
    if local_scores.empty:
        return
    ok = scores[scores["status"] == "OK"].copy()
    ok["strategy_name"] = ok["catalog_name"].map(local_strategy_name)
    merged = ok.merge(local_scores, on="strategy_name", how="inner", suffixes=("", "_local"))
    if merged.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(merged["f2_clean"], merged["score_value"], s=90, color="#2b8cbe", edgecolor="black", linewidth=0.4)
    for _, row in merged.sort_values("score_value", ascending=False).head(8).iterrows():
        ax.annotate(
            short_strategy_name(row["catalog_name"]),
            (row["f2_clean"], row["score_value"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Local f2_clean")
    ax.set_ylabel("Official score")
    ax.set_title("Local metric vs official score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_local_vs_official.png", dpi=160)
    plt.close(fig)


def make_figures(scores: pd.DataFrame, local_scores: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    ok = scores[scores["status"] == "OK"].copy()
    plot_score_by_catalog(ok, figures_dir)
    plot_match_vs_false(ok, figures_dir)
    plot_local_vs_official(scores, local_scores, figures_dir)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    scores = load_scores(paths)
    local_path = resolve_path(
        config["data"].get(
            "strategy_scores_path",
            "../06_catalog_strategy_comparison/outputs/scores/catalog_strategy_scores.csv",
        )
    )
    local_scores = pd.read_csv(local_path) if local_path.exists() else pd.DataFrame()
    make_figures(scores, local_scores, paths["report_figures_dir"])
    print(f"Figuras guardadas en {paths['report_figures_dir']}")


if __name__ == "__main__":
    main()
