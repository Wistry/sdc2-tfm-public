from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dirs, load_config, resolve_path


REPORT_STRATEGIES = [
    "A_baseline_current_full_raw",
    "B_sdc2_team_sofia_like_full_raw",
    "ML_ExtraTrees_full_f2",
    "ML_ExtraTrees_full_f0_5",
    "ML_XGBoost_full_f1",
    "ML_XGBoost_full_f0_5",
    "ML_XGBoost_full_conservative_fp",
    "ML_RandomForest_full_conservative_fp",
    "ML_GradientBoosting_no_position_f0_5",
    "ML_GradientBoosting_no_position_conservative_fp",
    "ML_XGBoost_no_position_conservative_fp",
]

LABEL_STRATEGIES = [
    "ML_ExtraTrees_full_f2",
    "ML_XGBoost_full_f1",
    "ML_XGBoost_full_conservative_fp",
    "ML_GradientBoosting_no_position_conservative_fp",
    "A_baseline_current_full_raw",
    "B_sdc2_team_sofia_like_full_raw",
]

FIG01_STRATEGIES = REPORT_STRATEGIES

FIG04_STRATEGIES = [
    "ML_ExtraTrees_full_f2",
    "ML_ExtraTrees_full_balanced_accuracy",
    "ML_ExtraTrees_full_f0_5",
    "ML_ExtraTrees_full_conservative_fp",
    "ML_XGBoost_full_f2",
    "ML_XGBoost_full_f1",
    "ML_XGBoost_full_f0_5",
    "ML_XGBoost_full_conservative_fp",
    "ML_GradientBoosting_no_position_f2",
    "ML_GradientBoosting_no_position_f0_5",
    "ML_GradientBoosting_no_position_conservative_fp",
]

SHORT_LABELS = {
    "A_baseline_current_full_raw": "SoFiA raw",
    "B_sdc2_team_sofia_like_full_raw": "SoFiA cons",
    "ML_ExtraTrees_full_f2": "ET full F2",
    "ML_ExtraTrees_full_balanced_accuracy": "ET bal",
    "ML_ExtraTrees_full_f0_5": "ET F0.5",
    "ML_ExtraTrees_full_conservative_fp": "ET cons",
    "ML_XGBoost_full_f2": "XGB F2",
    "ML_XGBoost_full_f1": "XGB full F1",
    "ML_XGBoost_full_f0_5": "XGB F0.5",
    "ML_XGBoost_full_conservative_fp": "XGB full cons",
    "ML_RandomForest_full_conservative_fp": "RF full cons",
    "ML_GradientBoosting_no_position_f2": "GB no-pos F2",
    "ML_GradientBoosting_no_position_f0_5": "GB no-pos F0.5",
    "ML_GradientBoosting_no_position_conservative_fp": "GB no-pos cons",
    "ML_XGBoost_no_position_conservative_fp": "XGB no-pos cons",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera figuras de estrategias de catalogo.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def short_strategy_name(name: str) -> str:
    if name in SHORT_LABELS:
        return SHORT_LABELS[name]
    text = name.removeprefix("ML_")
    text = text.replace("balanced_accuracy", "bal")
    text = text.replace("conservative_fp", "cons")
    text = text.replace("baseline_current_full_raw", "raw_permissive")
    text = text.replace("sdc2_team_sofia_like_full_raw", "raw_conservative")
    return text


def ordered_subset(scores: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    order = {name: idx for idx, name in enumerate(names)}
    subset = scores[scores["strategy_name"].isin(names)].copy()
    subset["_order"] = subset["strategy_name"].map(order)
    return subset.sort_values("_order").drop(columns=["_order"])


def load_scores(config: dict) -> pd.DataFrame:
    paths = ensure_dirs(config)
    scores_path = paths["scores_dir"] / "catalog_strategy_scores.csv"
    if not scores_path.exists():
        raise SystemExit(f"Falta {scores_path}. Ejecuta primero 02_score_filtered_catalogs.py")
    return pd.read_csv(scores_path)


def plot_stacked_composition(scores: pd.DataFrame, figures_dir: Path) -> None:
    subset = ordered_subset(scores, FIG01_STRATEGIES)
    labels = [short_strategy_name(name) for name in subset["strategy_name"]]
    y = range(len(subset))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y, subset["tp_clean"], color="#2b8cbe", label="TP clean")
    ax.barh(y, subset["fp_clean"], left=subset["tp_clean"], color="#f03b20", label="FP clean")
    ax.barh(
        y,
        subset["ambiguous"],
        left=subset["tp_clean"] + subset["fp_clean"],
        color="#bdbdbd",
        label="Ambiguous",
    )
    for idx, (_, row) in enumerate(subset.iterrows()):
        total = row["tp_clean"] + row["fp_clean"] + row["ambiguous"]
        if row["tp_clean"] > 35:
            ax.text(row["tp_clean"] / 2, idx, f"{int(row['tp_clean'])}", va="center", ha="center", color="white", fontsize=8)
        if row["fp_clean"] > 20:
            ax.text(row["tp_clean"] + row["fp_clean"] / 2, idx, f"{int(row['fp_clean'])}", va="center", ha="center", color="white", fontsize=8)
        if row["ambiguous"] > 50:
            ax.text(row["tp_clean"] + row["fp_clean"] + row["ambiguous"] / 2, idx, f"{int(row['ambiguous'])}", va="center", ha="center", color="#333333", fontsize=8)
        ax.text(total + 12, idx, f"{int(total)}", va="center", ha="left", fontsize=8)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidatos aceptados")
    ax.set_title("Composicion de candidatos aceptados por estrategia")
    ax.set_xlim(0, max((subset["tp_clean"] + subset["fp_clean"] + subset["ambiguous"]).max() * 1.12, 1))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig01_top_strategies_composition.png", dpi=160)
    plt.close(fig)


def plot_recall_vs_reliability(scores: pd.DataFrame, figures_dir: Path) -> None:
    marker_map = {"f1": "o", "f0_5": "P", "f2": "s", "balanced_accuracy": "^", "conservative_fp": "D", "raw": "X"}
    color_map = {"f1": "#756bb1", "f0_5": "#e6550d", "f2": "#2b8cbe", "balanced_accuracy": "#31a354", "conservative_fp": "#de2d26", "raw": "#636363"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for mode, group in scores.groupby("threshold_mode"):
        sizes = 80 + group["ambiguous_rate"].fillna(0) * 500
        ax.scatter(
            group["supervised_recall_clean"],
            group["reliability_clean"],
            s=sizes,
            marker=marker_map.get(mode, "o"),
            color=color_map.get(mode, "#969696"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            label=mode,
        )

    offsets = {
        "A_baseline_current_full_raw": (6, 8),
        "B_sdc2_team_sofia_like_full_raw": (6, -12),
        "ML_ExtraTrees_full_f2": (-45, -18),
        "ML_XGBoost_full_f1": (-45, 8),
        "ML_XGBoost_full_conservative_fp": (-50, -16),
        "ML_GradientBoosting_no_position_conservative_fp": (-78, 8),
    }
    for _, row in scores[scores["strategy_name"].isin(LABEL_STRATEGIES)].iterrows():
        offset = offsets.get(row["strategy_name"], (5, 5))
        ax.annotate(
            short_strategy_name(row["strategy_name"]),
            (row["supervised_recall_clean"], row["reliability_clean"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    ax.set_xlabel("Supervised recall clean")
    ax.set_ylabel("Reliability clean")
    ax.set_title("Recall limpio vs reliability limpia")
    ax.set_xlim(0.04, 1.03)
    ax.set_ylim(0.68, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(title="threshold", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig02_recall_vs_reliability.png", dpi=160)
    plt.close(fig)


def plot_tp_vs_ambiguous_rate(scores: pd.DataFrame, figures_dir: Path) -> None:
    marker_map = {"f1": "o", "f0_5": "P", "f2": "s", "balanced_accuracy": "^", "conservative_fp": "D", "raw": "X"}
    color_map = {"full": "#2b8cbe", "no_position": "#31a354", "raw": "#636363"}
    fig, ax = plt.subplots(figsize=(8, 6))

    for (feature_set, mode), group in scores.groupby(["feature_set", "threshold_mode"]):
        ax.scatter(
            group["ambiguous_rate"],
            group["tp_clean"],
            s=90,
            marker=marker_map.get(mode, "o"),
            color=color_map.get(feature_set, "#969696"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.4,
            label=f"{feature_set}/{mode}",
        )
    for _, row in scores[scores["strategy_name"].isin(REPORT_STRATEGIES)].iterrows():
        ax.annotate(
            short_strategy_name(row["strategy_name"]),
            (row["ambiguous_rate"], row["tp_clean"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75},
        )
    ax.set_xlabel("Ambiguous rate")
    ax.set_ylabel("TP clean accepted")
    ax.set_title("TP limpios recuperados vs proporcion de ambiguos")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures_dir / "fig03_tp_vs_ambiguous_rate.png", dpi=160)
    plt.close(fig)


def plot_threshold_effect(scores: pd.DataFrame, figures_dir: Path) -> None:
    subset = ordered_subset(scores, FIG04_STRATEGIES)
    labels = [short_strategy_name(name) for name in subset["strategy_name"]]
    y = range(len(subset))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(y, subset["tp_clean"], color="#2b8cbe", label="TP clean")
    ax.barh(y, subset["fp_clean"], left=subset["tp_clean"], color="#f03b20", label="FP clean")
    ax.barh(
        y,
        subset["ambiguous"],
        left=subset["tp_clean"] + subset["fp_clean"],
        color="#bdbdbd",
        label="Ambiguous",
    )
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Candidatos aceptados")
    ax.set_title("Efecto de la politica de threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "fig04_threshold_effect_by_model.png", dpi=160)
    plt.close(fig)


def make_figures(scores: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_stacked_composition(scores, figures_dir)
    plot_recall_vs_reliability(scores, figures_dir)
    plot_tp_vs_ambiguous_rate(scores, figures_dir)
    plot_threshold_effect(scores, figures_dir)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    scores = load_scores(config)
    figures_dir = resolve_path("outputs/report_figures")
    make_figures(scores, figures_dir)
    print(f"Figuras guardadas en {figures_dir}")


if __name__ == "__main__":
    main()
