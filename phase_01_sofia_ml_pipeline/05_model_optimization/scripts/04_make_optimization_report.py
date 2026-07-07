from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import ensure_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genera figuras de resumen de la optimizacion.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        warnings.warn(f"No existe {path}; se generara informe parcial.")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_json(path: Path) -> dict:
    if not path.exists():
        warnings.warn(f"No existe {path}; se omite.")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_bar_by_feature_set(df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    if df.empty or metric not in df.columns:
        return
    pivot = df.pivot_table(index="model", columns="feature_set", values=metric, aggfunc="max")
    ax = pivot.plot(kind="bar", figsize=(10, 5))
    ax.set_xlabel("Modelo")
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.legend(title="Feature set")
    plt.xticks(rotation=45, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_threshold_figure(df: pd.DataFrame, path: Path) -> None:
    required = {"model", "feature_set", "best_f2_threshold", "conservative_threshold"}
    if df.empty or not required.issubset(df.columns):
        return
    plot_df = df.copy()
    plot_df["model_key"] = plot_df["model"] + "_" + plot_df["feature_set"]
    plot_df = plot_df.set_index("model_key")[["best_f2_threshold", "conservative_threshold"]]
    ax = plot_df.plot(kind="bar", figsize=(11, 5))
    ax.set_xlabel("Modelo final")
    ax.set_ylabel("Threshold")
    ax.set_title("Thresholds F2 vs conservador")
    plt.xticks(rotation=45, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def load_final_metadata(final_models_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(final_models_dir.glob("*_metadata.json")):
        metadata = read_json(path)
        if metadata:
            rows.append(metadata)
    return rows


def threshold_tradeoff_table(metadata_rows: list[dict]) -> pd.DataFrame:
    rows = []
    policies = ["f2", "f0_5", "balanced_accuracy", "conservative_fp"]
    for meta in metadata_rows:
        thresholds = meta.get("thresholds", {})
        for policy in policies:
            values = thresholds.get(policy, {})
            if not isinstance(values, dict):
                continue
            rows.append({
                "model_key": meta.get("model_key"),
                "threshold_policy": policy,
                "threshold": values.get("threshold"),
                "tp": values.get("tp"),
                "fp": values.get("fp"),
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f0_5": values.get("f0_5"),
                "f2": values.get("f2"),
                "balanced_accuracy": values.get("balanced_accuracy"),
            })
    return pd.DataFrame(rows)


def save_tradeoff_figure(tradeoff: pd.DataFrame, path: Path) -> None:
    if tradeoff.empty:
        return
    plot_df = tradeoff[tradeoff["threshold_policy"].isin(["f2", "f0_5", "balanced_accuracy", "conservative_fp"])].copy()
    plot_df["label"] = plot_df["model_key"] + "\n" + plot_df["threshold_policy"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(plot_df))
    ax.bar([i - 0.2 for i in x], plot_df["tp"], width=0.4, label="TP")
    ax.bar([i + 0.2 for i in x], plot_df["fp"], width=0.4, label="FP")
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"], rotation=75, ha="right")
    ax.set_ylabel("Candidatos en test split")
    ax.set_title("Trade-off TP/FP por politica de threshold")
    ax.legend()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    report_figures = paths["report_figures_dir"]
    report_figures.mkdir(parents=True, exist_ok=True)

    opt_summary = read_csv(paths["optuna_dir"] / "optimization_summary.csv")
    metadata_rows = load_final_metadata(paths["final_models_dir"])
    tradeoff = threshold_tradeoff_table(metadata_rows)

    fig1 = report_figures / "fig01_optuna_pr_auc_by_model.png"
    fig2 = report_figures / "fig02_optuna_f2_by_model.png"
    fig3 = report_figures / "fig03_thresholds_by_model.png"
    fig4 = report_figures / "fig04_final_models_threshold_tradeoff.png"
    save_bar_by_feature_set(opt_summary, "test_average_precision", "PR-AUC test", fig1)
    save_bar_by_feature_set(opt_summary, "best_f2", "Mejor F2 test", fig2)
    save_threshold_figure(opt_summary, fig3)
    save_tradeoff_figure(tradeoff, fig4)

    print(f"Figuras guardadas en {report_figures}")


if __name__ == "__main__":
    main()
