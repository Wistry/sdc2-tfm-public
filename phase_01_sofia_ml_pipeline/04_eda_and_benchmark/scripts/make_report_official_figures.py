from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR.parent
    / "03_candidate_dataset"
    / "outputs"
    / "baseline_current_full"
    / "candidates_sofia_only.csv"
)

OUT_DIR = BASE_DIR / "outputs" / "eda" / "report_figures_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {-1: "Ambiguo", 0: "FP limpio", 1: "TP"}


# ============================================================
# Global style
# ============================================================

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 240,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.titlesize": 12,
})


# ============================================================
# Helpers
# ============================================================

def save_fig(fig: plt.Figure, filename: str) -> Path:
    path = OUT_DIR / filename
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def get_clean_df(df: pd.DataFrame) -> pd.DataFrame:
    clean = df[df["clean_label"].isin([0, 1])].copy()
    clean["class_name"] = clean["clean_label"].map({0: "FP limpio", 1: "TP"})
    return clean


def add_bar_labels(ax, total=None, top_margin=0.20):
    heights = [patch.get_height() for patch in ax.patches]
    if heights:
        ax.set_ylim(0, max(heights) * (1 + top_margin))

    for patch in ax.patches:
        h = patch.get_height()
        if total:
            text = f"{int(h)}\n({100 * h / total:.1f}%)"
        else:
            text = f"{int(h)}"

        ax.annotate(
            text,
            xy=(patch.get_x() + patch.get_width() / 2, h),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            clip_on=False,
        )


# ============================================================
# Class distribution figures
# ============================================================

def plot_class_distributions(df: pd.DataFrame) -> dict[str, Path]:
    paths = {}

    # Global distribution
    counts = df["clean_label"].value_counts().sort_index()
    labels = [LABEL_MAP[int(k)] for k in counts.index]

    fig, ax = plt.subplots(figsize=(5.4, 3.25))
    ax.bar(labels, counts.values, width=0.62)
    add_bar_labels(ax, total=len(df), top_margin=0.18)

    ax.set_title("Distribución global de etiquetas")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de candidatos")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    paths["class_all"] = save_fig(fig, "fig01_class_distribution_all.png")

    # Clean TP/FP distribution
    clean = get_clean_df(df)
    counts = clean["clean_label"].value_counts().sort_index()
    labels = [LABEL_MAP[int(k)] for k in counts.index]

    fig, ax = plt.subplots(figsize=(5.0, 3.1))
    ax.bar(labels, counts.values, width=0.55)
    add_bar_labels(ax, total=len(clean), top_margin=0.26)

    ax.set_title("Subconjunto limpio TP/FP")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Número de candidatos")
    ax.grid(axis="y", alpha=0.22)
    ax.set_axisbelow(True)

    paths["class_clean"] = save_fig(fig, "fig02_class_distribution_clean.png")

    return paths


# ============================================================
# Boxplot utilities
# ============================================================

def make_box_data(clean: pd.DataFrame, feature: str, transform: str | None = None):
    fp = clean.loc[clean["clean_label"] == 0, feature].dropna()
    tp = clean.loc[clean["clean_label"] == 1, feature].dropna()

    if transform == "log1p":
        fp = np.log1p(fp.clip(lower=0))
        tp = np.log1p(tp.clip(lower=0))

    return [fp, tp]


def draw_box(ax, data, title: str, ylabel: str):
    ax.boxplot(
        data,
        tick_labels=["FP limpio", "TP"],
        showfliers=True,
        widths=0.48,
        patch_artist=True,
        medianprops={"linewidth": 1.6, "color": "#e67e22"},
        boxprops={"linewidth": 1.0, "facecolor": "#7fb3d5", "alpha": 0.35},
        whiskerprops={"linewidth": 1.0},
        capprops={"linewidth": 1.0},
        flierprops={
            "marker": "o",
            "markersize": 2.8,
            "markerfacecolor": "none",
            "markeredgecolor": "#333333",
            "alpha": 0.55,
        },
    )

    # Diamond = mean. No numeric label, because the table already gives exact values.
    means = [np.mean(d) if len(d) else np.nan for d in data]
    for i, mean_value in enumerate(means, start=1):
        if np.isfinite(mean_value):
            ax.scatter(
                i,
                mean_value,
                marker="D",
                s=22,
                color="#ff7f0e",
                zorder=3,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.20)
    ax.set_axisbelow(True)


# ============================================================
# Feature separation figures
# ============================================================

def plot_feature_grid(df: pd.DataFrame) -> Path:
    clean = get_clean_df(df)

    specs = [
        ("w20", None, "w20", "Canales"),
        ("snr_max", None, "snr_max", "SNR máxima"),
        ("n_pix", "log1p", "log(1+n_pix)", "log(1 + voxels)"),
        ("f_sum", None, "f_sum", "Flujo integrado"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.3, 5.8))
    axes = axes.ravel()

    for ax, (feature, transform, title, ylabel) in zip(axes, specs):
        data = make_box_data(clean, feature, transform)
        draw_box(ax, data, title=title, ylabel=ylabel)

    fig.suptitle("Variables con separación TP/FP", y=0.995)
    fig.text(
        0.5,
        0.012,
        "El rombo naranja indica la media; la línea central de cada caja indica la mediana.",
        ha="center",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    return save_fig(fig, "fig04_feature_separation_grid.png")


# ============================================================
# Correlation heatmap
# ============================================================

def plot_correlation_heatmap(df: pd.DataFrame) -> Path:
    selected = [
        "n_pix",
        "f_sum",
        "snr",
        "snr_max",
        "w20",
        "w50",
        "wm50",
        "ell_maj",
        "ell_min",
        "rms",
        "err_f_sum",
        "err_z",
        "z",
        "freq",
        "x",
        "y",
    ]

    selected = [c for c in selected if c in df.columns]
    corr = df[selected].corr(numeric_only=True)

    n = len(selected)

    fig, ax = plt.subplots(figsize=(9.5, 8.2))

    im = ax.imshow(
        corr,
        cmap="RdBu",
        vmin=-1,
        vmax=1,
        aspect="equal",
    )

    ax.set_title(
        "Matriz de correlación - Variables SoFiA",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))

    ax.set_xticklabels(selected, rotation=90, ha="center", fontsize=8)
    ax.set_yticklabels(selected, fontsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(n):
        for j in range(n):
            value = corr.iloc[i, j]

            if abs(value) >= 0.995:
                text = "1" if value > 0 else "-1"
            else:
                text = f"{value:.2f}"

            text_color = "white" if abs(value) >= 0.70 else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )

    cbar = fig.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.16,
    )
    cbar.set_label("Correlación", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()

    return save_fig(fig, "fig09_correlation_heatmap_selected.png")

# ============================================================
# Main
# ============================================================

def main():
    df = pd.read_csv(DATA_PATH)

    if "clean_label" not in df.columns:
        raise ValueError("No existe la columna clean_label en el dataset.")

    paths = {}
    paths.update(plot_class_distributions(df))

    paths["feature_grid"] = plot_feature_grid(df)
    paths["corr"] = plot_correlation_heatmap(df)

    print(f"Figuras guardadas en: {OUT_DIR}")
    print()
    print("Figuras principales recomendadas:")
    print(f"- {paths['class_all']}")
    print(f"- {paths['class_clean']}")
    print(f"- {paths['feature_grid']}")
    print(f"- {paths['corr']}")


if __name__ == "__main__":
    main()
