from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from utils import (
    detect_label_column,
    detect_leakage_columns,
    ensure_dirs,
    get_numeric_feature_columns,
    load_candidates,
    load_config,
    plot_and_save,
    save_dataframe,
    save_json,
    split_clean_ambiguous,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EDA completo del dataset de candidatos SoFiA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--candidates-path", type=Path, default=None)
    return parser.parse_args()


def general_summary(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.columns.difference(numeric_cols)
    rows = [{
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_numeric_columns": len(numeric_cols),
        "n_categorical_columns": len(categorical_cols),
        "memory_mb": round(float(df.memory_usage(deep=True).sum()) / 1024**2, 4),
        "label_column": label_column,
        "duplicate_rows": int(df.duplicated().sum()),
    }]
    return pd.DataFrame(rows)


def class_distribution(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    counts = df[label_column].value_counts(dropna=False).sort_index()
    total = len(df)
    names = {-1: "ambiguous", 0: "FP", 1: "TP"}
    rows = []
    for label, count in counts.items():
        try:
            key = int(label) if pd.notna(label) and float(label).is_integer() else label
        except (TypeError, ValueError):
            key = label
        rows.append({
            "label": key,
            "class_name": names.get(key, "unknown"),
            "n": int(count),
            "percentage": 100.0 * int(count) / total if total else 0.0,
        })
    return pd.DataFrame(rows)


def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)
    rows = []
    for column in df.columns:
        n_missing = int(df[column].isna().sum())
        rows.append({
            "column": column,
            "missing_count": n_missing,
            "missing_percentage": 100.0 * n_missing / total if total else 0.0,
            "dtype": str(df[column].dtype),
        })
    return pd.DataFrame(rows).sort_values(["missing_count", "column"], ascending=[False, True])


def duplicates_report(df: pd.DataFrame) -> pd.DataFrame:
    position_sets = [
        ["x", "y", "z"],
        ["ra", "dec", "freq"],
        ["x_peak", "y_peak", "z_peak"],
    ]
    rows = [{"key": "full_row", "columns": "all", "duplicates": int(df.duplicated().sum())}]
    for columns in position_sets:
        existing = [column for column in columns if column in df.columns]
        if len(existing) == len(columns):
            rows.append({
                "key": "_".join(columns),
                "columns": ",".join(columns),
                "duplicates": int(df.duplicated(subset=columns).sum()),
            })
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame()
    summary = df[feature_columns].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary["skew"] = df[feature_columns].skew(numeric_only=True)
    summary["kurtosis"] = df[feature_columns].kurtosis(numeric_only=True)
    summary.insert(0, "column", summary.index)
    return summary.reset_index(drop=True)


def outlier_report(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    total = len(df)
    for column in feature_columns:
        values = df[column].dropna()
        if values.empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr if pd.notna(iqr) else np.nan
        upper = q3 + 1.5 * iqr if pd.notna(iqr) else np.nan
        count = int(((values < lower) | (values > upper)).sum()) if pd.notna(iqr) and iqr > 0 else 0
        rows.append({
            "column": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": count,
            "outlier_percentage": 100.0 * count / total if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["outlier_count", "column"], ascending=[False, True])


def top_correlations(df: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(feature_columns) < 2:
        return pd.DataFrame(), pd.DataFrame()
    corr = df[feature_columns].corr(numeric_only=True)
    rows = []
    columns = corr.columns.tolist()
    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = corr.loc[col_a, col_b]
            if pd.notna(value):
                rows.append({"feature_a": col_a, "feature_b": col_b, "correlation": value, "abs_correlation": abs(value)})
    pairs = pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)
    return corr, pairs


def feature_target_report(clean_df: pd.DataFrame, feature_columns: list[str], label_column: str) -> pd.DataFrame:
    if clean_df.empty:
        return pd.DataFrame()
    rows = []
    positives = clean_df[clean_df[label_column] == 1]
    negatives = clean_df[clean_df[label_column] == 0]
    for column in feature_columns:
        pos = positives[column].dropna()
        neg = negatives[column].dropna()
        if pos.empty or neg.empty:
            continue
        pooled = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
        cohen_d = (pos.mean() - neg.mean()) / pooled if pooled and pd.notna(pooled) else np.nan
        rows.append({
            "column": column,
            "mean_tp": pos.mean(),
            "mean_fp": neg.mean(),
            "mean_diff_tp_minus_fp": pos.mean() - neg.mean(),
            "cohen_d": cohen_d,
        })
    report = pd.DataFrame(rows)
    if not report.empty:
        report["abs_cohen_d"] = report["cohen_d"].abs()
        report = report.sort_values("abs_cohen_d", ascending=False)
    return report


def mutual_information_report(clean_df: pd.DataFrame, feature_columns: list[str], label_column: str, random_state: int) -> pd.DataFrame:
    if clean_df.empty or not feature_columns:
        return pd.DataFrame()
    X = clean_df[feature_columns].copy()
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    y = clean_df[label_column].astype(int)
    scores = mutual_info_classif(X, y, random_state=random_state)
    return pd.DataFrame({"column": feature_columns, "mutual_information": scores}).sort_values("mutual_information", ascending=False)


def make_plots(df: pd.DataFrame, clean_df: pd.DataFrame, label_column: str, eda_dir: Path, missing_df: pd.DataFrame, feature_report: pd.DataFrame, corr: pd.DataFrame) -> list[str]:
    figures: list[str] = []
    fig, ax = plt.subplots(figsize=(6, 4))
    df[label_column].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Distribucion de clean_label")
    ax.set_xlabel(label_column)
    ax.set_ylabel("n")
    path = eda_dir / "class_distribution_all.png"
    plot_and_save(fig, path)
    figures.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    clean_df[label_column].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Dataset limpio TP/FP")
    ax.set_xlabel(label_column)
    ax.set_ylabel("n")
    path = eda_dir / "class_distribution_clean.png"
    plot_and_save(fig, path)
    figures.append(str(path))

    top_missing = missing_df[missing_df["missing_count"] > 0].head(20)
    if not top_missing.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        top_missing.sort_values("missing_count").plot.barh(x="column", y="missing_count", ax=ax, legend=False)
        ax.set_title("Top missing columns")
        path = eda_dir / "top_missing_columns.png"
        plot_and_save(fig, path)
        figures.append(str(path))

    top_features = feature_report.head(8)["column"].tolist() if not feature_report.empty else []
    for column in top_features:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label, group in clean_df.groupby(label_column):
            group[column].dropna().plot(kind="hist", bins=30, alpha=0.45, ax=ax, label=str(label))
        ax.set_title(f"Histograma por clase: {column}")
        ax.legend()
        path = eda_dir / f"hist_by_class_{column}.png"
        plot_and_save(fig, path)
        figures.append(str(path))

        fig, ax = plt.subplots(figsize=(6, 4))
        clean_df.boxplot(column=column, by=label_column, ax=ax)
        ax.set_title(f"Boxplot por clase: {column}")
        fig.suptitle("")
        path = eda_dir / f"box_by_class_{column}.png"
        plot_and_save(fig, path)
        figures.append(str(path))

    if not corr.empty:
        cols = feature_report.head(15)["column"].tolist() if not feature_report.empty else corr.columns[:15].tolist()
        cols = [column for column in cols if column in corr.columns]
        if len(cols) >= 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            image = ax.imshow(corr.loc[cols, cols], cmap="coolwarm", vmin=-1, vmax=1)
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=90)
            ax.set_yticklabels(cols)
            fig.colorbar(image, ax=ax)
            ax.set_title("Heatmap correlaciones")
            path = eda_dir / "correlation_heatmap.png"
            plot_and_save(fig, path)
            figures.append(str(path))
    return figures


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_dirs(config)
    eda_dir = paths["eda_dir"]
    random_state = int(config.get("experiment", {}).get("random_state", 42))

    df = load_candidates(config, args.candidates_path)
    label_column = detect_label_column(df, config.get("data", {}).get("label_column", "clean_label"))
    clean_df, ambiguous_df = split_clean_ambiguous(
        df,
        label_column,
        int(config["data"].get("positive_label", 1)),
        int(config["data"].get("negative_label", 0)),
        int(config["data"].get("ambiguous_label", -1)),
    )

    leakage_df = detect_leakage_columns(df, label_column)
    feature_columns = get_numeric_feature_columns(df, label_column, leakage_df)

    summary_df = general_summary(df, label_column)
    class_df = class_distribution(df, label_column)
    missing_df = missing_values(df)
    duplicates_df = duplicates_report(df)
    numeric_df = numeric_summary(df, feature_columns)
    outliers_df = outlier_report(df, feature_columns)
    corr, corr_pairs = top_correlations(df, feature_columns)
    target_df = feature_target_report(clean_df, feature_columns, label_column)
    mi_df = mutual_information_report(clean_df, feature_columns, label_column, random_state)

    save_dataframe(summary_df, eda_dir / "general_summary.csv")
    save_dataframe(class_df, eda_dir / "class_distribution.csv")
    save_dataframe(missing_df, eda_dir / "missing_values.csv")
    save_dataframe(duplicates_df, eda_dir / "duplicates.csv")
    save_dataframe(numeric_df, eda_dir / "numeric_summary.csv")
    save_dataframe(outliers_df, eda_dir / "outliers_iqr.csv")
    save_dataframe(leakage_df, eda_dir / "leakage_candidates.csv")
    save_dataframe(corr, eda_dir / "correlation_matrix.csv")
    save_dataframe(corr_pairs, eda_dir / "top_correlated_pairs.csv")
    save_dataframe(target_df, eda_dir / "feature_target_effects.csv")
    save_dataframe(mi_df, eda_dir / "mutual_information.csv")
    save_dataframe(pd.DataFrame({"feature": feature_columns}), eda_dir / "numeric_features_non_leakage.csv")

    figures = make_plots(df, clean_df, label_column, eda_dir, missing_df, target_df, corr)
    save_json({
        "n_rows": int(len(df)),
        "n_clean_rows": int(len(clean_df)),
        "n_ambiguous_rows": int(len(ambiguous_df)),
        "n_feature_columns": int(len(feature_columns)),
        "figures": figures,
    }, eda_dir / "eda_summary.json")

    print(f"EDA guardado en {eda_dir}")


if __name__ == "__main__":
    main()
