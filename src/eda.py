"""Exploratory data analysis (thesis, Section 5.2).

Produces the class distribution, the PCA projection on the first two components,
the boxplot of `radius_mean` by diagnosis and the correlation heatmap of the
"mean" features.

Usage:
    python -m src.eda
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.data import get_features_and_target, load_data


def plot_class_distribution(df, out_path):
    counts = df[config.TARGET_COLUMN].value_counts().reindex(["B", "M"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.index, counts.values, color=[config.CLASS_COLORS[c] for c in counts.index])
    for i, v in enumerate(counts.values):
        ax.text(i, v + 5, str(v), ha="center")
    ax.set(title="Diagnosis distribution", xlabel="Diagnosis (B = benign, M = malignant)", ylabel="Count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_pca_projection(df, out_path):
    """PCA on the 30 standardised morphological features (id and diagnosis excluded)."""
    X, _ = get_features_and_target(df)
    pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
    scores = pipe.fit_transform(X)
    pca = pipe[-1]

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in ["B", "M"]:
        mask = df[config.TARGET_COLUMN] == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=14, alpha=0.7,
                   color=config.CLASS_COLORS[label], label=label)
    ev = pca.explained_variance_ratio_ * 100
    ax.set(title="PCA - projection on the first two principal components",
           xlabel=f"PC1 ({ev[0]:.1f}%)", ylabel=f"PC2 ({ev[1]:.1f}%)")
    ax.legend(title="Diagnosis")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_radius_boxplot(df, out_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x=config.TARGET_COLUMN, y="radius_mean", order=["B", "M"],
                hue=config.TARGET_COLUMN, palette=config.CLASS_COLORS, legend=False, ax=ax)
    ax.set(title="radius_mean by diagnosis", xlabel="Diagnosis", ylabel="radius_mean")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(df, out_path):
    """Clustered correlation heatmap of the ten `_mean` features (as R's heatmap())."""
    corr = df.filter(like="_mean").corr()
    grid = sns.clustermap(corr, cmap="bwr", vmin=-1, vmax=1, figsize=(8, 8),
                          annot=True, fmt=".2f", annot_kws={"size": 7})
    grid.figure.suptitle("Correlation between the mean features", y=1.02)
    grid.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(grid.figure)


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print(f"Shape after cleaning: {df.shape}")
    counts = df[config.TARGET_COLUMN].value_counts()
    print("\nDiagnosis distribution:")
    print(pd.concat([counts, (counts / len(df) * 100).round(1)], axis=1, keys=["n", "%"]))
    print("\nDescriptive statistics (first features):")
    print(df.drop(columns=[config.ID_COLUMN]).describe().T.head(10).round(3))

    plot_class_distribution(df, config.FIGURES_DIR / "diagnosis_distribution.png")
    plot_pca_projection(df, config.FIGURES_DIR / "pca_projection.png")
    plot_radius_boxplot(df, config.FIGURES_DIR / "radius_mean_boxplot.png")
    plot_correlation_heatmap(df, config.FIGURES_DIR / "correlation_heatmap.png")
    print(f"\nFigures saved to {config.FIGURES_DIR}")


if __name__ == "__main__":
    main()
