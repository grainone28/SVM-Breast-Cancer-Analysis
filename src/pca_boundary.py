"""Linear SVM on the first two principal components (thesis, Section 5.3, Figure 10).

A linear SVM (C = 0.1) is trained on PC1 and PC2 only, so that its decision
boundary can be drawn on the plane. It is a visual aid, not the model used for
the evaluation. Standardisation and PCA are fitted on the training set only.

Usage:
    python -m src.pca_boundary
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config
from src.data import get_features_and_target, load_data, stratified_partition


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    X, y = get_features_and_target(df)

    rng = np.random.default_rng(config.SEED)
    train_idx, test_idx = stratified_partition(y, config.TRAIN_FRACTION, rng)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    projection = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(X_train)
    train_scores = projection.transform(X_train)
    test_scores = projection.transform(X_test)
    print(f"Variance explained by PC1 + PC2: {projection[-1].explained_variance_ratio_.sum():.2%}")

    model = SVC(kernel="linear", C=config.PCA_BOUNDARY_C).fit(train_scores, y_train)
    print(f"Test accuracy using only PC1 and PC2: {model.score(test_scores, y_test):.3f}")

    # Decision regions on a grid
    x_range = np.linspace(train_scores[:, 0].min(), train_scores[:, 0].max(), 300)
    y_range = np.linspace(train_scores[:, 1].min(), train_scores[:, 1].max(), 300)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    for label in ["B", "M"]:
        mask = y_train.to_numpy() == label
        ax.scatter(train_scores[mask, 0], train_scores[mask, 1], s=14, alpha=0.7,
                   color=config.CLASS_COLORS[label], label=label)
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors="black", linestyles=["--", "-", "--"], linewidths=1)
    ax.set(title="Linear separation between classes on the PC1-PC2 plane", xlabel="PC1", ylabel="PC2")
    ax.legend(title="Diagnosis")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out_path = config.FIGURES_DIR / "pca_linear_boundary.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
