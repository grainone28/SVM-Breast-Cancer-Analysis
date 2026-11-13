"""Training and evaluation of SVMs with linear, RBF and polynomial kernels
(thesis, Sections 5.3-5.5).

For each of the 100 stratified Monte Carlo splits (70% train / 30% test):
  1. the hyperparameters of each kernel are tuned with 10-fold CV on the training set;
  2. the best model is evaluated on the held-out test set
     (accuracy, precision, recall, F1 and AUC, with "M" = malignant as positive class;
     the AUC is computed from the continuous SVM decision values);
  3. the number of support vectors and the selected hyperparameters are stored.

Usage:
    python -m src.train_evaluate [--n-splits 100] [--n-jobs -1]
"""

import argparse
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config
from src.data import get_features_and_target, load_data, monte_carlo_splits
from src.models import best_params, build_search, get_svc
from src.summarize import format_summary, hyperparameter_frequencies, summarize_metrics, support_vector_stats


def evaluate_split(split_id, train_idx, test_idx, X, y):
    """Tune and evaluate the three kernels on one train/test partition."""
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_test_bin = (y_test == config.POSITIVE_LABEL).astype(int)

    rows, roc_scores = [], {}
    for kernel, kernel_name in config.KERNEL_NAMES.items():
        search = build_search(kernel, seed=config.SEED + split_id).fit(X_train, y_train)
        model = search.best_estimator_
        svc = get_svc(model)
        params = best_params(search)

        y_pred = model.predict(X_test)
        # classes_ are sorted (["B", "M"]): positive decision values point to "M"
        scores = model.decision_function(X_test)
        n_sv = dict(zip(svc.classes_, svc.n_support_))

        rows.append({
            "kernel": kernel_name,
            "split": split_id + 1,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, pos_label=config.POSITIVE_LABEL, zero_division=0),
            "Recall": recall_score(y_test, y_pred, pos_label=config.POSITIVE_LABEL),
            "F1": f1_score(y_test, y_pred, pos_label=config.POSITIVE_LABEL),
            "AUC": roc_auc_score(y_test_bin, scores),
            "C": params.get("C"),
            "gamma": params.get("gamma"),
            "degree": params.get("degree"),
            "nSV_total": int(svc.n_support_.sum()),
            "nSV_B": int(n_sv[config.NEGATIVE_LABEL]),
            "nSV_M": int(n_sv[config.POSITIVE_LABEL]),
        })
        roc_scores[kernel_name] = (y_test_bin.to_numpy(), scores)
    return rows, roc_scores


def plot_roc_curves(roc_scores, out_path, split_id):
    fig, ax = plt.subplots(figsize=(6, 5))
    for kernel_name, (y_true, scores) in roc_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, lw=2, color=config.KERNEL_COLORS[kernel_name], label=f"{kernel_name} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    ax.set(title=f"ROC curves (test set of split {split_id})", xlabel="False positive rate (1 - specificity)",
           ylabel="True positive rate (sensitivity)")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_distribution(metrics, metric, out_path):
    order = [k for k in config.KERNEL_COLORS if k in set(metrics["kernel"])]
    data = [metrics.loc[metrics["kernel"] == k, metric] for k in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    box = ax.boxplot(data, patch_artist=True, widths=0.5)
    ax.set_xticks(range(1, len(order) + 1), order)
    for patch, k in zip(box["boxes"], order):
        patch.set_facecolor(config.KERNEL_COLORS[k])
        patch.set_alpha(0.7)
    ax.set(title=f"{metric} on the test sets of {metrics['split'].nunique()} MCCV splits", ylabel=metric)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(n_splits=config.N_SPLITS, n_jobs=-1):
    out_dir = config.RESULTS_DIR / "python"
    out_dir.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y = get_features_and_target(df)
    print(f"Features: {X.shape[1]} | splits: {n_splits}")

    splits = monte_carlo_splits(y, n_splits=n_splits)
    start = time.time()
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_split)(s, tr, te, X, y) for s, (tr, te) in enumerate(splits)
    )
    print(f"Done in {time.time() - start:.1f} s")

    metrics = pd.DataFrame([row for rows, _ in results for row in rows])
    summary = summarize_metrics(metrics)
    sv_stats = support_vector_stats(metrics)
    hp_freq = hyperparameter_frequencies(metrics)

    metrics.to_csv(out_dir / "mccv_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    sv_stats.to_csv(out_dir / "support_vectors.csv", index=False)
    hp_freq.to_csv(out_dir / "hyperparameters.csv", index=False)

    plot_roc_curves(results[-1][1], config.FIGURES_DIR / "roc_curves.png", split_id=n_splits)
    plot_metric_distribution(metrics, "F1", config.FIGURES_DIR / "f1_distribution.png")

    print("\nTest-set metrics - mean ± standard error\n")
    print(format_summary(summary).to_string())
    print("\nSupport vectors\n")
    print(sv_stats.to_string(index=False))
    print("\nSelected hyperparameters (counts)\n")
    print(hp_freq.to_string(index=False))
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-splits", type=int, default=config.N_SPLITS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    main(args.n_splits, args.n_jobs)
