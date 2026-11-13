"""Aggregate the per-split results of the Monte Carlo cross-validation.

Usage (summarise the results produced by the original R script):
    python -m src.summarize
"""

import numpy as np
import pandas as pd

from src import config

METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
KERNEL_ORDER = ["Linear", "RBF", "Polynomial"]


def _order(df):
    return df.set_index("kernel").loc[[k for k in KERNEL_ORDER if k in set(df["kernel"])]].reset_index()


def summarize_metrics(metrics):
    """Mean and standard error (std / sqrt(n_splits)) of every metric, per kernel."""
    cols = [m for m in METRICS if m in metrics.columns]
    grouped = metrics.groupby("kernel")[cols]
    mean = grouped.mean().add_suffix("_mean")
    se = (grouped.std() / np.sqrt(grouped.count())).add_suffix("_se")
    summary = pd.concat([mean, se], axis=1)
    summary = summary[[f"{m}_{s}" for m in cols for s in ("mean", "se")]].reset_index()
    return _order(summary)


def format_summary(summary):
    """Human readable 'mean ± se' table."""
    cols = [m for m in METRICS if f"{m}_mean" in summary.columns]
    table = pd.DataFrame({"kernel": summary["kernel"]})
    for m in cols:
        table[m] = [f"{a:.3f} ± {b:.4f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_se"])]
    return table.set_index("kernel")


def support_vector_stats(metrics):
    stats = metrics.groupby("kernel")["nSV_total"].agg(["mean", "min", "max"]).reset_index()
    return _order(stats)


def hyperparameter_frequencies(metrics):
    """How many times each hyperparameter value was selected across the splits."""
    rows = []
    for kernel, group in metrics.groupby("kernel"):
        for param in ["C", "gamma", "degree"]:
            values = group[param].dropna()
            for value, count in values.value_counts().sort_index().items():
                rows.append({"kernel": kernel, "parameter": param, "value": value, "count": count})
    return pd.DataFrame(rows)


def load_r_metrics(path=config.RESULTS_DIR / "r_original" / "mccv_metrics_R.csv"):
    """Read the csv written by the R script and align it to the Python column names."""
    df = pd.read_csv(path)
    df = df.rename(columns={"modello": "kernel", "Cost": "C", "Gamma": "gamma", "Degree": "degree",
                            "nSV_class1": "nSV_B", "nSV_class2": "nSV_M"})
    df["kernel"] = df["kernel"].replace({"Lineare": "Linear", "Polinomiale": "Polynomial"})
    return df


def main():
    metrics = load_r_metrics()
    out_dir = config.RESULTS_DIR / "r_original"
    summary = summarize_metrics(metrics)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print("Original R results (100 MCCV splits) - mean ± standard error\n")
    print(format_summary(summary).to_string())
    print("\nSupport vectors\n")
    print(support_vector_stats(metrics).to_string(index=False))
    print("\nSelected hyperparameters (counts)\n")
    print(hyperparameter_frequencies(metrics).to_string(index=False))


if __name__ == "__main__":
    main()
