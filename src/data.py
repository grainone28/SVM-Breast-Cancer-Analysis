"""Data loading, cleaning and preprocessing for the Breast Cancer Wisconsin (Diagnostic) dataset."""

import numpy as np
import pandas as pd

from src import config


def load_data(path=config.DATA_PATH):
    """Load the WDBC csv and drop the empty trailing column.

    The original file ends with a column made only of missing values
    (`X` in R, `Unnamed: 32` in pandas), which carries no information.
    """
    df = pd.read_csv(path)
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    df = df.drop(columns=empty_cols)

    if df.isna().sum().sum() != 0:
        raise ValueError("Unexpected missing values in the dataset.")
    return df


def get_features_and_target(df):
    """Split the dataframe into the 30 morphological features X and the target y.

    The sample identifier `id` is not a feature and is removed. Standardisation is
    not applied here: it is part of the model pipeline, so that it is fitted on
    the training data only.
    """
    y = df[config.TARGET_COLUMN]
    X = df.drop(columns=[config.TARGET_COLUMN, config.ID_COLUMN])
    return X, y


def stratified_partition(y, train_fraction, rng):
    """Return train/test positional indices of one stratified random split.

    Mirrors caret::createDataPartition: for each class, ceil(p * n_class)
    observations are sampled for training and the rest go to the test set.
    """
    y = np.asarray(y)
    train_idx = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n_train = int(np.ceil(train_fraction * len(idx)))
        train_idx.append(rng.choice(idx, size=n_train, replace=False))
    train_idx = np.sort(np.concatenate(train_idx))
    test_idx = np.setdiff1d(np.arange(len(y)), train_idx)
    return train_idx, test_idx


def monte_carlo_splits(y, n_splits=config.N_SPLITS, train_fraction=config.TRAIN_FRACTION, seed=config.SEED):
    """Generate the stratified Monte Carlo cross-validation partitions."""
    rng = np.random.default_rng(seed)
    return [stratified_partition(y, train_fraction, rng) for _ in range(n_splits)]
