"""Global configuration: paths, random seed, experimental settings and hyperparameter grids."""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "Cancer_Data.csv"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
ID_COLUMN = "id"
TARGET_COLUMN = "diagnosis"
POSITIVE_LABEL = "M"  # malignant: the clinically relevant class
NEGATIVE_LABEL = "B"  # benign
CLASS_COLORS = {"B": "steelblue", "M": "tomato"}

# --------------------------------------------------------------------------- #
# Experimental setup (same values as the original R script)
# --------------------------------------------------------------------------- #
SEED = 123
N_SPLITS = 100        # Monte Carlo cross-validation repetitions
TRAIN_FRACTION = 0.7  # stratified 70/30 train/test split
INNER_CV_FOLDS = 10   # folds used for hyperparameter tuning (e1071::tune default)

# Hyperparameter grids (C = `cost` in e1071).
# Kernels follow the libsvm parametrisation used by both e1071 and scikit-learn:
#   linear:      <x, x'>
#   rbf:         exp(-gamma * ||x - x'||^2)
#   polynomial:  (gamma * <x, x'> + coef0)^degree, with coef0 = 0
PARAM_GRIDS = {
    "linear": {"C": [0.01, 0.1, 1, 10, 100]},
    "rbf": {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]},
    "poly": {"C": [0.1, 1], "gamma": [0.01, 0.1], "degree": [2, 3, 4]},
}

KERNEL_NAMES = {"linear": "Linear", "rbf": "RBF", "poly": "Polynomial"}
KERNEL_COLORS = {"Linear": "steelblue", "RBF": "tomato", "Polynomial": "seagreen"}

# Linear SVM fitted on the first two principal components (thesis, Figure 10)
PCA_BOUNDARY_C = 0.1
