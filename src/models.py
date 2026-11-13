"""SVM model definitions and hyperparameter search."""

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config


def build_pipeline(kernel, **svc_params):
    """Standardisation followed by an SVM with the given kernel."""
    return Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel=kernel, **svc_params))])


def build_search(kernel, seed=config.SEED):
    """Return a GridSearchCV that tunes an SVM with the given kernel.

    10-fold stratified cross-validation on the training set, choosing the
    configuration with the highest accuracy (lowest classification error, as
    e1071::tune() in the original R analysis). The scaler is inside the
    pipeline, so it is re-fitted on each training fold.
    """
    param_grid = {f"svc__{name}": values for name, values in config.PARAM_GRIDS[kernel].items()}
    cv = StratifiedKFold(n_splits=config.INNER_CV_FOLDS, shuffle=True, random_state=seed)
    return GridSearchCV(build_pipeline(kernel), param_grid, cv=cv, scoring="accuracy", n_jobs=1)


def get_svc(fitted_pipeline):
    """Return the fitted SVC at the end of the pipeline."""
    return fitted_pipeline[-1]


def best_params(search):
    """Best hyperparameters without the pipeline prefix."""
    return {name.split("__")[-1]: value for name, value in search.best_params_.items()}
