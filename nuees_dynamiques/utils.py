# src/nuees_dynamiques/utils.py

import numpy as np


def set_seed(seed):
    np.random.seed(seed)


def validate_X(X):
    if not isinstance(X, np.ndarray):
        raise TypeError("X doit être un numpy.ndarray.")
    if X.ndim != 2:
        raise ValueError("X doit être 2D (n, d).")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X doit être numérique.")
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("X ne doit contenir ni NaN ni infini.")
    return True


def validate_labels(labels, n, k):
    if labels.shape != (n,):
        raise ValueError(f"labels doit avoir la forme ({n},).")
    if np.any(labels < 0) or np.any(labels >= k):
        raise ValueError(f"labels doit être entre 0 et {k-1}.")
    return True