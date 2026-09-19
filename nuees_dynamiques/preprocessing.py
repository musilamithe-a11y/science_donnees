# src/nuees_dynamiques/preprocessing.py

import numpy as np


def standardize(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def normalize_minmax(X):
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    rng = maxs - mins
    rng[rng == 0] = 1.0
    return (X - mins) / rng


def load_sklearn_dataset(name="iris"):
    """
    Charge un dataset sklearn et le retourne sous forme (X, y).
    Utilisé uniquement pour les tests / vérifications rapides.
    """
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    name = name.lower()
    if name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    elif name == "breast_cancer":
        d = load_breast_cancer()
    else:
        raise ValueError(f"Dataset inconnu : {name}")
    return d.data.astype(float), d.target.astype(int)