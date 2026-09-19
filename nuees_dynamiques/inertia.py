# src/nuees_dynamiques/inertia.py

from .distances import euclidean_distance_squared


def compute_inertia(X, labels, prototypes, distance_fn=euclidean_distance_squared):
    W = 0.0
    for i in range(X.shape[0]):
        W += distance_fn(X[i], prototypes[labels[i]])
    return float(W)


def inertia_converged(old_W, new_W, tol=1e-4):
    var = abs(old_W - new_W)
    return var < tol, var