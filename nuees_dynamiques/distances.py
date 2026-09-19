# src/nuees_dynamiques/distances.py

import numpy as np


def euclidean_distance(x, c):
    s = 0.0
    for l in range(len(x)):
        diff = x[l] - c[l]
        s += diff * diff
    return float(np.sqrt(s))


def euclidean_distance_squared(x, c):
    s = 0.0
    for l in range(len(x)):
        diff = x[l] - c[l]
        s += diff * diff
    return float(s)


def manhattan_distance(x, c):
    s = 0.0
    for l in range(len(x)):
        s += abs(x[l] - c[l])
    return float(s)


def minkowski_distance(x, c, p=3.0):
    s = 0.0
    for l in range(len(x)):
        s += abs(x[l] - c[l]) ** p
    return float(s ** (1.0 / p))


def compute_distance_matrix(X, prototypes, distance_fn=euclidean_distance_squared):
    n = X.shape[0]
    k = prototypes.shape[0]
    D = np.zeros((n, k), dtype=float)
    for i in range(n):
        for j in range(k):
            D[i, j] = distance_fn(X[i], prototypes[j])
    return D