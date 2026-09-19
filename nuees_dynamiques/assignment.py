# src/nuees_dynamiques/assignment.py

import numpy as np
from .distances import euclidean_distance_squared


def assign_clusters(X, prototypes, distance_fn=euclidean_distance_squared):
    n = X.shape[0]
    k = prototypes.shape[0]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = distance_fn(X[i], prototypes[0])
        for j in range(1, k):
            d = distance_fn(X[i], prototypes[j])
            if d < best_d:
                best_d = d
                best_j = j
        labels[i] = best_j
    return labels