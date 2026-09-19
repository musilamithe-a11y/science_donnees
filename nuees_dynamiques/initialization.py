# src/nuees_dynamiques/initialization.py

import numpy as np
from .distances import euclidean_distance_squared


def init_random(X, k, seed=None):
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) > n ({n}).")
    rng = np.random.RandomState(seed)
    idx = rng.choice(n, size=k, replace=False)
    return X[idx].copy()


def init_kmpp(X, k, seed=None, distance_fn=euclidean_distance_squared):
    n = X.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) > n ({n}).")
    rng = np.random.RandomState(seed)
    centers = [X[rng.randint(0, n)].copy()]
    for _ in range(1, k):
        d2 = np.zeros(n, dtype=float)
        for i in range(n):
            best = distance_fn(X[i], centers[0])
            for c in centers[1:]:
                d = distance_fn(X[i], c)
                if d < best:
                    best = d
            d2[i] = best
        total = float(np.sum(d2))
        if total <= 0.0:
            idx = rng.randint(0, n)
        else:
            probs = d2 / total
            idx = rng.choice(n, p=probs)
        centers.append(X[idx].copy())
    return np.array(centers)


def init_uniform(X, k, seed=None):
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)
    centers = np.zeros((k, d), dtype=float)
    for j in range(k):
        for l in range(d):
            centers[j, l] = rng.uniform(mins[l], maxs[l])
    return centers


def get_init_fn(name):
    name = name.lower()
    if name in ("random", "aleatoire", "aléatoire"):
        return init_random
    elif name in ("kmpp", "kmeans++", "k-means++"):
        return init_kmpp
    elif name in ("uniform", "uniforme"):
        return init_uniform
    raise ValueError(f"Initialisation inconnue : {name}")