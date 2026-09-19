# src/nuees_dynamiques/update.py

import numpy as np
from .distances import euclidean_distance_squared


def _reinit_empty(X, labels, k, old_prototypes, j, distance_fn, d):
    """Réinitialise un prototype vide au point le plus éloigné."""
    if old_prototypes is not None:
        worst_i = 0
        worst_d = -1.0
        for i in range(X.shape[0]):
            dist = distance_fn(X[i], old_prototypes[j])
            if dist > worst_d:
                worst_d = dist
                worst_i = i
        return X[worst_i].copy()
    return np.zeros(d, dtype=float)


def update_prototypes_centroide(X, labels, k, old_prototypes,
                                distance_fn=euclidean_distance_squared):
    n, d = X.shape
    new_protos = np.zeros((k, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            new_protos[j] = _reinit_empty(X, labels, k, old_prototypes, j,
                                          distance_fn, d)
        else:
            mean = np.zeros(d, dtype=float)
            for i in idx:
                for l in range(d):
                    mean[l] += X[i, l]
            mean /= len(idx)
            new_protos[j] = mean
    return new_protos, empty


def update_prototypes_mediane(X, labels, k, old_prototypes,
                              distance_fn=euclidean_distance_squared):
    n, d = X.shape
    new_protos = np.zeros((k, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            new_protos[j] = _reinit_empty(X, labels, k, old_prototypes, j,
                                          distance_fn, d)
        else:
            med = np.zeros(d, dtype=float)
            for l in range(d):
                col = sorted([X[i, l] for i in idx])
                m = len(col)
                if m % 2 == 0:
                    med[l] = 0.5 * (col[m // 2 - 1] + col[m // 2])
                else:
                    med[l] = col[m // 2]
            new_protos[j] = med
    return new_protos, empty


def update_prototypes_medoide(X, labels, k, old_prototypes,
                              distance_fn=euclidean_distance_squared):
    n, d = X.shape
    new_protos = np.zeros((k, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            new_protos[j] = _reinit_empty(X, labels, k, old_prototypes, j,
                                          distance_fn, d)
        else:
            best_i = idx[0]
            best_cost = float("inf")
            for i in idx:
                cost = 0.0
                for m in idx:
                    cost += distance_fn(X[i], X[m])
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            new_protos[j] = X[best_i].copy()
    return new_protos, empty


def update_prototypes(X, labels, k, old_prototypes,
                      prototype_type="centroide",
                      distance_fn=euclidean_distance_squared):
    pt = prototype_type.lower()
    if pt in ("centroide", "centroïde", "mean"):
        return update_prototypes_centroide(X, labels, k, old_prototypes, distance_fn)
    elif pt in ("mediane", "médiane", "median"):
        return update_prototypes_mediane(X, labels, k, old_prototypes, distance_fn)
    elif pt in ("medoide", "médoïde", "medoid"):
        return update_prototypes_medoide(X, labels, k, old_prototypes, distance_fn)
    raise ValueError(f"prototype_type inconnu : {prototype_type}")


def prototypes_have_moved(old_p, new_p, tol=1e-4):
    k, d = old_p.shape
    max_shift = 0.0
    for j in range(k):
        s = 0.0
        for l in range(d):
            diff = old_p[j, l] - new_p[j, l]
            s += diff * diff
        s = float(np.sqrt(s))
        if s > max_shift:
            max_shift = s
    return max_shift >= tol, max_shift