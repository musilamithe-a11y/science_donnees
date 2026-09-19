# src/nuees_dynamiques/nuees.py

import numpy as np
from .distances import euclidean_distance_squared
from .update import update_prototypes


# -------- point --------

def update_nuee_point(X, labels, k, old, prototype_type="centroide",
                      distance_fn=euclidean_distance_squared):
    return update_prototypes(X, labels, k, old,
                             prototype_type=prototype_type,
                             distance_fn=distance_fn)


def assign_nuee_point(X, nuees, distance_fn=euclidean_distance_squared):
    n = X.shape[0]
    k = nuees.shape[0]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = distance_fn(X[i], nuees[0])
        for j in range(1, k):
            d = distance_fn(X[i], nuees[j])
            if d < best_d:
                best_d = d
                best_j = j
        labels[i] = best_j
    return labels


# -------- ensemble_points --------

def update_nuee_ensemble_points(X, labels, k, old, m=3,
                                distance_fn=euclidean_distance_squared):
    n, d = X.shape
    new = np.zeros((k, m, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            if old is not None:
                new[j] = old[j]
            continue
        # centroïde
        c = np.zeros(d, dtype=float)
        for i in idx:
            for l in range(d):
                c[l] += X[i, l]
        c /= len(idx)
        # trier par distance
        dists = sorted([(i, distance_fn(X[i], c)) for i in idx], key=lambda t: t[1])
        for r in range(m):
            if r < len(dists):
                new[j, r] = X[dists[r][0]].copy()
            else:
                new[j, r] = c.copy()
    return new, empty


def assign_nuee_ensemble_points(X, nuees, distance_fn=euclidean_distance_squared):
    n = X.shape[0]
    k, m, d = nuees.shape
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            for r in range(m):
                d = distance_fn(X[i], nuees[j, r])
                if d < best_d:
                    best_d = d
                    best_j = j
        labels[i] = best_j
    return labels


# -------- distribution --------

def update_nuee_distribution(X, labels, k, old, reg=1e-6):
    n, d = X.shape
    means = np.zeros((k, d), dtype=float)
    covs = np.zeros((k, d, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            if old is not None:
                means[j] = old["means"][j]
                covs[j] = old["covs"][j]
            else:
                covs[j] = np.eye(d)
            continue
        mean = np.zeros(d, dtype=float)
        for i in idx:
            for l in range(d):
                mean[l] += X[i, l]
        mean /= len(idx)
        means[j] = mean
        cov = np.zeros((d, d), dtype=float)
        for i in idx:
            diff = X[i] - mean
            for a in range(d):
                for b in range(d):
                    cov[a, b] += diff[a] * diff[b]
        if len(idx) > 1:
            cov /= (len(idx) - 1)
        else:
            cov = np.eye(d)
        cov += reg * np.eye(d)
        covs[j] = cov
    return {"means": means, "covs": covs}, empty


def assign_nuee_distribution(X, nuees):
    n = X.shape[0]
    k = nuees["means"].shape[0]
    d = X.shape[1]
    invs = []
    for j in range(k):
        try:
            invs.append(np.linalg.inv(nuees["covs"][j]))
        except np.linalg.LinAlgError:
            invs.append(np.eye(d))
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            diff = X[i] - nuees["means"][j]
            d2 = float(diff @ invs[j] @ diff)
            if d2 < best_d:
                best_d = d2
                best_j = j
        labels[i] = best_j
    return labels


# -------- axe_factoriel --------

def update_nuee_axe(X, labels, k, old):
    n, d = X.shape
    points = np.zeros((k, d), dtype=float)
    dirs = np.zeros((k, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) < 2:
            empty.append(j)
            if old is not None:
                points[j] = old["points"][j]
                dirs[j] = old["dirs"][j]
            continue
        mean = np.zeros(d, dtype=float)
        for i in idx:
            for l in range(d):
                mean[l] += X[i, l]
        mean /= len(idx)
        cov = np.zeros((d, d), dtype=float)
        for i in idx:
            diff = X[i] - mean
            for a in range(d):
                for b in range(d):
                    cov[a, b] += diff[a] * diff[b]
        cov /= len(idx)
        eigvals, eigvecs = np.linalg.eigh(cov)
        direction = eigvecs[:, -1]
        points[j] = mean
        dirs[j] = direction
    return {"points": points, "dirs": dirs}, empty


def assign_nuee_axe(X, nuees):
    n = X.shape[0]
    k = nuees["points"].shape[0]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            diff = X[i] - nuees["points"][j]
            u = nuees["dirs"][j]
            proj = float(diff @ u)
            orth = diff - proj * u
            d2 = float(orth @ orth)
            if d2 < best_d:
                best_d = d2
                best_j = j
        labels[i] = best_j
    return labels


# -------- structure --------

def update_nuee_structure(X, labels, k, old):
    n, d = X.shape
    new = np.zeros((k, 2, d), dtype=float)
    empty = []
    for j in range(k):
        idx = [i for i in range(n) if labels[i] == j]
        if len(idx) == 0:
            empty.append(j)
            if old is not None:
                new[j] = old[j]
            continue
        best_pair = (idx[0], idx[0])
        best_d = -1.0
        for a in idx:
            for b in idx:
                if a < b:
                    diff = X[a] - X[b]
                    d2 = float(diff @ diff)
                    if d2 > best_d:
                        best_d = d2
                        best_pair = (a, b)
        new[j, 0] = X[best_pair[0]].copy()
        new[j, 1] = X[best_pair[1]].copy()
    return new, empty


def assign_nuee_structure(X, nuees):
    n = X.shape[0]
    k = nuees.shape[0]
    labels = np.zeros(n, dtype=int)
    for i in range(n):
        best_j = 0
        best_d = float("inf")
        for j in range(k):
            p1 = nuees[j, 0]
            p2 = nuees[j, 1]
            seg = p2 - p1
            sn = float(seg @ seg)
            if sn == 0:
                d2 = float((X[i] - p1) @ (X[i] - p1))
            else:
                t = float((X[i] - p1) @ seg) / sn
                t = max(0.0, min(1.0, t))
                proj = p1 + t * seg
                d2 = float((X[i] - proj) @ (X[i] - proj))
            if d2 < best_d:
                best_d = d2
                best_j = j
        labels[i] = best_j
    return labels


# -------- dispatcher --------

NUEES_TYPES = ["point", "ensemble_points", "distribution",
               "axe_factoriel", "structure"]


def update_nuees(X, labels, k, old, nuee_type="point",
                 prototype_type="centroide",
                 distance_fn=euclidean_distance_squared,
                 **kwargs):
    nt = nuee_type.lower()
    if nt == "point":
        return update_nuee_point(X, labels, k, old, prototype_type, distance_fn)
    elif nt == "ensemble_points":
        return update_nuee_ensemble_points(X, labels, k, old,
                                           m=kwargs.get("m", 3),
                                           distance_fn=distance_fn)
    elif nt == "distribution":
        return update_nuee_distribution(X, labels, k, old)
    elif nt == "axe_factoriel":
        return update_nuee_axe(X, labels, k, old)
    elif nt == "structure":
        return update_nuee_structure(X, labels, k, old)
    raise ValueError(f"nuee_type inconnu : {nuee_type}")


def assign_nuees(X, nuees, nuee_type="point",
                 distance_fn=euclidean_distance_squared):
    nt = nuee_type.lower()
    if nt == "point":
        return assign_nuee_point(X, nuees, distance_fn)
    elif nt == "ensemble_points":
        return assign_nuee_ensemble_points(X, nuees, distance_fn)
    elif nt == "distribution":
        return assign_nuee_distribution(X, nuees)
    elif nt == "axe_factoriel":
        return assign_nuee_axe(X, nuees)
    elif nt == "structure":
        return assign_nuee_structure(X, nuees)
    raise ValueError(f"nuee_type inconnu : {nuee_type}")