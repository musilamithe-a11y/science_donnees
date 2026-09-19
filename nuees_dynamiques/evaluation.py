# src/nuees_dynamiques/evaluation.py

import numpy as np
from .distances import euclidean_distance, euclidean_distance_squared


def silhouette_score(X, labels, distance_fn=euclidean_distance):
    n = X.shape[0]
    unique = np.unique(labels)
    k = len(unique)
    if k < 2:
        raise ValueError("Silhouette nécessite au moins 2 clusters.")
    if k == n:
        raise ValueError("Silhouette indéfinie si k = n.")

    clusters = {lab: [] for lab in unique}
    for i in range(n):
        clusters[labels[i]].append(i)

    s = np.zeros(n, dtype=float)
    for i in range(n):
        lab_i = labels[i]
        same = clusters[lab_i]
        if len(same) <= 1:
            a_i = 0.0
        else:
            tot = 0.0
            for j in same:
                if j != i:
                    tot += distance_fn(X[i], X[j])
            a_i = tot / (len(same) - 1)
        b_i = float("inf")
        for lab_o, members in clusters.items():
            if lab_o == lab_i or len(members) == 0:
                continue
            tot = 0.0
            for j in members:
                tot += distance_fn(X[i], X[j])
            avg = tot / len(members)
            if avg < b_i:
                b_i = avg
        denom = max(a_i, b_i)
        s[i] = 0.0 if denom == 0 else (b_i - a_i) / denom
    return float(np.mean(s))


def davies_bouldin_score(X, labels, prototypes, distance_fn=euclidean_distance):
    n = X.shape[0]
    k = prototypes.shape[0]

    clusters = {j: [] for j in range(k)}
    for i in range(n):
        clusters[labels[i]].append(i)

    S = np.zeros(k, dtype=float)
    for j in range(k):
        members = clusters[j]
        if len(members) == 0:
            continue
        tot = 0.0
        for i in members:
            tot += distance_fn(X[i], prototypes[j])
        S[j] = tot / len(members)

    db_vals = np.zeros(k, dtype=float)
    for i in range(k):
        max_r = 0.0
        for j in range(k):
            if i == j:
                continue
            m_ij = distance_fn(prototypes[i], prototypes[j])
            if m_ij == 0:
                r_ij = 0.0
            else:
                r_ij = (S[i] + S[j]) / m_ij
            if r_ij > max_r:
                max_r = r_ij
        db_vals[i] = max_r
    return float(np.mean(db_vals))