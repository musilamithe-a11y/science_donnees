# src/nuees_dynamiques/algorithm.py

import numpy as np
from .distances import euclidean_distance_squared
from .initialization import get_init_fn
from .nuees import update_nuees, assign_nuees, NUEES_TYPES
from .inertia import compute_inertia
from .convergence import check_convergence
from .utils import validate_X, validate_labels


def nuees_dynamiques(X, k, max_iter=100, tol=1e-4, seed=None,
                     init_method="kmpp",
                     nuee_type="point",
                     prototype_type="centroide",
                     distance_fn=euclidean_distance_squared,
                     verbose=False, m=3):
    validate_X(X)
    n, d = X.shape
    if k <= 0:
        raise ValueError("k doit être > 0.")
    if k > n:
        raise ValueError(f"k ({k}) > n ({n}).")
    if nuee_type not in NUEES_TYPES:
        raise ValueError(f"nuee_type doit être dans {NUEES_TYPES}")

    # init : on part toujours de k points
    init_fn = get_init_fn(init_method)
    points = init_fn(X, k, seed=seed)

    # adapter les nuées au type
    if nuee_type == "point":
        nuees = points
    elif nuee_type == "ensemble_points":
        nuees = np.zeros((k, m, d), dtype=float)
        for j in range(k):
            for r in range(m):
                nuees[j, r] = points[j]
    elif nuee_type == "distribution":
        nuees = {"means": points.copy(),
                 "covs": np.array([np.eye(d) for _ in range(k)])}
    elif nuee_type == "axe_factoriel":
        nuees = {"points": points.copy(),
                 "dirs": np.zeros((k, d), dtype=float)}
        rng = np.random.RandomState(seed)
        for j in range(k):
            v = rng.normal(size=d)
            v /= (np.linalg.norm(v) + 1e-12)
            nuees["dirs"][j] = v
    else:  # structure
        nuees = np.zeros((k, 2, d), dtype=float)
        for j in range(k):
            nuees[j, 0] = points[j]
            nuees[j, 1] = points[j]

    history = []
    stop_reason = "none"
    labels = None
    inertia = None
    old_nuees = None
    old_inertia = None

    for t in range(1, max_iter + 1):
        # affectation
        labels = assign_nuees(X, nuees, nuee_type=nuee_type,
                              distance_fn=distance_fn)
        validate_labels(labels, n, k)

        # mise à jour
        new_nuees, empty = update_nuees(X, labels, k, nuees,
                                        nuee_type=nuee_type,
                                        prototype_type=prototype_type,
                                        distance_fn=distance_fn,
                                        m=m)

        # inertie calculée toujours par rapport aux centroïdes
        centroids = np.zeros((k, d), dtype=float)
        for j in range(k):
            idx = [i for i in range(n) if labels[i] == j]
            if len(idx) == 0:
                continue
            for i in idx:
                centroids[j] += X[i]
            centroids[j] /= len(idx)
        inertia = compute_inertia(X, labels, centroids, distance_fn=distance_fn)
        history.append(inertia)

        if verbose:
            print(f"[{t:03d}] W = {inertia:.4f} | vides = {empty}")

        # convergence : on teste seulement pour les types où prototypes
        # et nuees ont la même forme numérique (point)
        if nuee_type == "point":
            stop, reason, _ = check_convergence(
                old_nuees, new_nuees, old_inertia, inertia,
                iteration=t, max_iter=max_iter, tol=tol
            )
        else:
            # pour les autres types : on s'arrête sur inertie stable ou max_iter
            if t >= max_iter:
                stop, reason = True, "max_iter"
            elif old_inertia is not None and abs(inertia - old_inertia) < tol:
                stop, reason = True, "inertia"
            else:
                stop, reason = False, "none"

        old_nuees = nuees
        old_inertia = inertia
        nuees = new_nuees

        if stop:
            stop_reason = reason
            break

    return {
        "labels": labels,
        "nuees": nuees,
        "history": history,
        "n_iter": len(history),
        "stop_reason": stop_reason,
        "final_inertia": inertia,
        "nuee_type": nuee_type,
        "prototype_type": prototype_type,
    }