# src/nuees_dynamiques/convergence.py

from .update import prototypes_have_moved
from .inertia import inertia_converged


def check_convergence(old_p, new_p, old_W, new_W,
                      iteration, max_iter, tol=1e-4):
    info = {"max_shift": None, "inertia_variation": None}
    if iteration >= max_iter:
        return True, "max_iter", info
    if old_p is not None:
        moved, shift = prototypes_have_moved(old_p, new_p, tol)
        info["max_shift"] = shift
        if not moved:
            return True, "prototypes", info
    if old_W is not None:
        conv, var = inertia_converged(old_W, new_W, tol)
        info["inertia_variation"] = var
        if conv:
            return True, "inertia", info
    return False, "none", info