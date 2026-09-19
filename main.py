# src/main.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from nuees_dynamiques.preprocessing import standardize, load_sklearn_dataset
from nuees_dynamiques.algorithm import nuees_dynamiques
from nuees_dynamiques.evaluation import silhouette_score, davies_bouldin_score
from nuees_dynamiques.visualization import (
    plot_nuees_2d, plot_convergence, plot_compare
)
from nuees_dynamiques.distances import (
    euclidean_distance_squared, euclidean_distance, manhattan_distance
)


RESULTS = "results"
FIG_DIR = os.path.join(RESULTS, "figures")

# ------------------------------------------------------------------
# Schéma de nommage des figures
# ------------------------------------------------------------------
# 01_partition_centroide.png
# 02_partition_mediane.png
# 03_partition_medoide.png
# 04_partition_ensemble.png
# 05_partition_gaussienne.png
# 06_partition_axe.png
# 07_partition_structure.png
# 01_courbe_centroide.png
# A_comparaison_silhouette.png
# B_comparaison_davies_bouldin.png
# ------------------------------------------------------------------

# Chaque variante est associée à :
#   - un identifiant court (pour les noms de fichiers)
#   - un libellé long (pour les titres)
VARIANTS = [
    # (nuee_type, prototype_type, distance_fit, distance_eval, code, libelle)
    ("point", "centroide", euclidean_distance_squared, euclidean_distance,
     "01_partition_centroide", "Nuée point — centroïde"),
    ("point", "mediane", manhattan_distance, manhattan_distance,
     "02_partition_mediane", "Nuée point — médiane"),
    ("point", "medoide", euclidean_distance_squared, euclidean_distance,
     "03_partition_medoide", "Nuée point — médoïde"),
    ("ensemble_points", "centroide", euclidean_distance_squared, euclidean_distance,
     "04_partition_ensemble", "Nuée ensemble de points (m=3)"),
    ("distribution", "centroide", euclidean_distance_squared, euclidean_distance,
     "05_partition_gaussienne", "Nuée distribution (gaussienne)"),
    ("axe_factoriel", "centroide", euclidean_distance_squared, euclidean_distance,
     "06_partition_axe", "Nuée axe factoriel"),
    ("structure", "centroide", euclidean_distance_squared, euclidean_distance,
     "07_partition_structure", "Nuée structure (segment)"),
]


def run_variant(X, k, nuee_type, prototype_type,
                distance_fit, distance_eval, seed=42):
    """Lance une variante et retourne ses métriques."""
    try:
        res = nuees_dynamiques(
            X, k=k, max_iter=100, tol=1e-6, seed=seed,
            init_method="kmpp",
            nuee_type=nuee_type,
            prototype_type=prototype_type,
            distance_fn=distance_fit,
            verbose=False,
        )
    except Exception as e:
        print(f"   ⚠️  Exécution échouée : {e}")
        return None

    # Silhouette
    try:
        sil = silhouette_score(X, res["labels"], distance_fn=distance_eval)
    except Exception:
        sil = float("nan")

    # Davies-Bouldin
    try:
        if nuee_type == "distribution":
            protos = res["nuees"]["means"]
        elif nuee_type == "axe_factoriel":
            protos = res["nuees"]["points"]
        elif nuee_type == "structure":
            protos = res["nuees"][:, 0]
        elif nuee_type == "ensemble_points":
            protos = res["nuees"][:, 0]
        else:
            protos = res["nuees"]
        db = davies_bouldin_score(X, res["labels"], protos,
                                  distance_fn=distance_eval)
    except Exception:
        db = float("nan")

    return {
        "nuee_type": nuee_type,
        "prototype_type": prototype_type,
        "inertia": res["final_inertia"],
        "silhouette": sil,
        "davies_bouldin": db,
        "n_iter": res["n_iter"],
        "stop_reason": res["stop_reason"],
        "labels": res["labels"],
        "nuees": res["nuees"],
        "history": res["history"],
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 70)
    print("NUÉES DYNAMIQUES — vérification sur Iris (chargé via sklearn)")
    print("=" * 70)

    # ---- 1) Charger Iris ----
    X, y = load_sklearn_dataset("iris")
    print(f"\nIris : {X.shape[0]} points, {X.shape[1]} dimensions")
    print(f"Classes réelles : {np.unique(y).tolist()}")

    # ---- 2) Standardiser ----
    X = standardize(X)

    # ---- 3) k = 3 ----
    k = 3

    # ---- 4) Lancer toutes les variantes ----
    rows = []
    for (nt, pt, dfit, deval, code, libelle) in VARIANTS:
        print(f"\n→ {libelle}")
        r = run_variant(X, k, nt, pt, dfit, deval, seed=42)
        if r is None:
            continue
        print(f"   W={r['inertia']:.4f} | "
              f"Sil={r['silhouette']:.4f} | "
              f"DB={r['davies_bouldin']:.4f} | "
              f"iter={r['n_iter']}")
        rows.append((code, libelle, r))

        # --- Figure partition ---
        try:
            plot_nuees_2d(
                X, r["labels"], r["nuees"], nuee_type=nt,
                title=libelle,
                output_path=os.path.join(FIG_DIR, f"{code}.png"),
            )
        except Exception as e:
            print(f"   ⚠️  Figure partition échouée : {e}")

        # --- Figure convergence ---
        try:
            conv_code = code.replace("partition", "courbe")
            plot_convergence(
                r["history"],
                title=f"Convergence — {libelle}",
                output_path=os.path.join(FIG_DIR, f"{conv_code}.png"),
            )
        except Exception as e:
            print(f"   ⚠️  Figure convergence échouée : {e}")

    # ---- 5) Tableau récapitulatif ----
    print("\n" + "=" * 90)
    print(f"{'Variante':<30}{'Inertie':>12}{'Silhouette':>14}"
          f"{'Davies-Bouldin':>18}{'Iter':>6}")
    print("-" * 90)
    for (_, libelle, r) in rows:
        print(f"{libelle:<30}"
              f"{r['inertia']:>12.4f}"
              f"{r['silhouette']:>14.4f}"
              f"{r['davies_bouldin']:>18.4f}"
              f"{r['n_iter']:>6}")
    print("=" * 90)

    # ---- 6) Comparaisons visuelles ----
    if rows:
        # On a besoin d'une liste de dicts pour plot_compare
        list_of_dicts = [
            {
                "nuee_type": r["nuee_type"],
                "prototype_type": r["prototype_type"],
                "silhouette": r["silhouette"],
                "davies_bouldin": r["davies_bouldin"],
            }
            for (_, _, r) in rows
        ]
        try:
            plot_compare(
                list_of_dicts, "silhouette",
                os.path.join(FIG_DIR, "A_comparaison_silhouette.png"),
                "Silhouette par variante (plus grand = mieux)",
                color="mediumseagreen",
            )
        except Exception as e:
            print(f"⚠️  Comparaison silhouette échouée : {e}")

        try:
            plot_compare(
                list_of_dicts, "davies_bouldin",
                os.path.join(FIG_DIR, "B_comparaison_davies_bouldin.png"),
                "Davies-Bouldin par variante (plus petit = mieux)",
                color="indianred",
            )
        except Exception as e:
            print(f"⚠️  Comparaison DB échouée : {e}")

    # ---- 7) Meilleure variante ----
    if rows:
        best_code, best_lib, best_r = max(
            rows, key=lambda t: t[2]["silhouette"]
        )
        print(f"\n🏆 Meilleure : {best_lib} "
              f"(Silhouette = {best_r['silhouette']:.4f})")

    print(f"\n📁 Figures dans : {FIG_DIR}")


if __name__ == "__main__":
    main()