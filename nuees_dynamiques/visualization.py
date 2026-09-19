# src/nuees_dynamiques/visualization.py

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def _colors(k):
    return plt.cm.tab10(np.linspace(0, 1, max(k, 10)))


def plot_nuee_point_2d(X, labels, nuees, title="Nuée point",
                       output_path=None):
    k = nuees.shape[0]
    colors = _colors(k)
    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.6,
                    color=colors[j], label=f"Cluster {j} ({len(idx)})")
    plt.scatter(nuees[:, 0], nuees[:, 1], s=250, marker="X",
                edgecolor="black", linewidth=2, color="red",
                zorder=5, label="Prototypes")
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def plot_nuee_ensemble_2d(X, labels, nuees, title="Nuée ensemble",
                          output_path=None):
    k, m, d = nuees.shape
    colors = _colors(k)
    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        plt.scatter(nuees[j, :, 0], nuees[j, :, 1], s=200, marker="X",
                    edgecolor="black", linewidth=2, color=colors[j],
                    zorder=5)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def _ellipse(mean, cov, n_std=2.0):
    eigvals, eigvecs = np.linalg.eigh(cov[:2, :2])
    order = eigvals.argsort()[::-1]
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.maximum(eigvals, 1e-12))
    t = np.linspace(0, 2 * np.pi, 100)
    ell = np.array([width / 2 * np.cos(t), height / 2 * np.sin(t)])
    R = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                  [np.sin(np.radians(angle)),  np.cos(np.radians(angle))]])
    return R @ ell + mean[:2, None]


def plot_nuee_distribution_2d(X, labels, nuees, title="Nuée distribution",
                              output_path=None):
    means = nuees["means"]; covs = nuees["covs"]
    k = means.shape[0]; colors = _colors(k)
    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.5,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        ell = _ellipse(means[j], covs[j], n_std=2.0)
        plt.plot(ell[0], ell[1], color=colors[j], linewidth=2)
        plt.scatter(means[j, 0], means[j, 1], s=150, marker="X",
                    color=colors[j], edgecolor="black", zorder=5)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def plot_nuee_axe_2d(X, labels, nuees, title="Nuée axe", output_path=None,
                     length=5.0):
    pts = nuees["points"]; dirs = nuees["dirs"]
    k = pts.shape[0]; colors = _colors(k)
    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        p = pts[j]; u = dirs[j]
        a = p - length * u; b = p + length * u
        plt.plot([a[0], b[0]], [a[1], b[1]], color=colors[j], linewidth=2.5)
        plt.scatter(p[0], p[1], s=150, marker="X",
                    color=colors[j], edgecolor="black", zorder=5)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def plot_nuee_structure_2d(X, labels, nuees, title="Nuée structure",
                           output_path=None):
    k = nuees.shape[0]; colors = _colors(k)
    plt.figure(figsize=(8, 6))
    for j in range(k):
        idx = np.where(labels == j)[0]
        if len(idx) == 0:
            continue
        plt.scatter(X[idx, 0], X[idx, 1], s=25, alpha=0.4,
                    color=colors[j], label=f"Cluster {j}")
    for j in range(k):
        p1 = nuees[j, 0]; p2 = nuees[j, 1]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]],
                 color=colors[j], linewidth=2.5)
        plt.scatter([p1[0], p2[0]], [p1[1], p2[1]], s=150, marker="X",
                    color=colors[j], edgecolor="black", zorder=5)
    plt.title(title); plt.xlabel("x1"); plt.ylabel("x2")
    plt.legend(fontsize=8); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def plot_nuees_2d(X, labels, nuees, nuee_type="point",
                  title=None, output_path=None):
    if title is None:
        title = f"Nuées — {nuee_type}"
    nt = nuee_type.lower()
    if nt == "point":
        plot_nuee_point_2d(X, labels, nuees, title, output_path)
    elif nt == "ensemble_points":
        plot_nuee_ensemble_2d(X, labels, nuees, title, output_path)
    elif nt == "distribution":
        plot_nuee_distribution_2d(X, labels, nuees, title, output_path)
    elif nt == "axe_factoriel":
        plot_nuee_axe_2d(X, labels, nuees, title, output_path)
    elif nt == "structure":
        plot_nuee_structure_2d(X, labels, nuees, title, output_path)


def plot_convergence(history, title="Convergence", output_path=None):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(history) + 1), history, marker="o",
             linewidth=2, color="steelblue")
    plt.title(title); plt.xlabel("Itération"); plt.ylabel("Inertie W")
    plt.grid(True, alpha=0.3); plt.xticks(range(1, len(history) + 1))
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()


def plot_compare(rows, key, output_path, title, color="mediumseagreen"):
    names = [f"{r['nuee_type']}/{r['prototype_type']}" for r in rows]
    vals = [r[key] for r in rows]
    plt.figure(figsize=(10, 5))
    plt.barh(names, vals, color=color)
    plt.title(title)
    plt.xlabel(key)
    plt.tight_layout()
    if output_path:
        _ensure(output_path); plt.savefig(output_path, dpi=150)
    plt.close()