"""Quick-look figures of the sparse cloud and the camera trajectory (saved as PNG)."""

import matplotlib

matplotlib.use("Agg")  # save figures without opening windows
import matplotlib.pyplot as plt
import numpy as np


def _subsample(X, max_points=30000, seed=0):
    if len(X) <= max_points:
        return X
    return X[np.random.default_rng(seed).choice(len(X), max_points, replace=False)]


def _equal_axes(ax, X):
    lo, hi = X.min(axis=0), X.max(axis=0)
    c, r = (lo + hi) / 2, max((hi - lo).max() / 2, 1e-9)
    for set_lim, k in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        set_lim(c[k] - r, c[k] + r)


def save_overview(X, centers, path, title="Sparse reconstruction"):
    """3D view plus XY / XZ / YZ projections in one figure."""
    Xs = _subsample(np.asarray(X))
    C = np.asarray(centers).reshape(-1, 3)
    fig = plt.figure(figsize=(16, 5))

    ax = fig.add_subplot(1, 4, 1, projection="3d")
    ax.scatter(*Xs.T, c=Xs[:, 2], s=1, cmap="viridis")
    ax.scatter(*C.T, c="red", marker="^", s=30)
    _equal_axes(ax, np.vstack([Xs, C]))
    ax.set_title(title)

    for k, (a, b, name) in enumerate([(0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ")], start=2):
        ax = fig.add_subplot(1, 4, k)
        ax.scatter(Xs[:, a], Xs[:, b], c=Xs[:, 2], s=1, cmap="viridis")
        ax.plot(C[:, a], C[:, b], "r^", ms=4)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Projection {name}")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
