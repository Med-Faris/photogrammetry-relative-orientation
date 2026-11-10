"""Point colours, cleaning and PLY export."""

import numpy as np


def point_colors(recon, image_paths, read_color):
    """Mean RGB colour of each 3D point over all images that observe it.

    Images are read one at a time, so memory stays low even with many photos.
    `read_color(path)` must return a BGR image at the same resolution used for SIFT.
    """
    n = len(recon.points)
    total = np.zeros((n, 3))
    count = np.zeros(n)
    by_image = {}
    for pid, track in enumerate(recon.tracks):
        for img, kp in track:
            by_image.setdefault(img, []).append((pid, kp))

    for img, obs in by_image.items():
        bgr = read_color(image_paths[img])
        h, w = bgr.shape[:2]
        pids, kps = map(np.array, zip(*obs))
        uv = np.rint(recon.kp[img][kps]).astype(int)
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        pids, uv = pids[inside], uv[inside]
        np.add.at(total, pids, bgr[uv[:, 1], uv[:, 0], ::-1])
        np.add.at(count, pids, 1)

    colors = np.full((n, 3), 255, dtype=np.uint8)
    seen = count > 0
    colors[seen] = np.clip(total[seen] / count[seen, None], 0, 255).astype(np.uint8)
    return colors


def remove_outliers(X, percentile=95.0):
    """Keep the points closest to the median, and return them centred on it.

    Returns (X_clean, mask, center) so colours can be filtered with the same mask.
    """
    X = np.asarray(X, dtype=float)
    finite = np.isfinite(X).all(axis=1)
    if not finite.any():
        return X[finite], finite, np.zeros(3)
    center = np.median(X[finite], axis=0)
    dist = np.linalg.norm(X - center, axis=1)
    dist[~finite] = np.inf
    mask = dist <= np.percentile(dist[finite], percentile)
    return X[mask] - center, mask, center


def write_ply(path, points, colors=None):
    """Binary little-endian PLY (smaller and much faster to write than ASCII)."""
    X = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(X)}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    data = np.empty(len(X), dtype=fields)
    data["x"], data["y"], data["z"] = X.T
    if colors is not None:
        C = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        data["red"], data["green"], data["blue"] = C.T

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())
