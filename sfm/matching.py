"""SIFT features, pairwise matching and epipolar filtering."""

from dataclasses import dataclass, field
from itertools import combinations

import cv2
import numpy as np


@dataclass
class PairMatch:
    """Inlier matches between images i and j (i < j), as keypoint index pairs."""

    i: int
    j: int
    n_good: int = 0
    idx_i: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    idx_j: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    F: np.ndarray = None

    @property
    def n_inliers(self):
        return len(self.idx_i)


def detect_features(images, n_features=0):
    """SIFT keypoints (as an (N, 2) array of pixel coordinates) and descriptors per image."""
    sift = cv2.SIFT_create(nfeatures=n_features)
    keypoints, descriptors = [], []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(np.array([k.pt for k in kp], dtype=np.float32).reshape(-1, 2))
        descriptors.append(des)
    return keypoints, descriptors


def candidate_pairs(n_images, max_gap=0):
    """Pairs of images to match.

    max_gap = 0: all pairs, n(n-1)/2 (exhaustive, O(n^2)).
    max_gap = g: each image is matched with the next g images only (O(n)),
    which works for photos taken in sequence around the object.
    """
    if max_gap <= 0:
        return list(combinations(range(n_images), 2))
    return [(i, j) for i in range(n_images) for j in range(i + 1, min(n_images, i + max_gap + 1))]


def match_pair(i, j, kp_i, kp_j, des_i, des_j, ratio=0.75, ransac_px=1.0):
    """Lowe ratio test, then RANSAC on the fundamental matrix."""
    res = PairMatch(i, j)
    if des_i is None or des_j is None or len(des_i) < 2 or len(des_j) < 2:
        return res

    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des_i, des_j, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    res.n_good = len(good)
    if len(good) < 8:
        return res

    qi = np.array([m.queryIdx for m in good])
    tj = np.array([m.trainIdx for m in good])
    F, mask = cv2.findFundamentalMat(kp_i[qi], kp_j[tj], cv2.FM_RANSAC, ransac_px, 0.999)
    if F is None or mask is None or F.shape != (3, 3):
        return res

    keep = mask.ravel().astype(bool)
    res.idx_i, res.idx_j, res.F = qi[keep], tj[keep], F
    return res


def match_all(keypoints, descriptors, pairs, ratio=0.75, ransac_px=1.0, verbose=True):
    """Match every candidate pair. Returns {(i, j): PairMatch}."""
    results = {}
    for i, j in pairs:
        r = match_pair(i, j, keypoints[i], keypoints[j], descriptors[i], descriptors[j], ratio, ransac_px)
        results[(i, j)] = r
        if verbose:
            print(f"  pair {i:03d}-{j:03d} | good = {r.n_good:5d} | inliers = {r.n_inliers:5d}")
    return results


def save_match_image(img_i, img_j, kp_i, kp_j, pm, path, max_lines=300):
    """Draw inlier matches side by side (for debugging and the README)."""
    sel = np.arange(pm.n_inliers)[:max_lines]
    k1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp_i[pm.idx_i[sel]]]
    k2 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp_j[pm.idx_j[sel]]]
    dm = [cv2.DMatch(k, k, 0) for k in range(len(sel))]
    out = cv2.drawMatches(img_i, k1, img_j, k2, dm, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), out)
