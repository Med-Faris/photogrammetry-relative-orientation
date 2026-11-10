"""Synthetic test: known cameras and points -> the reconstruction must recover the shape."""

import cv2
import numpy as np

from sfm import camera, export, matching
from sfm.reconstruction import Options, Reconstruction

W, H = 1512, 2016


def look_at(center, target=None):
    target = np.zeros(3) if target is None else target
    z = target - center
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ center


def synthetic_scene(n_cams=6, n_pts=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n_pts, 3))
    K = camera.intrinsics(W, H)
    cams, keypoints = [], []
    for k in range(n_cams):  # cameras on an arc around the object
        a = np.radians(-30 + 60 * k / (n_cams - 1))
        R, t = look_at(np.array([6 * np.sin(a), 0.5, -6 * np.cos(a)]))
        x = (K @ (R @ X.T + t[:, None])).T
        uv = x[:, :2] / x[:, 2:3] + rng.normal(0, 0.3, size=(n_pts, 2))
        cams.append((R, t))
        keypoints.append(uv.astype(np.float32))  # keypoint k of every image is 3D point k
    return X, K, cams, keypoints


def all_matches(keypoints):
    res = {}
    for i, j in matching.candidate_pairs(len(keypoints)):
        F, mask = cv2.findFundamentalMat(keypoints[i], keypoints[j], cv2.FM_RANSAC, 1.0, 0.999)
        keep = mask.ravel().astype(bool)
        idx = np.arange(len(keypoints[i]))[keep]
        res[(i, j)] = matching.PairMatch(i, j, len(idx), idx, idx, F)
    return res


def test_candidate_pairs():
    assert len(matching.candidate_pairs(5)) == 10
    assert matching.candidate_pairs(5, max_gap=1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert len(matching.candidate_pairs(10, max_gap=2)) == 17


def test_intrinsics_orientation_and_scale():
    portrait = camera.intrinsics(3024, 4032)
    landscape = camera.intrinsics(4032, 3024)
    half = camera.intrinsics(1512, 2016)
    assert np.isclose(portrait[0, 0], 3028) and np.isclose(landscape[0, 0], 3028)
    assert np.isclose(half[0, 0], 1514)
    assert np.allclose(landscape[:2, 2], [2016, 1512])


def test_full_reconstruction_recovers_shape():
    X_true, K, _, keypoints = synthetic_scene()
    recon = Reconstruction(K, keypoints, all_matches(keypoints)).run(verbose=False)

    assert len(recon.registered) == len(keypoints)
    assert recon.mean_reprojection_error() < 1.0

    # no duplicated points: each keypoint belongs to at most one 3D point
    assert len(recon.obs) == sum(len(t) for t in recon.tracks)

    # compare with the ground truth up to a similarity transform (scale is arbitrary)
    ids = np.array([t[0][1] for t in recon.tracks])  # keypoint index == true point index
    A, B = recon.points_array(), X_true[ids]
    A0, B0 = A - A.mean(0), B - B.mean(0)
    U, S, Vt = np.linalg.svd(A0.T @ B0)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    s = np.trace(np.diag(S) @ D) / np.sum(A0 ** 2)
    err = np.linalg.norm(s * A0 @ (U @ D @ Vt) - B0, axis=1)
    assert np.median(err) < 0.02  # object size is 2 units


def test_ply_roundtrip(tmp_path):
    X = np.array([[0, 0, 0], [1, 2, 3]], dtype=float)
    C = np.array([[255, 0, 0], [0, 128, 255]], dtype=np.uint8)
    path = tmp_path / "c.ply"
    export.write_ply(path, X, C)
    raw = path.read_bytes()
    header, body = raw.split(b"end_header\n")
    assert b"element vertex 2" in header
    data = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                      ("r", "u1"), ("g", "u1"), ("b", "u1")])
    assert np.allclose(np.c_[data["x"], data["y"], data["z"]], X)
    assert (np.c_[data["r"], data["g"], data["b"]] == C).all()


def test_bundle_adjustment_removes_drift():
    _, K, _, keypoints = synthetic_scene(seed=1)
    m = all_matches(keypoints)
    no_ba = Reconstruction(K, keypoints, m, Options(bundle_adjustment=False)).run(verbose=False)
    ba = Reconstruction(K, keypoints, m).run(verbose=False)
    assert ba.mean_reprojection_error() < 0.5  # noise is 0.3 px
    assert ba.mean_reprojection_error() < no_ba.mean_reprojection_error()
