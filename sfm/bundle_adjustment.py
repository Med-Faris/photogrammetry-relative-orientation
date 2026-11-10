"""Bundle adjustment: refine all camera poses and 3D points together.

Minimises the squared reprojection error of every observation (sparse
least squares); observations that stay far off are removed afterwards.
The first camera stays fixed to keep the frame. Without this step, small pose
errors add up as images are registered one by one.
"""

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix


def _project(points, rvecs, tvecs, K):
    """Project each 3D point with its own camera (rvecs/tvecs given per observation)."""
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = np.where(theta > 0, rvecs / theta, 0.0)
    dot = np.sum(points * v, axis=1, keepdims=True)
    cos, sin = np.cos(theta), np.sin(theta)
    Xc = cos * points + sin * np.cross(v, points) + dot * (1 - cos) * v + tvecs  # Rodrigues
    x = Xc[:, :2] / Xc[:, 2:3]
    return x * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def bundle_adjust(recon, max_nfev=50, verbose=False):
    """Refine recon.R, recon.t and recon.points in place. Returns (error before, error after) in px."""
    cams = sorted(recon.registered)
    fixed = cams[0]
    free = [c for c in cams if c != fixed]
    cam_col = {c: k for k, c in enumerate(free)}

    cam_idx, pt_idx, uv = [], [], []
    for pid, track in enumerate(recon.tracks):
        for img, kp in track:
            cam_idx.append(img)
            pt_idx.append(pid)
            uv.append(recon.kp[img][kp])
    cam_idx, pt_idx, uv = np.array(cam_idx), np.array(pt_idx), np.asarray(uv, dtype=np.float64)
    n_pts = len(recon.points)

    rvec0 = {c: cv2.Rodrigues(recon.R[c])[0].ravel() for c in cams}
    x0 = np.concatenate([np.concatenate([rvec0[c], recon.t[c]]) for c in free]
                        + [recon.points_array().ravel()])
    n_cam_params = 6 * len(free)
    fixed_pose = np.concatenate([rvec0[fixed], recon.t[fixed]])
    is_fixed = cam_idx == fixed
    col = np.array([cam_col.get(c, 0) for c in cam_idx])

    def residuals(x):
        poses = x[:n_cam_params].reshape(-1, 6)
        pts = x[n_cam_params:].reshape(n_pts, 3)
        p = np.where(is_fixed[:, None], fixed_pose, poses[col] if len(free) else fixed_pose)
        return (_project(pts[pt_idx], p[:, :3], p[:, 3:], recon.K) - uv).ravel()

    # Sparsity: each residual (u, v) depends on its camera's 6 parameters and its point's 3.
    obs = np.arange(len(uv))
    free_obs = obs[~is_fixed]
    r_cam = np.concatenate([2 * free_obs + d for d in (0, 1) for _ in range(6)])
    c_cam = np.concatenate([6 * col[~is_fixed] + k for _ in (0, 1) for k in range(6)])
    r_pt = np.concatenate([2 * obs + d for d in (0, 1) for _ in range(3)])
    c_pt = np.concatenate([n_cam_params + 3 * pt_idx + k for _ in (0, 1) for k in range(3)])
    rows, cols = np.concatenate([r_cam, r_pt]), np.concatenate([c_cam, c_pt])
    A = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(2 * len(uv), len(x0)))

    before = np.linalg.norm(residuals(x0).reshape(-1, 2), axis=1).mean()
    res = least_squares(residuals, x0, jac_sparsity=A, loss="linear",
                        x_scale="jac", method="trf", max_nfev=max_nfev, verbose=2 if verbose else 0)
    after = np.linalg.norm(res.fun.reshape(-1, 2), axis=1).mean()

    poses = res.x[:n_cam_params].reshape(-1, 6)
    for c, k in cam_col.items():
        recon.R[c] = cv2.Rodrigues(poses[k, :3])[0]
        recon.t[c] = poses[k, 3:].copy()
    recon.points = list(res.x[n_cam_params:].reshape(n_pts, 3))
    return float(before), float(after)
