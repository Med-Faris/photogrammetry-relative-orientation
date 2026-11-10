"""Incremental reconstruction: initial pair -> PnP registration -> triangulation.

Convention: a camera maps a world point X to camera coordinates R @ X + t and
projects it with P = K [R | t]. The first camera of the initial pair is the
world frame, so the scale of the result is arbitrary.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .bundle_adjustment import bundle_adjust


@dataclass
class Options:
    min_init_inliers: int = 50    # inliers needed for the initial pair
    min_pnp_points: int = 6       # 2D-3D correspondences needed to register an image
    pnp_reproj_px: float = 8.0    # RANSAC threshold for solvePnPRansac
    max_reproj_px: float = 4.0    # a new 3D point is kept only if it reprojects within this error
    min_angle_deg: float = 1.0    # ... and if its two viewing rays are at least this far apart
    bundle_adjustment: bool = True  # refine poses and points as the model grows, and at the end
    ba_growth: float = 1.3          # run a local refinement when the number of cameras grows by 30 %


class Reconstruction:
    def __init__(self, K, keypoints, matches, options=None):
        self.K = K
        self.kp = keypoints
        self.matches = matches  # {(i, j): PairMatch} with i < j
        self.opt = options or Options()
        n = len(keypoints)
        self.R = [None] * n
        self.t = [None] * n
        self.registered = []
        self.points = []           # list of 3D points (np.ndarray of shape (3,))
        self.tracks = []           # for each point, list of (image, keypoint) observations
        self.obs = {}              # (image, keypoint) -> point id

    # ------------------------------------------------------------------ helpers
    def _pair(self, a, b):
        """Inlier keypoint indices (in a, in b) for images a and b, whatever their order."""
        if (a, b) in self.matches:
            m = self.matches[(a, b)]
            return m.idx_i, m.idx_j
        if (b, a) in self.matches:
            m = self.matches[(b, a)]
            return m.idx_j, m.idx_i
        return None

    def P(self, i):
        return self.K @ np.hstack([self.R[i], self.t[i].reshape(3, 1)])

    def center(self, i):
        return -self.R[i].T @ self.t[i]

    def _reproj_error(self, i, X, uv):
        x = (self.P(i) @ np.c_[X, np.ones(len(X))].T).T
        return np.linalg.norm(x[:, :2] / x[:, 2:3] - uv, axis=1)

    def _add_observation(self, pid, img, kp):
        key = (img, kp)
        if key in self.obs:
            return
        self.obs[key] = pid
        self.tracks[pid].append(key)

    # --------------------------------------------------------- initialisation
    def initialize(self):
        """Pick the pair with the most inliers and fix its relative pose."""
        best = max(self.matches.values(), key=lambda m: m.n_inliers)
        if best.F is None or best.n_inliers < self.opt.min_init_inliers:
            raise RuntimeError("No image pair has enough inliers to start the reconstruction.")

        i, j = best.i, best.j
        pts_i, pts_j = self.kp[i][best.idx_i], self.kp[j][best.idx_j]
        # Estimate E directly with the known K (5-point + RANSAC). This is more accurate
        # than E = K^T F K, because F is estimated without knowing K.
        E, _ = cv2.findEssentialMat(pts_i, pts_j, self.K, cv2.RANSAC, 0.999, self.opt.max_reproj_px / 2)
        if E is None or E.shape != (3, 3):
            E = self.K.T @ best.F @ self.K
        _, R, t, _ = cv2.recoverPose(E, pts_i, pts_j, self.K)

        self.R[i], self.t[i] = np.eye(3), np.zeros(3)
        self.R[j], self.t[j] = R, t.ravel()
        self.registered = [i, j]
        added = self.triangulate(i, j)
        return i, j, best.n_inliers, added

    # ------------------------------------------------------------ triangulation
    def triangulate(self, a, b):
        """Triangulate the matches between two registered images.

        Matches whose keypoints already belong to a 3D point extend that point's
        track instead of creating a duplicate point. New points must be in front
        of both cameras, reproject within max_reproj_px and have enough parallax.
        """
        pair = self._pair(a, b)
        if pair is None:
            return 0
        idx_a, idx_b = pair

        new_a, new_b = [], []
        for ka, kb in zip(idx_a, idx_b):
            pa, pb = self.obs.get((a, ka)), self.obs.get((b, kb))
            if pa is not None and pb is not None:
                continue
            if pa is not None:
                self._add_observation(pa, b, kb)
            elif pb is not None:
                self._add_observation(pb, a, ka)
            else:
                new_a.append(ka)
                new_b.append(kb)

        if not new_a:
            return 0
        new_a, new_b = np.array(new_a), np.array(new_b)
        uv_a, uv_b = self.kp[a][new_a], self.kp[b][new_b]

        Xh = cv2.triangulatePoints(self.P(a), self.P(b), uv_a.T.astype(np.float64), uv_b.T.astype(np.float64))
        X = (Xh[:3] / Xh[3]).T

        depth_a = (self.R[a] @ X.T).T[:, 2] + self.t[a][2]
        depth_b = (self.R[b] @ X.T).T[:, 2] + self.t[b][2]
        ray_a = X - self.center(a)
        ray_b = X - self.center(b)
        cos = np.sum(ray_a * ray_b, axis=1) / (np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1))
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        err = np.maximum(self._reproj_error(a, X, uv_a), self._reproj_error(b, X, uv_b))

        ok = (np.isfinite(X).all(axis=1) & (depth_a > 0) & (depth_b > 0)
              & (err < self.opt.max_reproj_px) & (angle > self.opt.min_angle_deg))

        for Xp, ka, kb in zip(X[ok], new_a[ok], new_b[ok]):
            pid = len(self.points)
            self.points.append(Xp)
            self.tracks.append([])
            self._add_observation(pid, a, int(ka))
            self._add_observation(pid, b, int(kb))
        return int(ok.sum())

    # ------------------------------------------------------------ registration
    def correspondences_2d3d(self, img):
        """2D keypoints of `img` matched to existing 3D points, through registered images."""
        kp_ids, pids = [], []
        seen_kp, seen_pid = set(), set()
        for r in self.registered:
            pair = self._pair(img, r)
            if pair is None:
                continue
            for k_img, k_r in zip(*pair):
                pid = self.obs.get((r, int(k_r)))
                if pid is None or pid in seen_pid or k_img in seen_kp:
                    continue
                seen_pid.add(pid)
                seen_kp.add(k_img)
                kp_ids.append(int(k_img))
                pids.append(pid)
        return np.array(kp_ids, dtype=int), np.array(pids, dtype=int)

    def register(self, img, kp_ids, pids):
        """Estimate the pose of `img` with PnP + RANSAC, then refine it on the inliers."""
        pts2d = self.kp[img][kp_ids].astype(np.float64)
        pts3d = np.asarray(self.points, dtype=np.float64)[pids]
        try:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                pts3d, pts2d, self.K, None,
                iterationsCount=1000, reprojectionError=self.opt.pnp_reproj_px,
                confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return 0
        if not ok or inl is None or len(inl) < self.opt.min_pnp_points:
            return 0

        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(pts3d[inl], pts2d[inl], self.K, None, rvec, tvec)
        R, _ = cv2.Rodrigues(rvec)
        self.R[img], self.t[img] = R, tvec.ravel()
        self.registered.append(img)
        for k in inl:  # the image now observes these 3D points too
            self._add_observation(int(pids[k]), img, int(kp_ids[k]))
        return len(inl)

    def refine(self, verbose=True, max_nfev=50):
        """Drop bad observations, bundle-adjust, then drop the ones that are still bad.

        Removing outliers first matters: least squares is pulled away by a few wrong matches.
        """
        removed = self.remove_bad_observations()
        before, after = bundle_adjust(self, max_nfev=max_nfev)
        removed += self.remove_bad_observations()
        if verbose:
            print(f"  bundle adjustment: {before:.2f} px -> {after:.2f} px "
                  f"({len(self.registered)} cameras, {removed} bad observations removed)")

    def observation_errors(self):
        """All observations as arrays (point id, image, keypoint, reprojection error in px)."""
        obs = np.array([(pid, img, kp) for pid, track in enumerate(self.tracks) for img, kp in track],
                       dtype=int).reshape(-1, 3)
        err = np.zeros(len(obs))
        X = self.points_array()
        for img in np.unique(obs[:, 1]):
            sel = obs[:, 1] == img
            err[sel] = self._reproj_error(img, X[obs[sel, 0]], self.kp[img][obs[sel, 2]])
        return obs, err

    def remove_bad_observations(self):
        """Remove observations above max_reproj_px; points left with < 2 views are deleted."""
        obs, err = self.observation_errors()
        good = obs[err < self.opt.max_reproj_px]
        tracks = [[] for _ in self.tracks]
        for pid, img, kp in good:
            tracks[pid].append((int(img), int(kp)))
        alive = [pid for pid, t in enumerate(tracks) if len(t) >= 2]
        self.points = [self.points[pid] for pid in alive]
        self.tracks = [tracks[pid] for pid in alive]
        self.obs = {key: new for new, track in enumerate(self.tracks) for key in track}
        return int(len(obs) - len(good))

    def run(self, verbose=True):
        i, j, n_inl, added = self.initialize()
        if verbose:
            print(f"Initial pair {i:03d}-{j:03d} | inliers = {n_inl} | points = {added}")
        next_ba = 3

        remaining = set(range(len(self.kp))) - set(self.registered)
        failed = set()
        while remaining - failed:
            # next image: the one that sees the most existing 3D points
            best = None
            for img in remaining - failed:
                kp_ids, pids = self.correspondences_2d3d(img)
                if len(kp_ids) >= self.opt.min_pnp_points and (best is None or len(kp_ids) > len(best[1])):
                    best = (img, kp_ids, pids)
            if best is None:
                break

            img, kp_ids, pids = best
            n_inl = self.register(img, kp_ids, pids)
            if n_inl == 0:
                failed.add(img)
                if verbose:
                    print(f"Image {img:03d} | 2D-3D = {len(kp_ids):5d} | PnP failed")
                continue

            remaining.discard(img)
            failed.clear()  # new points may now let a failed image register
            new = sum(self.triangulate(img, r) for r in self.registered if r != img)
            if verbose:
                print(f"Image {img:03d} | 2D-3D = {len(kp_ids):5d} | PnP inliers = {n_inl:5d} | new points = {new}")
            if self.opt.bundle_adjustment and len(self.registered) >= next_ba:
                self.refine(verbose, max_nfev=20)
                next_ba = max(next_ba + 1, int(len(self.registered) * self.opt.ba_growth))

        if self.opt.bundle_adjustment:
            self.refine(verbose, max_nfev=100)
        return self

    # ------------------------------------------------------------------ results
    def points_array(self):
        return np.asarray(self.points, dtype=np.float64).reshape(-1, 3)

    def mean_reprojection_error(self):
        """Mean reprojection error (px) over all observations of all points."""
        _, err = self.observation_errors()
        return float(err.mean()) if len(err) else float("nan")
