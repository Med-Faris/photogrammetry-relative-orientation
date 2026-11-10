"""3D reconstruction by relative orientation (incremental SfM) from a folder of photos.

Examples
--------
All pairs (small sets, ~20 images):
    python reconstruct.py data/pyramid --output results/pyramid

Large sets: match each image with its 2 next neighbours and use one image in two:
    python reconstruct.py photos/map --pair-gap 2 --step 2 --max-size 2000
"""

import argparse
import json
import time
from pathlib import Path

from sfm import camera, export, io, matching, plotting
from sfm.reconstruction import Options, Reconstruction


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", type=Path, help="folder with the photos, taken in order around the object")
    p.add_argument("-o", "--output", type=Path, default=Path("results"), help="output folder")
    p.add_argument("--pair-gap", type=int, default=0,
                   help="match each image with its next N images only (0 = all pairs)")
    p.add_argument("--step", type=int, default=1, help="use one image every STEP images")
    p.add_argument("--max-size", type=int, default=None,
                   help="downscale images so their long side is at most this (faster)")
    p.add_argument("--focal-px", type=float, default=camera.IPHONE13_FOCAL_PX,
                   help="focal length in pixels at the reference resolution (default: iPhone 13)")
    p.add_argument("--ref-size", type=int, nargs=2, default=camera.IPHONE13_REF_SIZE,
                   metavar=("W", "H"), help="resolution at which --focal-px is given")
    p.add_argument("--ratio", type=float, default=0.75, help="Lowe ratio test threshold")
    p.add_argument("--ransac-px", type=float, default=1.0, help="RANSAC threshold for F (pixels)")
    p.add_argument("--max-reproj", type=float, default=Options.max_reproj_px,
                   help="max reprojection error of a new 3D point (pixels)")
    p.add_argument("--min-angle", type=float, default=Options.min_angle_deg,
                   help="min triangulation angle of a new 3D point (degrees)")
    p.add_argument("--no-ba", action="store_true",
                   help="disable bundle adjustment (faster, but errors accumulate)")
    p.add_argument("--keep-percentile", type=float, default=95.0,
                   help="keep this percentage of points closest to the centre in the clean cloud")
    p.add_argument("--save-matches", type=int, default=0, metavar="N",
                   help="save match images for pairs with at least N inliers (0 = none)")
    p.add_argument("-q", "--quiet", action="store_true", help="print a summary only")
    return p.parse_args()


def main():
    a = parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    verbose = not a.quiet

    paths = io.list_images(a.images, a.step)
    images = [io.read_image(p, a.max_size) for p in paths]
    h, w = images[0].shape
    K = camera.intrinsics(w, h, a.focal_px, tuple(a.ref_size))
    print(f"{len(paths)} images, {w}x{h} px, focal = {K[0, 0]:.1f} px")

    keypoints, descriptors = matching.detect_features(images)
    print(f"SIFT: {sum(len(k) for k in keypoints)} keypoints in total")

    pairs = matching.candidate_pairs(len(paths), a.pair_gap)
    print(f"Matching {len(pairs)} pairs ({'all pairs' if a.pair_gap <= 0 else f'gap = {a.pair_gap}'})")
    matches = matching.match_all(keypoints, descriptors, pairs, a.ratio, a.ransac_px, verbose)
    if a.save_matches:
        (out / "matches").mkdir(exist_ok=True)
        for (i, j), m in matches.items():
            if m.n_inliers >= a.save_matches:
                matching.save_match_image(images[i], images[j], keypoints[i], keypoints[j], m,
                                          out / "matches" / f"matches_{i:03d}_{j:03d}.jpg")

    opt = Options(max_reproj_px=a.max_reproj, min_angle_deg=a.min_angle, bundle_adjustment=not a.no_ba)
    recon = Reconstruction(K, keypoints, matches, opt).run(verbose)
    X = recon.points_array()
    reproj = recon.mean_reprojection_error()
    print(f"Cameras: {len(recon.registered)}/{len(paths)} | 3D points: {len(X)} | "
          f"mean reprojection error: {reproj:.2f} px")

    colors = export.point_colors(recon, paths, lambda p: io.read_image(p, a.max_size, color=True))
    export.write_ply(out / "cloud_rgb.ply", X, colors)
    X_clean, mask, center = export.remove_outliers(X, a.keep_percentile)
    export.write_ply(out / "cloud_clean_rgb.ply", X_clean, colors[mask])

    centers = [recon.center(i) - center for i in sorted(recon.registered)]
    plotting.save_overview(X_clean, centers, out / "overview.png", title=a.images.name)

    summary = {
        "images": len(paths), "image_size": [w, h], "focal_px": round(float(K[0, 0]), 1),
        "pairs_matched": len(pairs), "cameras_registered": len(recon.registered),
        "points_3d": len(X), "points_clean": int(mask.sum()),
        "mean_reprojection_error_px": round(reproj, 3), "runtime_s": round(time.time() - t0, 1),
        "options": {k: v for k, v in vars(a).items() if k not in ("images", "output")},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved to {out}/ in {summary['runtime_s']} s")


if __name__ == "__main__":
    main()
