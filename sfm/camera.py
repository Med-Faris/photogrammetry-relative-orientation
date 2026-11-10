"""Camera intrinsics."""

import numpy as np

# Approximate iPhone 13 main camera, for 3024 x 4032 images (see report, section 3.2).
IPHONE13_FOCAL_PX = 3028.0
IPHONE13_REF_SIZE = (3024, 4032)  # (width, height)


def intrinsics(width, height, focal_px=IPHONE13_FOCAL_PX, ref_size=IPHONE13_REF_SIZE):
    """Pinhole matrix K for an image of size (width, height).

    The focal length is given in pixels for a reference resolution and is scaled
    with the image's long side, so it stays correct for portrait and landscape
    images and for downscaled copies. The principal point is the image centre.
    """
    scale = max(width, height) / float(max(ref_size))
    f = focal_px * scale
    return np.array(
        [[f, 0.0, width / 2.0],
         [0.0, f, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
