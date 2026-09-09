"""
Distance from two cameras, by triangulation.

Two cameras a known distance apart see the same object at slightly different
horizontal positions. That shift -- the disparity -- shrinks as the object
gets further away, and inverting it gives depth:

    Z = focal_length_px * baseline_m / disparity_px

This is what human eyes do, and it is genuinely a distance measurement rather
than an inference from appearance, which is why it can be trusted in a way a
monocular depth network cannot.

THE LIMIT, AND IT IS FUNDAMENTAL
--------------------------------
Differentiating that formula gives the error in depth for a given error in
disparity:

    dZ = (Z^2 / (f * B)) * d(disparity)

Depth error grows with the SQUARE of distance. With KITTI's numbers
(f = 721 px, B = 0.54 m, so f*B = 389), a single pixel of disparity error is
worth:

      10 m  ->  0.26 m
      20 m  ->  1.03 m
      30 m  ->  2.31 m
      50 m  ->  6.43 m
      80 m  -> 16.4  m

So stereo is excellent close up and degrades quadratically. That is the
opposite failure mode from a monocular depth network (roughly constant
relative error) and much worse than LiDAR (constant ~2 cm at any range).
A wider baseline pushes the problem further out but cannot remove it.
"""

import cv2
import numpy as np


def depth_from_disparity(disparity_px, focal_length_px, baseline_m):
    """Metres. Non-positive disparities become NaN -- they mean the matcher
    failed, not that the object is infinitely far, and letting them through
    as huge depths would quietly poison any average."""
    disparity = np.asarray(disparity_px, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = focal_length_px * baseline_m / disparity
    depth[disparity <= 0] = np.nan
    return depth


def depth_uncertainty_m(depth_m, focal_length_px, baseline_m, disparity_error_px=1.0):
    """How much depth error one pixel of matching error costs, at this range.

    Use it to attach an honest error bar to every stereo measurement rather
    than reporting a bare number that is worth centimetres up close and tens
    of metres far away.
    """
    return (np.asarray(depth_m, dtype=np.float64) ** 2
            * disparity_error_px / (focal_length_px * baseline_m))


def make_matcher(num_disparities=128, block_size=5):
    """Semi-global block matching.

    SGBM rather than plain block matching: it adds a smoothness penalty along
    several directions, which is what keeps disparity stable across the flat,
    textureless surfaces that cover most of a car. Plain block matching fails
    exactly there, on car doors and road, which is where we need it most.

    num_disparities caps the nearest measurable distance --  with f*B = 389,
    128 disparities means anything closer than about 3 m is out of range.
    """
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,      # must be divisible by 16
        blockSize=block_size,
        # Smoothness penalties, scaled to the block size as OpenCV recommends;
        # P2 > P1 penalises large jumps harder than small ones, so surfaces
        # stay smooth while real depth edges survive.
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,                     # reject matches that disagree left-to-right
        uniquenessRatio=10,                  # reject matches with a close runner-up
        speckleWindowSize=100,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def compute_disparity(left_bgr, right_bgr, matcher=None):
    """Disparity in pixels, as float32 with NaN where matching failed."""
    matcher = matcher or make_matcher()
    left_gray = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
    # OpenCV returns disparity scaled by 16 in fixed point.
    raw = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
    raw[raw <= 0] = np.nan
    return raw


def depth_map_to_points(depth_m, focal_length_px, cx, cy, stride=2):
    """Dense depth map -> (3, N) camera-frame points plus their (2, N) pixels.

    This is what lets stereo feed the same localisation code as LiDAR: once a
    depth map becomes a point cloud, nothing downstream needs to know which
    sensor produced it. `stride` subsamples, because a full 1242x375 map is
    ~460k points per frame and the vehicle-level maths gains nothing from
    that density.
    """
    h, w = depth_m.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    z = depth_m[0:h:stride, 0:w:stride]
    valid = np.isfinite(z) & (z > 0)
    xs, ys, z = xs[valid], ys[valid], z[valid]
    x = (xs - cx) * z / focal_length_px
    y = (ys - cy) * z / focal_length_px
    return np.stack([x, y, z]), np.stack([xs, ys]).astype(np.float64)
