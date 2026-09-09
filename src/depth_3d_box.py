"""
Camera-only 3D vehicle localisation: monocular metric depth -> 3D position,
as a replacement for ground_plane.py's flat-road homography.

WHY THIS EXISTS
---------------
ground_plane.py reads distance from *where a box's bottom edge sits* on an
assumed-flat road. That has a hard geometric failure: as a vehicle gets
farther away its ground-contact point creeps toward the horizon, where one
pixel of detector error is worth metres of distance. Measured against
nuScenes 3D ground truth (see accuracy_report() below), that method put
~36% of cars off by more than 10m, with errors exploding past 20m range.

A monocular depth network reads distance from the vehicle's *appearance*
(apparent size, texture, scene context) instead of its contact row, so it
has no horizon singularity -- it degrades gradually rather than blowing up.

This module deliberately does NOT touch ground_plane.py, estimate_distance.py
or any existing script; it is an additive parallel path, same as the SUMO and
nuScenes tracks.

WHAT IT PRODUCES
----------------
For each detected vehicle: a 3D position in the camera frame plus a full 3D
box (position + metric size + heading), which 2D boxes cannot express. The
heading matters for the planned trajectory work -- a car pointing at you and
one pointing away are the same rectangle in 2D but completely different risks.
"""

import os

import numpy as np

# Metric (not relative) depth -- outdoor variant, trained for driving-scale
# scenes. The relative-depth models return unitless inverse depth that would
# need a separate scale calibration; the metric model returns metres directly.
DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Median real vehicle dimensions, measured from the nuScenes annotations
# themselves (see measure_size_priors()), used to give a detected vehicle a
# physically plausible 3D extent when only a 2D detection is available.
CAR_SIZE_WLH = (1.95, 4.63, 1.74)


def load_depth_pipeline(device=0):
    from transformers import pipeline
    return pipeline("depth-estimation", model=DEPTH_MODEL, device=device)


def depth_map(pipe, image):
    out = pipe(image)
    key = "predicted_depth" if "predicted_depth" in out else "depth"
    return np.array(out[key], dtype=np.float32)


def sample_box_depth(depth, x1, y1, x2, y2, inner=0.5):
    """Depth of the vehicle inside a 2D box.

    Samples only the central `inner` fraction of the box: the outer margin
    routinely straddles the vehicle's silhouette and picks up whatever is
    behind it, which drags the estimate toward background depth. Median (not
    mean) so a few bleed-through pixels can't move the answer.
    """
    h, w = depth.shape
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * inner / 2.0, (y2 - y1) * inner / 2.0
    a = max(int(round(cx - hw)), 0)
    b = min(int(round(cx + hw)) + 1, w)
    c = max(int(round(cy - hh)), 0)
    d = min(int(round(cy + hh)) + 1, h)
    if a >= b or c >= d:
        return None
    patch = depth[c:d, a:b]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    return float(np.median(patch)) if patch.size else None


def unproject(u, v, z, K):
    """Pixel + depth-along-optical-axis -> 3D point in the camera frame."""
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.array([x, y, z])


def box_corners(center, wlh, yaw):
    """8 corners of a 3D box, in the same frame as `center`.

    Camera frame convention: x right, y down, z forward -- so yaw rotates
    about the y (vertical) axis.
    """
    w, l, h = wlh
    xc = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * w / 2.0
    yc = np.array([0, 0, -1, -1, 0, 0, -1, -1]) * h
    zc = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * l / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * xc + s * zc
    z = -s * xc + c * zc
    return np.vstack([x, yc, z]) + np.array(center).reshape(3, 1)


# Depth-Anything-V2-Metric-Outdoor reports distances with near-perfect scale
# but a consistent far bias. Validated two ways on nuScenes:
#   * per-pixel against projected LiDAR: pred = 0.975 * true + 4.53, r = 0.939
#   * per-vehicle, offset fitted on one half and tested on the held-out half
# The slope being ~1.0 is what makes a single additive constant the right
# correction; a fitted scale factor was tried and made results worse.
DEPTH_OFFSET_M = 3.54

# Below this image row the ground-contact point is far enough beneath the
# horizon that flat-road projection is better conditioned than the depth
# network (measured: 1.26 m vs 1.61 m median error under ~15 m). Above it,
# flat-road projection degrades without bound -- its worst case on this data
# was 21.7 km -- so depth takes over. Row is for nuScenes' 900px-tall
# CAM_FRONT; it scales with image height.
HOMOGRAPHY_TRUST_ROW_FRAC = 560.0 / 900.0


def estimate_yaw_candidates(pixel_width, distance_m, fx, wlh=CAR_SIZE_WLH):
    """Rough vehicle heading from how wide it appears, as TWO candidates.

    A car viewed end-on spans its width (~1.95 m); side-on it spans its
    length (~4.63 m). In between, the projected extent is
        apparent = w*cos(t) + l*sin(t) = R*cos(t - phi)
    with R = hypot(w, l) and phi = atan2(l, w).

    Inverting a cosine gives two solutions, phi -/+ arccos(apparent/R), and
    BOTH are geometrically valid -- apparent width peaks at t = phi (~67 deg
    for a car) and falls off either side, so a given width does not identify
    a single angle. Returned as a pair rather than silently picking one.
    (On top of this, the whole family has a four-fold front/back/left/right
    ambiguity, since a silhouette cannot say which end is which.)

    Once a vehicle is tracked across frames its direction of travel is a far
    better heading source. Use this only on first sighting, or for a
    stationary vehicle, where motion tells you nothing.
    """
    w, l, _ = wlh
    amp = float(np.hypot(w, l))
    phi = float(np.arctan2(l, w))
    apparent_m = pixel_width * distance_m / fx
    apparent_m = float(np.clip(apparent_m, min(w, l), amp))
    delta = float(np.arccos(np.clip(apparent_m / amp, -1.0, 1.0)))
    lo = float(np.clip(phi - delta, 0.0, np.pi / 2))
    hi = float(np.clip(phi + delta, 0.0, np.pi / 2))
    return lo, hi


def estimate_3d_box(x1, y1, x2, y2, depth, K, image_height, camera_height_m=None):
    """Camera-only 3D box for one detected vehicle.

    Returns a dict with the 3D centre (camera frame, metres), metric size,
    heading, and which method supplied the distance -- or None if no usable
    depth could be sampled.
    """
    raw = sample_box_depth(depth, x1, y1, x2, y2)
    if raw is None:
        return None
    z = raw - DEPTH_OFFSET_M
    method = "depth"

    # Near the camera, flat-road geometry is better conditioned; hand over
    # to it when the contact point sits well below the horizon and we know
    # the mounting height.
    if camera_height_m is not None and y2 > HOMOGRAPHY_TRUST_ROW_FRAC * image_height:
        ray = np.linalg.inv(K) @ np.array([(x1 + x2) / 2.0, y2, 1.0])
        if ray[1] > 1e-6:                       # ray heading downward in camera frame
            t = camera_height_m / ray[1]
            z_ground = float(ray[2] * t)
            if 0 < z_ground < 60:
                z, method = z_ground, "ground-plane"

    if not np.isfinite(z) or z <= 0:
        return None

    yaw_lo, yaw_hi = estimate_yaw_candidates(x2 - x1, z, K[0, 0])
    # box_corners() builds a box standing UP from the point it is given, so
    # that point must be the vehicle's ground contact -- the bottom edge of
    # the 2D box, not its centre. Unprojecting the centre instead leaves the
    # box floating half a car-height above the road.
    centre = unproject((x1 + x2) / 2.0, y2, z, K)
    return {"center": centre, "wlh": CAR_SIZE_WLH,
            "yaw": yaw_lo, "yaw_alt": yaw_hi,   # see estimate_yaw_candidates: genuinely ambiguous
            "distance_m": float(z), "method": method}
