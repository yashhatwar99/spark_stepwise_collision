"""
Pixel -> real-world ground-plane position, for a fixed camera looking down
at a flat road from a height, at a downward tilt angle.

This is a homography in spirit: it maps any pixel that lies on the road
surface to a real-world (X, Z) position in meters (X = sideways offset from
directly under the camera's view direction, Z = forward distance along the
ground). The difference from a "4 clicked points" homography is *how* the
mapping is built: from an explicit camera model (height/tilt/focal length)
instead of matched point pairs, since we have no ground-truth survey data
for any of these cameras.

Different videos come from physically different cameras (an elevated
intersection surveillance camera vs. an in-cabin dashcam), so camera
parameters are grouped into named CameraProfile presets rather than one
fixed set of constants. All values are assumptions, not measured -- kept
individually documented so they can be corrected if better information
becomes available. Treat all outputs as approximate.
"""

import math
from dataclasses import dataclass


@dataclass
class CameraProfile:
    height_m: float
    tilt_deg: float
    hfov_deg: float
    # Fraction of frame height below which the camera's own vehicle (hood,
    # dashboard) occludes the road -- any detected box bottom at or past this
    # row can't be its true ground-contact point, so distance there can't be
    # trusted. None for cameras with no such self-occlusion (e.g. a pole-
    # mounted external camera).
    hood_cutoff_frac: float = None


# Elevated mast-arm traffic-signal camera (used for headon-nn.mp4).
# Height: typical mast-arm mounting height (9-11m is common).
# Tilt: estimated by eye from where the horizon (tree line / distant lights)
# sits in that frame -- well above the image's vertical center, consistent
# with a fairly steep downward tilt.
# HFOV: typical for a wide security/traffic camera lens.
INTERSECTION_CAM = CameraProfile(height_m=9.0, tilt_deg=18.0, hfov_deg=70.0)

# In-cabin dashcam (used for the CCD clips, e.g. 000004.mp4, 000011.mp4).
# Height: mounted near the windshield/rearview mirror, roughly eye height
# inside the car -- much lower than a pole-mounted camera.
# Tilt: estimated from 000004.mp4, where the horizon sits close to the
# image's vertical center, consistent with a near-level mount.
# HFOV: a middle-of-the-road guess for a dashcam lens. Note 000011.mp4
# specifically shows visible barrel (fisheye) distortion at the frame edges
# that this simple rectilinear pinhole model does NOT correct for -- treat
# distance estimates for off-center objects in that clip as unreliable until
# a distortion correction step is added.
# hood_cutoff_frac: measured directly from 000004.mp4 -- the dashboard hump
# is visible starting around row 600 of 720 (~0.83 of frame height) in every
# frame from this camera.
DASHCAM = CameraProfile(height_m=1.3, tilt_deg=5.0, hfov_deg=90.0, hood_cutoff_frac=0.83)


def _focal_length_px(image_width, camera):
    return (image_width / 2) / math.tan(math.radians(camera.hfov_deg / 2))


def is_ground_contact_occluded(v, image_height, camera):
    """True if a box bottom at row v is at/past the camera's own hood --
    meaning the vehicle's true ground-contact point is hidden, so any
    ground-plane distance computed from this point can't be trusted."""
    if camera.hood_cutoff_frac is None:
        return False
    return v >= camera.hood_cutoff_frac * image_height


def pixel_to_ground(u, v, image_width, image_height, camera):
    """Map an image pixel (u, v), assumed to lie on the flat road plane,
    to a real-world (X, Z) position in meters relative to the camera.

    Returns None if the pixel is at or above the horizon (no ground-plane
    intersection exists there -- e.g. sky, or a point too far to be valid)
    or if it's occluded by the camera's own hood (see is_ground_contact_occluded).
    """
    if is_ground_contact_occluded(v, image_height, camera):
        return None

    focal_px = _focal_length_px(image_width, camera)
    u_c = image_width / 2
    v_c = image_height / 2

    beta = math.atan((v - v_c) / focal_px)
    alpha = math.radians(camera.tilt_deg) + beta  # total depression angle below horizontal

    if alpha <= 0:
        return None

    z = camera.height_m / math.tan(alpha)
    x = (u - u_c) * z / focal_px
    return x, z


def measure_ground_position(x1, y1, x2, y2, image_width, image_height, camera):
    """Given a detection box, return (status, position) where status is
    "ok", "too_close" (ground-contact point occluded by ego hood), or
    "invalid" (above horizon). position is (x, z) in meters, or None."""
    bottom_center_u = (x1 + x2) / 2
    bottom_center_v = y2

    if is_ground_contact_occluded(bottom_center_v, image_height, camera):
        return "too_close", None

    ground_pos = pixel_to_ground(bottom_center_u, bottom_center_v, image_width, image_height, camera)
    if ground_pos is None:
        return "invalid", None

    return "ok", ground_pos
