"""
Detection + LiDAR -> a 3D box in the camera's frame.

The division of labour is the point of this module. A camera measures the
*angle* to a vehicle almost perfectly (checked against ground truth: error
in hundredths of a degree) but cannot measure how far along that angle the
vehicle sits. LiDAR measures exactly that distance. Neither sensor can do
the other's job, so each is asked only for what it is good at.

Position is found by averaging the visible points and then pushing the
result away from the camera by a quarter of a car length -- the laser only
ever hits the near face, so the points sit systematically in front of where
the vehicle's middle actually is.

A minimum-area-rectangle fit was tried as the alternative and measured
WORSE for position (1.68 m vs 1.04 m median): fitting a hull to a one-sided
partial scan moves the centre more than plain averaging does. It was the
only route to a geometric heading, but that came out at 55-59 degrees error,
so it has been removed rather than left in as dead weight. Heading now comes
from tracked direction of travel.
"""

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Box3D:
    """A vehicle's 3D box in camera coordinates (x right, y down, z forward)."""

    center: np.ndarray          # (3,) metres -- the middle of the vehicle
    size: np.ndarray            # (3,) width, length, height in metres
    yaw: float                  # radians about the vertical axis; 0 = facing along +z
    distance_m: float           # forward distance, the number risk logic cares about
    n_lidar_points: int
    source: str                 # which estimator produced it
    yaw_is_measured: bool       # False when yaw is a placeholder awaiting tracking

    @property
    def bearing_deg(self):
        return float(np.degrees(np.arctan2(self.center[0], self.center[2])))


def localize(points_xyz, camera_height_m, size_prior=None, yaw=None):
    """Build a 3D box from the LiDAR points already isolated onto one vehicle.

    `points_xyz` must be the foreground points for a single vehicle, in the
    camera frame -- see lidar.points_in_box_2d + lidar.separate_foreground.
    Pass `yaw` (radians) once a tracker knows the vehicle's direction of
    travel; without it the box gets a placeholder orientation that is
    visibly wrong for any vehicle not driving straight away from us.
    Returns None when there is nothing to work with.
    """
    size = np.array(size_prior if size_prior is not None else config.CAR_SIZE_WLH)
    n = points_xyz.shape[1]
    if n == 0:
        return None

    # Ground-plane coordinates: x across, z along the line of sight.
    xz = np.stack([points_xyz[0, :], points_xyz[2, :]], axis=1)

    centroid = xz.mean(axis=0)
    # Push away from the camera: the visible surface is the near face,
    # roughly a quarter of a car length in front of the true centre.
    bearing = centroid / (np.linalg.norm(centroid) + 1e-9)
    centre = centroid + bearing * (size[1] * 0.25)
    cx, cz = float(centre[0]), float(centre[1])

    yaw_measured = yaw is not None
    if not yaw_measured:
        # Placeholder: lay the vehicle along our own line of sight. This is
        # a guess, not a measurement, and it is the single most visible
        # error in the rendered boxes -- a crossing vehicle gets a box
        # pointing at the camera, so the box overhangs the car badly.
        yaw = float(np.arctan2(bearing[0], bearing[1]))

    # Stand the box on the road. The road sits camera_height_m below the
    # camera, and y points down, so that is the base; the centre is half a
    # vehicle height above it. Taken from geometry rather than from the
    # points because a sparse sweep often misses both roof and wheels.
    cy = camera_height_m - size[2] / 2.0

    return Box3D(
        center=np.array([cx, cy, cz]),
        size=size,
        yaw=float(yaw),
        distance_m=float(cz),
        n_lidar_points=int(n),
        source="lidar-centroid",
        yaw_is_measured=yaw_measured,
    )


def box_corners(box):
    """The box's 8 corners in camera coordinates, for drawing or overlap tests.

    Order: 0-3 are the front face (in the vehicle's own facing direction),
    4-7 the rear, each running bottom-left, bottom-right, top-right, top-left.
    """
    w, l, h = box.size
    x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * w / 2.0
    y = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * h / 2.0
    z = np.array([1, 1, 1, 1, -1, -1, -1, -1]) * l / 2.0

    c, s = np.cos(box.yaw), np.sin(box.yaw)
    # Rotate about the vertical (y) axis: x and z mix, y is untouched.
    xr = c * x + s * z
    zr = -s * x + c * z
    return np.vstack([xr, y, zr]) + box.center.reshape(3, 1)
