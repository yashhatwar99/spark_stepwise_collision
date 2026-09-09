"""
The tight-fit geometry: 2D box + size + orientation -> 3D position.

DERIVATION
----------
A corner of the object, X in its own frame, lands in the camera at
    P = R(yaw) @ X + T
and projects to
    u = fx * P_x / P_z + cx        v = fy * P_y / P_z + cy

Requiring that corner to sit exactly on the 2D box's left edge, u = x1:

    fx * (RX + T)_x / (RX + T)_z + cx = x1
    fx * (RX)_x + fx * T_x = (x1 - cx) * (RX)_z + (x1 - cx) * T_z
    fx * T_x - (x1 - cx) * T_z = (x1 - cx) * (RX)_z - fx * (RX)_x

T appears only linearly, so each of the four edges contributes one row of
A T = b and the position is a least-squares solve. No iteration, no
optimiser, no depth measurement anywhere.

THE CATCH, AND WHY THIS ENUMERATES
----------------------------------
Which corner touches which edge depends on the viewing angle -- seen from
behind, a car's rear corners are extremal; seen from the side, its front
ones are. The assignment cannot be known in advance, so every plausible one
is solved and the winner is whichever reprojects closest to the observed
box. That search is the reason this is not a one-liner.
"""

import numpy as np


def rotation_y(yaw):
    """Rotation about the camera's vertical axis (y points down)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0.0, s],
                     [0.0, 1.0, 0.0],
                     [-s, 0.0, c]])


def object_corners(wlh):
    """The 8 corners in the object's own frame, origin at its base centre.

    y is down, so the roof sits at -h and the wheels at 0. The box then
    stands on the ground plane instead of floating with its middle on it.
    Corners 0-3 are the base, 4-7 the roof.
    """
    w, l, h = wlh
    x = np.array([1, 1, -1, -1, 1, 1, -1, -1]) * w / 2.0
    y = np.array([0, 0, 0, 0, -1, -1, -1, -1]) * h
    z = np.array([1, -1, -1, 1, 1, -1, -1, 1]) * l / 2.0
    return np.stack([x, y, z])          # (3, 8)


def _edge_equation(rotated_corner, edge_value, K, axis):
    """One row of A and one entry of b, for a corner pinned to one edge.

    `axis` is 0 for a vertical edge (x1/x2, constraining u) and 1 for a
    horizontal one (y1/y2, constraining v).
    """
    f = K[axis, axis]
    offset = edge_value - K[axis, 2]
    row = np.zeros(3)
    row[axis] = f
    row[2] = -offset
    b = offset * rotated_corner[2] - f * rotated_corner[axis]
    return row, b


def solve_translation(box_2d, wlh, yaw, K, corner_choice):
    """Least-squares 3D position for one corner-to-edge assignment.

    `corner_choice` names which corner touches (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = box_2d
    rotated = rotation_y(yaw) @ object_corners(wlh)

    rows, rhs = [], []
    for edge_value, axis, corner_index in (
        (x1, 0, corner_choice[0]),
        (y1, 1, corner_choice[1]),
        (x2, 0, corner_choice[2]),
        (y2, 1, corner_choice[3]),
    ):
        row, b = _edge_equation(rotated[:, corner_index], edge_value, K, axis)
        rows.append(row)
        rhs.append(b)

    try:
        translation, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return translation


def project_box(translation, wlh, yaw, K):
    """The 3D box's 8 corners in pixels, or None if it is behind the camera."""
    points = rotation_y(yaw) @ object_corners(wlh) + np.asarray(translation).reshape(3, 1)
    if points[2].min() <= 0.1:
        return None
    uv = K @ points
    return uv[:2] / uv[2]


def reprojection_error(translation, box_2d, wlh, yaw, K):
    """Total pixel gap between the projected box's extremes and the 2D box."""
    uv = project_box(translation, wlh, yaw, K)
    if uv is None:
        return np.inf
    predicted = np.array([uv[0].min(), uv[1].min(), uv[0].max(), uv[1].max()])
    return float(np.abs(predicted - np.asarray(box_2d, dtype=float)).sum())


# Assignments worth trying. The unrestricted space is 8^4 = 4096 per object
# per angle, far more than the geometry permits: a vertical edge is touched
# by a corner that is genuinely leftmost or rightmost, and the bottom edge of
# a vehicle standing on the road is a wheel corner, never a roof one.
# Restricting bottom and side edges to base corners and the top edge to roof
# corners leaves a few hundred assignments and discards none a real vehicle
# can produce.
_BASE = (0, 1, 2, 3)
_ROOF = (4, 5, 6, 7)
CANDIDATE_ASSIGNMENTS = [
    (left, top, right, bottom)
    for left in _BASE
    for right in _BASE
    if right != left
    for top in _ROOF
    for bottom in _BASE
]


def fit_box(box_2d, wlh, yaw, K, max_range_m=90.0):
    """Best 3D position for a known size and orientation.

    Returns (translation, reprojection_error); (None, inf) if nothing fit.
    """
    best, best_error = None, np.inf
    for assignment in CANDIDATE_ASSIGNMENTS:
        translation = solve_translation(box_2d, wlh, yaw, K, assignment)
        if translation is None or not np.isfinite(translation).all():
            continue
        # Reject nonsense before paying for a projection: behind the camera,
        # or implausibly far away.
        if not (0.5 < translation[2] < max_range_m):
            continue
        error = reprojection_error(translation, box_2d, wlh, yaw, K)
        if error < best_error:
            best, best_error = translation, error
    return best, best_error


def fit_box_search_yaw(box_2d, wlh, K, yaw_candidates=None, max_range_m=90.0):
    """Fit when orientation is unknown too, by trying angles.

    Reprojection error alone picks the angle here, and it is a weak signal --
    several orientations fit one rectangle nearly as well, the same
    front/back ambiguity that defeated shape fitting earlier in this project.
    Pass a tracked heading instead whenever one exists.
    """
    if yaw_candidates is None:
        yaw_candidates = np.linspace(-np.pi, np.pi, 37)      # every 10 degrees
    best, best_yaw, best_error = None, None, np.inf
    for yaw in yaw_candidates:
        translation, error = fit_box(box_2d, wlh, float(yaw), K, max_range_m)
        if error < best_error:
            best, best_yaw, best_error = translation, float(yaw), error
    return best, best_yaw, best_error
