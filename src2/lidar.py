"""
LiDAR points, moved into the camera's frame and projected onto its image.

The transform is done the long way round -- through the global frame, using
each sensor's own ego pose -- rather than the tempting shortcut of going
straight from LiDAR to camera. The LiDAR spins at 20 Hz and the camera fires
at 12 Hz, so the two readings in a keyframe are captured tens of
milliseconds apart, and the car keeps moving in between (roughly 9 m/s in
these scenes, so ~0.2 m of ego motion). Ignoring that would smear every
point by about the accuracy we are trying to achieve.

Chain: lidar sensor -> ego(at lidar time) -> global -> ego(at camera time)
       -> camera
"""

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


def load_lidar_in_camera(nusc, sample, camera_channel, lidar_channel="LIDAR_TOP"):
    """Returns (points_xyz, pixels_uv) with the LiDAR sweep in camera
    coordinates, keeping only points in front of the camera that land inside
    the image.

    points_xyz is (3, N) in metres, camera frame (x right, y down, z forward).
    pixels_uv is (2, N), the same points projected to pixel coordinates.
    """
    lidar_sd = nusc.get("sample_data", sample["data"][lidar_channel])
    cam_sd = nusc.get("sample_data", sample["data"][camera_channel])

    pc = LidarPointCloud.from_file(f"{nusc.dataroot}/{lidar_sd['filename']}")

    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    pc.rotate(Quaternion(lidar_cs["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_cs["translation"]))

    lidar_pose = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
    pc.rotate(Quaternion(lidar_pose["rotation"]).rotation_matrix)
    pc.translate(np.array(lidar_pose["translation"]))

    cam_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    pc.translate(-np.array(cam_pose["translation"]))
    pc.rotate(Quaternion(cam_pose["rotation"]).rotation_matrix.T)

    cam_cs = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    pc.translate(-np.array(cam_cs["translation"]))
    pc.rotate(Quaternion(cam_cs["rotation"]).rotation_matrix.T)

    points = pc.points[:3, :]
    K = np.array(cam_cs["camera_intrinsic"])

    infront = points[2, :] > 0.5      # behind the lens, or on it, cannot project
    points = points[:, infront]

    uv = K @ points
    uv = uv[:2] / uv[2]

    width, height = cam_sd["width"], cam_sd["height"]
    on_image = (uv[0] >= 0) & (uv[0] < width) & (uv[1] >= 0) & (uv[1] < height)
    return points[:, on_image], uv[:, on_image]


def points_in_box_2d(points_xyz, pixels_uv, box, shrink=0.15):
    """LiDAR points whose pixels fall inside a 2D detection box.

    The box is shrunk toward its centre first. A detector's box hugs the
    vehicle's silhouette, so its outer edge straddles the boundary between
    vehicle and whatever is behind it -- points sampled there belong to the
    background as often as to the vehicle.
    """
    x1, y1, x2, y2 = box
    mx, my = shrink * (x2 - x1), shrink * (y2 - y1)
    inside = (
        (pixels_uv[0] >= x1 + mx) & (pixels_uv[0] <= x2 - mx)
        & (pixels_uv[1] >= y1 + my) & (pixels_uv[1] <= y2 - my)
    )
    return points_xyz[:, inside]


def separate_foreground(points_xyz, depth_gap=2.0, min_cluster=2):
    """Pick out the vehicle from the other things sharing its 2D box.

    Anything directly behind a vehicle -- road, wall, another car -- projects
    into the same box, so the points arrive as several groups spread along
    the depth axis. Splitting wherever there is a gap larger than a car is
    long, then keeping the group with the most points, picks the object that
    actually fills the box.

    Largest-group rather than nearest-group was chosen on measurement: nearest
    gave 2.35 m median error, largest 1.51 m, because a thin foreground
    object (a pole, a mirror, the edge of a closer car) is nearer but
    contributes few points.
    """
    if points_xyz.shape[1] == 0:
        return points_xyz

    order = np.argsort(points_xyz[2, :])
    ordered = points_xyz[:, order]
    depths = ordered[2, :]

    split_at = np.where(np.diff(depths) > depth_gap)[0] + 1
    groups = np.split(np.arange(ordered.shape[1]), split_at)

    substantial = [g for g in groups if len(g) >= min_cluster] or groups
    biggest = max(substantial, key=len)
    return ordered[:, biggest]


def remove_ground(points_xyz, camera_height_m, clearance_m=0.4):
    """Drop points lying on the road surface.

    A 2D detection box does not tightly hug a vehicle -- road is visible
    around its silhouette, and those returns project inside the box. They
    matter more than their number suggests: road points form a continuous
    ramp in depth, so they connect to the vehicle's own cluster instead of
    splitting away from it, and they stretch the footprint used to fit a
    heading. Measured effect of leaving them in: heading error 58.9 degrees,
    barely better than guessing.

    In camera coordinates y points down, so the road sits at
    y = camera_height_m. Anything within `clearance_m` of that is ground.
    Vehicles keep their wheels but lose the tarmac.
    """
    if points_xyz.shape[1] == 0:
        return points_xyz
    above_road = points_xyz[1, :] < (camera_height_m - clearance_m)
    # If nothing survives, the vehicle was probably represented only by
    # low returns; hand back what we had rather than nothing at all.
    return points_xyz[:, above_road] if above_road.any() else points_xyz


def is_consistent_with_box(points_xyz, box_2d, focal_length_px, max_width_m=7.0):
    """Do these points plausibly belong to the object filling this 2D box?

    Guards against the failure that matters most: when a vehicle returns no
    laser points at all -- common past 45 m, where 40% of vehicles are missed
    entirely -- whatever lies behind it does fall inside its 2D box, and gets
    reported with total confidence as the vehicle's position.

    The camera settles it. A detection box of a given pixel width, at depth
    Z, implies a real-world width of `box_width * Z / focal_length`. If the
    points sat on the vehicle that fills the box, that width comes out
    vehicle-sized; if they sit on a wall much further back, the implied width
    is far too large for any vehicle. So the camera's angular measurement
    audits the LiDAR's depth measurement -- each sensor checking the other
    where it is strong.
    """
    if points_xyz.shape[1] == 0:
        return False
    x1, _, x2, _ = box_2d
    depth = float(np.median(points_xyz[2, :]))
    implied_width_m = (x2 - x1) * depth / focal_length_px
    return implied_width_m <= max_width_m
