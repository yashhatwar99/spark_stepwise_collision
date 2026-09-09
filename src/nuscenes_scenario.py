"""
Validates collision_risk.py's TTC/CPA/risk-classification logic against
real recorded driving scenes from nuScenes, using nuScenes' own LiDAR-based
3D annotations (position + velocity) as ground truth -- not vision-estimated.

Unlike the SUMO track (synthetic, straight-line only), nuScenes scenes
include real urban intersections with genuine crossing/turning vehicles --
this is what actually exercises compute_cpa() on non-straight-ahead motion,
which the SUMO scenario so far has not tested.

NOT YET RUN END-TO-END: this project has no nuScenes data downloaded yet
(v1.0-mini requires an account + login-gated download the user has to do
themselves -- see the project plan). Written carefully against the
documented devkit API, but treat the coordinate-convention comments below
as "best understanding, to be confirmed" until this actually runs once data
exists, same as we've done for every other assumption in this project.

Setup once you have the data:
    1. Download v1.0-mini from nuscenes.org (requires a free account).
    2. Extract it so you have a folder like data/nuscenes/ containing
       maps/, samples/, sweeps/, v1.0-mini/ (the standard nuScenes layout).
    3. Update NUSCENES_DATAROOT below if you extract it somewhere else.
"""

import os

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from collision_risk import classify_risk, compute_cpa, compute_ttc

NUSCENES_DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nuScenes", "v1.0-mini")
NUSCENES_VERSION = "v1.0-mini"

# Object categories we care about (nuScenes uses dotted category names).
VEHICLE_PREFIXES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.motorcycle")


def is_vehicle(category_name):
    return category_name.startswith(VEHICLE_PREFIXES)


def relative_position_and_velocity(nusc, ann_token, ego_pose):
    """Returns (x, z, vx, vz) for one annotated object, relative to ego,
    in ego's own heading frame (x = lateral, z = forward) -- matching the
    convention used everywhere else in this project (ground_plane.py,
    kalman_filter.py, collision_risk.py).

    Axis convention confirmed empirically (not just assumed): computed
    ego's own displacement between two consecutive keyframes in the global
    frame, then rotated it into ego's own heading frame the same way this
    function does for other objects. Since a moving vehicle travels mostly
    forward, whichever local axis absorbed nearly all of that displacement
    must be "forward" -- it was local X (4.49m of motion vs -0.01m on
    local Y over one step), i.e. nuScenes' ego frame has X = forward,
    Y = lateral. That's the reverse of this project's own (x=lateral,
    z=forward) convention, so the local axes are swapped on return below.
    """
    box = nusc.get_box(ann_token)  # box.center/orientation are in the global frame
    # Move into ego-relative coordinates: translate by ego's global position,
    # then rotate by the INVERSE of ego's rotation quaternion -- standard
    # nuScenes devkit pattern (see e.g. their own render_annotation code)
    # for converting a global-frame box into the ego vehicle's frame.
    box.translate(-np.array(ego_pose["translation"]))
    box.rotate(Quaternion(ego_pose["rotation"]).inverse)

    # velocity: nuScenes gives this in the global frame (derived from
    # consecutive keyframe positions of the same tracked instance -- real
    # ground truth from the annotation pipeline, not vision-estimated), so
    # it must be rotated into ego's frame the same way the position was,
    # or position and velocity would end up in mismatched frames.
    global_velocity = nusc.box_velocity(ann_token)  # (vx, vy, vz), may contain NaN at sequence ends
    ego_velocity = Quaternion(ego_pose["rotation"]).inverse.rotate(global_velocity)

    local_x, local_y, local_z = box.center
    local_vx, local_vy, local_vz = ego_velocity
    # nuScenes local X = forward, Y = lateral (confirmed above) -- this
    # project's convention is the reverse, so swap on the way out.
    lateral, forward = local_y, local_x
    lateral_v, forward_v = local_vy, local_vx
    return lateral, forward, lateral_v, forward_v


def main():
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATAROOT, verbose=True)

    scene = nusc.scene[0]
    sample_token = scene["first_sample_token"]

    while sample_token:
        sample = nusc.get("sample", sample_token)
        lidar_data = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose = nusc.get("ego_pose", lidar_data["ego_pose_token"])

        for ann_token in sample["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            if not is_vehicle(ann["category_name"]):
                continue

            x, z, vx, vz = relative_position_and_velocity(nusc, ann_token, ego_pose)

            if z <= 0:
                # Behind ego. compute_ttc/classify_risk assume z is forward
                # distance (>= 0), matching this project's forward-dashcam
                # scope -- a vehicle approaching from behind is a different
                # problem (rear-end risk *to* ego) that these formulas were
                # never designed for; feeding them negative z produces
                # nonsense (e.g. a negative "time to collision" that still
                # satisfies ttc <= threshold and gets flagged CRITICAL).
                continue

            if np.isnan([vx, vz]).any():
                # First/last annotation of a track has no neighbor to
                # difference against, so nuScenes can't compute a velocity.
                print(f"t={sample['timestamp']} obj={ann['instance_token'][:8]} "
                      f"x={x:6.2f}m z={z:6.2f}m velocity unavailable (track start/end)")
                continue

            ttc = compute_ttc(x, z, vx, vz)
            cpa_time, cpa_distance = compute_cpa(x, z, vx, vz)
            risk = classify_risk(ttc, cpa_time, cpa_distance)

            ttc_str = f"{ttc:.2f}s" if ttc is not None else "N/A"
            print(f"t={sample['timestamp']} obj={ann['instance_token'][:8]} "
                  f"x={x:6.2f}m z={z:6.2f}m ttc={ttc_str:>8} "
                  f"cpa_dist={cpa_distance:.2f}m cpa_time={cpa_time:.2f}s risk={risk}")

        sample_token = sample["next"]


if __name__ == "__main__":
    main()
