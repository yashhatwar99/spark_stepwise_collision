"""
Same vehicles, same code, two different depth sensors.

The localisation path takes 3D points and does not care which sensor made
them, so stereo can be swapped in for LiDAR without touching detection,
foreground separation, or the box estimator. That makes a genuinely fair
comparison possible: run both through the identical pipeline on identical
detections, and the only difference left is the sensor.

LiDAR is treated as the reference rather than as truth in the strict sense --
KITTI's raw drives ship no object labels -- but at ~2 cm per point it is far
more accurate than either estimate needs to resolve.

Usage:
    python -m src2.compare_stereo_lidar --frames 30
"""

import argparse

import numpy as np

from . import config
from .detection import VehicleDetector
from .kitti_source import KittiStereoSource
from .lidar import points_in_box_2d, remove_ground, separate_foreground
from .localize import localize
from .stereo import compute_disparity, depth_from_disparity, depth_map_to_points, make_matcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--step", type=int, default=5)
    args = parser.parse_args()

    source = KittiStereoSource()
    detector = VehicleDetector()
    matcher = make_matcher()
    f, b, cx, cy = source.focal_length_px, source.baseline_m, source.cx, source.cy

    # KITTI's cameras sit ~1.65 m above the road.
    camera_height_m = 1.65

    rows = []
    for i in range(0, min(args.frames * args.step, len(source)), args.step):
        frame = source.frame(i)
        if frame.lidar_points_cam is None:
            continue

        depth = depth_from_disparity(compute_disparity(frame.left, frame.right, matcher), f, b)
        stereo_pts, stereo_uv = depth_map_to_points(depth, f, cx, cy)

        # The LiDAR points and their pixels must stay index-aligned, so both
        # are built from one mask rather than from two separate filters.
        pts = frame.lidar_points_cam[:, frame.lidar_points_cam[2, :] > 1.0]
        uv = source.P2 @ np.vstack([pts, np.ones((1, pts.shape[1]))])
        uv = uv[:2] / uv[2]
        h, w = frame.left.shape[:2]
        on_image = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
        lidar_pts, lidar_uv = pts[:, on_image], uv[:, on_image]

        for det in detector.detect(frame.left):
            box_2d = (det.x1, det.y1, det.x2, det.y2)

            def locate(points, pixels):
                p = points_in_box_2d(points, pixels, box_2d)
                p = remove_ground(p, camera_height_m)
                p = separate_foreground(p, config.FOREGROUND_DEPTH_GAP)
                return localize(p, camera_height_m)

            from_lidar = locate(lidar_pts, lidar_uv)
            from_stereo = locate(stereo_pts, stereo_uv)
            if from_lidar is None or from_stereo is None:
                continue
            if not (3 < from_lidar.distance_m < 70):
                continue
            rows.append((
                from_lidar.distance_m,
                float(np.hypot(from_stereo.center[0] - from_lidar.center[0],
                               from_stereo.center[2] - from_lidar.center[2])),
                from_lidar.n_lidar_points,
                from_stereo.n_lidar_points,
            ))

    a = np.array(rows)
    if not len(a):
        print("no vehicles compared")
        return
    reference, disagreement, n_lidar, n_stereo = a.T

    print(f"vehicles compared : {len(a)}")
    print(f"points per vehicle: LiDAR median {int(np.median(n_lidar))}, "
          f"stereo median {int(np.median(n_stereo))}")
    print()
    print("STEREO vs LIDAR, same detections, same localisation code")
    print(f"{'range':>10s} {'n':>5s} {'median gap':>12s} {'90th':>8s} {'within 1m':>10s}")
    for lo, hi in [(3, 10), (10, 20), (20, 30), (30, 50), (50, 70)]:
        m = (reference >= lo) & (reference < hi)
        if m.sum() < 3:
            continue
        d = disagreement[m]
        print(f"  {lo:2d}-{hi:2d} m {m.sum():5d} {np.median(d):11.2f}m "
              f"{np.percentile(d, 90):7.2f}m {100 * (d < 1).mean():9.0f}%")
    print()
    print(f"overall median disagreement: {np.median(disagreement):.2f} m")


if __name__ == "__main__":
    main()
