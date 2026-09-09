"""
How accurately does LiDAR + camera place a vehicle in 3D?

Scored against nuScenes' LiDAR-verified boxes. By default the detection
boxes are taken from ground truth, which isolates the *localisation* step --
otherwise a missed detection and a mislocated vehicle would be mixed into
one number and neither could be diagnosed. Pass --detector to measure the
real end-to-end pipeline instead.

Usage (from the project root):
    python -m src2.evaluate_localization
    python -m src2.evaluate_localization --detector
"""

import argparse
from collections import defaultdict

import numpy as np

from . import config
from .lidar import (is_consistent_with_box, points_in_box_2d, remove_ground,
                    separate_foreground)
from .localize import localize
from .nuscenes_source import NuScenesSource

MAX_RANGE_M = 60.0


def iou_2d(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def summarise(name, errors):
    errors = np.asarray(errors)
    if not len(errors):
        return
    print(f"  {name:22s} n={len(errors):5d}  median {np.median(errors):5.2f} m  "
          f"mean {errors.mean():5.2f} m  90th {np.percentile(errors, 90):6.2f} m  "
          f"within 1m {100 * (errors < 1).mean():3.0f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=3, help="evaluate every Nth frame")
    parser.add_argument("--detector", action="store_true",
                        help="use real YOLO detections instead of ground-truth 2D boxes")
    args = parser.parse_args()

    source = NuScenesSource(load_lidar=True)
    detector = None
    if args.detector:
        from .detection import VehicleDetector
        detector = VehicleDetector()

    by_points = defaultdict(list)
    by_range = defaultdict(list)
    all_errors = []
    yaw_errors = []
    no_points = 0
    considered = 0

    for frame in source.frames(step=args.step):
        truths = source.ground_truth_boxes(frame.sample_token)
        detections_2d = None
        if detector is not None:
            dets = detector.detect(frame.image)
            detections_2d = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

        for truth in truths:
            distance = truth.distance_to_center
            if not (0 < distance <= MAX_RANGE_M) or truth.visibility < 2:
                continue
            projected = source.project_to_image(truth, frame.intrinsics, frame.image.shape)
            if projected is None:
                continue

            if detections_2d is not None:
                # Only score vehicles the detector actually found; recall is
                # already measured separately in evaluate_detection.py.
                best = max(detections_2d, key=lambda d: iou_2d(projected, d), default=None)
                if best is None or iou_2d(projected, best) < 0.4:
                    continue
                box_2d = best
            else:
                box_2d = projected

            considered += 1
            points = points_in_box_2d(frame.lidar_points, frame.lidar_pixels, box_2d)
            points = remove_ground(points, frame.camera_height_m)
            points = separate_foreground(points, config.FOREGROUND_DEPTH_GAP)
            if not is_consistent_with_box(points, box_2d, frame.intrinsics[0, 0]):
                points = points[:, :0]      # points are background, not this vehicle
            estimate = localize(points, frame.camera_height_m)
            if estimate is None:
                no_points += 1
                continue

            # Error in the ground plane -- height is not what collision
            # geometry turns on, and including it would flatter the result.
            error = float(np.hypot(estimate.center[0] - truth.center[0],
                                   estimate.center[2] - truth.center[2]))
            all_errors.append(error)

            n = estimate.n_lidar_points
            bucket = ("1-4" if n < 5 else "5-19" if n < 20 else
                      "20-59" if n < 60 else "60-149" if n < 150 else "150+")
            by_points[bucket].append(error)
            band = ("0-15m" if distance < 15 else "15-30m" if distance < 30
                    else "30-45m" if distance < 45 else "45-60m")
            by_range[band].append(error)

            if estimate.yaw_is_measured:
                gt_yaw = np.arctan2(
                    truth.corners[0, 0] - truth.corners[0, 4],
                    truth.corners[2, 0] - truth.corners[2, 4],
                )
                delta = abs((estimate.yaw - gt_yaw + np.pi / 2) % np.pi - np.pi / 2)
                yaw_errors.append(np.degrees(delta))

    print(f"2D boxes from     : {'YOLO detections' if args.detector else 'ground truth'}")
    print(f"vehicles considered: {considered}")
    print(f"  located          : {len(all_errors)}  ({100 * len(all_errors) / max(considered, 1):.0f}%)")
    print(f"  no LiDAR points  : {no_points}  ({100 * no_points / max(considered, 1):.0f}%)")
    print()
    print("POSITION ERROR (ground plane)")
    summarise("overall", all_errors)
    print()
    print("  by LiDAR points on the vehicle:")
    for bucket in ("1-4", "5-19", "20-59", "60-149", "150+"):
        summarise(f"    {bucket} points", by_points[bucket])
    print()
    print("  by range:")
    for band in ("0-15m", "15-30m", "30-45m", "45-60m"):
        summarise(f"    {band}", by_range[band])

    if yaw_errors:
        y = np.array(yaw_errors)
        print()
        print(f"HEADING (only where >= {config.MIN_POINTS_FOR_YAW} points made it measurable)")
        print(f"  n={len(y)}  median {np.median(y):.1f} deg  "
              f"within 10 deg {100 * (y < 10).mean():.0f}%  "
              f"within 15 deg {100 * (y < 15).mean():.0f}%")


if __name__ == "__main__":
    main()
