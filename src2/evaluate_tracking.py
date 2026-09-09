"""
Does tracked direction of travel actually give a usable heading?

This is the question the whole tracking step exists to answer. Geometry got
55-60 degrees, which is no better than guessing; published detectors report
31-65 degrees averaged over classes. If motion-derived heading lands near a
few degrees, the 3D box is finished.

Frames are walked in order within each scene and the tracker is reset at
scene boundaries -- tracking across a cut would associate unrelated vehicles
and quietly invent motion.

Usage (from the project root):
    python -m src2.evaluate_tracking
    python -m src2.evaluate_tracking --detector --min-speed 2
"""

import argparse
from collections import defaultdict

import numpy as np

from . import config
from .lidar import (is_consistent_with_box, points_in_box_2d, remove_ground,
                    separate_foreground)
from .localize import localize
from .nuscenes_source import NuScenesSource
from .tracking import Tracker

MAX_RANGE_M = 45.0


def ground_truth_heading(truth):
    """GT direction of travel in camera axes, as a yaw about the vertical.

    nuScenes orders box corners with the first four on the face the vehicle
    faces, so front-centre minus rear-centre is the way it points.
    """
    front = truth.corners[:, 0:4].mean(axis=1)
    rear = truth.corners[:, 4:8].mean(axis=1)
    return float(np.arctan2(front[0] - rear[0], front[2] - rear[2]))


def angle_error_deg(a, b):
    """Smallest angle between two headings, in degrees, over the full circle.

    Full circle rather than modulo 180: a vehicle driving towards you and one
    driving away are the same box but opposite situations, and collision
    prediction depends on telling them apart.
    """
    return float(np.degrees(abs((a - b + np.pi) % (2 * np.pi) - np.pi)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", action="store_true",
                        help="use YOLO detections rather than ground-truth 2D boxes")
    parser.add_argument("--min-speed", type=float, default=config.MIN_SPEED_FOR_HEADING,
                        help="m/s below which a heading is refused as noise")
    parser.add_argument("--scenes", type=int, default=10)
    args = parser.parse_args()

    source = NuScenesSource(load_lidar=True)
    detector = None
    if args.detector:
        from .detection import VehicleDetector
        detector = VehicleDetector()

    tracker = Tracker()
    current_scene = None
    errors, by_speed = [], defaultdict(list)
    refused = matched = 0
    scenes_seen = set()

    for frame in source.frames():
        if frame.scene_name != current_scene:
            scenes_seen.add(frame.scene_name)
            if len(scenes_seen) > args.scenes:
                break
            current_scene = frame.scene_name
            tracker = Tracker()      # never associate across a scene cut

        truths = [t for t in source.ground_truth_boxes(frame.sample_token)
                  if 0 < t.distance_to_center <= MAX_RANGE_M and t.visibility >= 2]
        if not truths:
            continue

        boxes_2d = None
        if detector is not None:
            boxes_2d = [(d.x1, d.y1, d.x2, d.y2) for d in detector.detect(frame.image)]

        estimates, paired_truth = [], []
        for truth in truths:
            projected = source.project_to_image(truth, frame.intrinsics, frame.image.shape)
            if projected is None:
                continue
            box_2d = projected
            if boxes_2d is not None:
                best, best_iou = None, 0.0
                for candidate in boxes_2d:
                    ix1, iy1 = max(projected[0], candidate[0]), max(projected[1], candidate[1])
                    ix2, iy2 = min(projected[2], candidate[2]), min(projected[3], candidate[3])
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    if inter <= 0:
                        continue
                    union = ((projected[2] - projected[0]) * (projected[3] - projected[1])
                             + (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]) - inter)
                    score = inter / union
                    if score > best_iou:
                        best, best_iou = candidate, score
                if best is None or best_iou < 0.4:
                    continue
                box_2d = best

            points = points_in_box_2d(frame.lidar_points, frame.lidar_pixels, box_2d)
            points = remove_ground(points, frame.camera_height_m)
            points = separate_foreground(points, config.FOREGROUND_DEPTH_GAP)
            if not is_consistent_with_box(points, box_2d, frame.intrinsics[0, 0]):
                continue
            box = localize(points, frame.camera_height_m)
            if box is None:
                continue
            estimates.append(box)
            paired_truth.append(truth)

        if not estimates:
            continue

        # Track in the global frame -- see tracking.py for why camera-frame
        # tracking would report parked cars as moving.
        centres_cam = np.stack([b.center for b in estimates], axis=1)
        centres_global = frame.to_global(centres_cam)
        ids = tracker.update([centres_global[:2, i] for i in range(centres_global.shape[1])],
                             frame.timestamp_s)

        for i, track_id in enumerate(ids):
            yaw = tracker.heading_in_camera(track_id, frame, min_speed=args.min_speed)
            if yaw is None:
                refused += 1
                continue
            matched += 1
            error = angle_error_deg(yaw, ground_truth_heading(paired_truth[i]))
            errors.append(error)
            speed = tracker.tracks[track_id].speed or 0.0
            band = ("1-3 m/s" if speed < 3 else "3-8 m/s" if speed < 8 else "8+ m/s")
            by_speed[band].append(error)

    print(f"2D boxes from : {'YOLO detections' if detector else 'ground truth'}")
    print(f"scenes        : {len(scenes_seen)}")
    print(f"heading given : {matched}")
    print(f"heading refused (too slow to read): {refused}")
    print()
    if not errors:
        print("no headings produced")
        return
    e = np.array(errors)
    print("HEADING ERROR (full circle, so front/back confusion counts as an error)")
    print(f"  median {np.median(e):5.1f} deg   mean {e.mean():5.1f} deg")
    print(f"  within  5 deg: {100 * (e < 5).mean():3.0f}%")
    print(f"  within 10 deg: {100 * (e < 10).mean():3.0f}%")
    print(f"  within 15 deg: {100 * (e < 15).mean():3.0f}%")
    print(f"  over   90 deg: {100 * (e > 90).mean():3.0f}%   (pointing the wrong way entirely)")
    print()
    print("  by speed:")
    for band in ("1-3 m/s", "3-8 m/s", "8+ m/s"):
        if by_speed[band]:
            b = np.array(by_speed[band])
            print(f"    {band:9s} n={len(b):4d}  median {np.median(b):5.1f} deg  "
                  f"within 15 deg {100 * (b < 15).mean():3.0f}%")


if __name__ == "__main__":
    main()
