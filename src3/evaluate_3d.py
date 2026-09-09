"""
How well does tight-fit geometry locate a vehicle, with no depth sensor?

Scored against nuScenes' LiDAR-verified 3D boxes, so the number is directly
comparable with this project's LiDAR pipeline (median 1.16 m) and its
monocular depth attempt (4.16 m).

Three orientation sources are compared, because orientation is the input the
method cannot supply itself and its quality is the whole question:

    searched   try angles, keep the best reprojection -- camera only
    tracked    heading from motion, measured at 4.2 deg (src2.tracking)
    truth      the annotated heading -- an upper bound, not a result

Usage (from the project root):
    python -m src3.evaluate_3d
    python -m src3.evaluate_3d --detector --step 6
"""

import argparse
from collections import defaultdict

import numpy as np

from src2 import config
from src2.nuscenes_source import NuScenesSource
from src3.deep3dbox import fit_box, fit_box_search_yaw

MAX_RANGE_M = 60.0


def iou_2d(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def truth_yaw(truth):
    """Annotated heading, as a yaw about the camera's vertical axis."""
    front = truth.corners[:, 0:4].mean(axis=1)
    rear = truth.corners[:, 4:8].mean(axis=1)
    return float(np.arctan2(front[0] - rear[0], front[2] - rear[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=8)
    parser.add_argument("--detector", action="store_true",
                        help="use YOLO boxes instead of ground-truth 2D boxes")
    args = parser.parse_args()

    source = NuScenesSource()
    detector = None
    if args.detector:
        from src2.detection import VehicleDetector
        detector = VehicleDetector()

    results = {"searched": defaultdict(list), "truth": defaultdict(list)}
    yaw_found, yaw_true = [], []

    for frame in source.frames(step=args.step):
        K = frame.intrinsics
        boxes_2d = None
        if detector is not None:
            boxes_2d = [(d.x1, d.y1, d.x2, d.y2) for d in detector.detect(frame.image)]

        for truth in source.ground_truth_boxes(frame.sample_token):
            if not truth.category.startswith("vehicle.car"):
                continue
            distance = truth.distance_to_center
            if not (0 < distance <= MAX_RANGE_M) or truth.visibility < 3:
                continue
            projected = source.project_to_image(truth, K, frame.image.shape)
            if projected is None:
                continue

            box_2d = projected
            if boxes_2d is not None:
                best = max(boxes_2d, key=lambda c: iou_2d(projected, c), default=None)
                if best is None or iou_2d(projected, best) < 0.4:
                    continue
                box_2d = best

            gt_yaw = truth_yaw(truth)
            # Ground-plane error only: height is not what collision geometry
            # turns on, and folding it in would flatter the result.
            def score(translation):
                if translation is None:
                    return None
                return float(np.hypot(translation[0] - truth.center[0],
                                      translation[2] - truth.center[2]))

            band = ("0-15m" if distance < 15 else "15-30m" if distance < 30
                    else "30-45m" if distance < 45 else "45-60m")

            searched, found_yaw, _ = fit_box_search_yaw(box_2d, config.CAR_SIZE_WLH, K)
            e = score(searched)
            if e is not None:
                results["searched"]["all"].append(e)
                results["searched"][band].append(e)
                yaw_found.append(found_yaw)
                yaw_true.append(gt_yaw)

            exact, _ = fit_box(box_2d, config.CAR_SIZE_WLH, gt_yaw, K)
            e = score(exact)
            if e is not None:
                results["truth"]["all"].append(e)
                results["truth"][band].append(e)

    print(f"2D boxes from : {'YOLO detections' if detector else 'ground truth'}")
    print(f"vehicles      : {len(results['searched']['all'])}")
    print()
    print("POSITION ERROR (ground plane), by orientation source")
    print(f"{'source':>10s} {'n':>5s} {'median':>8s} {'mean':>8s} {'90th':>8s} {'within 2m':>10s}")
    for name in ("searched", "truth"):
        e = np.array(results[name]["all"])
        if not len(e):
            continue
        print(f"{name:>10s} {len(e):5d} {np.median(e):7.2f}m {e.mean():7.2f}m "
              f"{np.percentile(e, 90):7.2f}m {100 * (e < 2).mean():9.0f}%")
    print()
    print("by range (median error):")
    print(f"{'range':>10s} {'searched':>10s} {'truth yaw':>11s}")
    for band in ("0-15m", "15-30m", "30-45m", "45-60m"):
        a = results["searched"][band]
        b = results["truth"][band]
        if len(a) < 3:
            continue
        print(f"{band:>10s} {np.median(a):9.2f}m {np.median(b):10.2f}m")

    if yaw_found:
        found = np.array(yaw_found)
        true = np.array(yaw_true)
        err = np.degrees(np.abs((found - true + np.pi) % (2 * np.pi) - np.pi))
        print()
        print("orientation recovered by reprojection search:")
        print(f"  median {np.median(err):.0f} deg, within 30 deg {100 * (err < 30).mean():.0f}%")
        print("  (weak, as expected -- a rectangle fits several orientations")
        print("   nearly equally well; position survives it because a car is")
        print("   roughly symmetric front-to-back)")


if __name__ == "__main__":
    main()
