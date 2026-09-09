"""
How good is the detector, actually?

Scores YOLO's 2D boxes against nuScenes' LiDAR-verified vehicles, matching
by IoU. This exists because "the pictures look right" is not a measurement,
and because every later stage inherits whatever the detector misses -- a
vehicle never detected can never be ranged, tracked, or risk-assessed.

Reported separately by range and by day/night, since a single averaged
number would hide exactly the cases that matter.

Usage (from the project root):
    python -m src2.evaluate_detection
    python -m src2.evaluate_detection --step 5 --min-visibility 3
"""

import argparse
from collections import defaultdict
from pathlib import Path

from .detection import VehicleDetector
from .nuscenes_source import NuScenesSource

DEFAULT_IOU_MATCH = 0.4
MAX_RANGE_M = 60.0


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match(detections, truths, iou_threshold=DEFAULT_IOU_MATCH):
    """Greedy highest-IoU matching, each detection used at most once.

    Greedy rather than optimal (Hungarian) assignment: with vehicles rarely
    overlapping heavily in image space the two agree, and greedy keeps the
    scoring simple enough to trust by reading it.
    """
    det_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
    used = set()
    matched = []
    for t_index, t_box in truths:
        best_iou, best_j = 0.0, None
        for j, d_box in enumerate(det_boxes):
            if j in used:
                continue
            score = iou(t_box, d_box)
            if score > best_iou:
                best_iou, best_j = score, j
        if best_j is not None and best_iou >= iou_threshold:
            used.add(best_j)
            matched.append((t_index, best_j, best_iou))
    return matched, used


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=4, help="evaluate every Nth frame")
    parser.add_argument("--min-visibility", type=int, default=3,
                        help="lowest nuScenes visibility bucket to require of a "
                             "ground-truth vehicle (1=0-40%% visible ... 4=80-100%%)")
    parser.add_argument("--weights", default=None, help="override detector weights")
    parser.add_argument("--conf", type=float, default=None, help="override confidence threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="override detector input size")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU_MATCH,
                        help="IoU required to call a detection a match. 0.4 rather "
                             "than the usual 0.5: nuScenes boxes span a vehicle's "
                             "full 3D extent including hidden parts, while the "
                             "detector boxes only what it can see, so 0.5 rejects "
                             "matches that are in fact correct detections")
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector(weights=args.weights, min_confidence=args.conf, imgsz=args.imgsz)

    stats = defaultdict(lambda: {"truth": 0, "hit": 0})
    total_truth = total_hit = total_det = total_matched = 0
    frames_done = 0

    for frame in source.frames(step=args.step):
        detections = detector.detect(frame.image)
        boxes = source.ground_truth_boxes(frame.sample_token)

        # Two ground-truth sets, deliberately different:
        #  * `truths` -- the strict set the detector is held to for RECALL
        #    (close enough and visible enough that missing one is a real fault)
        #  * `lenient` -- every labelled vehicle, however distant or occluded,
        #    used only to decide PRECISION. Scoring a correct detection of a
        #    70m or barely-visible vehicle as a false positive would blame the
        #    detector for the evaluation's own range cutoff.
        truths = []
        meta = []
        lenient = []
        for b in boxes:
            projected = source.project_to_image(b, frame.intrinsics, frame.image.shape)
            if projected is None:
                continue
            lenient.append((len(lenient), projected))
            distance = b.distance_to_center
            if b.visibility < args.min_visibility or not (0 < distance <= MAX_RANGE_M):
                continue
            truths.append((len(truths), projected))
            meta.append((distance, frame.is_night))

        matched, _ = match(detections, truths, args.iou)
        hit_indices = {t for t, _, _ in matched}
        _, used = match(detections, lenient, args.iou)

        for i, (distance, is_night) in enumerate(meta):
            band = ("0-15m" if distance < 15 else "15-30m" if distance < 30
                    else "30-45m" if distance < 45 else "45-60m")
            for key in (band, "night" if is_night else "day"):
                stats[key]["truth"] += 1
                if i in hit_indices:
                    stats[key]["hit"] += 1

        total_truth += len(truths)
        total_hit += len(hit_indices)
        total_det += len(detections)
        total_matched += len(used)
        frames_done += 1

    print(f"frames evaluated : {frames_done} (every {args.step} of {len(source)})")
    print(f"detector         : {Path(detector.weights).name}, conf>={detector.min_confidence}, "
          f"imgsz={detector.imgsz}")
    print(f"scoring          : IoU>={args.iou}, vehicles within {MAX_RANGE_M:.0f}m, "
          f"visibility>={args.min_visibility}")
    print()

    recall = total_hit / total_truth if total_truth else float("nan")
    precision = total_matched / total_det if total_det else float("nan")
    print(f"ground-truth vehicles : {total_truth}")
    print(f"detections made       : {total_det}")
    print(f"RECALL    {recall:6.1%}   (of real vehicles, how many were found)")
    print(f"PRECISION {precision:6.1%}   (of detections, how many were real vehicles)")
    print()

    print("recall by range:")
    for band in ("0-15m", "15-30m", "30-45m", "45-60m"):
        s = stats[band]
        if s["truth"]:
            print(f"   {band:8s} {s['hit']:4d}/{s['truth']:4d}  {s['hit']/s['truth']:6.1%}")
    print()
    print("recall by lighting:")
    for band in ("day", "night"):
        s = stats[band]
        if s["truth"]:
            print(f"   {band:8s} {s['hit']:4d}/{s['truth']:4d}  {s['hit']/s['truth']:6.1%}")


if __name__ == "__main__":
    main()
