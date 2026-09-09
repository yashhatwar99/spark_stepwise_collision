"""
Stage 1 in isolation: YOLO only.

No 3D, no geometry, no tracking, no prediction -- just the detector's 2D
boxes on nuScenes camera frames. Everything downstream inherits whatever
this stage misses, so it is worth being able to look at it on its own: a
vehicle never detected can never be boxed in 3D, tracked, or predicted.

Writes to output3/testing/.

Usage (from the project root):
    python -m src3.test_detection --start 40 --count 8
    python -m src3.test_detection --start 40 --count 8 --conf 0.2
"""

import argparse

import cv2

from src2 import config
from src2.detection import VehicleDetector
from src2.nuscenes_source import NuScenesSource

OUT_DIR = config.PROJECT_ROOT / "output3" / "testing"

# One colour per class, so a truck misread as a car is visible at a glance.
CLASS_COLORS = {
    "car": (0, 220, 0),
    "truck": (255, 160, 0),
    "bus": (255, 80, 200),
    "motorcycle": (0, 200, 255),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--conf", type=float, default=None,
                        help="override the confidence threshold")
    parser.add_argument("--weights", default=None, help="override detector weights")
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector(weights=args.weights, min_confidence=args.conf)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"detector : {detector.weights}")
    print(f"settings : conf>={detector.min_confidence}, imgsz={detector.imgsz}")
    print(f"writing  : {OUT_DIR}")
    print()

    total = 0
    for index in range(args.start, min(args.start + args.count * args.step, len(source)), args.step):
        frame = source.frame(index)
        detections = detector.detect(frame.image)
        canvas = frame.image.copy()

        for det in detections:
            colour = CLASS_COLORS.get(det.class_name, (200, 200, 200))
            x1, y1, x2, y2 = (int(v) for v in (det.x1, det.y1, det.x2, det.y2))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            # Put the label inside the frame when the box is near the top edge,
            # otherwise it is drawn off-screen and silently lost.
            ty = y1 - 4 if y1 - th - 6 >= 0 else y2 + th + 4
            cv2.rectangle(canvas, (x1, ty - th - base + 2), (x1 + tw + 4, ty + base - 2),
                          colour, -1)
            cv2.putText(canvas, label, (x1 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 1, cv2.LINE_AA)

        counts = {}
        for det in detections:
            counts[det.class_name] = counts.get(det.class_name, 0) + 1
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "nothing detected"

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(canvas, f"STAGE 1: YOLO 2D DETECTION ONLY", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {index}  {frame.scene_name}"
                            f"{'  [night]' if frame.is_night else ''}   {summary}",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        path = OUT_DIR / f"stage1_detect_{index:04d}.jpg"
        cv2.imwrite(str(path), canvas)
        total += len(detections)
        print(f"  frame {index:3d}  {len(detections):2d} detections ({summary})  -> {path.name}")

    print()
    print(f"{total} detections across the frames rendered")


if __name__ == "__main__":
    main()
