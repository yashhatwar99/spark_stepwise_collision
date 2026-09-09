"""
Collision warning drawn on the image, using no distance at all.

Each tracked vehicle gets its box projected forward in time -- the same box
grown at its measured rate and slid along its measured bearing drift. Where
that future box lands says whether it is coming at us or past us, and the
colour says how urgently.

Usage (from the project root):
    python -m src2.render_collision --start 0 --count 12
"""

import argparse

import cv2
import numpy as np

from . import config
from .detection import VehicleDetector
from .drawing import draw_caption
from .looming import BoxTracker
from .nuscenes_source import NuScenesSource

RISK_COLORS = {
    "CRITICAL": (0, 0, 255),
    "WARNING": (0, 165, 255),
    "CLEAR": (0, 200, 0),
    "PASSING": (160, 160, 160),
    "NO CLOSING": (110, 110, 110),
}
HORIZONS_S = (1.0, 2.0, 3.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=12)
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector()
    tracker = BoxTracker()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scene = None
    for index in range(args.start, min(args.start + args.count, len(source))):
        frame = source.frame(index)
        if frame.scene_name != scene:
            scene = frame.scene_name
            tracker = BoxTracker()

        fx = float(frame.intrinsics[0, 0])
        cx = float(frame.intrinsics[0, 2])
        detections = detector.detect(frame.image)
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        ids = tracker.update(boxes, frame.timestamp_s, fx, cx)

        canvas = frame.image.copy()
        counts = {}
        for box, track_id in zip(boxes, ids):
            track = tracker.tracks[track_id]
            risk = track.risk()
            counts[risk] = counts.get(risk, 0) + 1
            color = RISK_COLORS[risk]

            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

            # The trajectory: the same box at 1, 2 and 3 seconds ahead, drawn
            # progressively fainter. Only for things actually closing --
            # extrapolating a box that is not growing would draw a confident
            # future for a vehicle that is simply sitting there.
            if risk in ("CRITICAL", "WARNING", "CLEAR"):
                for k, horizon in enumerate(HORIZONS_S):
                    future = track.predict_box(horizon)
                    if future is None:
                        continue
                    fx1, fy1, fx2, fy2 = (int(v) for v in future)
                    if fx2 - fx1 < 2 or fy2 - fy1 < 2:
                        continue
                    faded = tuple(int(c * (0.75 ** (k + 1))) for c in color)
                    cv2.rectangle(canvas, (fx1, fy1), (fx2, fy2), faded, 1)
                    cv2.putText(canvas, f"{horizon:.0f}s", (fx2 + 2, fy1 + 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, faded, 1, cv2.LINE_AA)

            ttc = track.ttc()
            rate = track.bearing_rate()
            label = f"#{track_id} {risk}"
            if ttc is not None:
                label += f" ttc {ttc:.1f}s"
            if rate is not None:
                label += f" drift {rate:.3f}"
            cv2.putText(canvas, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2, cv2.LINE_AA)

        summary = "  ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        draw_caption(canvas, [
            f"frame {index}  {frame.scene_name}   NO DISTANCE USED",
            f"{summary or 'nothing detected'}",
            "faint boxes = predicted position at 1s / 2s / 3s",
        ])
        out = config.OUTPUT_DIR / f"collision_{index:04d}.jpg"
        cv2.imwrite(str(out), canvas)
        print(f"  frame {index:3d}  {summary or '-'}  -> {out.name}")


if __name__ == "__main__":
    main()
