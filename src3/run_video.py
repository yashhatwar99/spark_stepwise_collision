"""
Car-to-car collision prediction on a fixed camera, from a plain video file.

WHY THIS IS NOT THE nuScenes PIPELINE
-------------------------------------
Everything else in src2/src3 answers "will something hit US" from a camera
riding on the ego vehicle. A fixed traffic camera is never going to be hit,
so that question is meaningless here. The question this footage actually
poses is whether two OTHER vehicles will hit each other -- which needs their
paths compared against each other rather than against the observer.

Three things the nuScenes pipeline relies on are missing from an arbitrary
video, so none of them is assumed:

  * Camera intrinsics. Unknown, so no metric 3D box is attempted. Everything
    here is in pixels.
  * Ego motion. Not needed -- the camera is fixed, which is the one case
    where the camera frame IS the world frame.
  * A level camera. This one looks steeply down, which breaks the
    "vehicle is upright in the camera frame" assumption the 3D geometry in
    deep3dbox.py depends on. Attempting a 3D box here would produce
    confident nonsense.

WHAT IT DOES INSTEAD
--------------------
Tracks each vehicle's 2D box, extrapolates it forward, and reports a warning
when two vehicles' PREDICTED boxes overlap. For a camera looking down on a
roughly flat intersection, overlap in the image is a fair proxy for two
vehicles occupying the same piece of road -- not exact, because perspective
compresses distance unevenly, but honest about what it is measuring.

Usage:
    python -m src3.run_video --video "data/collision 1.mp4"
    python -m src3.run_video --video "data/collision 1.mp4" --conf 0.2 --horizon 1.5
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from src2 import config
from src2.detection import VehicleDetector
from src2.looming import BoxTracker

OUT_DIR = config.PROJECT_ROOT / "output3" / "testing"


def box_overlap(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / min(area_a, area_b)      # relative to the smaller box


def predict_box(track, seconds_ahead, window=4):
    """Where a tracked box will be, by extrapolating its recent motion.

    Position and size are both carried forward: a vehicle approaching the
    camera grows, and ignoring that would shrink its predicted footprint just
    as it becomes most relevant.
    """
    if len(track.boxes) < 2:
        return None
    n = min(window + 1, len(track.boxes))
    dt = track.timestamps[-1] - track.timestamps[-n]
    if dt <= 1e-6:
        return None
    old, now = np.array(track.boxes[-n]), np.array(track.boxes[-1])
    return tuple(now + (now - old) / dt * seconds_ahead)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--horizon", type=float, default=1.5,
                        help="seconds ahead to predict")
    parser.add_argument("--steps", type=int, default=3,
                        help="how many intermediate predictions to draw")
    parser.add_argument("--overlap", type=float, default=0.15,
                        help="predicted-box overlap that counts as a conflict")
    args = parser.parse_args()

    path = Path(args.video)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = VehicleDetector(min_confidence=args.conf)
    tracker = BoxTracker()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"video_{path.stem.replace(' ', '_')}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))

    print(f"video    : {path}  ({width}x{height}, {fps:.0f} fps)")
    print(f"detector : conf>={args.conf}")
    print(f"predicting {args.horizon:.1f}s ahead, conflict when predicted boxes "
          f"overlap by {args.overlap:.0%}")
    print()

    frame_index = 0
    conflict_frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp = frame_index / fps

        detections = detector.detect(frame)
        boxes = [(d.x1, d.y1, d.x2, d.y2) for d in detections]
        # Bearing is meaningless for a fixed camera, but BoxTracker wants the
        # intrinsics for it; nominal values keep the tracker happy without
        # anything downstream depending on them.
        ids = tracker.update(boxes, timestamp, float(width), width / 2.0)

        # Predict every track, then look for pairs whose futures collide.
        futures = {}
        for track_id in ids:
            predicted = predict_box(tracker.tracks[track_id], args.horizon)
            if predicted is not None:
                futures[track_id] = predicted

        in_conflict = set()
        for i, a in futures.items():
            for j, b in futures.items():
                if i >= j:
                    continue
                if box_overlap(a, b) >= args.overlap:
                    in_conflict.add(i)
                    in_conflict.add(j)

        for box, track_id in zip(boxes, ids):
            danger = track_id in in_conflict
            colour = (0, 0, 255) if danger else (0, 220, 0)
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(frame, f"#{track_id}", (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

            track = tracker.tracks[track_id]
            for step in range(1, args.steps + 1):
                ahead = args.horizon * step / args.steps
                predicted = predict_box(track, ahead)
                if predicted is None:
                    continue
                fade = 0.75 ** step
                faded = tuple(int(c * fade) for c in colour)
                px1, py1, px2, py2 = (int(v) for v in predicted)
                cv2.rectangle(frame, (px1, py1), (px2, py2), faded, 1)

        if in_conflict:
            conflict_frames.append(frame_index)
            cv2.rectangle(frame, (0, 0), (width, 34), (0, 0, 160), -1)
            cv2.putText(frame, f"PREDICTED CONFLICT  ids {sorted(in_conflict)}",
                        (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                        cv2.LINE_AA)
        else:
            cv2.putText(frame, f"{len(boxes)} tracked", (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"t={timestamp:.1f}s  fixed camera, pixels only (no 3D)",
                    (8, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()
    print(f"{frame_index} frames -> {out_path}")
    if conflict_frames:
        first = conflict_frames[0] / fps
        print(f"conflict predicted on {len(conflict_frames)} frames, "
              f"first at t={first:.1f}s (frame {conflict_frames[0]})")
    else:
        print("no conflict predicted")


if __name__ == "__main__":
    main()
