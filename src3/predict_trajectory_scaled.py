"""
predict_trajectory.py, but tracking with the range-scaled gate.

Identical pipeline and drawing -- YOLO -> 3D box -> track -> predict -> draw
-- except the tracker is src3.tracking_scaled.RangeScaledTracker instead of
src2.tracking.Tracker. Kept as a separate script so the original stays
available for side-by-side comparison.

Usage (from the project root):
    python -m src3.predict_trajectory_scaled --start 39 --count 12
    python -m src3.predict_trajectory_scaled --start 39 --count 12 --out output5/scaled_gate
"""

import argparse

import cv2
import numpy as np

from src2 import config
from src2.detection import VehicleDetector
from src2.nuscenes_source import NuScenesSource
from src3.deep3dbox import project_box
from src3.predict_trajectory import base_centre_pixel, detect_boxes, draw_wireframe
from src3.tracking_scaled import RangeScaledTracker

FRAME_DT = 0.5          # nuScenes keyframe spacing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=39)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=3,
                        help="how many frames ahead to predict (0.5 s each)")
    parser.add_argument("--out", default="output5/scaled_gate",
                        help="output folder, relative to the project root")
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector()
    tracker = RangeScaledTracker()
    out_dir = config.PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = None
    previous_ego = None
    for index in range(args.start, min(args.start + args.count, len(source))):
        frame = source.frame(index)
        if frame.scene_name != scene:
            scene = frame.scene_name
            tracker = RangeScaledTracker()      # a scene cut is not vehicle motion
            previous_ego = None

        boxes = detect_boxes(frame, detector)
        canvas = frame.image.copy()

        ego_now = frame.cam_to_global_t
        ego_velocity = np.zeros(3)
        if previous_ego is not None:
            ego_velocity = (ego_now - previous_ego) / FRAME_DT
        previous_ego = ego_now.copy()

        ids = []
        if boxes:
            centres_global = frame.to_global(np.stack([b["translation"] for b in boxes], axis=1))
            ranges = [float(b["translation"][2]) for b in boxes]
            ids = tracker.update(
                [centres_global[:2, i] for i in range(centres_global.shape[1])],
                frame.timestamp_s,
                ranges,
            )

        predicted = 0
        for i, (box, track_id) in enumerate(zip(boxes, ids)):
            distance = float(box["translation"][2])
            colour = ((0, 90, 255) if distance < 15
                      else (0, 200, 255) if distance < 30 else (0, 220, 0))

            uv = project_box(box["translation"], config.CAR_SIZE_WLH, box["yaw"], frame.intrinsics)
            if uv is None or not np.isfinite(uv).all():
                continue
            draw_wireframe(canvas, uv, colour)

            velocity = tracker.tracks[track_id].velocity()
            label = f"#{track_id} {distance:.0f}m"
            if velocity is not None:
                label += f" {float(np.hypot(velocity[0], velocity[1])):.0f}m/s"
            cv2.putText(canvas, label, (int(box["det"].x1), max(int(box["det"].y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 2, cv2.LINE_AA)
            if velocity is None:
                continue
            predicted += 1

            here = centres_global[:, i]
            trail = [base_centre_pixel(uv)]
            for step in range(1, args.horizon + 1):
                ahead = step * FRAME_DT
                future_global = np.array([here[0] + velocity[0] * ahead,
                                          here[1] + velocity[1] * ahead,
                                          here[2]])
                future_ego = ego_now + ego_velocity * ahead
                relative = frame.cam_to_global_R.T @ (future_global - future_ego)
                future_uv = project_box(relative, config.CAR_SIZE_WLH, box["yaw"], frame.intrinsics)
                if future_uv is None or not np.isfinite(future_uv).all():
                    continue
                faded = tuple(int(c * 0.8 ** step) for c in colour)
                draw_wireframe(canvas, future_uv, faded, 2 if step == args.horizon else 1)
                trail.append(base_centre_pixel(future_uv))
                if step == args.horizon:
                    px, py = trail[-1]
                    cv2.putText(canvas, f"+{ahead:.1f}s", (px + 4, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, faded, 1, cv2.LINE_AA)
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array(trail)], False, colour, 1, cv2.LINE_AA)
                for px, py in trail[1:]:
                    cv2.circle(canvas, (px, py), 2, colour, -1)

        cv2.putText(canvas, f"frame {index}  {frame.scene_name}   camera only   range-scaled gate",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(boxes)} boxes, {predicted} predicted "
                            f"{args.horizon * FRAME_DT:.1f}s ahead  (bright = now, faded = future)",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        path = out_dir / f"traj_{index:04d}.jpg"
        cv2.imwrite(str(path), canvas)
        print(f"  frame {index:3d}  {len(boxes)} boxes, {predicted} predicted -> {path.name}")


if __name__ == "__main__":
    main()
