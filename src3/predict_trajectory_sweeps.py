"""
Future 3D boxes using every camera image (~10 fps), not just keyframes (2 fps).

Same pipeline as predict_trajectory_scaled.py -- YOLO -> 3D box -> track ->
predict -> draw -- with three changes for the higher frame rate:

  * frames come from sweep_source.NuScenesSweepSource (all images, each with
    its own ego pose);
  * tracking uses tracking_timewindow.TimeWindowTracker, which measures speed
    by fitting a line through the last 1.5 s of positions, instead of taking
    two endpoints 4 frames apart;
  * every time step uses real timestamps. Gaps between images vary from
    50 ms to 200 ms, so a fixed frame spacing would be wrong.

Keyframe range is given in NuScenesSource indices, so outputs line up with
the 2 fps results. Keyframe images are saved as traj_0047.jpg, and the sweeps
after them as traj_0047_s1.jpg, traj_0047_s2.jpg, ...

Usage (from the project root):
    python -m src3.predict_trajectory_sweeps --first 39 --last 60 --save-from 47
"""

import argparse
from collections import defaultdict

import cv2
import numpy as np

from src2 import config
from src2.detection import VehicleDetector
from src3.deep3dbox import project_box
from src3.predict_trajectory import base_centre_pixel, detect_boxes, draw_wireframe
from src3.sweep_source import NuScenesSweepSource
from src3.tracking_timewindow import TimeWindowTracker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=39, help="first keyframe (starts tracking)")
    parser.add_argument("--last", type=int, default=60, help="last keyframe")
    parser.add_argument("--save-from", type=int, default=47,
                        help="only save images from this keyframe on; earlier ones warm the tracker up")
    parser.add_argument("--horizon", type=float, default=1.5, help="seconds ahead to predict")
    parser.add_argument("--steps", type=int, default=3, help="future boxes drawn per car")
    parser.add_argument("--keyframes-only", action="store_true",
                        help="process every image but save only the keyframe ones")
    parser.add_argument("--out", default="output5/fps10")
    args = parser.parse_args()

    source = NuScenesSweepSource()
    detector = VehicleDetector()
    tracker = TimeWindowTracker()
    out_dir = config.PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = source.images_between_keyframes(args.first, args.last)
    print(f"{len(tokens)} camera images between keyframes {args.first} and {args.last}")

    previous_ego = previous_time = None
    sweep_counter = defaultdict(int)
    key_cars = key_predicted = 0

    for token in tokens:
        frame = source.frame(token)
        boxes = detect_boxes(frame, detector)
        canvas = frame.image.copy()

        # Ego velocity from the previous IMAGE, using the real time gap.
        ego_now = frame.cam_to_global_t
        ego_velocity = np.zeros(3)
        if previous_ego is not None and frame.timestamp_s > previous_time:
            ego_velocity = (ego_now - previous_ego) / (frame.timestamp_s - previous_time)
        previous_ego, previous_time = ego_now.copy(), frame.timestamp_s

        ids = []
        if boxes:
            centres_global = frame.to_global(np.stack([b["translation"] for b in boxes], axis=1))
            ids = tracker.update(
                [centres_global[:2, i] for i in range(centres_global.shape[1])],
                frame.timestamp_s,
                [float(b["translation"][2]) for b in boxes],
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
            for step in range(1, args.steps + 1):
                ahead = args.horizon * step / args.steps
                future_global = np.array([here[0] + velocity[0] * ahead,
                                          here[1] + velocity[1] * ahead,
                                          here[2]])
                relative = frame.cam_to_global_R.T @ (future_global - (ego_now + ego_velocity * ahead))
                future_uv = project_box(relative, config.CAR_SIZE_WLH, box["yaw"], frame.intrinsics)
                if future_uv is None or not np.isfinite(future_uv).all():
                    continue
                faded = tuple(int(c * 0.8 ** step) for c in colour)
                draw_wireframe(canvas, future_uv, faded, 2 if step == args.steps else 1)
                trail.append(base_centre_pixel(future_uv))
                if step == args.steps:
                    px, py = trail[-1]
                    cv2.putText(canvas, f"+{ahead:.1f}s", (px + 4, py),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, faded, 1, cv2.LINE_AA)
            if len(trail) > 1:
                cv2.polylines(canvas, [np.array(trail)], False, colour, 1, cv2.LINE_AA)
                for px, py in trail[1:]:
                    cv2.circle(canvas, (px, py), 2, colour, -1)

        k = frame.key_index
        if frame.is_key_frame:
            name = f"traj_{k:04d}.jpg"
        else:
            sweep_counter[k] += 1
            name = f"traj_{k:04d}_s{sweep_counter[k]}.jpg"

        if k < args.save_from:
            continue
        if frame.is_key_frame:
            key_cars += len(boxes)
            key_predicted += predicted

        kind = "keyframe" if frame.is_key_frame else "sweep"
        cv2.putText(canvas, f"keyframe {k}  {kind}  t={frame.timestamp_s % 100:.2f}s  "
                            f"{frame.scene_name}   10 fps, time-window speed",
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{len(boxes)} boxes, {predicted} predicted {args.horizon:.1f}s ahead "
                            f"(bright = now, faded = future)",
                    (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        if frame.is_key_frame or not args.keyframes_only:
            cv2.imwrite(str(out_dir / name), canvas)
        print(f"  {name:22s} {kind:8s} {len(boxes):2d} boxes, {predicted:2d} predicted")

    print()
    if key_cars:
        print(f"keyframes {args.save_from}-{args.last}: {key_predicted} of {key_cars} cars predicted "
              f"({100 * key_predicted / key_cars:.0f}%)")
    print(f"tracks created: {tracker._next_id}")


if __name__ == "__main__":
    main()
