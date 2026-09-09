"""
Draw the 3D boxes on the camera image, with heading from tracked motion.

Frames are walked in order so the tracker can see each vehicle move -- a
single frame cannot give a heading, and heading is what makes a box look
like it belongs to the car rather than merely sitting near it.

A box whose heading is measured is drawn solid with an arrow; one whose
vehicle is too slow to read is drawn thin and marked, rather than being
dressed up as if we knew which way it faced.

Usage (from the project root):
    python -m src2.render_3d                    # first 8 frames of scene 1
    python -m src2.render_3d --start 60 --count 6
"""

import argparse

import cv2
import numpy as np

from . import config
from .detection import VehicleDetector
from .drawing import draw_caption
from .lidar import (is_consistent_with_box, points_in_box_2d, remove_ground,
                    separate_foreground)
from .localize import box_corners, localize
from .nuscenes_source import NuScenesSource
from .tracking import Tracker

EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def confidence_color(n_points):
    if n_points >= 20:
        return (0, 220, 0)
    if n_points >= 5:
        return (0, 200, 255)
    return (0, 90, 255)


def draw_box_3d(image, corners_2d, color, thickness=2):
    pts = corners_2d.T.astype(int)
    for a, b in EDGES:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    # Mark the face the vehicle is driving towards, so a wrong heading is
    # obvious on sight rather than hidden inside a symmetric wireframe.
    front = pts[[0, 1, 2, 3]]
    cv2.polylines(image, [front], True, color, thickness + 1, cv2.LINE_AA)


def project(points_3d, K):
    uv = K @ points_3d
    return uv[:2] / uv[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--no-points", action="store_true")
    args = parser.parse_args()

    source = NuScenesSource(load_lidar=True)
    detector = VehicleDetector()
    tracker = Tracker()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    current_scene = None
    for index in range(args.start, min(args.start + args.count, len(source))):
        frame = source.frame(index)
        if frame.scene_name != current_scene:
            current_scene = frame.scene_name
            tracker = Tracker()       # a scene cut is not vehicle motion

        canvas = frame.image.copy()
        K = frame.intrinsics

        if not args.no_points:
            for u, v in zip(frame.lidar_pixels[0], frame.lidar_pixels[1]):
                cv2.circle(canvas, (int(u), int(v)), 1, (80, 80, 80), -1)

        boxes, dets = [], []
        for det in detector.detect(frame.image):
            box_2d = (det.x1, det.y1, det.x2, det.y2)
            points = points_in_box_2d(frame.lidar_points, frame.lidar_pixels, box_2d)
            points = remove_ground(points, frame.camera_height_m)
            points = separate_foreground(points, config.FOREGROUND_DEPTH_GAP)
            if not is_consistent_with_box(points, box_2d, K[0, 0]):
                continue
            box = localize(points, frame.camera_height_m)
            if box is not None:
                boxes.append(box)
                dets.append(det)

        if not boxes:
            continue

        centres = frame.to_global(np.stack([b.center for b in boxes], axis=1))
        ids = tracker.update([centres[:2, i] for i in range(centres.shape[1])],
                             frame.timestamp_s)

        measured = 0
        for box, det, track_id in zip(boxes, dets, ids):
            yaw = tracker.heading_in_camera(track_id, frame)
            if yaw is not None:
                # Motion has supplied the one thing the single-frame estimate
                # could not; swap the placeholder heading for the real one.
                box.yaw = yaw
                box.yaw_is_measured = True
                measured += 1

            corners = box_corners(box)
            if corners[2].min() <= 0.5:
                continue
            uv = project(corners, K)
            if not np.isfinite(uv).all():
                continue

            color = confidence_color(box.n_lidar_points)
            draw_box_3d(canvas, uv, color, thickness=2 if box.yaw_is_measured else 1)

            speed = tracker.tracks[track_id].speed
            label = f"#{track_id} {box.distance_m:.1f}m"
            if box.yaw_is_measured and speed is not None:
                label += f" {speed:.0f}m/s"
            else:
                label += " heading?"
            cv2.putText(canvas, label, (int(det.x1), max(int(det.y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        draw_caption(canvas, [
            f"frame {index}  {frame.scene_name}{'  [night]' if frame.is_night else ''}",
            f"{len(boxes)} boxes, {measured} with measured heading (thick = heading known)",
        ])
        out = config.OUTPUT_DIR / f"box3d_{index:04d}.jpg"
        cv2.imwrite(str(out), canvas)
        print(f"  frame {index:3d}  {len(boxes)} boxes, {measured} headed -> {out.name}")


if __name__ == "__main__":
    main()
