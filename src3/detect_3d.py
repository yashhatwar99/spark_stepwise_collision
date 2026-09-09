"""
Detect vehicles and draw a 3D box around each, from a single camera image.

Pipeline, end to end, with no depth sensor anywhere:
    YOLO                -> 2D box
    size statistics     -> the vehicle's dimensions
    reprojection search -> its orientation
    tight-fit geometry  -> its 3D position          (src3.deep3dbox)

Measured against nuScenes ground truth by src3.evaluate_3d: 2.01 m median
position error. For comparison, this project's LiDAR pipeline gets 1.16 m
and its monocular depth attempt got 4.16 m -- so geometry on a plain camera
lands much closer to a laser than a depth network does.

Usage (from the project root):
    python -m src3.detect_3d --index 60
    python -m src3.detect_3d --start 40 --count 6
"""

import argparse

import cv2
import numpy as np

from src2 import config
from src2.detection import VehicleDetector
from src2.drawing import draw_caption
from src2.nuscenes_source import NuScenesSource
from src3.deep3dbox import fit_box_search_yaw, object_corners, project_box, rotation_y

# Base 0-3, roof 4-7, matching deep3dbox.object_corners.
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def colour_for(distance_m):
    if distance_m < 15:
        return (0, 90, 255)       # near -- red
    if distance_m < 30:
        return (0, 200, 255)      # mid  -- amber
    return (0, 220, 0)            # far  -- green


def draw_3d(image, uv, colour):
    pts = uv.T.astype(int)
    for a, b in EDGES:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
    # Shade the base so the box reads as standing on the road rather than
    # floating in the air.
    cv2.polylines(image, [pts[[0, 1, 2, 3]]], True, colour, 3, cv2.LINE_AA)


def render(frame, detector, draw_2d=True):
    K = frame.intrinsics
    canvas = frame.image.copy()
    drawn = 0

    for det in detector.detect(frame.image):
        box_2d = (det.x1, det.y1, det.x2, det.y2)
        translation, yaw, error = fit_box_search_yaw(box_2d, config.CAR_SIZE_WLH, K)
        if translation is None:
            continue
        uv = project_box(translation, config.CAR_SIZE_WLH, yaw, K)
        if uv is None or not np.isfinite(uv).all():
            continue
        # A box whose reprojection misses the detection badly is a fit that
        # failed, not a vehicle located; drawing it would look confident and
        # be wrong.
        if error > 0.5 * ((det.x2 - det.x1) + (det.y2 - det.y1)):
            continue

        distance = float(translation[2])
        colour = colour_for(distance)
        if draw_2d:
            cv2.rectangle(canvas, (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)), (90, 90, 90), 1)
        draw_3d(canvas, uv, colour)
        cv2.putText(canvas, f"{det.class_name} {distance:.1f}m",
                    (int(det.x1), max(int(det.y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
        drawn += 1

    draw_caption(canvas, [
        f"frame {frame.sample_index}  {frame.scene_name}   CAMERA ONLY - no lidar, no stereo",
        f"{drawn} vehicles boxed by tight-fit geometry (Deep3DBox)",
        "red <15m   amber 15-30m   green >30m",
    ])
    return canvas, drawn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    source = NuScenesSource()
    detector = VehicleDetector()
    out_dir = config.PROJECT_ROOT / "output3"
    out_dir.mkdir(parents=True, exist_ok=True)

    indices = ([args.index] if args.index is not None
               else list(range(args.start, args.start + args.count * args.step, args.step)))
    for index in indices:
        canvas, drawn = render(source.frame(index), detector)
        path = out_dir / f"box3d_{index:04d}.jpg"
        cv2.imwrite(str(path), canvas)
        print(f"  frame {index:3d}  {drawn} vehicles -> {path.name}")


if __name__ == "__main__":
    main()
