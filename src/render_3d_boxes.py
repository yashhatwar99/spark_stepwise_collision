"""
Camera-only 3D vehicle boxes, drawn on a real image.

Runs the full monocular path -- YOLO 2D detection -> metric depth -> 3D box
(position, metric size, heading) -- on nuScenes CAM_FRONT frames. Nothing
here reads the nuScenes 3D annotations; they are used only by the separate
evaluation script to score this output. The camera intrinsics ARE taken
from nuScenes, since a calibrated camera is a fair assumption (any real
deployment calibrates its own camera once).
"""

import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from depth_3d_box import depth_map, estimate_3d_box, load_depth_pipeline, box_corners  # noqa: E402

from nuscenes.nuscenes import NuScenes  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nuScenes", "v1.0-mini")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")
VEHICLE_CLASS_IDS = {2, 5, 7}       # COCO: car, bus, truck
MIN_CONFIDENCE = 0.4

# 12 edges of a box, as index pairs into the 8 corners from box_corners().
EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]


def project(points_3d, K):
    uv = K @ points_3d
    return uv[:2] / uv[2]


def draw_box_3d(img, corners_2d, color, thickness=2):
    pts = corners_2d.T.astype(int)
    for a, b in EDGES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, thickness, cv2.LINE_AA)
    # Fill the front face lightly so orientation is readable at a glance.
    face = pts[[0, 1, 2, 3]]
    overlay = img.copy()
    cv2.fillPoly(overlay, [face], color)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)


def render(sample_index=10, out_name="camera_only_3d_boxes.jpg"):
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)
    sample = nusc.sample[sample_index]
    sd = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    K = np.array(cs["camera_intrinsic"])
    cam_h = cs["translation"][2]
    img_path = os.path.join(nusc.dataroot, sd["filename"])

    detector = YOLO("yolov8n.pt")
    pipe = load_depth_pipeline()

    pil = Image.open(img_path)
    depth = depth_map(pipe, pil)
    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    result = detector(img, verbose=False, conf=MIN_CONFIDENCE)[0]

    drawn = 0
    for det in result.boxes:
        if int(det.cls) not in VEHICLE_CLASS_IDS:
            continue
        x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
        est = estimate_3d_box(x1, y1, x2, y2, depth, K, H, camera_height_m=cam_h)
        if est is None:
            continue

        corners = box_corners(est["center"], est["wlh"], est["yaw"])
        if corners[2].min() <= 0.5:            # behind or on top of the camera
            continue
        uv = project(corners, K)
        if not np.isfinite(uv).all() or uv.min() < -4 * W or uv.max() > 5 * W:
            continue

        # colour by distance so the depth estimate is visually checkable
        d = est["distance_m"]
        color = (0, 200, 0) if d > 25 else (0, 200, 255) if d > 12 else (0, 0, 255)
        draw_box_3d(img, uv, color)
        label = f"{d:.1f}m {est['method'][:5]}"
        cv2.putText(img, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        drawn += 1

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out_path, img)
    print(f"{drawn} vehicles drawn as 3D boxes -> {out_path}")
    return out_path


if __name__ == "__main__":
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    render(idx)
