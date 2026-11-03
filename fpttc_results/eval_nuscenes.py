"""
Per-car time-to-collision from FP-TTC, tested on nuScenes: data FP-TTC never saw.

For every keyframe of nuScenes mini (10 scenes), FP-TTC gets the frame just
before the keyframe and the keyframe itself, the same "past frames" pairing the
model was trained on. YOLOv8s finds cars in the keyframe. Each detected car is
matched to its LiDAR-verified annotated box, and the per-car methods are scored
against the true answer.

True answer for a car, from the annotations (no camera measurement involved):
    Z        forward distance of the box centre, camera frame
    closing  speed at which Z shrinks = -(z of object velocity minus camera
             velocity, in camera axes). Object velocity comes from the
             annotations, camera velocity from the vehicle's own pose.
    TTC      Z / closing, only when the car is getting closer
    scale    1 - closing * dt / Z   (what the model's scale map should say)

Usage (from the project root):
    python fpttc_results/eval_nuscenes.py
"""

import argparse
import time

import cv2
import numpy as np
import pandas as pd
import torch
from nuscenes.utils.data_classes import Box  # noqa: F401  (kept for type clarity)

import common as C
from src2 import config
from src2.detection import VehicleDetector
from src3.sweep_source import NuScenesSweepSource

CONFIGS = [("fpttc_kitti", (640, 320)), ("fpttc_kitti", (640, 352)),
           ("fpttc_kitti", (960, 544)), ("fpttc_mix", (640, 352))]
CAR_CLASSES = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.motorcycle")
MIN_Z, MAX_Z = 3.0, 70.0


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def true_cars(nusc, frame_key, frame_prev, dt, img_w, img_h):
    """LiDAR-verified cars visible in the keyframe, with true distance, closing
    speed and TTC. Returns a list of dicts."""
    sample = nusc.get("sample", frame_key.sample_token)
    sd_token = sample["data"][config.CAMERA_CHANNEL]
    _, boxes, _ = nusc.get_sample_data(sd_token)
    v_cam = (frame_key.cam_to_global_t - frame_prev.cam_to_global_t) / dt   # global axes
    out = []
    for g in boxes:
        if not g.name.startswith("vehicle."):
            continue
        corners = g.corners()
        z = float(g.center[2])
        if corners[2].min() <= 0.1 or not (MIN_Z <= z <= MAX_Z):
            continue
        uv = frame_key.intrinsics @ corners
        uv = uv[:2] / uv[2]
        box = (max(uv[0].min(), 0), max(uv[1].min(), 0), min(uv[0].max(), img_w), min(uv[1].max(), img_h))
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            continue
        v_obj = nusc.box_velocity(g.token)
        if not np.isfinite(v_obj[:2]).all():
            continue
        v_obj = np.array([v_obj[0], v_obj[1], 0.0])
        v_rel_cam = frame_key.cam_to_global_R.T @ (v_obj - v_cam)
        closing = -float(v_rel_cam[2])
        out.append(dict(name=g.name, box=box, z=z, closing=closing,
                        ttc_true=z / closing if closing > 0.05 else np.nan,
                        s_true=1.0 - closing * dt / z, lateral=float(g.center[0])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10 ** 6, help="keyframes to use (quick test)")
    args = ap.parse_args()

    src = NuScenesSweepSource()
    nusc = src.nusc
    detector = VehicleDetector()
    models = {wn: C.load_model(wn) for wn in sorted({c[0] for c in CONFIGS})}

    rows, frames_info = [], []
    used = 0
    t_start = time.time()
    for k, sample in enumerate(nusc.sample):
        if used >= args.limit:
            break
        tok = sample["data"][config.CAMERA_CHANNEL]
        prev_tok = nusc.get("sample_data", tok)["prev"]
        if not prev_tok:
            continue
        key, prev = src.frame(tok), src.frame(prev_tok)
        dt = key.timestamp_s - prev.timestamp_s
        if not (0.03 <= dt <= 0.30):
            continue
        used += 1
        H, W = key.image.shape[:2]
        cars = true_cars(nusc, key, prev, dt, W, H)

        t0 = time.perf_counter()
        dets = detector.detect(key.image)
        torch.cuda.synchronize()
        t_yolo = time.perf_counter() - t0
        dboxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

        # recall of the detector on the annotated cars of the four classes
        eligible = [c for c in cars if c["name"] in CAR_CLASSES]
        hit = sum(1 for c in eligible if any(iou(c["box"], b) >= 0.4 for b in dboxes))
        frames_info.append(dict(keyframe=k, scene=key.scene_name, dt=dt, n_true=len(eligible),
                                n_found=hit, n_detections=len(dets), yolo_seconds=t_yolo))

        # match each detection to its best annotated car
        matches = []
        for b in dboxes:
            best = max(cars, key=lambda c: iou(c["box"], b), default=None)
            if best is not None and iou(best["box"], b) >= 0.4:
                matches.append((b, best))

        rgb_prev = cv2.cvtColor(prev.image, cv2.COLOR_BGR2RGB)
        rgb_key = cv2.cvtColor(key.image, cv2.COLOR_BGR2RGB)
        for wn, sz in CONFIGS:
            w0, h0 = sz
            t0 = time.perf_counter()
            scale, flow = C.run_pair(models[wn], rgb_prev, rgb_key, size=sz)
            torch.cuda.synchronize()
            t_fp = time.perf_counter() - t0
            fx, fy = w0 / W, h0 / H
            for b, c in matches:
                bx = (b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy)
                rec = dict(keyframe=k, scene=key.scene_name, weights=wn, width=w0, height=h0, dt=dt,
                           category=c["name"], z=c["z"], closing=c["closing"], ttc_true=c["ttc_true"],
                           s_true=c["s_true"], lateral=c["lateral"], box_w=b[2] - b[0], box_h=b[3] - b[1],
                           fpttc_seconds=t_fp)
                rec["s_median_box100"], _ = C.car_scale_median(scale, bx, 1.0)
                rec["s_median_box70"], _ = C.car_scale_median(scale, bx, 0.7)
                rec["s_median_box50"], _ = C.car_scale_median(scale, bx, 0.5)
                rec["s_zoom_box70"], rec["zoom_inlier_share"], _ = C.car_scale_zoom(flow, bx, 0.7)
                rows.append(rec)

        if used % 20 == 0:
            print(f"keyframes {used}   matched cars so far {len(rows) // len(CONFIGS)}   "
                  f"{time.time() - t_start:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(C.RESULTS_DIR / "nuscenes_cars.csv", index=False)
    pd.DataFrame(frames_info).to_csv(C.RESULTS_DIR / "nuscenes_frames.csv", index=False)
    print(f"saved. keyframes used {used}, matched cars {len(rows) // len(CONFIGS)}")


if __name__ == "__main__":
    main()
