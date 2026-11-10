"""
Test Method 2 (learned head) against Method 1 (median) on nuScenes, data that
FP-TTC and the head never saw. Same setup as fpttc_results/eval_nuscenes.py:
for each keyframe, FP-TTC gets the frame just before it and the keyframe, YOLO
finds cars in the keyframe, each detection is matched to its annotated car, and
both methods are scored against the true TTC from the annotations. Both methods
get exactly the same boxes from the same model run.

Usage (from the project root, with the project .venv):
    python per_car_ttc/learned_head/eval_head_nuscenes.py
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import head_model as H  # noqa: E402
import eval_nuscenes as EN  # noqa: E402  (fpttc_results/eval_nuscenes.py)
import build_summary as BS  # noqa: E402
from src2 import config  # noqa: E402
from src2.detection import VehicleDetector  # noqa: E402
from src3.sweep_source import NuScenesSweepSource  # noqa: E402

C = H.C
OUT = HERE / "output"


def main():
    src = NuScenesSweepSource()
    nusc = src.nusc
    detector = VehicleDetector()
    model = C.load_model("fpttc_kitti")
    reader = H.FeatureReader(model)
    head, _ = H.load_head()

    rows = []
    t_head, n_head = 0.0, 0
    t_start = time.time()
    used = 0
    for k, sample in enumerate(nusc.sample):
        tok = sample["data"][config.CAMERA_CHANNEL]
        prev_tok = nusc.get("sample_data", tok)["prev"]
        if not prev_tok:
            continue
        key, prev = src.frame(tok), src.frame(prev_tok)
        dt = key.timestamp_s - prev.timestamp_s
        if not (0.03 <= dt <= 0.30):
            continue
        used += 1
        Hh, Ww = key.image.shape[:2]
        cars = EN.true_cars(nusc, key, prev, dt, Ww, Hh)
        dboxes = [(d.x1, d.y1, d.x2, d.y2) for d in detector.detect(key.image)]
        matches = []
        for b in dboxes:
            best = max(cars, key=lambda c: EN.iou(c["box"], b), default=None)
            if best is not None and EN.iou(best["box"], b) >= 0.4:
                matches.append((b, best))
        if not matches:
            continue
        scale, feat = reader(cv2.cvtColor(prev.image, cv2.COLOR_BGR2RGB),
                             cv2.cvtColor(key.image, cv2.COLOR_BGR2RGB), H.SIZE)
        fx, fy = H.SIZE[0] / Ww, H.SIZE[1] / Hh
        bx = [(b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy) for b, _ in matches]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s_head = H.predict_s(head, feat, bx)
        torch.cuda.synchronize()
        t_head += time.perf_counter() - t0
        n_head += len(bx)
        for (b, c), box_m, sh in zip(matches, bx, s_head):
            rows.append(dict(keyframe=k, dt=dt, z=c["z"], ttc_true=c["ttc_true"], s_true=c["s_true"],
                             s_head=float(sh), s_median_box70=C.car_scale_median(scale, box_m, 0.7)[0],
                             s_median_box50=C.car_scale_median(scale, box_m, 0.5)[0]))
        if used % 40 == 0:
            print(f"keyframes {used}   cars {len(rows)}   {time.time() - t_start:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "nuscenes_head_vs_median.csv", index=False)
    dts = df.dt.to_numpy(float)
    tt = df.ttc_true.to_numpy(float)
    print(f"\nnuScenes: {len(df)} matched cars from {used} keyframes (640x320, fpttc_kitti)")
    print("                         scale error   TTC error median / p90    danger found / precision (TTC < 3 s)")
    report = {}
    for m, label in [("median_box70", "Median, inner 70% (Method 1)"), ("head", "Learned head (Method 2)")]:
        r = BS.car_metrics(df, m, dts, ttc_true=tt)
        report[m] = r
        e, dg = r["ttc_abs_error_s"], r["danger"]
        print(f"  {label:<30s} {r['scale_error_percent_median']:5.2f}%      "
              f"{e['median']:5.2f} s / {e['p90']:5.2f} s            {dg['recall']:.0%} / {dg['precision']:.0%}")
    print("\n  TTC error by true TTC range (median |error|, seconds)")
    for rng_ in report["head"]["ttc_by_true_ttc"]:
        a = report["median_box70"]["ttc_by_true_ttc"][rng_]
        b = report["head"]["ttc_by_true_ttc"][rng_]
        print(f"    {rng_:>8s}   Method 1 {a.get('median', float('nan')):5.2f}  (n={a['n']})   Method 2 {b.get('median', float('nan')):5.2f}  (n={b['n']})")
    print(f"\n  head time per car: {1000 * t_head / max(n_head, 1):.3f} ms")
    (OUT / "nuscenes_eval.json").write_text(json.dumps(report, indent=1, default=float))
    print(f"saved {OUT / 'nuscenes_eval.json'} and nuscenes_head_vs_median.csv")


if __name__ == "__main__":
    main()
