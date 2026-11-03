"""
Example pictures for the slides and the web page:
  kitti_example.jpg     input frame, predicted scale map, true scale map, error
  nuscenes_example.jpg  detected cars labelled with predicted and true TTC
  fpttc_output_*.jpg    the same KITTI frame as FP-TTC's own per-pixel TTC colour map

Usage (from the project root):  python fpttc_results/make_figures.py
"""

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import common as C
import eval_kitti as K
import eval_nuscenes as N
from src2 import config
from src2.detection import VehicleDetector
from src3.sweep_source import NuScenesSweepSource

DANGER = 3.0


def kitti_example(image_id):
    kt = K.Kitti()
    model = C.load_model("fpttc_kitti")
    g = K.ground_truth(kt, image_id)
    a, b = kt.image(image_id, 9), kt.image(image_id, 10)
    scale, _ = C.run_pair(model, a, b, size=(960, 288))
    h, w = g["gt"].shape
    pred = cv2.resize(scale, (w, h))
    gt = np.where(g["mask"], g["gt"], np.nan)
    err = np.where(g["mask"], np.abs(np.log(pred) - np.log(g["gt"])), np.nan)
    mid = np.nanmean(err) * 1e4

    fig, ax = plt.subplots(2, 2, figsize=(13, 5.6))
    ax[0, 0].imshow(b); ax[0, 0].set_title(f"KITTI frame {image_id:06d} (held out from training)")
    im = ax[0, 1].imshow(pred, cmap="RdYlBu", vmin=0.85, vmax=1.15); ax[0, 1].set_title("FP-TTC scale map")
    ax[1, 0].imshow(gt, cmap="RdYlBu", vmin=0.85, vmax=1.15); ax[1, 0].set_title("true scale, from stereo and flow")
    ie = ax[1, 1].imshow(err, cmap="magma", vmin=0, vmax=0.05); ax[1, 1].set_title(f"error |log ratio|   Mid error {mid:.0f} (x1e-4)")
    for x in ax.ravel():
        x.axis("off")
    fig.colorbar(im, ax=ax[:, 1], fraction=0.02, pad=0.01, label="scale: below 1 = getting closer")
    fig.savefig(C.FIGURES_DIR / "kitti_example.jpg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)

    # FP-TTC's own style output: colour map of time to collision
    ttc = C.ttc_from_scale(scale, 0.1)
    cmap = plt.get_cmap("inferno").copy(); cmap.set_bad((0.6, 0.6, 0.6))
    fig, ax = plt.subplots(1, 2, figsize=(11, 2.9))
    ax[0].imshow(cv2.resize(b, (960, 288))); ax[0].axis("off"); ax[0].set_title("input")
    im = ax[1].imshow(np.ma.masked_invalid(ttc), cmap=cmap, vmin=0, vmax=10); ax[1].axis("off")
    ax[1].set_title("time to collision per pixel (seconds, grey = not approaching)")
    fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01)
    fig.savefig(C.FIGURES_DIR / "fpttc_output_ttc.jpg", dpi=110, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)
    return mid


def nuscenes_example():
    n = pd.read_csv(C.RESULTS_DIR / "nuscenes_cars.csv")
    n = n[(n.weights == "fpttc_kitti") & (n.width == 640) & (n.height == 352)]
    cand = n.groupby("keyframe").apply(lambda g: pd.Series(dict(
        total=len(g), danger=int((g.ttc_true < DANGER).sum()),
        near=int(((g.ttc_true < 6) & (g.z < 25) & (g.box_w > 150)).sum()))), include_groups=False)
    cand = cand[(cand.total >= 3) & (cand.danger >= 1)].sort_values(["near", "total"], ascending=False)
    k = int(cand.index[0])

    src = NuScenesSweepSource()
    nusc = src.nusc
    sample = nusc.sample[k]
    tok = sample["data"][config.CAMERA_CHANNEL]
    key, prev = src.frame(tok), src.frame(nusc.get("sample_data", tok)["prev"])
    dt = key.timestamp_s - prev.timestamp_s
    H, W = key.image.shape[:2]
    cars = N.true_cars(nusc, key, prev, dt, W, H)
    dets = VehicleDetector().detect(key.image)
    model = C.load_model("fpttc_kitti")
    scale, _ = C.run_pair(model, cv2.cvtColor(prev.image, cv2.COLOR_BGR2RGB),
                          cv2.cvtColor(key.image, cv2.COLOR_BGR2RGB), size=(640, 352))
    img = key.image.copy()
    fx, fy = 640 / W, 352 / H
    shown = 0
    for d in sorted(dets, key=lambda d: -(d.x2 - d.x1) * (d.y2 - d.y1)):
        if shown >= 5:
            break
        b = (d.x1, d.y1, d.x2, d.y2)
        best = max(cars, key=lambda c: N.iou(c["box"], b), default=None)
        if best is None or N.iou(best["box"], b) < 0.4:
            continue
        s, _ = C.car_scale_median(scale, (b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy), 0.7)
        tp = float(C.ttc_from_scale(np.array([s]), dt)[0])
        tt = best["ttc_true"]
        danger = np.isfinite(tp) and tp < DANGER
        col = (40, 40, 235) if danger else ((0, 170, 255) if np.isfinite(tp) and tp < 6 else (60, 190, 60))
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 4)
        label = (f"{tp:.1f}s" if np.isfinite(tp) else "no TTC") + (f"  (true {tt:.1f}s)" if np.isfinite(tt) else "  (true: not closing)")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        y0 = max(int(b[1]) - 8, th + 4)
        cv2.rectangle(img, (int(b[0]), y0 - th - 6), (int(b[0]) + tw + 6, y0 + 4), col, -1)
        cv2.putText(img, label, (int(b[0]) + 3, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        shown += 1
    cv2.imwrite(str(C.FIGURES_DIR / "nuscenes_example.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return k, shown, key.scene_name


def main():
    C.FIGURES_DIR.mkdir(exist_ok=True)
    # a held-out KITTI image with several cars
    c = pd.read_csv(C.RESULTS_DIR / "kitti_cars.csv")
    c = c[(c.protocol == "past_09_10") & (c.split == "held_out_40") & (c.weights == "fpttc_kitti") & (c.width == 960)]
    best = c[c.s_true < 0.97].groupby("image").size().sort_values(ascending=False).index[0]
    print("KITTI example image", best, "Mid error", round(kitti_example(int(best)), 1))
    print("nuScenes example", nuscenes_example())


if __name__ == "__main__":
    main()
