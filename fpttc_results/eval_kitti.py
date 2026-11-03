"""
Accuracy of FP-TTC on the KITTI scene-flow training set, which has ground truth.

Ground truth is rebuilt exactly as the repo's own validation code does it
(fpttc/utils/trainer.py, eval_epoch): true scale = distance in the later frame
divided by distance in the earlier frame, from stereo disparity at both times
and the true optical flow. Two input protocols are run: "past" feeds frames _09 and _10, which is how
the model is trained (it predicts the scale for the NEXT interval, 10 to 11,
from the two frames before it); "script" feeds _10 and _11, which is what the
repo's evaluation script does. The 40 images
whose number is divisible by 5 are the held-out set: the released
fpttc_kitti weights were trained on the other 160.

Reports, for each weights file and each input size:
  * Mid error (the paper's metric), overall, by region, by distance
  * optical-flow accuracy of the flow FP-TTC computes inside itself
  * per-car scale and time-to-collision error, by three methods
  * time per frame pair

Usage (from the project root):
    python fpttc_results/eval_kitti.py            # everything, about 10-15 minutes
    python fpttc_results/eval_kitti.py --limit 5  # quick test
"""

import argparse
import io
import json
import time
import zipfile

import cv2
import numpy as np
import torch

import common as C
from dataloader.load import triangulation  # the repo's own helper

ZIP_MAIN = C.ROOT / "data" / "kitti_scene_flow" / "data_scene_flow.zip"
ZIP_CALIB = C.ROOT / "data" / "kitti_scene_flow" / "data_scene_flow_calib.zip"
PAST_DIR = C.ROOT / "data" / "kitti_scene_flow" / "multiview" / "training" / "image_2"
DT = 0.1  # KITTI frames are 10 Hz

WEIGHTS = ["fpttc_kitti", "fpttc_mix"]
SIZES = [(640, 320), (960, 288), (1248, 384)]   # (width, height)
DEPTH_BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]


class Kitti:
    def __init__(self):
        self.zm = zipfile.ZipFile(ZIP_MAIN)
        self.zc = zipfile.ZipFile(ZIP_CALIB)

    def _png(self, name, flags):
        return cv2.imdecode(np.frombuffer(self.zm.read(name), np.uint8), flags)

    def image(self, i, k):  # k = 9, 10 or 11, returns RGB uint8
        if k == 9:  # the earlier frame lives in the multiview archive, fetched separately
            bgr = cv2.imread(str(PAST_DIR / f"{i:06d}_09.png"), cv2.IMREAD_COLOR)
        else:
            bgr = self._png(f"training/image_2/{i:06d}_{k}.png", cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def disparity(self, which, i):
        d = self._png(f"training/disp_occ_{which}/{i:06d}_10.png", cv2.IMREAD_ANYDEPTH)
        return d.astype(np.float32) / 256.0

    def flow(self, i):
        f = self._png(f"training/flow_occ/{i:06d}_10.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
        f = f[:, :, ::-1].astype(np.float32)
        return (f[:, :, :2] - 2 ** 15) / 64.0, f[:, :, 2]

    def objects(self, i):
        return self._png(f"training/obj_map/{i:06d}_10.png", cv2.IMREAD_UNCHANGED)

    def calib(self, i):
        txt = self.zc.read(f"training/calib_cam_to_cam/{i:06d}.txt").decode()
        P = {}
        for line in txt.splitlines():
            if line.startswith(("P_rect_02", "P_rect_03")):
                P[line[:9]] = np.array(line.split(":")[1].split(), dtype=np.float64).reshape(3, 4)
        p2, p3 = P["P_rect_02"], P["P_rect_03"]
        bl = p2[0, 3] / p2[0, 0] - p3[0, 3] / p3[0, 0]
        return dict(fl=p2[0, 0], cx=p2[0, 2], cy=p2[1, 2], bl=bl)


def ground_truth(kt, i):
    """True scale per pixel, copied from the repo's validation routine."""
    flow, valid = kt.flow(i)
    d1, d2 = kt.disparity(0, i), kt.disparity(1, i)
    cal = kt.calib(i)
    valid_flow = valid > 0.99
    valid = valid_flow & (d1 > 0) & (d2 > 0)
    d1[d1 <= 0] = 1e6
    d2[d2 <= 0] = 1e6
    h, w = d1.shape
    xs, ys = np.meshgrid(range(w), range(h))
    xs, ys = xs.astype(float), ys.astype(float)
    kw = dict(bl=cal["bl"], fl=cal["fl"], cx=cal["cx"], cy=cal["cy"])
    P0 = triangulation(d1, xs, ys, **kw)
    P1 = triangulation(d2, xs + flow[:, :, 0], ys + flow[:, :, 1], **kw)
    z0 = P0[2].reshape(h, w).astype(np.float32)
    valid = valid & (z0 > 0)
    dz = (P1 - P0)[2].reshape(h, w)
    gt = 1.0 + dz / z0
    mask = (gt < 3) & (gt > 0.3) & valid
    return dict(gt=gt, mask=mask, depth=z0, flow=flow, valid_flow=valid_flow)


def flow_errors(pred_flow, size, g):
    w0, h0 = size
    h, w = g["gt"].shape
    u = cv2.resize(pred_flow[0], (w, h)) * (w / w0)
    v = cv2.resize(pred_flow[1], (w, h)) * (h / h0)
    epe = np.hypot(u - g["flow"][:, :, 0], v - g["flow"][:, :, 1])[g["valid_flow"]]
    mag = np.hypot(g["flow"][:, :, 0], g["flow"][:, :, 1])[g["valid_flow"]]
    return float(epe.mean()), float(np.mean((epe > 3) & (epe > 0.05 * mag)))


def log_err(pred, g, sel=None):
    m = g["mask"] if sel is None else g["mask"] & sel
    if m.sum() == 0:
        return np.nan, 0
    return float(np.abs(np.log(np.clip(pred, 1e-3, None)) - np.log(g["gt"]))[m].mean()), int(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    kt = Kitti()
    models = {wn: C.load_model(wn) for wn in WEIGHTS}
    configs = [(wn, sz) for wn in WEIGHTS for sz in SIZES]

    img_rows, car_rows = [], []
    t_start = time.time()
    for i in range(args.limit):
        split = "held_out_40" if i % 5 == 0 else "trained_on_160"
        g = ground_truth(kt, i)
        h, w = g["gt"].shape
        f09, f10, f11 = kt.image(i, 9), kt.image(i, 10), kt.image(i, 11)
        protocols = {"past_09_10": (f09, f10), "script_10_11": (f10, f11)}
        obj = kt.objects(i)
        fg = obj > 0
        depth = g["depth"]

        # the trivial answer "nothing changes" (s = 1) as a yardstick
        base_err, _ = log_err(np.ones_like(g["gt"]), g)

        objs = []
        for oid in [v for v in np.unique(obj) if v != 0]:
            m = obj == oid
            mv = m & g["mask"]
            if mv.sum() < 300:
                continue
            ys_, xs_ = np.where(m)
            objs.append(dict(oid=int(oid), mask=m, box=(xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1),
                             s_true=float(np.median(g["gt"][mv])), z=float(np.median(depth[mv])),
                             area=int(m.sum())))

        for (proto, (a, b)), (wn, sz) in [(pr, cf) for pr in protocols.items() for cf in configs]:
            w0, h0 = sz
            t0 = time.perf_counter()
            scale, flow = C.run_pair(models[wn], a, b, size=sz)
            torch.cuda.synchronize()
            dt_run = time.perf_counter() - t0

            pred = cv2.resize(scale, (w, h))
            mid, npx = log_err(pred, g)
            mid_fg, _ = log_err(pred, g, fg)
            mid_bg, _ = log_err(pred, g, ~fg)
            row = dict(image=i, split=split, protocol=proto, weights=wn, width=w0, height=h0, mid_err=mid,
                       mid_err_foreground=mid_fg, mid_err_background=mid_bg, valid_pixels=npx,
                       baseline_no_change=base_err, seconds=dt_run)
            for lo, hi in DEPTH_BINS:
                row[f"mid_err_depth_{lo}_{int(min(hi, 999))}"] = log_err(pred, g, (depth >= lo) & (depth < hi))[0]
            row["flow_epe"], row["flow_fl_all"] = flow_errors(flow, sz, g)
            img_rows.append(row)

            fx, fy = w0 / w, h0 / h
            for o in objs:
                bx = (o["box"][0] * fx, o["box"][1] * fy, o["box"][2] * fx, o["box"][3] * fy)
                small = cv2.resize(o["mask"].astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST) > 0
                rec = dict(image=i, split=split, protocol=proto, weights=wn, width=w0, height=h0, object=o["oid"],
                           depth_m=o["z"], area_px=o["area"], s_true=o["s_true"])
                rec["s_median_box100"], _ = C.car_scale_median(scale, bx, 1.0)
                rec["s_median_box70"], _ = C.car_scale_median(scale, bx, 0.7)
                rec["s_median_box50"], _ = C.car_scale_median(scale, bx, 0.5)
                rec["s_median_mask"], _ = C.car_scale_mask_median(scale, small)
                rec["s_zoom_box70"], rec["zoom_inlier_share"], _ = C.car_scale_zoom(flow, bx, 0.7)
                car_rows.append(rec)

        if i % 10 == 0:
            el = time.time() - t_start
            print(f"image {i + 1}/{args.limit}   {el:.0f}s elapsed", flush=True)

    import pandas as pd
    pd.DataFrame(img_rows).to_csv(C.RESULTS_DIR / "kitti_images.csv", index=False)
    pd.DataFrame(car_rows).to_csv(C.RESULTS_DIR / "kitti_cars.csv", index=False)
    print("saved kitti_images.csv and kitti_cars.csv")


if __name__ == "__main__":
    main()
