"""
True TTC for the cases we have labels for, so a method's output can be shown
next to the answer:

  KittiCase(i)       a KITTI scene-flow training pair (frames _09 and _10, the later one is _10).
                     True TTC = median of dt / (1 - true s) over the pixels of KITTI's measured
                     depth inside the inner part of each box.
  NuScenesCase(k)    one nuScenes front-camera frame (keyframe number k) and the frame before it.
                     True TTC = distance / closing speed of the labelled car whose box overlaps the
                     detection by at least 0.4 (the same matching as fpttc_results/eval_nuscenes.py).

Each case has: .name, .prev_bgr, .curr_bgr (the two frames), .dt (seconds between them), and
.truth(boxes) -> one dict per box with
    has_truth    False if there is no label for that box
    true_ttc_s   seconds, or None when the car is not getting closer
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttc_utils import BOX_FRAC, C  # noqa: E402  (also puts fpttc_results on sys.path)

KITTI_DT = 0.1


def kitti_box_truth(gt, mask, box, frac=BOX_FRAC, dt=KITTI_DT, min_pixels=50):
    """Median true TTC inside the inner box (box in KITTI pixels), over pixels with valid truth.
    Returns (ttc seconds or inf if not approaching, number of truth pixels), or (None, n) if too few pixels."""
    x1, y1, x2, y2 = C.inner_box(box, frac)
    H, W = gt.shape
    x1, x2 = max(int(x1), 0), min(int(np.ceil(x2)), W)
    y1, y2 = max(int(y1), 0), min(int(np.ceil(y2)), H)
    s = gt[y1:y2, x1:x2][mask[y1:y2, x1:x2]]
    if s.size < min_pixels:
        return None, int(s.size)
    t = np.where(s < 1.0, dt / (1.0 - np.minimum(s, 0.999999)), np.inf)
    return float(np.median(t)), int(s.size)


class KittiCase:
    def __init__(self, i, kt=None):
        import eval_kitti as K
        kt = kt or K.Kitti()                  # pass one Kitti() in when making many cases
        self.name = f"kitti_{i:06d}"
        self.dt = KITTI_DT
        self.prev_bgr = cv2.cvtColor(kt.image(i, 9), cv2.COLOR_RGB2BGR)
        self.curr_bgr = cv2.cvtColor(kt.image(i, 10), cv2.COLOR_RGB2BGR)
        self.g = K.ground_truth(kt, i)
        self.where = f"KITTI training pair {i:06d} (frames _09 and _10)"

    def truth(self, boxes):
        out = []
        for b in boxes:
            t, n = kitti_box_truth(self.g["gt"], self.g["mask"], b)
            out.append(dict(has_truth=t is not None, true_ttc_s=t if t is not None and np.isfinite(t) else None,
                            truth_pixels=n))
        return out


class NuScenesCase:
    def __init__(self, k):
        import eval_nuscenes as EN
        from src2 import config
        from src3.sweep_source import NuScenesSweepSource
        self.EN = EN
        src = NuScenesSweepSource()
        nusc = src.nusc
        tok = nusc.sample[k]["data"][config.CAMERA_CHANNEL]
        prev_tok = nusc.get("sample_data", tok)["prev"]
        if not prev_tok:
            raise SystemExit(f"nuScenes frame {k} has no earlier camera frame; pick another number")
        key, prev = src.frame(tok), src.frame(prev_tok)
        self.dt = key.timestamp_s - prev.timestamp_s
        if not (0.03 <= self.dt <= 0.30):
            raise SystemExit(f"nuScenes frame {k}: the two frames are {self.dt:.2f} s apart (allowed 0.03 to 0.30 s); pick another number")
        self.name = f"nuscenes_{k:04d}"
        self.prev_bgr, self.curr_bgr = prev.image, key.image
        H, W = key.image.shape[:2]
        self.cars = EN.true_cars(nusc, key, prev, self.dt, W, H)
        self.where = f"nuScenes frame {k}, scene {key.scene_name}, {self.dt:.2f} s after the previous frame"

    def truth(self, boxes):
        out = []
        for b in boxes:
            best = max(self.cars, key=lambda c: self.EN.iou(c["box"], b), default=None)
            if best is None or self.EN.iou(best["box"], b) < 0.4:
                out.append(dict(has_truth=False, true_ttc_s=None))
            else:
                t = best["ttc_true"]
                out.append(dict(has_truth=True, true_ttc_s=float(t) if np.isfinite(t) else None, distance_m=round(best["z"], 1)))
        return out
