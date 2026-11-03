"""
Shared pieces for measuring FP-TTC: load the model, run it on a frame pair,
and turn its per-pixel output into one number per car.

This file imports from fpttc/ and does not edit it. (Two of the authors' files
there carry small local patches so the model runs without mmcv's CUDA op:
scale_net/scale_encoder.py and scale_net/utils/multi_scale_deformable_attn_function.py,
plus the added mmcv_compat.py, view_results.py and run_kitti_eval.sh.)

Conventions used everywhere in this folder
  scale s   next distance / current distance, per pixel or per car.
            s < 1 means getting closer.
  TTC       dt / (1 - s), seconds, only defined when s < 1 (the paper's Eq. 1,
            with s = eta, the model's predicted next/current distance ratio).
  Mid error mean |log(predicted s) - log(true s)|, the paper's accuracy
            metric. Reported times 10,000, as the paper does (59.35 means
            0.005935 in log units, about 0.6 percent).
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
# fpttc/ first, so `import fpttc.fp_ttc` finds the model code and not the
# clone folder itself.
for p in (str(ROOT), str(ROOT / "fpttc")):
    if p in sys.path:
        sys.path.remove(p)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fpttc"))

from fpttc.fp_ttc import FpTTC  # noqa: E402

WEIGHTS_DIR = ROOT / "fpttc" / "pretrained"
RESULTS_DIR = ROOT / "fpttc_results" / "results"
FIGURES_DIR = ROOT / "fpttc_results" / "figures"

# Settings copied from the repo's own test.sh. The README's evaluation
# command leaves these at defaults that do not match the released weights.
MODEL_KW = dict(num_scales=2, feature_channels=128, upsample_factor=4, num_head=1,
                ffn_dim_expansion=4, num_transformer_layers=6, reg_refine=False, train=False)
RUN_KW = dict(attn_type="swin", attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
              prop_radius_list=[-1, 1], num_reg_refine=1, testing=False)


def load_model(weights="fpttc_kitti", device="cuda"):
    model = FpTTC(**MODEL_KW).to(device).eval()
    ck = torch.load(WEIGHTS_DIR / f"{weights}.pth.tar", map_location=device, weights_only=True)
    model.load_state_dict({k.replace("module.", ""): v for k, v in ck["net"].items()})
    return model


@torch.no_grad()
def run_pair(model, rgb0, rgb1, size=(640, 320), device="cuda"):
    """rgb0 = earlier frame, rgb1 = later frame, uint8 RGB arrays.
    size = (width, height) the frames are squeezed to; both must be
    multiples of 32.

    Returns scale (H, W) and flow (2, H, W) as numpy arrays at that size.
    Flow is in pixels at that size, from rgb0 to rgb1.
    """
    w, h = size
    a = cv2.resize(rgb0, (w, h))
    b = cv2.resize(rgb1, (w, h))
    a = torch.from_numpy(a).permute(2, 0, 1).float().unsqueeze(0).to(device)
    b = torch.from_numpy(b).permute(2, 0, 1).float().unsqueeze(0).to(device)
    scale, flow, _ = model(a, b, **RUN_KW)
    return scale[0, 0].cpu().numpy(), flow[0].cpu().numpy()


# ---------------------------------------------------------------- per-car

def inner_box(box, frac):
    """Shrink an (x1, y1, x2, y2) box about its centre to `frac` of its size."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 2 * frac, (y2 - y1) / 2 * frac
    return cx - hw, cy - hh, cx + hw, cy + hh


def _pixels(box, shape):
    h, w = shape
    x1, y1, x2, y2 = box
    xs = np.arange(max(int(np.floor(x1)), 0), min(int(np.ceil(x2)), w))
    ys = np.arange(max(int(np.floor(y1)), 0), min(int(np.ceil(y2)), h))
    return xs, ys


def car_scale_median(scale, box, frac=0.5):
    """Median of the scale map inside the shrunken box. Returns (s, n_pixels)."""
    xs, ys = _pixels(inner_box(box, frac), scale.shape)
    if len(xs) == 0 or len(ys) == 0:
        return np.nan, 0
    patch = scale[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    return float(np.median(patch)), patch.size


def car_scale_mask_median(scale, mask):
    """Median of the scale map over an exact car mask (oracle upper bound)."""
    vals = scale[mask]
    return (float(np.median(vals)), int(mask.sum())) if vals.size else (np.nan, 0)


def car_scale_zoom(flow, box, frac=0.7, step=2, thr=1.0, min_points=20):
    """Fit one zoom to the flow vectors inside the shrunken box (RANSAC).

    Returns (s, inlier_share, n_points) with s = 1/k, so it reads like the
    scale map: s < 1 means the car is growing, i.e. getting closer.
    """
    _, h, w = flow.shape
    xs, ys = _pixels(inner_box(box, frac), (h, w))
    xs, ys = xs[::step], ys[::step]
    if len(xs) < 3 or len(ys) < 3:
        return np.nan, 0.0, 0
    gx, gy = np.meshgrid(xs, ys)
    old = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    new = old + np.stack([flow[0, gy.ravel(), gx.ravel()], flow[1, gy.ravel(), gx.ravel()]], 1)
    if len(old) < min_points:
        return np.nan, 0.0, len(old)
    M, inl = cv2.estimateAffinePartial2D(old, new.astype(np.float32), method=cv2.RANSAC,
                                         ransacReprojThreshold=thr, maxIters=2000, confidence=0.99)
    if M is None:
        return np.nan, 0.0, len(old)
    k = float(np.hypot(M[0, 0], M[1, 0]))
    return 1.0 / k, float(inl.mean()), len(old)


def ttc_from_scale(s, dt):
    """Seconds until the car reaches the camera plane, at constant closing
    speed: dt / (1 - s), the paper's Eq. 1. `s` is the model's scale output
    (next distance / current distance). NaN when the car is not getting
    closer (s >= 1).

    Not for the zoom-fit method: its s = 1/k is the ratio between the two
    frames already seen, whose TTC is dt / (k - 1) = dt * s / (1 - s)."""
    s = np.asarray(s, dtype=np.float64)
    out = np.full(s.shape, np.nan)
    ok = s < 1.0
    out[ok] = dt / (1.0 - s[ok])
    return out


def summarise(values):
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), mean=float(v.mean()), median=float(np.median(v)),
                p90=float(np.percentile(v, 90)), max=float(v.max()))


class Timer:
    """GPU-aware stopwatch: waits for the GPU before reading the clock."""

    def __init__(self, cuda=True):
        self.cuda = cuda and torch.cuda.is_available()

    def now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.perf_counter()
