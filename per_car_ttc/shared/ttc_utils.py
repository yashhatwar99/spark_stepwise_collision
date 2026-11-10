"""
Pieces every per-car method uses: the TTC map, the median inside a car's box,
and the picture of boxes labelled with their TTC. Each method has its own folder
(median/, learned_head/, ...) and imports this.

TTC is the paper's Eq. 1, dt / (1 - s), where s is the model's scale output
(next distance / current distance; below 1 = getting closer).
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # the project root
sys.path.insert(0, str(ROOT / "fpttc_results"))

import common as C  # noqa: E402,F401  (loads FP-TTC from fpttc/, sets up sys.path)
from src2.detection import VehicleDetector  # noqa: E402,F401

BOX_FRAC = 0.7      # inner part of the YOLO box that is read
DANGER_S = 3.0      # colour a car red below this many seconds (display only)


def ttc_map(scale, dt):
    """Per-pixel TTC in seconds, dt / (1 - s). Not approaching (s >= 1): +inf."""
    s = np.asarray(scale, dtype=np.float64)
    out = np.full(s.shape, np.inf)
    ok = s < 1.0
    out[ok] = dt / (1.0 - s[ok])
    return out


def car_ttc(ttc, box, frac=BOX_FRAC):
    """Median of the TTC map inside the inner `frac` of box (x1, y1, x2, y2, in
    the map's pixels). Returns (ttc_seconds, share_of_pixels_approaching, n_pixels).
    ttc_seconds is inf when the middle pixel is not approaching, nan if the box is empty."""
    xs, ys = C._pixels(C.inner_box(box, frac), ttc.shape)
    if len(xs) == 0 or len(ys) == 0:
        return float("nan"), 0.0, 0
    patch = ttc[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
    return float(np.median(patch)), float(np.isfinite(patch).mean()), int(patch.size)


def draw(curr_bgr, cars, font=0.8, stagger=False):
    """Boxes on the later frame, each labelled with its TTC (cars: dicts with box and ttc_s, None = not approaching).
    A car may carry its own "label" and "color" (BGR), which replace the default ones.
    font: text size. stagger: every second label goes under its box, so labels of neighbouring cars do not overlap."""
    img = curr_bgr.copy()
    for i, c in enumerate(cars):
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        t = c["ttc_s"]
        if t is None:
            col, label = (150, 150, 150), "not closing"
        else:
            col = (40, 40, 235) if t < DANGER_S else ((0, 170, 255) if t < 6 else (60, 190, 60))
            label = f"{t:.1f} s" if t <= 99 else ">99 s"
        col, label = c.get("color", col), c.get("label", label)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        thick = 2 if font >= 0.7 else 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, thick)
        if stagger and i % 2 == 1:
            y0 = min(y2 + th + 8, img.shape[0] - 2)             # label under the box
        else:
            y0 = max(y1, th + 8)                                # label above the box
        cv2.rectangle(img, (x1, y0 - th - 8), (x1 + tw + 8, y0), col, -1)
        cv2.putText(img, label, (x1 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, font, (255, 255, 255), thick, cv2.LINE_AA)
    return img
