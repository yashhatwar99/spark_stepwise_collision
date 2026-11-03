"""
Per-car time to collision, median method.

Two frames from the same camera go into FP-TTC, which gives a scale map s (one
number per pixel: next distance / current distance, below 1 = getting closer).
That is turned into a TTC map with the paper's Eq. 1, TTC = dt / (1 - s), where
pixels that are not getting closer (s >= 1) are +infinity. YOLO draws a box on
each car; the median of the TTC map inside the inner 70% of the box is that
car's TTC. Because the median only looks at the middle value, it gives the same
answer as taking the median of s and converting after.

Not in this first version: validity checks, spread (IQR), tracking, smoothing.

Usage (from the project root, with the project .venv):
    python per_car_ttc/median_ttc.py FRAME_EARLIER FRAME_LATER --dt 0.1
    python per_car_ttc/median_ttc.py a.png b.png --dt 0.1 --weights fpttc_mix --frac 0.7

dt is the real time in seconds between the two frames. Both frames must come
from the same camera. FP-TTC was trained on frames 0.1 s apart.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "fpttc_results"))

import common as C  # noqa: E402  (loads FP-TTC from fpttc/, sets up sys.path)
from src2.detection import VehicleDetector  # noqa: E402

BOX_FRAC = 0.7      # inner part of the YOLO box that is read
DANGER_S = 3.0      # colour a car red below this many seconds (display only)
OUT_DIR = HERE / "output" / "median"     # one output folder per method (output/<method>/)


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


def run(prev_bgr, curr_bgr, dt, model, detector, size=(640, 320), frac=BOX_FRAC):
    """prev_bgr, curr_bgr: the two frames as read by cv2.imread. Returns
    (scale map, TTC map, cars), cars sorted with the soonest collision first.
    Each car: class_name, confidence, box (x1, y1, x2, y2 in the later frame's
    pixels), ttc_s (None = not approaching), approaching_share, pixels."""
    rgb0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    rgb1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
    scale, _ = C.run_pair(model, rgb0, rgb1, size=size)
    ttc = ttc_map(scale, dt)
    H, W = curr_bgr.shape[:2]
    fx, fy = size[0] / W, size[1] / H
    cars = []
    for d in detector.detect(curr_bgr):
        t, share, n = car_ttc(ttc, (d.x1 * fx, d.y1 * fy, d.x2 * fx, d.y2 * fy), frac)
        cars.append(dict(class_name=d.class_name, confidence=round(float(d.confidence), 3),
                         box=[round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
                         ttc_s=round(t, 2) if np.isfinite(t) else None,
                         approaching_share=round(share, 3), pixels=n))
    cars.sort(key=lambda c: (c["ttc_s"] is None, c["ttc_s"] if c["ttc_s"] is not None else 0.0))
    return scale, ttc, cars


def draw(curr_bgr, cars):
    img = curr_bgr.copy()
    for c in cars:
        x1, y1, x2, y2 = (int(v) for v in c["box"])
        t = c["ttc_s"]
        if t is None:
            col, label = (150, 150, 150), "not closing"
        else:
            col = (40, 40, 235) if t < DANGER_S else ((0, 170, 255) if t < 6 else (60, 190, 60))
            label = f"{t:.1f} s"
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        y0 = max(y1, th + 8)
        cv2.rectangle(img, (x1, y0 - th - 8), (x1 + tw + 8, y0), col, -1)
        cv2.putText(img, label, (x1 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def side_by_side(curr_bgr, scale, cars, dt, frac, path):
    """Left: the later frame with each car's TTC. Right: the per-pixel TTC map
    with the same cars; the dashed box is the inner part of the box that is read."""
    H, W = curr_bgr.shape[:2]
    # display only: the map is stretched back to the photo's size; the car
    # numbers themselves come from the model-size map in run()
    ttc_full = ttc_map(cv2.resize(scale, (W, H)), dt)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad((0.6, 0.6, 0.6))                       # grey = not approaching
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.0 * H / W + 0.7))
    ax[0].imshow(cv2.cvtColor(draw(curr_bgr, cars), cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"later frame: TTC per car (median of the inner {frac:.0%} of each box)")
    im = ax[1].imshow(np.ma.masked_invalid(np.where(np.isfinite(ttc_full), ttc_full, np.nan)),
                      cmap=cmap, vmin=0, vmax=10)
    ax[1].set_title("TTC per pixel (seconds), grey = not approaching")
    for c in cars:
        x1, y1, x2, y2 = c["box"]
        ix1, iy1, ix2, iy2 = C.inner_box((x1, y1, x2, y2), frac)
        ax[1].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.0))
        ax[1].add_patch(plt.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, fill=False, ec="white", lw=1.2, ls="--"))
        label = "not closing" if c["ttc_s"] is None else f"{c['ttc_s']:.1f} s"
        ax[1].text(x1, y1 - 3, label, color="white", fontsize=8, va="bottom",
                   bbox=dict(fc="black", ec="none", alpha=0.65, pad=1.5))
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01, label="seconds")
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("earlier", help="earlier frame (image file)")
    ap.add_argument("later", help="later frame (image file); cars are detected on this one")
    ap.add_argument("--dt", type=float, default=0.1, help="seconds between the two frames (default 0.1)")
    ap.add_argument("--weights", default="fpttc_kitti", choices=["fpttc_kitti", "fpttc_mix"])
    ap.add_argument("--width", type=int, default=640, help="model input width, multiple of 32")
    ap.add_argument("--height", type=int, default=320, help="model input height, multiple of 32")
    ap.add_argument("--frac", type=float, default=BOX_FRAC, help="inner part of the box to read (default 0.7)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    a = ap.parse_args()

    prev, curr = cv2.imread(a.earlier), cv2.imread(a.later)
    if prev is None or curr is None:
        raise SystemExit("could not read one of the images")
    if prev.shape != curr.shape:
        raise SystemExit(f"the two frames must be the same size, got {prev.shape} and {curr.shape}")

    model = C.load_model(a.weights)
    detector = VehicleDetector()
    scale, _, cars = run(prev, curr, a.dt, model, detector, size=(a.width, a.height), frac=a.frac)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(a.later).stem
    side_by_side(curr, scale, cars, a.dt, a.frac, str(out / f"{stem}_ttc.jpg"))
    (out / f"{stem}_ttc.json").write_text(json.dumps(dict(dt=a.dt, weights=a.weights, frac=a.frac, cars=cars), indent=1))

    print(f"{len(cars)} cars, dt = {a.dt} s, weights = {a.weights}, inner box = {a.frac:.0%}\n")
    print("  TTC (s)   class        conf   approaching   box (x1, y1, x2, y2)")
    for c in cars:
        t = "not closing" if c["ttc_s"] is None else f"{c['ttc_s']:7.2f}"
        print(f"  {t:>11s}   {c['class_name']:<10s}  {c['confidence']:.2f}   {c['approaching_share']:>9.0%}    {c['box']}")
    print(f"\nsaved {out / (stem + '_ttc.jpg')}\n      {out / (stem + '_ttc.json')}")


if __name__ == "__main__":
    main()
