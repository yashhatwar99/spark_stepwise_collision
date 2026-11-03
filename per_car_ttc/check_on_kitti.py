"""
Check the median method against KITTI's own measured answer.

For a KITTI scene-flow training image, run the method on the frames before it
(past protocol: frames _09 and _10 predict the step after _10), then compare each
detected car's TTC with the TTC that KITTI's stereo depth gives for the same
pixels: the median of dt / (1 - s_true) over the valid truth pixels inside the
same inner box, with s_true = disparity at _10 / disparity at _11 (the repo's
own ground_truth()).

Usage (from the project root, with the project .venv):
    python per_car_ttc/check_on_kitti.py 25 10 0
    (any image numbers 0 to 199; the held-out 40 are the multiples of 5)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import median_ttc as M  # noqa: E402
import eval_kitti as K  # noqa: E402  (fpttc_results/eval_kitti.py)

C = M.C
DT = 0.1


def truth_ttc(gt, mask, box_img, frac):
    """Median true TTC inside the inner box (box in KITTI pixels), over valid truth pixels."""
    x1, y1, x2, y2 = C.inner_box(box_img, frac)
    H, W = gt.shape
    x1, x2 = max(int(x1), 0), min(int(np.ceil(x2)), W)
    y1, y2 = max(int(y1), 0), min(int(np.ceil(y2)), H)
    patch, ok = gt[y1:y2, x1:x2], mask[y1:y2, x1:x2]
    s = patch[ok]
    if s.size < 50:
        return None, int(s.size)
    t = np.where(s < 1.0, DT / (1.0 - np.minimum(s, 0.999999)), np.inf)
    return float(np.median(t)), int(s.size)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=int, nargs="+", help="KITTI training image numbers")
    ap.add_argument("--weights", default="fpttc_kitti")
    a = ap.parse_args()

    kt = K.Kitti()
    model = C.load_model(a.weights)
    detector = M.VehicleDetector()
    diffs = []
    for i in a.images:
        rgb0, rgb1 = kt.image(i, 9), kt.image(i, 10)
        prev, curr = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR), cv2.cvtColor(rgb1, cv2.COLOR_RGB2BGR)
        g = K.ground_truth(kt, i)
        _, _, cars = M.run(prev, curr, DT, model, detector)
        split = "held out" if i % 5 == 0 else "seen in training"
        print(f"\nKITTI image {i:06d}  ({split})")
        print("  predicted TTC   KITTI TTC    difference   truth pixels   box")
        for c in cars:
            tt, n = truth_ttc(g["gt"], g["mask"], c["box"], M.BOX_FRAC)
            pt = np.inf if c["ttc_s"] is None else c["ttc_s"]
            if tt is None:
                print(f"  {'not closing' if c['ttc_s'] is None else format(pt, '10.2f'):>13s}   {'(too few truth pixels)':<22s}  {n:>8d}      {c['box']}")
                continue
            d = (pt - tt) if np.isfinite(pt) and np.isfinite(tt) else None
            if d is not None:
                diffs.append(abs(d))
            fmt = lambda v: "not closing" if not np.isfinite(v) else f"{v:.2f}"
            print(f"  {fmt(pt):>13s}   {fmt(tt):>11s}   {('%+.2f' % d) if d is not None else '-':>10s}   {n:>10d}   {c['box']}")
    if diffs:
        print(f"\ncars with a finite TTC in both: {len(diffs)}; median |difference| = {np.median(diffs):.2f} s")


if __name__ == "__main__":
    main()
