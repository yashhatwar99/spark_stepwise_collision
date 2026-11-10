"""
Per-car time to collision, Method 1: median.

Two frames from the same camera go into FP-TTC, which gives a scale map s (one
number per pixel: next distance / current distance, below 1 = getting closer).
That is turned into a TTC map with the paper's Eq. 1, TTC = dt / (1 - s), where
pixels that are not getting closer (s >= 1) are +infinity. YOLO draws a box on
each car; the median of the TTC map inside the inner 70% of the box is that
car's TTC. Because the median only looks at the middle value, it gives the same
answer as taking the median of s and converting after.

Not in this first version: validity checks, spread (IQR), tracking, smoothing.

Usage (from the project root, with the project .venv):
    # any two frames from the same camera
    python per_car_ttc/median/median_ttc.py FRAME_EARLIER FRAME_LATER --dt 0.1

    # a case that has the true answer: the output also shows the true TTC and
    # "predicted | true" on every car
    python per_car_ttc/median/median_ttc.py --kitti 25        # KITTI training pair 0..199
    python per_car_ttc/median/median_ttc.py --kitti_all       # all 200 KITTI pairs -> output/kitti/ and all_cars.csv
    python per_car_ttc/median/median_ttc.py --nuscenes 262    # nuScenes front-camera frame number

dt is the real time in seconds between the two frames (the labelled cases know it).
Both frames must come from the same camera. FP-TTC was trained on frames 0.1 s apart.
Output goes to per_car_ttc/median/output/.
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
sys.path.insert(0, str(HERE.parent / "shared"))

from ttc_utils import BOX_FRAC, C, VehicleDetector, car_ttc, draw, ttc_map  # noqa: E402,F401
import panels  # noqa: E402

OUT_DIR = HERE / "output"


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


def side_by_side(curr_bgr, scale, cars, dt, frac, path):
    """No truth available. Left: the later frame with each car's TTC. Right: the per-pixel TTC map
    with the same cars; the dashed box is the inner part of the box that is read."""
    H, W = curr_bgr.shape[:2]
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.0 * H / W + 0.7))
    ax[0].imshow(cv2.cvtColor(draw(curr_bgr, cars), cv2.COLOR_BGR2RGB))
    ax[0].set_title(f"later frame: TTC per car (median of the inner {frac:.0%} of each box)")
    im = panels.ttc_panel(ax[1], curr_bgr, scale, cars, dt, frac)
    ax[1].set_title("TTC per pixel (seconds), grey = not approaching")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01, label="seconds")
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)


def process(prev, curr, dt, stem, case, model, detector, args, out, print_table=True):
    """Run the method on one frame pair, save the picture and the JSON, optionally print the table. Returns the cars."""
    size = (args.width, args.height)
    scale, _, cars = run(prev, curr, dt, model, detector, size=size, frac=args.frac)
    if case is not None:
        panels.add_truth(cars, case.truth([c["box"] for c in cars]))
    out.mkdir(parents=True, exist_ok=True)
    pic = out / f"{stem}_ttc.jpg"
    if case is not None:
        titles = dict(a=f"median method: TTC per car (inner {args.frac:.0%} of each box)", b="true TTC per car",
                      c="median method on the TTC per pixel map (grey = not approaching)",
                      d="predicted | true  (green = close, orange = off, red = far off)")
        panels.four_panels(curr, scale, cars, dt, str(pic), case.where, titles, frac=args.frac)
    else:
        side_by_side(curr, scale, cars, dt, args.frac, str(pic))
    (out / f"{stem}_ttc.json").write_text(json.dumps(dict(dt=dt, weights=args.weights, frac=args.frac, cars=cars), indent=1))
    if not print_table:
        return cars

    print(f"{len(cars)} cars, dt = {dt:.3f} s, weights = {args.weights}, inner box = {args.frac:.0%}\n")
    if case is None:
        print("  TTC (s)   class        conf   approaching   box (x1, y1, x2, y2)")
        for c in cars:
            t = "not closing" if c["ttc_s"] is None else f"{c['ttc_s']:7.2f}"
            print(f"  {t:>11s}   {c['class_name']:<10s}  {c['confidence']:.2f}   {c['approaching_share']:>9.0%}    {c['box']}")
    else:
        print("  predicted    true       difference   class        box (x1, y1, x2, y2)")
        diffs = []
        for c in cars:
            p = "not closing" if c["ttc_s"] is None else f"{c['ttc_s']:.2f}"
            if not c["has_truth"]:
                tr, df = "no truth", ""
            else:
                tr = "not closing" if c["true_ttc_s"] is None else f"{c['true_ttc_s']:.2f}"
                df = "" if c["error_s"] is None else f"{c['error_s']:+.2f}"
                if c["error_s"] is not None:
                    diffs.append(abs(c["error_s"]))
            print(f"  {p:>11s}  {tr:>11s}  {df:>9s}    {c['class_name']:<10s}  {c['box']}")
        if diffs:
            print(f"\ncars with a TTC in both: {len(diffs)}; median |difference| = {np.median(diffs):.2f} s")
    print(f"\nsaved {pic}\n      {pic.with_suffix('.json')}")
    return cars


def run_all_kitti(args, model, detector):
    """The four-panel picture for every KITTI training pair (0..199), into output/kitti/, plus one table of all cars."""
    import csv
    import truth
    import eval_kitti as K
    kt = K.Kitti()
    out = Path(args.out) / "kitti"
    rows = []
    for i in range(200):
        case = truth.KittiCase(i, kt)
        cars = process(case.prev_bgr, case.curr_bgr, case.dt, case.name, case, model, detector, args, out, print_table=False)
        for c in cars:
            rows.append(dict(image=i, split="held_out" if i % 5 == 0 else "seen_in_training", class_name=c["class_name"],
                             confidence=c["confidence"], predicted_ttc_s=c["ttc_s"], has_truth=c["has_truth"],
                             true_ttc_s=c["true_ttc_s"], error_s=c["error_s"], box=c["box"]))
        if i % 20 == 0:
            print(f"image {i}/200   {len(rows)} cars so far", flush=True)
    path = out / "all_cars.csv"
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    both = [abs(r["error_s"]) for r in rows if r["error_s"] is not None]
    held = [abs(r["error_s"]) for r in rows if r["error_s"] is not None and r["split"] == "held_out"]
    print(f"\n200 pictures saved in {out}")
    print(f"{len(rows)} detected cars; {sum(r['has_truth'] for r in rows)} have truth; {len(both)} have a TTC in both")
    print(f"median |difference|: all {np.median(both):.2f} s   held-out images only {np.median(held):.2f} s ({len(held)} cars)")
    print(f"table of every car: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("earlier", nargs="?", help="earlier frame (image file)")
    ap.add_argument("later", nargs="?", help="later frame (image file); cars are detected on this one")
    ap.add_argument("--kitti", type=int, help="use KITTI training pair number 0..199 (comes with the true TTC)")
    ap.add_argument("--kitti_all", action="store_true", help="do all 200 KITTI training pairs, pictures go to output/kitti/")
    ap.add_argument("--nuscenes", type=int, help="use nuScenes front-camera frame number (comes with the true TTC)")
    ap.add_argument("--dt", type=float, default=0.1, help="seconds between the two frames (image files only; default 0.1)")
    ap.add_argument("--weights", default="fpttc_kitti", choices=["fpttc_kitti", "fpttc_mix"])
    ap.add_argument("--width", type=int, default=640, help="model input width, multiple of 32")
    ap.add_argument("--height", type=int, default=320, help="model input height, multiple of 32")
    ap.add_argument("--frac", type=float, default=BOX_FRAC, help="inner part of the box to read (default 0.7)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    a = ap.parse_args()

    model = C.load_model(a.weights)
    detector = VehicleDetector()
    if a.kitti_all:
        run_all_kitti(a, model, detector)
        return

    case = None
    if a.kitti is not None or a.nuscenes is not None:
        if a.kitti is not None and a.nuscenes is not None:
            raise SystemExit("use either --kitti or --nuscenes, not both")
        import truth
        case = truth.KittiCase(a.kitti) if a.kitti is not None else truth.NuScenesCase(a.nuscenes)
        prev, curr, dt, stem = case.prev_bgr, case.curr_bgr, case.dt, case.name
    else:
        if not (a.earlier and a.later):
            raise SystemExit("give two image files, or --kitti N, --kitti_all, or --nuscenes N")
        prev, curr = cv2.imread(a.earlier), cv2.imread(a.later)
        if prev is None or curr is None:
            raise SystemExit("could not read one of the images")
        dt, stem = a.dt, Path(a.later).stem
    if prev.shape != curr.shape:
        raise SystemExit(f"the two frames must be the same size, got {prev.shape} and {curr.shape}")
    process(prev, curr, dt, stem, case, model, detector, a, Path(a.out))


if __name__ == "__main__":
    main()
