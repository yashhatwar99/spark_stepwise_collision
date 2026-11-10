"""
Per-car time to collision, Method 2: a small learned head on FP-TTC's scale
encoder output (SF_t). The decoder's output is not used for the per-car number.

Two frames from the same camera go into FP-TTC. A forward hook copies SF_t.
YOLO draws a box on each car of the later frame; the box is cut out of SF_t
(RoIAlign, 7 x 7 x 128) and the trained head turns it into one scale s. Then
TTC = dt / (1 - s), the paper's Eq. 1 (s >= 1 means not getting closer).

Train the head first:  python per_car_ttc/learned_head/train_head.py

Usage (from the project root, with the project .venv):
    # any two frames from the same camera
    python per_car_ttc/learned_head/learned_head_ttc.py FRAME_EARLIER FRAME_LATER --dt 0.1

    # a case that has the true answer: the output also shows the true TTC and
    # "predicted | true" on every car
    python per_car_ttc/learned_head/learned_head_ttc.py --kitti 25
    python per_car_ttc/learned_head/learned_head_ttc.py --kitti_all       # all 200 pairs -> output/kitti/ and all_cars.csv
    python per_car_ttc/learned_head/learned_head_ttc.py --nuscenes 262

The head was trained for the fpttc_kitti weights at 640x320 on frames 0.1 s
apart, so those are the only settings offered. In the pictures the per-pixel
map is the model's OWN TTC map (from its decoder), for reference, because the
head gives a number per car, not a map. For KITTI pairs the picture says
whether the head learned from that image: it learned from the images whose
number is not divisible by 5 and does not end in 1; the ones divisible by 5 are
the held-out test images.
Output goes to per_car_ttc/learned_head/output/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "shared"))

import head_model as H  # noqa: E402
import ttc_utils as T  # noqa: E402
import panels  # noqa: E402

C = H.C
OUT_DIR = HERE / "output"


def head_role(i):
    """What the head did with KITTI image i in training. Same rule as train_head.role()."""
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def ttc_from_s(s, dt):
    return float(dt / (1.0 - s)) if s < 1.0 else None


def run(prev_bgr, curr_bgr, dt, reader, head, detector):
    """Returns (scale map from the decoder, cars). Each car: class_name, confidence,
    box (later frame's pixels), s, ttc_s (None = not approaching), median_ttc_s (Method 1 on the
    same box, for comparison)."""
    rgb0 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB)
    rgb1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
    scale, feat = reader(rgb0, rgb1, H.SIZE)
    ttc_pix = T.ttc_map(scale, dt)
    Hh, Ww = curr_bgr.shape[:2]
    fx, fy = H.SIZE[0] / Ww, H.SIZE[1] / Hh
    dets = detector.detect(curr_bgr)
    boxes = [(d.x1 * fx, d.y1 * fy, d.x2 * fx, d.y2 * fy) for d in dets]
    s_head = H.predict_s(head, feat, boxes) if boxes else np.zeros(0)
    cars = []
    for d, bx, s in zip(dets, boxes, s_head):
        t_med, _, _ = T.car_ttc(ttc_pix, bx, T.BOX_FRAC)
        t = ttc_from_s(float(s), dt)
        cars.append(dict(class_name=d.class_name, confidence=round(float(d.confidence), 3),
                         box=[round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
                         s=round(float(s), 4), ttc_s=round(t, 2) if t is not None else None,
                         median_ttc_s=round(t_med, 2) if np.isfinite(t_med) else None))
    cars.sort(key=lambda c: (c["ttc_s"] is None, c["ttc_s"] if c["ttc_s"] is not None else 0.0))
    return scale, cars


def side_by_side(curr_bgr, scale, cars, dt, path):
    """No truth available. Left: the later frame with each car's TTC from the head. Right: the model's own per-pixel TTC map."""
    Hh, Ww = curr_bgr.shape[:2]
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.0 * Hh / Ww + 0.7))
    ax[0].imshow(cv2.cvtColor(T.draw(curr_bgr, cars), cv2.COLOR_BGR2RGB))
    ax[0].set_title("later frame: TTC per car from the learned head (scale encoder output)")
    im = panels.ttc_panel(ax[1], curr_bgr, scale, cars, dt)
    ax[1].set_title("for reference: the model's own TTC per pixel (grey = not approaching)")
    for a in ax:
        a.axis("off")
    fig.colorbar(im, ax=ax[1], fraction=0.025, pad=0.01, label="seconds")
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)


def add_median_error(cars):
    for c in cars:
        c["median_error_s"] = None
        if c.get("has_truth") and c["median_ttc_s"] is not None and c["true_ttc_s"] is not None:
            c["median_error_s"] = round(c["median_ttc_s"] - c["true_ttc_s"], 2)


def process(prev, curr, dt, stem, case, reader, head, detector, out, extra_title="", print_table=True):
    """Run the method on one frame pair, save the picture and the JSON, optionally print the table. Returns the cars."""
    scale, cars = run(prev, curr, dt, reader, head, detector)
    if case is not None:
        panels.add_truth(cars, case.truth([c["box"] for c in cars]))
        add_median_error(cars)
    out.mkdir(parents=True, exist_ok=True)
    pic = out / f"{stem}_ttc.jpg"
    if case is not None:
        titles = dict(a="learned head (Method 2): TTC per car", b="true TTC per car",
                      c="the model's own TTC per pixel (reference); labels = learned head",
                      d="predicted | true  (green = close, orange = off, red = far off)")
        panels.four_panels(curr, scale, cars, dt, str(pic), case.where + extra_title, titles)
    else:
        side_by_side(curr, scale, cars, dt, str(pic))
    (out / f"{stem}_ttc.json").write_text(json.dumps(dict(dt=dt, cars=cars), indent=1))
    if not print_table:
        return cars

    print(f"{len(cars)} cars, dt = {dt:.3f} s\n")
    if case is None:
        print("  head TTC (s)   Method 1 (median)   class        conf   box (x1, y1, x2, y2)")
        for c in cars:
            t = "not closing" if c["ttc_s"] is None else f"{c['ttc_s']:.2f}"
            m = "not closing" if c["median_ttc_s"] is None else f"{c['median_ttc_s']:.2f}"
            print(f"  {t:>12s}   {m:>17s}   {c['class_name']:<10s}  {c['confidence']:.2f}   {c['box']}")
    else:
        print("  head TTC   median TTC   true TTC    head error   median error   class        box (x1, y1, x2, y2)")
        dh, dm = [], []
        for c in cars:
            f = lambda v: "not closing" if v is None else f"{v:.2f}"
            tr = "no truth" if not c["has_truth"] else f(c["true_ttc_s"])
            eh = "" if c["error_s"] is None else f"{c['error_s']:+.2f}"
            em = "" if c["median_error_s"] is None else f"{c['median_error_s']:+.2f}"
            if c["error_s"] is not None:
                dh.append(abs(c["error_s"]))
            if c["median_error_s"] is not None:
                dm.append(abs(c["median_error_s"]))
            print(f"  {f(c['ttc_s']):>11s}  {f(c['median_ttc_s']):>11s}  {tr:>10s}  {eh:>10s}  {em:>12s}   {c['class_name']:<10s}  {c['box']}")
        if dh:
            print(f"\ncars with a TTC in both: {len(dh)}; median |difference|: head {np.median(dh):.2f} s, median method {np.median(dm):.2f} s")
    print(f"\nsaved {pic}\n      {pic.with_suffix('.json')}")
    return cars


def run_all_kitti(reader, head, detector, out_root):
    """The four-panel picture for every KITTI training pair (0..199), into output/kitti/, plus one table of all cars."""
    import truth
    import eval_kitti as K
    kt = K.Kitti()
    out = out_root / "kitti"
    rows = []
    for i in range(200):
        case = truth.KittiCase(i, kt)
        role = head_role(i)
        txt = {"test": "held out: the head never saw this image", "quiz": "the head used this image to pick its best round",
               "train": "the head learned from this image"}[role]
        cars = process(case.prev_bgr, case.curr_bgr, case.dt, case.name, case, reader, head, detector, out,
                       extra_title=f"  [{txt}]", print_table=False)
        for c in cars:
            rows.append(dict(image=i, head_role=role, class_name=c["class_name"], confidence=c["confidence"],
                             head_ttc_s=c["ttc_s"], median_ttc_s=c["median_ttc_s"], has_truth=c["has_truth"],
                             true_ttc_s=c["true_ttc_s"], head_error_s=c["error_s"], median_error_s=c["median_error_s"],
                             box=c["box"]))
        if i % 20 == 0:
            print(f"image {i}/200   {len(rows)} cars so far", flush=True)
    path = out / "all_cars.csv"
    with open(path, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n200 pictures saved in {out}")
    print(f"{len(rows)} detected cars; {sum(r['has_truth'] for r in rows)} have truth")
    for label, sel in [("all images", lambda r: True), ("held-out images only (the head never saw them)", lambda r: r["head_role"] == "test")]:
        h = [abs(r["head_error_s"]) for r in rows if sel(r) and r["head_error_s"] is not None]
        m = [abs(r["median_error_s"]) for r in rows if sel(r) and r["median_error_s"] is not None]
        print(f"median |difference| from the true TTC, {label}: head {np.median(h):.2f} s, median method {np.median(m):.2f} s ({len(h)} cars)")
    print(f"table of every car: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("earlier", nargs="?", help="earlier frame (image file)")
    ap.add_argument("later", nargs="?", help="later frame (image file); cars are detected on this one")
    ap.add_argument("--kitti", type=int, help="use KITTI training pair number 0..199 (comes with the true TTC)")
    ap.add_argument("--kitti_all", action="store_true", help="do all 200 KITTI training pairs, pictures go to output/kitti/")
    ap.add_argument("--nuscenes", type=int, help="use nuScenes front-camera frame number (comes with the true TTC)")
    ap.add_argument("--dt", type=float, default=0.1, help="seconds between the two frames (image files only; default 0.1)")
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    a = ap.parse_args()

    if not H.WEIGHTS_FILE.exists():
        raise SystemExit(f"no trained head at {H.WEIGHTS_FILE}; run train_head.py first")
    model = C.load_model("fpttc_kitti")
    reader = H.FeatureReader(model)
    head, _ = H.load_head()
    detector = T.VehicleDetector()
    if a.kitti_all:
        run_all_kitti(reader, head, detector, Path(a.out))
        return

    case, extra = None, ""
    if a.kitti is not None or a.nuscenes is not None:
        if a.kitti is not None and a.nuscenes is not None:
            raise SystemExit("use either --kitti or --nuscenes, not both")
        import truth
        case = truth.KittiCase(a.kitti) if a.kitti is not None else truth.NuScenesCase(a.nuscenes)
        prev, curr, dt, stem = case.prev_bgr, case.curr_bgr, case.dt, case.name
        if a.kitti is not None:
            extra = "  [" + {"test": "held out: the head never saw this image",
                             "quiz": "the head used this image to pick its best round",
                             "train": "the head learned from this image"}[head_role(a.kitti)] + "]"
    else:
        if not (a.earlier and a.later):
            raise SystemExit("give two image files, or --kitti N, --kitti_all, or --nuscenes N")
        prev, curr = cv2.imread(a.earlier), cv2.imread(a.later)
        if prev is None or curr is None:
            raise SystemExit("could not read one of the images")
        dt, stem = a.dt, Path(a.later).stem
    if prev.shape != curr.shape:
        raise SystemExit(f"the two frames must be the same size, got {prev.shape} and {curr.shape}")
    process(prev, curr, dt, stem, case, reader, head, detector, Path(a.out), extra_title=extra)


if __name__ == "__main__":
    main()
