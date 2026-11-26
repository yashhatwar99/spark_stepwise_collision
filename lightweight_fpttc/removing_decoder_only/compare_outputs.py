"""
The full FP-TTC and the light FP-TTC side by side on the same frames.

For each frame pair: the top row is the full model, the bottom row is the light model (new decoder).
  left      the later frame with each car's TTC (median of the TTC map inside the inner 70% of its YOLO box)
  middle    the per-pixel TTC map (from the model's scale map), grey = not approaching
  right     only when the true answer is known: "predicted | true" on each car
            (green = close, orange = off, red = far off)
For KITTI pairs the title also gives the Mid error of each model on that picture.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_decoder_only/compare_outputs.py FRAME_EARLIER FRAME_LATER --dt 0.1
    python lightweight_fpttc/removing_decoder_only/compare_outputs.py --kitti 25
    python lightweight_fpttc/removing_decoder_only/compare_outputs.py --kitti_all        # all 200 pairs -> output/kitti/ and all_cars.csv
    python lightweight_fpttc/removing_decoder_only/compare_outputs.py --nuscenes 262
Output goes to lightweight_fpttc/removing_decoder_only/output/.
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
sys.path.insert(0, str(HERE.parents[1] / "per_car_ttc" / "shared"))

import light_model as L  # noqa: E402
import ttc_utils as T  # noqa: E402
import panels  # noqa: E402

C = L.C
OUT_DIR = HERE / "output"
DT_KITTI = 0.1


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def car_list(dets, scale320, dt, size, frame_shape):
    ttc = T.ttc_map(scale320, dt)
    Hh, Ww = frame_shape[:2]
    fx, fy = size[0] / Ww, size[1] / Hh
    cars = []
    for d in dets:
        t, _, _ = T.car_ttc(ttc, (d.x1 * fx, d.y1 * fy, d.x2 * fx, d.y2 * fy), T.BOX_FRAC)
        cars.append(dict(class_name=d.class_name, confidence=round(float(d.confidence), 3),
                         box=[round(d.x1, 1), round(d.y1, 1), round(d.x2, 1), round(d.y2, 1)],
                         ttc_s=round(t, 2) if np.isfinite(t) else None))
    return cars


def run_both(prev_bgr, curr_bgr, dt, runner, detector):
    """Returns (full scale map, light scale map, cars with the full model's TTC, cars with the light model's TTC)."""
    scale_f, _, _ = runner.full(cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB))
    scale_l = runner.light_from_captured()
    dets = detector.detect(curr_bgr)
    return (scale_f, scale_l, car_list(dets, scale_f, dt, L.SIZE, curr_bgr.shape),
            car_list(dets, scale_l, dt, L.SIZE, curr_bgr.shape))


def figure(curr, scale_f, scale_l, cars_f, cars_l, dt, path, where, truth, mid=None):
    H, W = curr.shape[:2]
    cols = 3 if truth else 2
    fig, ax = plt.subplots(2, cols, figsize=(8.0 * cols, 2 * (7.6 * H / W) + 1.8))
    for r, (name, scale, cars) in enumerate((("full FP-TTC", scale_f, cars_f), ("light FP-TTC", scale_l, cars_l))):
        ax[r, 0].imshow(cv2.cvtColor(T.draw(curr, cars), cv2.COLOR_BGR2RGB))
        ax[r, 0].set_title(f"{name}: TTC per car")
        im = panels.ttc_panel(ax[r, 1], curr, scale, cars, dt)
        ax[r, 1].set_title(f"{name}: TTC per pixel")
        fig.colorbar(im, ax=ax[r, 1], fraction=0.03, pad=0.01, label="seconds")
        if truth:
            ax[r, 2].imshow(cv2.cvtColor(T.draw(curr, panels.together_labelled(cars), font=0.55, stagger=True), cv2.COLOR_BGR2RGB))
            ax[r, 2].set_title(f"{name}: predicted | true")
    for a in ax.ravel():
        a.axis("off")
    extra = f"   Mid error: full {mid[0]:.0f}, light {mid[1]:.0f} (lower is better)" if mid else ""
    fig.suptitle(f"{where}, dt = {dt:.2f} s{extra}\nlight = new decoder (Conv3); per-pixel maps: grey = not approaching; right column: green = close, orange = off, red = far off", fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


def mid_error(case, scale320):
    import eval_kitti as K
    gt, mask = case.g["gt"], case.g["mask"]
    h, w = gt.shape
    return K.log_err(cv2.resize(scale320, (w, h)), dict(gt=gt, mask=mask))[0] * 1e4


def process(prev, curr, dt, stem, case, runner, detector, out, extra_title="", print_table=True, kitti_index=None):
    scale_f, scale_l, cars_f, cars_l = run_both(prev, curr, dt, runner, detector)
    mid = None
    if case is not None:
        truths = case.truth([c["box"] for c in cars_f])
        panels.add_truth(cars_f, truths)
        panels.add_truth(cars_l, truths)
        if kitti_index is not None:
            mid = (mid_error(case, scale_f), mid_error(case, scale_l))
    out.mkdir(parents=True, exist_ok=True)
    pic = out / f"{stem}_full_vs_light.jpg"
    figure(curr, scale_f, scale_l, cars_f, cars_l, dt, str(pic), (case.where if case else stem) + extra_title, case is not None, mid)
    (out / f"{stem}_full_vs_light.json").write_text(json.dumps(dict(dt=dt, mid_error=mid, full=cars_f, light=cars_l), indent=1))
    if print_table:
        f = lambda v: "not closing" if v is None else f"{v:.2f}"
        print(f"{len(cars_f)} cars, dt = {dt:.3f} s" + (f"   Mid error: full {mid[0]:.0f}, light {mid[1]:.0f}" if mid else "") + "\n")
        if case is None:
            print("  full TTC    light TTC   class        box (x1, y1, x2, y2)")
            for a, b in zip(cars_f, cars_l):
                print(f"  {f(a['ttc_s']):>10s}  {f(b['ttc_s']):>10s}   {a['class_name']:<10s}  {a['box']}")
        else:
            print("  full TTC    light TTC   true TTC    full error  light error  class        box")
            for a, b in zip(cars_f, cars_l):
                tr = "no truth" if not a["has_truth"] else f(a["true_ttc_s"])
                ea = "" if a["error_s"] is None else f"{a['error_s']:+.2f}"
                eb = "" if b["error_s"] is None else f"{b['error_s']:+.2f}"
                print(f"  {f(a['ttc_s']):>10s}  {f(b['ttc_s']):>10s}  {tr:>10s}  {ea:>10s}  {eb:>11s}   {a['class_name']:<10s}  {a['box']}")
        print(f"\nsaved {pic}")
    return cars_f, cars_l, mid


def run_all_kitti(runner, detector, out_root):
    import truth
    import eval_kitti as K
    kt = K.Kitti()
    out = out_root / "kitti"
    rows, mids = [], []
    for i in range(200):
        case = truth.KittiCase(i, kt)
        txt = {"test": "held out", "quiz": "used to pick the light decoder's best round", "train": "the light decoder learned from this image"}[role(i)]
        cf, cl, mid = process(case.prev_bgr, case.curr_bgr, case.dt, case.name, case, runner, detector, out,
                              extra_title=f"  [{txt}]", print_table=False, kitti_index=i)
        mids.append(dict(image=i, role=role(i), mid_full=mid[0], mid_light=mid[1]))
        for a, b in zip(cf, cl):
            rows.append(dict(image=i, role=role(i), class_name=a["class_name"], confidence=a["confidence"], full_ttc_s=a["ttc_s"],
                             light_ttc_s=b["ttc_s"], has_truth=a["has_truth"], true_ttc_s=a["true_ttc_s"],
                             full_error_s=a["error_s"], light_error_s=b["error_s"], box=a["box"]))
        if i % 20 == 0:
            print(f"image {i}/200   {len(rows)} cars so far", flush=True)
    for name, data in (("all_cars.csv", rows), ("mid_error_per_image.csv", mids)):
        with open(out / name, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"\n200 pictures saved in {out}")
    for label, sel in (("all images", lambda r: True), ("held-out images only", lambda r: r["role"] == "test")):
        mf = np.mean([m["mid_full"] for m in mids if sel(m)])
        ml = np.mean([m["mid_light"] for m in mids if sel(m)])
        ef = [abs(r["full_error_s"]) for r in rows if sel(r) and r["full_error_s"] is not None]
        el = [abs(r["light_error_s"]) for r in rows if sel(r) and r["light_error_s"] is not None]
        print(f"{label}: Mid error full {mf:.1f}, light {ml:.1f}; median |TTC difference| full {np.median(ef):.2f} s, light {np.median(el):.2f} s ({len(ef)} cars)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("earlier", nargs="?")
    ap.add_argument("later", nargs="?")
    ap.add_argument("--kitti", type=int)
    ap.add_argument("--kitti_all", action="store_true")
    ap.add_argument("--nuscenes", type=int)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--out", default=str(OUT_DIR))
    a = ap.parse_args()

    if not L.WEIGHTS_FILE.exists():
        raise SystemExit(f"no trained light decoder at {L.WEIGHTS_FILE}; run train_light_decoder.py first")
    model = C.load_model("fpttc_kitti")
    dec, _ = L.load_decoder()
    runner = L.Runner(model, dec)
    detector = T.VehicleDetector()
    if a.kitti_all:
        run_all_kitti(runner, detector, Path(a.out))
        return

    case, extra, kidx = None, "", None
    if a.kitti is not None or a.nuscenes is not None:
        import truth
        case = truth.KittiCase(a.kitti) if a.kitti is not None else truth.NuScenesCase(a.nuscenes)
        prev, curr, dt, stem = case.prev_bgr, case.curr_bgr, case.dt, case.name
        if a.kitti is not None:
            kidx = a.kitti
            extra = "  [" + {"test": "held out", "quiz": "used to pick the light decoder's best round",
                             "train": "the light decoder learned from this image"}[role(a.kitti)] + "]"
    else:
        if not (a.earlier and a.later):
            raise SystemExit("give two image files, or --kitti N, --kitti_all, or --nuscenes N")
        prev, curr = cv2.imread(a.earlier), cv2.imread(a.later)
        if prev is None or curr is None:
            raise SystemExit("could not read one of the images")
        dt, stem = a.dt, Path(a.later).stem
    if prev.shape != curr.shape:
        raise SystemExit(f"the two frames must be the same size, got {prev.shape} and {curr.shape}")
    process(prev, curr, dt, stem, case, runner, detector, Path(a.out), extra_title=extra, kitti_index=kidx)


if __name__ == "__main__":
    main()
