"""
Output pictures on the KITTI frames: full FP-TTC, version A and version B (flow + affine removed), one row each.

  left      the later frame with each car's TTC (median of the TTC map inside the inner 70% of its YOLO box)
  middle    the per-pixel TTC map (grey = not approaching)
  right     "predicted | true" on each car (green = close, orange = off, red = far off)
The title gives the Mid error of each model on that picture.
A and B use the Conv3 retrained on KITTI (weights/versions_conv3_retrained.pt, made by evaluate_versions.py).

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine_decoder/compare_outputs_versions.py            # the 40 held-out KITTI images
    python lightweight_fpttc/removing_flow_affine_decoder/compare_outputs_versions.py --all      # all 200
    python lightweight_fpttc/removing_flow_affine_decoder/compare_outputs_versions.py --kitti 25
Output goes to lightweight_fpttc/removing_flow_affine_decoder/output/kitti/ (pictures, a Mid error CSV and a per-car CSV).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "removing_decoder_only"))
sys.path.insert(0, str(HERE.parents[1] / "per_car_ttc" / "shared"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402
import compare_outputs as CO  # noqa: E402  (car list and Mid error code of the first light version)
import ttc_utils as T  # noqa: E402
import panels  # noqa: E402

C = L.C
OUT = HERE / "output" / "kitti"
NAMES = ["full FP-TTC", "version A (scale_conv kept)", "version B (scale_conv removed)"]


def figure(curr, maps, cars, dt, path, where, mids, names=None, note=None):
    names = names or NAMES
    n = len(maps)
    H, W = curr.shape[:2]
    fig, ax = plt.subplots(n, 3, figsize=(24, n * (7.6 * H / W) + 2.2))
    for r in range(n):
        ax[r, 0].imshow(cv2.cvtColor(T.draw(curr, cars[r]), cv2.COLOR_BGR2RGB))
        ax[r, 0].set_title(f"{names[r]}: TTC per car")
        im = panels.ttc_panel(ax[r, 1], curr, maps[r], cars[r], dt)
        ax[r, 1].set_title(f"{names[r]}: TTC per pixel")
        fig.colorbar(im, ax=ax[r, 1], fraction=0.03, pad=0.01, label="seconds")
        ax[r, 2].imshow(cv2.cvtColor(T.draw(curr, panels.together_labelled(cars[r]), font=0.55, stagger=True), cv2.COLOR_BGR2RGB))
        ax[r, 2].set_title(f"{names[r]}: predicted | true")
    for a in ax.ravel():
        a.axis("off")
    fig.suptitle(f"{where}, dt = {dt:.2f} s   Mid error: full {mids[0]:.0f}, A {mids[1]:.0f}, B {mids[2]:.0f} (lower is better)\n"
                 "A and B = flow + affine removed, Conv3 retrained on KITTI; per-pixel maps: grey = not approaching; right column: green = close, orange = off, red = far off",
                 fontsize=10)
    fig.savefig(path, dpi=80, bbox_inches="tight", pil_kwargs={"quality": 88})
    plt.close(fig)


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kitti", type=int)
    ap.add_argument("--conv3", help="folder name (distillation_only, distill_then_kitti): rows = full, A trained directly on KITTI, A with that folder's Conv3")
    a = ap.parse_args()
    import truth
    import eval_kitti as K

    dev = "cuda"
    model = C.load_model("fpttc_kitti")
    states = torch.load(HERE / "weights" / "versions_conv3_retrained.pt", map_location=dev)
    out_dir, names = OUT, NAMES
    if a.conv3:
        ck = torch.load(HERE / a.conv3 / "weights" / "conv3.pt", map_location=dev)
        heads = [M.build_head(model, "A", states["A"], dev), M.build_head(model, "A", ck["state"], dev)]
        names = ["full FP-TTC", "version A, trained directly on KITTI truth", f"version A, {a.conv3.replace('_', ' ')}"]
        out_dir = HERE / a.conv3 / "output" / "kitti"
    else:
        heads = [M.build_head(model, v, states[v], dev) for v in ("A", "B")]      # before the Runner, which adds a hook
    runner = L.Runner(model, None)
    detector = T.VehicleDetector()
    kt = K.Kitti()
    idx = [a.kitti] if a.kitti is not None else (list(range(200)) if a.all else [i for i in range(200) if role(i) == "test"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mids_rows, car_rows = [], []
    for n, i in enumerate(idx):
        case = truth.KittiCase(i, kt)
        prev, curr, dt = case.prev_bgr, case.curr_bgr, case.dt
        r0, r1 = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), cv2.cvtColor(curr, cv2.COLOR_BGR2RGB)
        scale_f, _, _ = runner.full(r0, r1)
        pa, pb = L.prepare(r0, r1, device=dev)
        with torch.no_grad():
            m0, m1 = M.features(model, pa, pb)
            maps = [scale_f] + [h(m0, m1)[0, 0].cpu().numpy() for h in heads]
        dets = detector.detect(curr)
        cars = [CO.car_list(dets, m, dt, L.SIZE, curr.shape) for m in maps]
        truths = case.truth([c["box"] for c in cars[0]])
        for cs in cars:
            panels.add_truth(cs, truths)
        mids = [CO.mid_error(case, m) for m in maps]
        txt = {"test": "held out", "quiz": "used to pick the best round", "train": "Conv3 learned from this image"}[role(i)]
        figure(curr, maps, cars, dt, str(out_dir / f"{case.name}_{'compare' if a.conv3 else 'full_vs_A_vs_B'}.jpg"), f"{case.where}  [{txt}]", mids, names)
        mids_rows.append(dict(image=i, role=role(i), mid_full=mids[0], mid_A=mids[1], mid_B=mids[2]) if not a.conv3
                         else dict(image=i, role=role(i), mid_full=mids[0], mid_A_direct=mids[1], mid_this=mids[2]))
        for k in range(len(cars[0])):
            car_rows.append(dict(image=i, role=role(i), class_name=cars[0][k]["class_name"], box=cars[0][k]["box"],
                                 true_ttc_s=cars[0][k]["true_ttc_s"], full_ttc_s=cars[0][k]["ttc_s"],
                                 **({"A_direct_ttc_s": cars[1][k]["ttc_s"], "this_ttc_s": cars[2][k]["ttc_s"]} if a.conv3 else
                                    {"A_ttc_s": cars[1][k]["ttc_s"], "B_ttc_s": cars[2][k]["ttc_s"]})))
        if n % 10 == 0:
            print(f"image {n}/{len(idx)}", flush=True)
    for name, data in (("mid_error_per_image.csv", mids_rows), ("all_cars.csv", car_rows)):
        with open(out_dir / name, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"{len(idx)} pictures saved in {out_dir}")
    print("mean Mid error:", {k: round(float(np.mean([m[k] for m in mids_rows])), 1) for k in mids_rows[0] if k.startswith("mid_")})


if __name__ == "__main__":
    main()
