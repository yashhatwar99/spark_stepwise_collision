"""
Output pictures on the KITTI frames for the decoder-removed model taught by distillation.

Rows: the full FP-TTC, the decoder-removed model with Conv3 trained on KITTI truth (removing_decoder_only), and the decoder-removed
model with Conv3 distilled (this folder). Columns: TTC per car, TTC per pixel, predicted | true (green = close, orange = off,
red = far off). The title gives the Mid error of each on that picture.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_decoder_only/distillation/compare_outputs_distill.py          # the 40 held-out KITTI images
    python lightweight_fpttc/removing_decoder_only/distillation/compare_outputs_distill.py --all    # all 200
Output: lightweight_fpttc/removing_decoder_only/distillation/output/kitti/ (pictures, Mid error CSV, per-car CSV).
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LW = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(LW / "removing_flow_affine_decoder"))
sys.path.insert(0, str(LW.parent / "per_car_ttc" / "shared"))

import light_model as L  # noqa: E402
import compare_outputs as CO  # noqa: E402
import compare_outputs_versions as CV  # noqa: E402  (the three-row figure)
import ttc_utils as T  # noqa: E402
import panels  # noqa: E402

C = L.C
OUT = HERE / "output" / "kitti"
NAMES = ["full FP-TTC", "decoder removed, Conv3 trained on KITTI truth", "decoder removed, Conv3 distilled"]


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--kitti", type=int)
    a = ap.parse_args()
    import truth
    import eval_kitti as K

    dev = "cuda"
    model = C.load_model("fpttc_kitti")
    dec, _ = L.load_decoder(device=dev)
    ck = torch.load(HERE / "weights" / "conv3.pt", map_location=dev)
    dec_d = L.LightDecoder().to(dev)
    dec_d.load_state_dict(ck["state"])
    dec_d.eval()
    runner = L.Runner(model, dec)
    detector = T.VehicleDetector()
    kt = K.Kitti()
    idx = [a.kitti] if a.kitti is not None else (list(range(200)) if a.all else [i for i in range(200) if role(i) == "test"])
    OUT.mkdir(parents=True, exist_ok=True)
    mids_rows, car_rows = [], []
    for n, i in enumerate(idx):
        case = truth.KittiCase(i, kt)
        prev, curr, dt = case.prev_bgr, case.curr_bgr, case.dt
        scale_f, sf, ft = runner.full(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), cv2.cvtColor(curr, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            maps = [scale_f, runner.light_from_captured(), dec_d(sf[None], ft[None])[0, 0].cpu().numpy()]
        dets = detector.detect(curr)
        cars = [CO.car_list(dets, m, dt, L.SIZE, curr.shape) for m in maps]
        truths = case.truth([c["box"] for c in cars[0]])
        for cs in cars:
            panels.add_truth(cs, truths)
        mids = [CO.mid_error(case, m) for m in maps]
        txt = {"test": "held out", "quiz": "used to pick the best round", "train": "Conv3 learned from this image"}[role(i)]
        CV.figure(curr, maps, cars, dt, str(OUT / f"{case.name}_compare.jpg"), f"{case.where}  [{txt}]", mids, NAMES, note="decoder replaced by Conv3, flow + affine kept")
        mids_rows.append(dict(image=i, role=role(i), mid_full=mids[0], mid_decoder_removed_direct=mids[1], mid_decoder_removed_distilled=mids[2]))
        for k in range(len(cars[0])):
            car_rows.append(dict(image=i, role=role(i), class_name=cars[0][k]["class_name"], box=cars[0][k]["box"],
                                 true_ttc_s=cars[0][k]["true_ttc_s"], full_ttc_s=cars[0][k]["ttc_s"],
                                 direct_ttc_s=cars[1][k]["ttc_s"], distilled_ttc_s=cars[2][k]["ttc_s"]))
        if n % 10 == 0:
            print(f"image {n}/{len(idx)}", flush=True)
    for name, data in (("mid_error_per_image.csv", mids_rows), ("all_cars.csv", car_rows)):
        with open(OUT / name, "w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    print(f"{len(idx)} pictures saved in {OUT}")
    print("mean Mid error:", {k: round(float(np.mean([m[k] for m in mids_rows])), 1) for k in mids_rows[0] if k.startswith("mid_")})


if __name__ == "__main__":
    main()
