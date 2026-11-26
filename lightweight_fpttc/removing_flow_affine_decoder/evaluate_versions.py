"""
The two versions of the light model without flow + affine, measured against the full model and the first light version.

  A  only flow + affine removed. The scale encoder's own scale_conv stays and makes the query from the image features.
  B  flow + affine AND scale_conv removed. The query is one fixed constant.

Both keep the scale encoder (released weights, frozen) and use Conv3 on its output. Each is measured twice:
  asis        Conv3 exactly as trained in removing_decoder_only
  retrained   Conv3 trained again on this version's own SF_t (the encoder stays frozen); the quiz images pick the best round

Measured: parameters, time (GPU, CPU), Mid error on KITTI (40 held-out images), per-car TTC on KITTI and nuScenes cars.
Writes results/versions_summary.json and results/versions_summary.md.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine_decoder/evaluate_versions.py
    python lightweight_fpttc/removing_flow_affine_decoder/evaluate_versions.py --skip_nuscenes --skip_cpu
Needs cache/kitti_features.pt (train_no_flow.py makes it) and the cache and weights of removing_decoder_only.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "removing_decoder_only"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402
import eval_kitti as K  # noqa: E402
import evaluate_light as E  # noqa: E402  (Mid error, timing and per-car code of the first light version)

C = L.C
RES = HERE / "results"
CACHE_FEATS = HERE / "cache" / "kitti_features.pt"
CACHE_PART1 = HERE.parent / "removing_decoder_only" / "cache" / "kitti_light.pt"
N = 200
DT = 0.1
BATCH, LR, WD = 4, 1e-4, 1e-4
VARIANTS = ["A_asis", "A_retrained", "B_asis", "B_retrained"]


def getf(F_, i, dev):
    return [x[i:i + 1].to(dev).float() for x in F_["f0"]], [x[i:i + 1].to(dev).float() for x in F_["f1"]]


@torch.no_grad()
def all_sf(head, F_, dev):
    head.eval()
    return [head.sf(*getf(F_, i, dev)).half().cpu() for i in range(N)]


def image_loss(conv3, sf, ft, d, i, dev):
    logs = conv3.log_scale(sf.to(dev).float(), ft.to(dev).float())
    gt, mask = d["gt"][i].to(dev), d["mask"][i].to(dev)
    big = F.interpolate(logs, size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
    return (big - torch.log(gt.float())).abs()[mask].mean()


def retrain_conv3(conv3, sfs, F_, d, dev, epochs=40, patience=8):
    """Train Conv3 again on cached SF_t (encoder frozen). Quiz images choose the best round."""
    role = E.role
    train = [i for i in range(N) if role(i) == "train"]
    quiz = [i for i in range(N) if role(i) == "quiz"]
    ft = F_["f1"][1]
    opt = torch.optim.AdamW(conv3.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def qloss():
        conv3.eval()
        return float(np.mean([float(image_loss(conv3, sfs[i], ft[i:i + 1], d, i, dev)) for i in quiz]))

    start = best = qloss()
    state = {k: v.clone() for k, v in conv3.state_dict().items()}
    bad, rounds = 0, 0
    for ep in range(epochs):
        conv3.train()
        order = rng.permutation(train)
        for b in range(0, len(order), BATCH):
            chunk = order[b:b + BATCH]
            loss = sum(image_loss(conv3, sfs[int(i)], ft[int(i):int(i) + 1], d, int(i), dev) for i in chunk) / len(chunk)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conv3.parameters(), 1.0)
            opt.step()
        sched.step()
        q = qloss()
        rounds = ep + 1
        if q < best - 1e-6:
            best, bad = q, 0
            state = {k: v.clone() for k, v in conv3.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 5 == 0:
            print(f"    round {ep + 1}  quiz error {q * 1e4:.1f}  (best {best * 1e4:.1f})", flush=True)
        if bad >= patience:
            break
    conv3.load_state_dict(state)
    conv3.eval()
    return dict(start_quiz_e4=start * 1e4, best_quiz_e4=best * 1e4, rounds=rounds)


@torch.no_grad()
def maps_from(conv3, sfs, F_, dev):
    ft = F_["f1"][1]
    return [conv3(sfs[i].to(dev).float(), ft[i:i + 1].to(dev).float())[0, 0].cpu().numpy() for i in range(N)]


def bench(device, rgb0, rgb1, n, warm, states):
    model = C.load_model("fpttc_kitti", device)
    heads = {v: M.build_head(model, v, states[v], device) for v in ("A", "B")}
    dec, _ = L.load_decoder(device=device)
    runner = L.Runner(model, dec)
    a, b = L.prepare(rgb0, rgb1, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)

    def run_head(h):
        m0, m1 = M.features(model, a, b)
        return h(m0, m1)

    jobs = [("full", lambda: runner.forward_full(a, b)), ("decoder_removed", lambda: runner.forward_light(a, b)),
            ("A", lambda: run_head(heads["A"])), ("B", lambda: run_head(heads["B"]))]
    out = {}
    for name, fn in jobs:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ms = E.median_ms(fn, sync, n, warm)
        out[name] = dict(ms=ms, fps=1000 / ms)
        if device == "cuda":
            out[name]["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {device} {name}: {ms:.0f} ms", flush=True)
    del model, heads, dec, runner
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_nuscenes", action="store_true")
    ap.add_argument("--skip_cpu", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    dev = "cuda"
    S = {}

    F_ = torch.load(CACHE_FEATS, weights_only=False)
    d = torch.load(CACHE_PART1, weights_only=False)
    sf_full = d.pop("sf")
    d.pop("ft")
    prev = json.loads((HERE.parent / "removing_decoder_only" / "results" / "summary.json").read_text())

    model = C.load_model("fpttc_kitti")
    heads = {v: M.build_head(model, v, None, dev) for v in ("A", "B")}
    S["parameters"] = {v: M.parameter_report(model, heads[v]) for v in heads}
    for v in heads:
        p = S["parameters"][v]
        print(f"version {v}: parameters {p['light']:,} (full {p['full']:,})", flush=True)

    # ---------------------------------------------------------- SF_t of each version, and how far it is from the full model's
    sfs = {v: all_sf(heads[v], F_, dev) for v in heads}
    S["sf_difference_percent"] = {}
    for v in heads:
        r = [100 * float((sfs[v][i].float() - sf_full[i:i + 1].float()).abs().mean() / sf_full[i:i + 1].float().abs().mean()) for i in range(N)]
        S["sf_difference_percent"][v] = float(np.mean(r))
        print(f"version {v}: SF_t differs from the full model's by {np.mean(r):.2f} % on average", flush=True)
    del sf_full

    # ---------------------------------------------------------- Conv3: as it is, and trained again
    conv3 = {}
    S["retraining"] = {}
    for v in heads:
        conv3[f"{v}_asis"] = heads[v].conv3
        c3 = L.LightDecoder().to(dev)
        c3.load_state_dict(heads[v].conv3.state_dict())
        print(f"retraining Conv3 for version {v} ...", flush=True)
        S["retraining"][v] = retrain_conv3(c3, sfs[v], F_, d, dev)
        conv3[f"{v}_retrained"] = c3
        print(f"  quiz error {S['retraining'][v]['start_quiz_e4']:.1f} -> {S['retraining'][v]['best_quiz_e4']:.1f}", flush=True)
    states = {v: {k: x.detach().clone() for k, x in conv3[f"{v}_retrained"].state_dict().items()} for v in heads}
    torch.save(states, HERE / "weights" / "versions_conv3_retrained.pt")

    # ---------------------------------------------------------- Mid error on KITTI
    maps = {k: maps_from(conv3[k], sfs[k[0]], F_, dev) for k in VARIANTS}
    full_maps = [d["scale_full"][i].numpy() for i in range(N)]
    me = {k: E.mid_errors(maps[k], d) for k in VARIANTS}
    S["mid_error"] = {}
    for name, sel in (("held_out_40", lambda r: r == "test"), ("other_160", lambda r: r != "test"), ("all_200", lambda r: True)):
        m = me["A_asis"].role.map(sel)
        S["mid_error"][name] = dict(full=prev["mid_error"][name]["full"], decoder_removed=prev["mid_error"][name]["light"])
        for k in VARIANTS:
            S["mid_error"][name][k] = float(me[k][m].mid.mean())
            S["mid_error"][name][k + "_fg"] = float(me[k][m].fg.mean())
            S["mid_error"][name][k + "_bg"] = float(me[k][m].bg.mean())
    print("Mid error held-out 40:", {k: round(v, 1) for k, v in S["mid_error"]["held_out_40"].items() if not k.endswith(("_fg", "_bg"))}, flush=True)

    # ---------------------------------------------------------- per-car TTC, KITTI cars
    rows = []
    for c in d["cars"]:
        i = c["image"]
        r = dict(image=i, role=E.role(i), s_true=c["s_true"], s_full=C.car_scale_median(full_maps[i], c["box"], 0.7)[0])
        for k in VARIANTS:
            r["s_" + k] = C.car_scale_median(maps[k][i], c["box"], 0.7)[0]
        rows.append(r)
    kc = pd.DataFrame(rows)
    kc.to_csv(RES / "kitti_cars_versions.csv", index=False)
    S["kitti_cars"] = {}
    for name, sel in (("held_out_40", kc.role == "test"), ("all_200", kc.role.notna())):
        S["kitti_cars"][name] = {m: E.car_stats(kc[sel], m, DT) for m in ["full"] + VARIANTS}

    # ---------------------------------------------------------- time
    kt = K.Kitti()
    rgb0, rgb1 = kt.image(25, 9), kt.image(25, 10)
    del heads, model
    torch.cuda.empty_cache()
    print("timing (GPU) ...", flush=True)
    S["time_gpu"] = bench("cuda", rgb0, rgb1, 30, 5, states)
    if not a.skip_cpu:
        print("timing (CPU, a few runs) ...", flush=True)
        S["time_cpu"] = bench("cpu", rgb0, rgb1, 3, 1, states)

    # ---------------------------------------------------------- per-car TTC, nuScenes cars
    if not a.skip_nuscenes:
        import eval_nuscenes as EN
        from src2 import config
        from src2.detection import VehicleDetector
        from src3.sweep_source import NuScenesSweepSource
        src = NuScenesSweepSource()
        nusc = src.nusc
        detector = VehicleDetector()
        model = C.load_model("fpttc_kitti")
        hs = {k: M.build_head(model, k[0], states[k[0]] if k.endswith("retrained") else None, dev) for k in VARIANTS}
        runner = L.Runner(model, None)
        rows, used = [], 0
        for k_, sample in enumerate(nusc.sample):
            tok = sample["data"][config.CAMERA_CHANNEL]
            prev_tok = nusc.get("sample_data", tok)["prev"]
            if not prev_tok:
                continue
            key, pv = src.frame(tok), src.frame(prev_tok)
            dt = key.timestamp_s - pv.timestamp_s
            if not (0.03 <= dt <= 0.30):
                continue
            used += 1
            Hh, Ww = key.image.shape[:2]
            cars = EN.true_cars(nusc, key, pv, dt, Ww, Hh)
            matches = []
            for dd in detector.detect(key.image):
                b = (dd.x1, dd.y1, dd.x2, dd.y2)
                best = max(cars, key=lambda c: EN.iou(c["box"], b), default=None)
                if best is not None and EN.iou(best["box"], b) >= 0.4:
                    matches.append((b, best))
            if not matches:
                continue
            r0, r1 = cv2.cvtColor(pv.image, cv2.COLOR_BGR2RGB), cv2.cvtColor(key.image, cv2.COLOR_BGR2RGB)
            scale_f, _, _ = runner.full(r0, r1)
            pa, pb = L.prepare(r0, r1, device=dev)
            m0, m1 = M.features(model, pa, pb)
            with torch.no_grad():
                smaps = {k: hs[k](m0, m1)[0, 0].cpu().numpy() for k in VARIANTS}
            fx, fy = L.SIZE[0] / Ww, L.SIZE[1] / Hh
            for b, c in matches:
                bx = (b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy)
                r = dict(keyframe=k_, dt=dt, z=c["z"], ttc_true=c["ttc_true"], s_true=c["s_true"],
                         s_full=C.car_scale_median(scale_f, bx, 0.7)[0])
                for k in VARIANTS:
                    r["s_" + k] = C.car_scale_median(smaps[k], bx, 0.7)[0]
                rows.append(r)
            if used % 40 == 0:
                print(f"keyframes {used}   cars {len(rows)}", flush=True)
        nc = pd.DataFrame(rows)
        nc.to_csv(RES / "nuscenes_cars_versions.csv", index=False)
        S["nuscenes_cars"] = {m: E.car_stats(nc, m, nc.dt.to_numpy(float), nc.ttc_true.to_numpy(float)) for m in ["full"] + VARIANTS}

    (RES / "versions_summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_md(S)
    print(f"\nsaved {RES / 'versions_summary.json'} and {RES / 'versions_summary.md'}")


def write_md(S):
    t = S["time_gpu"]
    lines = ["# Light FP-TTC without flow + affine: version A and version B", "",
             "- **A**: only flow + affine removed. The encoder's own `scale_conv` stays and makes the query from the image features.",
             "- **B**: flow + affine and `scale_conv` removed. The query is one fixed constant.",
             "- *asis*: Conv3 from `removing_decoder_only`, not trained again. *retrained*: Conv3 trained again on this version's SF_t (encoder frozen).", "",
             "## Size and speed", "",
             "| | Full FP-TTC | Decoder removed | A | B |", "|---|---|---|---|---|"]
    p = S["parameters"]
    lines.append(f"| Parameters | {p['A']['full'] / 1e6:.2f} M | 6.83 M | {p['A']['light'] / 1e6:.2f} M | {p['B']['light'] / 1e6:.2f} M |")
    lines.append(f"| Time per pair, GPU | {t['full']['ms']:.0f} ms | {t['decoder_removed']['ms']:.0f} ms | {t['A']['ms']:.0f} ms | {t['B']['ms']:.0f} ms |")
    if "time_cpu" in S:
        c = S["time_cpu"]
        lines.append(f"| Time per pair, CPU | {c['full']['ms'] / 1000:.1f} s | {c['decoder_removed']['ms'] / 1000:.1f} s | {c['A']['ms'] / 1000:.1f} s | {c['B']['ms'] / 1000:.1f} s |")
    lines.append(f"| Peak GPU memory | {t['full']['peak_gpu_gb']:.2f} GB | {t['decoder_removed']['peak_gpu_gb']:.2f} GB | {t['A']['peak_gpu_gb']:.2f} GB | {t['B']['peak_gpu_gb']:.2f} GB |")
    lines += ["", f"SF_t difference from the full model: A {S['sf_difference_percent']['A']:.2f} %, B {S['sf_difference_percent']['B']:.2f} %.", "",
              "## Accuracy (Mid error, lower is better)", "",
              "| | Full | Decoder removed | A asis | A retrained | B asis | B retrained |", "|---|---|---|---|---|---|---|"]
    for name, label in (("held_out_40", "KITTI held-out 40"), ("other_160", "KITTI other 160"), ("all_200", "KITTI all 200")):
        m = S["mid_error"][name]
        lines.append(f"| {label} | {m['full']:.1f} | {m['decoder_removed']:.1f} | " + " | ".join(f"{m[k]:.1f}" for k in VARIANTS) + " |")
    ks = ["full"] + VARIANTS
    lines += ["", "## Per-car TTC (median method)", "", "| | Full | A asis | A retrained | B asis | B retrained |", "|---|---|---|---|---|---|"]
    srcs = [("KITTI cars, held-out 40", S["kitti_cars"]["held_out_40"]), ("KITTI cars, all 200", S["kitti_cars"]["all_200"])]
    if "nuscenes_cars" in S:
        srcs.append(("nuScenes cars", S["nuscenes_cars"]))
    for label, g in srcs:
        lines.append(f"| {label}: scale error | " + " | ".join(f"{g[k]['scale_err_pct']:.2f}%" for k in ks) + " |")
        lines.append(f"| {label}: TTC error (middle) | " + " | ".join(f"{g[k]['ttc_med']:.2f} s" for k in ks) + " |")
        lines.append(f"| {label}: cars under 3 s found | " + " | ".join(f"{100 * g[k]['found']:.0f}%" for k in ks) + " |")
        lines.append(f"| {label}: precision | " + " | ".join(f"{100 * g[k]['precision']:.0f}%" for k in ks) + " |")
    (RES / "versions_summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
