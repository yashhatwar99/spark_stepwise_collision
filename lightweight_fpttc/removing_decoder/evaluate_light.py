"""
Full FP-TTC against the light FP-TTC (new decoder): size, speed, memory and accuracy.

  parameters        counted per part of the model
  time and memory   one KITTI pair at 640x320, GPU (and CPU), the model alone, median of many runs
  accuracy          Mid error (the paper's metric) on KITTI: the 40 held-out images, the other 160, all 200
  per-car TTC       the median method on both scale maps, KITTI cars (from the same cache) and nuScenes cars

Writes lightweight_fpttc/removing_decoder/results/summary.json and summary.md.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_decoder/evaluate_light.py
    python lightweight_fpttc/removing_decoder/evaluate_light.py --skip_nuscenes
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import light_model as L  # noqa: E402
import eval_kitti as K  # noqa: E402
import build_summary as BS  # noqa: E402

C = L.C
CACHE = HERE / "cache" / "kitti_light.pt"
RES = HERE / "results"
DT = 0.1


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def median_ms(fn, sync, n, warm):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts))


def bench(device, rgb0, rgb1, n, warm):
    model = C.load_model("fpttc_kitti", device)
    dec, _ = L.load_decoder(device=device)
    runner = L.Runner(model, dec)
    a, b = L.prepare(rgb0, rgb1, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    out = {}
    for name, fn in (("full", lambda: runner.forward_full(a, b)), ("light", lambda: runner.forward_light(a, b))):
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ms = median_ms(fn, sync, n, warm)
        out[name] = dict(ms=ms, fps=1000 / ms)
        if device == "cuda":
            out[name]["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 1e9
    del model, dec, runner
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def mid_errors(scale_maps, d):
    """Mid error (x1e4) per image: the paper's metric, at KITTI's picture size, same as eval_kitti.py."""
    rows = []
    for i in range(len(scale_maps)):
        gt, mask, fg = d["gt"][i].float().numpy(), d["mask"][i].numpy(), d["fg"][i].numpy()
        h, w = gt.shape
        g = dict(gt=gt, mask=mask)
        pred = cv2.resize(scale_maps[i], (w, h))
        rows.append(dict(image=i, role=role(i), mid=K.log_err(pred, g)[0] * 1e4,
                         fg=K.log_err(pred, g, fg)[0] * 1e4, bg=K.log_err(pred, g, ~fg)[0] * 1e4))
    return pd.DataFrame(rows)


def car_stats(d_, method, dt, ttc_true=None):
    m = BS.car_metrics(d_, method, dt, ttc_true=ttc_true)
    return dict(cars=m["cars"], scale_err_pct=m["scale_error_percent_median"], ttc_med=m["ttc_abs_error_s"]["median"],
                found=m["danger"]["recall"], precision=m["danger"]["precision"], ttc_cars=m["ttc_cars"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip_nuscenes", action="store_true")
    ap.add_argument("--skip_cpu", action="store_true")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    dev = "cuda"
    S = {}

    model = C.load_model("fpttc_kitti")
    dec, ck = L.load_decoder()
    runner = L.Runner(model, dec)
    S["parameters"] = L.parameter_report(model, dec)
    p = S["parameters"]
    print(f"parameters: full {p['full']:,}   light {p['light']:,}   (removed {p['removed']:,}, Conv3 {p['conv3']:,})", flush=True)

    # ------------------------------------------------------------------ accuracy on KITTI (from the cache)
    d = torch.load(CACHE, weights_only=False)
    n_img = len(d["gt"])
    full_maps = [d["scale_full"][i].numpy() for i in range(n_img)]
    with torch.no_grad():
        light_maps = [dec(d["sf"][i:i + 1].to(dev), d["ft"][i:i + 1].to(dev))[0, 0].cpu().numpy() for i in range(n_img)]
    mf, ml = mid_errors(full_maps, d), mid_errors(light_maps, d)
    S["mid_error"] = {}
    for name, sel in (("held_out_40", lambda r: r == "test"), ("other_160", lambda r: r != "test"), ("all_200", lambda r: True)):
        m = mf.role.map(sel)
        S["mid_error"][name] = dict(full=float(mf[m].mid.mean()), light=float(ml[m].mid.mean()),
                                    full_fg=float(mf[m].fg.mean()), light_fg=float(ml[m].fg.mean()),
                                    full_bg=float(mf[m].bg.mean()), light_bg=float(ml[m].bg.mean()), images=int(m.sum()))
    print("Mid error (x1e4), held-out 40:", {k: round(v, 1) for k, v in S["mid_error"]["held_out_40"].items() if k in ("full", "light")}, flush=True)

    # ------------------------------------------------------------------ per-car TTC on KITTI cars
    rows = []
    for c in d["cars"]:
        i = c["image"]
        rows.append(dict(image=i, role=role(i), s_true=c["s_true"],
                         s_full=C.car_scale_median(full_maps[i], c["box"], 0.7)[0],
                         s_light=C.car_scale_median(light_maps[i], c["box"], 0.7)[0]))
    kc = pd.DataFrame(rows)
    S["kitti_cars"] = {}
    for name, sel in (("held_out_40", kc.role == "test"), ("all_200", kc.role.notna())):
        S["kitti_cars"][name] = {m: car_stats(kc[sel], m, DT) for m in ("full", "light")}
    kc.to_csv(RES / "kitti_cars_full_vs_light.csv", index=False)

    # ------------------------------------------------------------------ time and memory
    kt = K.Kitti()
    rgb0, rgb1 = kt.image(25, 9), kt.image(25, 10)
    del runner, model
    torch.cuda.empty_cache()
    print("timing the models (GPU) ...", flush=True)
    S["time_gpu"] = bench("cuda", rgb0, rgb1, n=30, warm=5)
    if not a.skip_cpu:
        print("timing the models (CPU, a few runs) ...", flush=True)
        S["time_cpu"] = bench("cpu", rgb0, rgb1, n=3, warm=1)

    # ------------------------------------------------------------------ per-car TTC on nuScenes
    if not a.skip_nuscenes:
        import eval_nuscenes as EN
        from src2 import config
        from src2.detection import VehicleDetector
        from src3.sweep_source import NuScenesSweepSource
        src = NuScenesSweepSource()
        nusc = src.nusc
        detector = VehicleDetector()
        model = C.load_model("fpttc_kitti")
        runner = L.Runner(model, dec)
        rows, used = [], 0
        for k, sample in enumerate(nusc.sample):
            tok = sample["data"][config.CAMERA_CHANNEL]
            prev_tok = nusc.get("sample_data", tok)["prev"]
            if not prev_tok:
                continue
            key, prev = src.frame(tok), src.frame(prev_tok)
            dt = key.timestamp_s - prev.timestamp_s
            if not (0.03 <= dt <= 0.30):
                continue
            used += 1
            Hh, Ww = key.image.shape[:2]
            cars = EN.true_cars(nusc, key, prev, dt, Ww, Hh)
            matches = []
            for dd in detector.detect(key.image):
                b = (dd.x1, dd.y1, dd.x2, dd.y2)
                best = max(cars, key=lambda c: EN.iou(c["box"], b), default=None)
                if best is not None and EN.iou(best["box"], b) >= 0.4:
                    matches.append((b, best))
            if not matches:
                continue
            scale_f, _, _ = runner.full(cv2.cvtColor(prev.image, cv2.COLOR_BGR2RGB), cv2.cvtColor(key.image, cv2.COLOR_BGR2RGB))
            scale_l = runner.light_from_captured()
            fx, fy = L.SIZE[0] / Ww, L.SIZE[1] / Hh
            for b, c in matches:
                bx = (b[0] * fx, b[1] * fy, b[2] * fx, b[3] * fy)
                rows.append(dict(keyframe=k, dt=dt, z=c["z"], ttc_true=c["ttc_true"], s_true=c["s_true"],
                                 s_full=C.car_scale_median(scale_f, bx, 0.7)[0], s_light=C.car_scale_median(scale_l, bx, 0.7)[0]))
            if used % 40 == 0:
                print(f"keyframes {used}   cars {len(rows)}", flush=True)
        nc = pd.DataFrame(rows)
        nc.to_csv(RES / "nuscenes_cars_full_vs_light.csv", index=False)
        S["nuscenes_cars"] = {m: car_stats(nc, m, nc.dt.to_numpy(float), nc.ttc_true.to_numpy(float)) for m in ("full", "light")}

    S["decoder_training"] = ck["info"]
    (RES / "summary.json").write_text(json.dumps(S, indent=1, default=float))
    write_md(S)
    print(f"\nsaved {RES / 'summary.json'} and {RES / 'summary.md'}")


def change(full, light, lower_is_better=True, unit="", pct=True, digits=2):
    d = light - full
    rel = f" ({100 * d / full:+.0f}%)" if pct and full else ""
    better = (d < 0) == lower_is_better
    same = abs(d) < 1e-9 or (full and abs(d / full) < 0.005)          # under half a percent: no real change
    word = "same" if same else ("better" if better else "worse")
    return f"{d:+.{digits}f}{unit}{rel}, {word}"


def write_md(S):
    p, g, md = S["parameters"], S["time_gpu"], []
    add = md.append
    add("# Full FP-TTC against light FP-TTC (new decoder)\n")
    add("Everything else is the same as the full model (frozen, released `fpttc_kitti` weights). Only the decoder is replaced by Conv3 on Concat(SF_t, x_t), trained on KITTI.\n")
    add("| | Full FP-TTC | Light FP-TTC | Change |\n|---|---|---|---|")
    add(f"| Parameters | {p['full'] / 1e6:.2f} M | {p['light'] / 1e6:.2f} M | {change(p['full'] / 1e6, p['light'] / 1e6, True, ' M')} |")
    add(f"| Model size (32-bit) | {4 * p['full'] / 1e6:.0f} MB | {4 * p['light'] / 1e6:.0f} MB | {change(4 * p['full'] / 1e6, 4 * p['light'] / 1e6, True, ' MB', digits=0)} |")
    add(f"| Time per pair, GPU | {g['full']['ms']:.0f} ms | {g['light']['ms']:.0f} ms | {change(g['full']['ms'], g['light']['ms'], True, ' ms', digits=0)} |")
    add(f"| Pairs per second, GPU | {g['full']['fps']:.1f} | {g['light']['fps']:.1f} | {change(g['full']['fps'], g['light']['fps'], False, '', digits=1)} |")
    if "time_cpu" in S:
        c = S["time_cpu"]
        add(f"| Time per pair, CPU | {c['full']['ms'] / 1000:.1f} s | {c['light']['ms'] / 1000:.1f} s | {change(c['full']['ms'] / 1000, c['light']['ms'] / 1000, True, ' s', digits=1)} |")
    add(f"| Peak GPU memory | {g['full']['peak_gpu_gb']:.2f} GB | {g['light']['peak_gpu_gb']:.2f} GB | {change(g['full']['peak_gpu_gb'], g['light']['peak_gpu_gb'], True, ' GB')} |")
    for k, label in (("held_out_40", "40 held-out images"), ("other_160", "other 160 images"), ("all_200", "all 200 images")):
        m = S["mid_error"][k]
        add(f"| Mid error, {label} (lower is better) | {m['full']:.1f} | {m['light']:.1f} | {change(m['full'], m['light'], True, '', digits=1)} |")
    m = S["mid_error"]["held_out_40"]
    add(f"| Mid error on cars, held-out | {m['full_fg']:.1f} | {m['light_fg']:.1f} | {change(m['full_fg'], m['light_fg'], True, '', digits=1)} |")
    add(f"| Mid error on background, held-out | {m['full_bg']:.1f} | {m['light_bg']:.1f} | {change(m['full_bg'], m['light_bg'], True, '', digits=1)} |")
    for key, label in (("kitti_cars", "KITTI cars"), ("nuscenes_cars", "nuScenes cars")):
        if key not in S:
            continue
        groups = S[key].items() if key == "kitti_cars" else [("", S[key])]
        for gname, v in groups:
            tag = f"{label}{', ' + gname.replace('_', ' ') if gname else ''} ({v['full']['cars']} cars)"
            add(f"| Per-car scale error, {tag} | {v['full']['scale_err_pct']:.2f}% | {v['light']['scale_err_pct']:.2f}% | {change(v['full']['scale_err_pct'], v['light']['scale_err_pct'], True, '%')} |")
            add(f"| Per-car TTC error (middle), {tag} | {v['full']['ttc_med']:.2f} s | {v['light']['ttc_med']:.2f} s | {change(v['full']['ttc_med'], v['light']['ttc_med'], True, ' s')} |")
            add(f"| Cars under 3 s found, {tag} | {100 * v['full']['found']:.0f}% | {100 * v['light']['found']:.0f}% | {change(100 * v['full']['found'], 100 * v['light']['found'], False, ' pts', pct=False, digits=0)} |")
            add(f"| Precision, {tag} | {100 * v['full']['precision']:.0f}% | {100 * v['light']['precision']:.0f}% | {change(100 * v['full']['precision'], 100 * v['light']['precision'], False, ' pts', pct=False, digits=0)} |")
    add("\n## Parameters by part (full model)\n")
    add("| Part | Parameters | In the light model |\n|---|---|---|")
    for name, n in p["groups"].items():
        gone = name in ("matching-clue attention", "decoder + upsample")
        add(f"| {name} | {n:,} | {'removed' if gone else 'kept, frozen'} |")
    add(f"| Conv3 (new) | {p['conv3']:,} | added, trained |")
    (RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf8")


if __name__ == "__main__":
    main()
