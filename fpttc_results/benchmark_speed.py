"""
Speed, memory and size of FP-TTC on this laptop (RTX 3050, 4 GB).

Measures, each with warm-up runs first and the GPU synchronised before every
clock reading:
  * time per frame pair at several input sizes
  * GPU memory at each size
  * time per module (which part of the network is slow)
  * parameters per module and file size on disk
  * batch size, half precision, and CPU
  * the whole camera-to-per-car-TTC pipeline: YOLOv8s + FP-TTC + per-car step

Run on an otherwise idle GPU, or the timings are not comparable.
Usage (from the project root):  python fpttc_results/benchmark_speed.py
"""

import json
import platform
import time
from collections import defaultdict

import numpy as np
import torch

import common as C
from src2.detection import VehicleDetector
import cv2

SIZES = [(480, 256), (640, 320), (640, 352), (960, 288), (960, 544), (1248, 384), (1280, 704)]
RUNS, WARM = 30, 5


def sync():
    torch.cuda.synchronize()


def time_runs(fn, runs=RUNS, warm=WARM):
    for _ in range(warm):
        fn()
    sync()
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    return dict(mean_ms=float(ts.mean()), median_ms=float(np.median(ts)), std_ms=float(ts.std()),
                p95_ms=float(np.percentile(ts, 95)), min_ms=float(ts.min()), runs=runs,
                fps=float(1000.0 / np.median(ts)))


def pair(w, h, batch=1, device="cuda"):
    return (torch.rand(batch, 3, h, w, device=device) * 255, torch.rand(batch, 3, h, w, device=device) * 255)


def main():
    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, cuda=torch.version.cuda,
               python=platform.python_version(), note="deformable attention runs in pure PyTorch "
               "(mmcv_compat.py), slower than the authors' CUDA kernel")
    model = C.load_model("fpttc_kitti")

    # ---- size on disk and parameters
    out["parameters_total"] = sum(p.numel() for p in model.parameters())
    out["file_size_mb"] = {f.name: round(f.stat().st_size / 1e6, 1) for f in C.WEIGHTS_DIR.glob("*.pth.tar")}
    parts = {
        "backbone CNN": model.cnet,
        "backbone transformer": model.featnet,
        "flow / matching": model.corrnet,
        "initial scale (conv_corr)": model.conv_corr,
        "scale encoder": model.scalenet.encoder,
        "matching-clue attention": model.scalenet.gma,
        "decoder + upsample": model.scalenet.decoder,
    }
    out["parameters_by_module"] = {k: sum(p.numel() for p in m.parameters()) for k, m in parts.items()}

    # ---- time and memory by input size
    out["by_size"] = []
    for (w, h) in SIZES:
        try:
            a, b = pair(w, h)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                r = time_runs(lambda: model(a, b, **C.RUN_KW))
            r.update(width=w, height=h, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
            out["by_size"].append(r)
            print(f"{w}x{h}: {r['median_ms']:.0f} ms  {r['fps']:.1f} FPS  {r['peak_gpu_gb']:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out["by_size"].append(dict(width=w, height=h, error="out of GPU memory"))
            print(f"{w}x{h}: out of memory", flush=True)
            torch.cuda.empty_cache()

    # ---- per-module time at the default size
    times, start = defaultdict(float), {}
    hooks = []
    def make_hooks(n):
        # hooks must return None: a pre-hook that returns a value replaces the module's inputs
        def pre(module, args):
            sync()
            start[n] = time.perf_counter()

        def post(module, args, output):
            sync()
            times[n] += time.perf_counter() - start[n]
        return pre, post

    for n, mod in parts.items():
        pre, post = make_hooks(n)
        hooks.append(mod.register_forward_pre_hook(pre))
        hooks.append(mod.register_forward_hook(post))
    a, b = pair(640, 320)
    with torch.no_grad():
        for _ in range(WARM):
            model(a, b, **C.RUN_KW)
        times.clear()
        N = 20
        sync(); t0 = time.perf_counter()
        for _ in range(N):
            model(a, b, **C.RUN_KW)
        sync(); total = (time.perf_counter() - t0) / N * 1000
    for h_ in hooks:
        h_.remove()
    out["by_module_640x320"] = {n: dict(ms=times[n] / N * 1000, percent=100 * times[n] / N * 1000 / total) for n in parts}
    out["by_module_640x320"]["(total, with timing overhead)"] = dict(ms=total, percent=100.0)

    # ---- batch size
    out["by_batch_640x320"] = []
    for bs in (1, 2, 4, 8):
        try:
            a, b = pair(640, 320, bs)
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                r = time_runs(lambda: model(a, b, **C.RUN_KW), runs=15)
            r.update(batch=bs, ms_per_pair=r["median_ms"] / bs, peak_gpu_gb=torch.cuda.max_memory_allocated() / 1e9)
            out["by_batch_640x320"].append(r)
            print(f"batch {bs}: {r['ms_per_pair']:.0f} ms/pair  {r['peak_gpu_gb']:.2f} GB", flush=True)
        except torch.cuda.OutOfMemoryError:
            out["by_batch_640x320"].append(dict(batch=bs, error="out of GPU memory"))
            torch.cuda.empty_cache()

    # ---- half precision
    a, b = pair(640, 320)
    try:
        with torch.no_grad():
            ref = model(a, b, **C.RUN_KW)[0]
            def half():
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(a, b, **C.RUN_KW)
            r = time_runs(half, runs=20)
            alt = half()[0].float()
        r["mean_abs_scale_difference_vs_fp32"] = float((alt - ref).abs().mean())
        out["half_precision_640x320"] = r
        print(f"fp16: {r['median_ms']:.0f} ms", flush=True)
    except Exception as e:  # noqa: BLE001
        out["half_precision_640x320"] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")

    # ---- CPU
    cpu_model = C.load_model("fpttc_kitti", device="cpu")
    a, b = pair(640, 320, device="cpu")
    ts = []
    with torch.no_grad():
        cpu_model(a, b, **C.RUN_KW)
        for _ in range(3):
            t = time.perf_counter(); cpu_model(a, b, **C.RUN_KW); ts.append(time.perf_counter() - t)
    out["cpu_640x320"] = dict(median_ms=float(np.median(ts) * 1000), runs=3, threads=torch.get_num_threads())
    print(f"cpu: {np.median(ts) * 1000:.0f} ms", flush=True)
    del cpu_model

    # ---- whole pipeline on a real frame: YOLO + FP-TTC + per-car median
    img = cv2.imread(str(C.ROOT / "data" / "nuScenes" / "v1.0-mini" / "samples" / "CAM_FRONT"
                         / sorted((C.ROOT / "data" / "nuScenes" / "v1.0-mini" / "samples" / "CAM_FRONT").iterdir())[40].name))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    det = VehicleDetector()
    t_y = time_runs(lambda: det.detect(img), runs=30)
    dets = det.detect(img)
    scale, flow = C.run_pair(model, rgb, rgb, size=(640, 352))
    H, W = img.shape[:2]
    boxes = [(d.x1 * 640 / W, d.y1 * 352 / H, d.x2 * 640 / W, d.y2 * 352 / H) for d in dets]
    t_f = time_runs(lambda: C.run_pair(model, rgb, rgb, size=(640, 352)), runs=20)

    def per_car():
        return [C.car_scale_median(scale, b_, 0.5) for b_ in boxes]
    t_m = time_runs(per_car, runs=200, warm=10)
    out["pipeline_640x352"] = dict(yolov8s=t_y, fpttc_including_resize=t_f, per_car_median=t_m,
                                   cars_in_frame=len(dets),
                                   total_median_ms=t_y["median_ms"] + t_f["median_ms"] + t_m["median_ms"])
    out["pipeline_640x352"]["fps"] = 1000.0 / out["pipeline_640x352"]["total_median_ms"]
    print("pipeline:", round(out["pipeline_640x352"]["total_median_ms"]), "ms", flush=True)

    (C.RESULTS_DIR / "speed.json").write_text(json.dumps(out, indent=2))
    print("saved speed.json")


if __name__ == "__main__":
    main()
