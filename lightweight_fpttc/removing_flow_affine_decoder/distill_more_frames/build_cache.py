"""
Teacher answers and version A inputs for the extra KITTI frames (raw drive 0009, image_02).

Pairs: neighbouring frames (j, j+1), 0.1 s apart. Frames 374 to 405 are left out (and every pair that uses one): they lie within one
second of KITTI scene flow's held-out scene 10 (raw frames 384/385) and quiz scene 11 (raw frames 394/395), found by check_overlap.py.
For every pair, saved to cache/ as memory-mapped files (they are too big to hold in memory together):
    raw_sf.npy          SF_t of version A (frozen scale encoder, query from the image features), fp16, (n, 128, 80, 160)
    raw_ft.npy          the later frame's fine image features x_t, fp16, (n, 128, 80, 160)
    raw_logteacher.npy  log of the FULL model's scale map (the teacher), float32, (n, 320, 640)
    raw_pairs.json      the frame numbers of every pair

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine_decoder/distill_more_frames/build_cache.py
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE.parents[1] / "removing_decoder_only"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402

C = L.C
RAW = HERE.parents[2] / "data" / "kitti" / "2011_09_26" / "2011_09_26_drive_0009_sync" / "image_02" / "data"
CACHE = HERE / "cache"
SKIP = range(374, 406)          # raw frame numbers left out (near the held-out and quiz scenes)


def read(j):
    return cv2.cvtColor(cv2.imread(str(RAW / f"{j:010d}.png")), cv2.COLOR_BGR2RGB)


def main():
    n_frames = len(list(RAW.glob("*.png")))
    pairs = [(j, j + 1) for j in range(n_frames - 1) if j not in SKIP and j + 1 not in SKIP]
    print(f"{n_frames} raw frames -> {len(pairs)} pairs (frames {SKIP.start}-{SKIP.stop - 1} left out)", flush=True)
    CACHE.mkdir(exist_ok=True)
    n = len(pairs)
    sf_mm = np.lib.format.open_memmap(CACHE / "raw_sf.npy", mode="w+", dtype=np.float16, shape=(n, 128, 80, 160))
    ft_mm = np.lib.format.open_memmap(CACHE / "raw_ft.npy", mode="w+", dtype=np.float16, shape=(n, 128, 80, 160))
    tc_mm = np.lib.format.open_memmap(CACHE / "raw_logteacher.npy", mode="w+", dtype=np.float32, shape=(n, 320, 640))

    dev = "cuda"
    model = C.load_model("fpttc_kitti")
    head = M.build_head(model, "A", None, dev)            # before the Runner, which adds a hook to the model's encoder
    runner = L.Runner(model, None)
    t0 = time.time()
    for k, (a, b) in enumerate(pairs):
        r0, r1 = read(a), read(b)
        scale, _, _ = runner.full(r0, r1)                  # the full model: the teacher
        pa, pb = L.prepare(r0, r1, device=dev)
        with torch.no_grad():
            m0, m1 = M.features(model, pa, pb)
            sf = head.sf(m0, m1)
        sf_mm[k] = sf[0].half().cpu().numpy()
        ft_mm[k] = m1[-1][0].half().cpu().numpy()
        tc_mm[k] = np.log(np.clip(scale, 1e-6, None)).astype(np.float32)
        if k % 25 == 0:
            print(f"pair {k}/{n}   {time.time() - t0:.0f}s", flush=True)
    for m in (sf_mm, ft_mm, tc_mm):
        m.flush()
    (CACHE / "raw_pairs.json").write_text(json.dumps(pairs))
    print(f"saved {n} pairs in {CACHE}", flush=True)


if __name__ == "__main__":
    main()
