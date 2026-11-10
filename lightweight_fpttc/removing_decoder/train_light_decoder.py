"""
Train the light decoder (Conv3) on KITTI. Everything else in FP-TTC stays frozen.

What it does
  1. Run the full FP-TTC once on each of the 200 KITTI training pairs (frames _09 and _10, the same
     "past" pairing the model was trained on) and save: SF_t, x_t (the image features of the later
     frame), the full model's own scale map, KITTI's true scale map and its valid mask, the car mask.
     Saved to lightweight_fpttc/removing_decoder/cache/ (about 1.5 GB, not committed). Skipped if it exists.
  2. Learn: Conv3 turns Concat(SF_t, x_t) into a scale map. Loss = mean |log guess - log true| over the
     pixels where KITTI has a true scale (the paper's Eq. 10), at KITTI's picture size.
  3. Quiz images pick the best round. The 40 held-out images (numbers divisible by 5, the paper's test
     split) are used only for the final numbers.

Images:  held-out test = number divisible by 5 (40);  quiz = number ending in 1 (20);  training = the rest (140).

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_decoder/train_light_decoder.py
    python lightweight_fpttc/removing_decoder/train_light_decoder.py --rebuild_cache
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import light_model as L  # noqa: E402
import eval_kitti as K  # noqa: E402  (fpttc_results/eval_kitti.py)

C = L.C
CACHE = HERE / "cache" / "kitti_light.pt"
N_IMAGES = 200
BATCH = 4
EPOCHS = 120
PATIENCE = 30
LR, WD = 3e-4, 1e-4


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def build_cache():
    print("running the full FP-TTC once on the 200 KITTI pairs ...", flush=True)
    kt = K.Kitti()
    runner = L.Runner(C.load_model("fpttc_kitti"))
    sf_all = ft_all = None
    scales = torch.zeros(N_IMAGES, L.SIZE[1], L.SIZE[0])
    gts, masks, fgs, cars = [], [], [], []
    t0 = time.time()
    for i in range(N_IMAGES):
        g = K.ground_truth(kt, i)
        h, w = g["gt"].shape
        scale, sf, ft = runner.full(kt.image(i, 9), kt.image(i, 10), L.SIZE)
        if sf_all is None:
            sf_all = torch.zeros(N_IMAGES, *sf.shape, dtype=torch.float16)
            ft_all = torch.zeros(N_IMAGES, *ft.shape, dtype=torch.float16)
            print("SF_t and x_t shape per image:", tuple(sf.shape), tuple(ft.shape), flush=True)
        sf_all[i], ft_all[i] = sf.half().cpu(), ft.half().cpu()
        scales[i] = torch.from_numpy(scale)
        obj = kt.objects(i)
        gts.append(torch.from_numpy(np.where(g["mask"], g["gt"], 1.0)).half())
        masks.append(torch.from_numpy(g["mask"]))
        fgs.append(torch.from_numpy(obj > 0))
        fx, fy = L.SIZE[0] / w, L.SIZE[1] / h
        for oid in [v for v in np.unique(obj) if v != 0]:
            m = obj == oid
            mv = m & g["mask"]
            if mv.sum() < 300:                                  # same rule as the other results
                continue
            ys, xs = np.where(m)
            cars.append(dict(image=i, oid=int(oid), s_true=float(np.median(g["gt"][mv])),
                             box=[float(xs.min() * fx), float(ys.min() * fy), float((xs.max() + 1) * fx), float((ys.max() + 1) * fy)]))
        if i % 25 == 0:
            print(f"  image {i}/{N_IMAGES}   {time.time() - t0:.0f}s", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(sf=sf_all, ft=ft_all, scale_full=scales, gt=gts, mask=masks, fg=fgs, cars=cars), CACHE)
    print(f"saved {CACHE}  ({len(cars)} cars)", flush=True)


def image_loss(dec, sf, ft, gt, mask):
    """Mean |log guess - log true| for one image, at KITTI's picture size, over pixels with a true scale."""
    logs = dec.log_scale(sf[None], ft[None])
    logs = F.interpolate(logs, size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
    return (logs - torch.log(gt.float())).abs()[mask].mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    dev = "cuda"

    if a.rebuild_cache or not CACHE.exists():
        build_cache()
    d = torch.load(CACHE, weights_only=False)
    sf, ft, gt, mask = d["sf"], d["ft"], d["gt"], d["mask"]
    train_idx = [i for i in range(N_IMAGES) if role(i) == "train"]
    quiz_idx = [i for i in range(N_IMAGES) if role(i) == "quiz"]
    print(f"images: train {len(train_idx)}, quiz {len(quiz_idx)}, test {sum(role(i) == 'test' for i in range(N_IMAGES))}", flush=True)

    # per-channel scaling of the inputs, from the training images only
    s1 = torch.zeros(2 * L.CH, device=dev)
    s2 = torch.zeros(2 * L.CH, device=dev)
    n = 0
    for i in train_idx:
        x = torch.cat([sf[i], ft[i]], 0).float().to(dev)
        s1 += x.sum((1, 2))
        s2 += (x * x).sum((1, 2))
        n += x.shape[1] * x.shape[2]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(1e-8).sqrt()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dec = L.LightDecoder().to(dev)
    dec.mean.copy_(mean.view(1, -1, 1, 1))
    dec.std.copy_(std.view(1, -1, 1, 1))
    print("light decoder parameters:", L.count(dec), flush=True)
    opt = torch.optim.AdamW(dec.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    def get(i):
        return sf[i].to(dev), ft[i].to(dev), gt[i].to(dev), mask[i].to(dev)

    @torch.no_grad()
    def quiz_loss():
        dec.eval()
        return float(np.mean([float(image_loss(dec, *get(i))) for i in quiz_idx]))

    best, best_state, bad, log = 1e9, None, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        dec.train()
        order = rng.permutation(train_idx)
        run = 0.0
        for b in range(0, len(order), BATCH):
            loss = sum(image_loss(dec, *get(int(i))) for i in order[b:b + BATCH]) / len(order[b:b + BATCH])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            run += float(loss) * len(order[b:b + BATCH])
        sched.step()
        tr, qz = run / len(order), quiz_loss()
        log.append(dict(epoch=ep + 1, train=tr, quiz=qz))
        if qz < best - 1e-6:
            best, bad = qz, 0
            best_state = {k: v.detach().clone() for k, v in dec.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"round {ep + 1:3d}   train error {tr * 1e4:7.1f}   quiz error {qz * 1e4:7.1f}{'   (best)' if bad == 0 else ''}   {time.time() - t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("quiz error stopped improving, stopping", flush=True)
            break

    dec.load_state_dict(best_state)
    L.save_decoder(dec, info=dict(quiz_error_e4=best * 1e4, rounds=len(log), train_images=len(train_idx)))
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    (out / "training_log.json").write_text(json.dumps(log, indent=1))
    print(f"\nsaved {L.WEIGHTS_FILE}  (best quiz error {best * 1e4:.1f}; the same units as the Mid error)")
    print("next: python lightweight_fpttc/removing_decoder/evaluate_light.py")


if __name__ == "__main__":
    main()
