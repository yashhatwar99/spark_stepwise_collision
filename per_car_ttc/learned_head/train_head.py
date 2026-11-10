"""
Train the small head of Method 2 on KITTI. FP-TTC stays frozen.

What it does
  1. Run FP-TTC once on each of the 200 KITTI training pairs (frames _09 and _10,
     the same "past" pairing the model was trained on) and save SF_t, the model's
     own scale map, and every labelled car: its box and its true scale. Saved to
     per_car_ttc/learned_head/cache/ (about 0.8 GB, not committed). Skipped if it exists.
  2. Learn: for each training car, crop SF_t inside the car's box (with the box
     drawn a bit too big, too small or shifted, many ways, because YOLO's boxes
     are never exact), let the head guess log s, compare with the true log s
     (loss |log guess - log true|, the paper's Eq. 10) and nudge the head.
  3. Quiz images pick the best round. The 40 held-out images (numbers divisible
     by 5, the paper's test split) are touched only at the very end.
  4. Compare with the Median method on the held-out cars.

Images:  held-out test = number divisible by 5 (40);  quiz = number ending in 1 (20);
         training = the rest (140).

Usage (from the project root, with the project .venv):
    python per_car_ttc/learned_head/train_head.py
    python per_car_ttc/learned_head/train_head.py --rebuild_cache
Saves the head to per_car_ttc/learned_head/weights/ and results to per_car_ttc/learned_head/output/.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import head_model as H  # noqa: E402
import eval_kitti as K  # noqa: E402  (fpttc_results/eval_kitti.py)
import build_summary as BS  # noqa: E402

C = H.C
DT = 0.1
CACHE = HERE / "cache" / "kitti_features.pt"
OUT = HERE / "output"
N_IMAGES = 200
AUG_PER_CAR = 10          # jittered boxes per car per round
BATCH = 64
EPOCHS = 80
PATIENCE = 20
LR, WD = 2e-3, 1e-2


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def build_cache():
    print("running FP-TTC once on the 200 KITTI pairs ...", flush=True)
    kt = K.Kitti()
    model = C.load_model("fpttc_kitti")
    reader = H.FeatureReader(model)
    feats = None
    scales = torch.zeros(N_IMAGES, H.SIZE[1], H.SIZE[0])
    cars = []
    t0 = time.time()
    for i in range(N_IMAGES):
        g = K.ground_truth(kt, i)
        h, w = g["gt"].shape
        scale, feat = reader(kt.image(i, 9), kt.image(i, 10), H.SIZE)
        if feats is None:
            feats = torch.zeros(N_IMAGES, *feat.shape, dtype=torch.float16)
            print("SF_t shape per image:", tuple(feat.shape), flush=True)
        feats[i] = feat.half().cpu()
        scales[i] = torch.from_numpy(scale)
        obj = kt.objects(i)
        fx, fy = H.SIZE[0] / w, H.SIZE[1] / h
        for oid in [v for v in np.unique(obj) if v != 0]:
            m = obj == oid
            mv = m & g["mask"]
            if mv.sum() < 300:                         # same rule as the Median results
                continue
            ys, xs = np.where(m)
            box = (xs.min() * fx, ys.min() * fy, (xs.max() + 1) * fx, (ys.max() + 1) * fy)
            cars.append(dict(image=i, oid=int(oid), box=[float(b) for b in box],
                             s_true=float(np.median(g["gt"][mv]))))
        if i % 25 == 0:
            print(f"  image {i}/{N_IMAGES}   {time.time() - t0:.0f}s", flush=True)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(feats=feats, scales=scales, cars=cars), CACHE)
    print(f"saved {CACHE}  ({len(cars)} cars)", flush=True)


def jitter(box, rng):
    """The box drawn a bit too big/small and shifted, like a detector would."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sw, sh = np.exp(rng.uniform(-0.22, 0.22, 2))
    cx += rng.uniform(-0.12, 0.12) * w
    cy += rng.uniform(-0.12, 0.12) * h
    w, h = max(w * sw, 6.0), max(h * sh, 6.0)
    nx1, nx2 = np.clip([cx - w / 2, cx + w / 2], 0, H.SIZE[0])
    ny1, ny2 = np.clip([cy - h / 2, cy + h / 2], 0, H.SIZE[1])
    if nx2 - nx1 < 4 or ny2 - ny1 < 4:
        return tuple(box)
    return (float(nx1), float(ny1), float(nx2), float(ny2))


def batch_crops(feats, items, device):
    """items: list of (image index, box). Returns (N, 128, 7, 7) in the same order."""
    by_img = {}
    for n, (i, b) in enumerate(items):
        by_img.setdefault(i, []).append((n, b))
    out = torch.empty(len(items), H.CHANNELS, H.CROP, H.CROP, device=device)
    for i, lst in by_img.items():
        boxes = torch.tensor([b for _, b in lst], dtype=torch.float32, device=device)
        out[[n for n, _ in lst]] = H.crop(feats[i], boxes)
    return out


@torch.no_grad()
def quiz_error(head, feats, cars, device, seed=123, n_jitter=4):
    """Mean |log guess - log true| over the cars: the tight box and a few fixed jittered boxes."""
    head.eval()
    rng = np.random.default_rng(seed)
    items, ys = [], []
    for c in cars:
        for k in range(n_jitter + 1):
            items.append((c["image"], tuple(c["box"]) if k == 0 else jitter(c["box"], rng)))
            ys.append(math.log(c["s_true"]))
    pred = head(batch_crops(feats, items, device)).cpu().numpy()
    return float(np.mean(np.abs(pred - np.array(ys))))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild_cache", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    device = "cuda"

    if a.rebuild_cache or not CACHE.exists():
        build_cache()
    data = torch.load(CACHE, weights_only=False)
    feats = data["feats"].to(device)                       # fp16 on the GPU
    scales = data["scales"].numpy()
    cars = data["cars"]
    train = [c for c in cars if role(c["image"]) == "train"]
    quiz = [c for c in cars if role(c["image"]) == "quiz"]
    test = [c for c in cars if role(c["image"]) == "test"]
    print(f"cars: train {len(train)}, quiz {len(quiz)}, test {len(test)}  (all {len(cars)})", flush=True)

    # per-channel scaling of the features, from the training images only
    tr_imgs = [i for i in range(N_IMAGES) if role(i) == "train"]
    s1 = torch.zeros(H.CHANNELS, device=device)
    s2 = torch.zeros(H.CHANNELS, device=device)
    n = 0
    for i in tr_imgs:
        f = feats[i].float()
        s1 += f.sum((1, 2))
        s2 += (f * f).sum((1, 2))
        n += f.shape[1] * f.shape[2]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(1e-8).sqrt()

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    head = H.Head().to(device)
    head.mean.copy_(mean.view(1, -1, 1, 1))
    head.std.copy_(std.view(1, -1, 1, 1))
    print("head parameters:", sum(p.numel() for p in head.parameters()), flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    best, best_state, bad = 1e9, None, 0
    log = []
    t0 = time.time()
    for ep in range(a.epochs):
        head.train()
        items, ys = [], []
        for c in train:
            for _ in range(AUG_PER_CAR):
                items.append((c["image"], jitter(c["box"], rng)))
                ys.append(math.log(c["s_true"]))
        ys = torch.tensor(ys, dtype=torch.float32, device=device)
        perm = rng.permutation(len(items))
        run_loss = 0.0
        for b in range(0, len(perm), BATCH):
            idx = perm[b:b + BATCH]
            crops = batch_crops(feats, [items[j] for j in idx], device)
            loss = (head(crops) - ys[idx]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            run_loss += float(loss) * len(idx)
        sched.step()
        tr_loss = run_loss / len(perm)
        qe = quiz_error(head, feats, quiz, device)
        log.append(dict(epoch=ep + 1, train=tr_loss, quiz=qe))
        if qe < best - 1e-5:
            best, bad = qe, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        if ep % 5 == 0 or bad == 0:
            print(f"round {ep + 1:3d}   train error {tr_loss * 100:5.2f}%   quiz error {qe * 100:5.2f}%"
                  f"{'   (best)' if bad == 0 else ''}   {time.time() - t0:.0f}s", flush=True)
        if bad >= PATIENCE:
            print("quiz error stopped improving, stopping", flush=True)
            break

    head.load_state_dict(best_state)
    head.eval()
    H.save_head(head, info=dict(quiz_error=best, rounds=len(log), train_cars=len(train),
                                quiz_cars=len(quiz), aug_per_car=AUG_PER_CAR))
    print(f"\nsaved {H.WEIGHTS_FILE}  (best quiz error {best * 100:.2f}%)", flush=True)

    # ------------------------------------------------------------------ held-out test
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    rows = []
    for c in test:
        boxes = [("tight", tuple(c["box"]))] + [(f"jit{k}", jitter(c["box"], rng)) for k in range(5)]
        s_head = H.predict_s(head, feats[c["image"]], [b for _, b in boxes])
        for (kind, b), sh in zip(boxes, s_head):
            sm, _ = C.car_scale_median(scales[c["image"]], b, 0.7)
            rows.append(dict(image=c["image"], oid=c["oid"], boxes=kind, s_true=c["s_true"],
                             s_head=float(sh), s_median_box70=sm))
    df = pd.DataFrame(rows)
    report = {}
    print("\nHeld-out KITTI cars (never used for learning or for choosing the best round)")
    for kind, sel in [("tight mask boxes", df.boxes == "tight"), ("jittered boxes (like YOLO)", df.boxes != "tight")]:
        d = df[sel]
        print(f"\n  {kind}: {len(d)} boxes on {d[['image', 'oid']].drop_duplicates().shape[0]} cars")
        print("                     scale error   TTC error median / p90    danger found / precision (TTC < 3 s)")
        for m, label in [("median_box70", "Median (Method 1)"), ("head", "Learned head (Method 2)")]:
            r = BS.car_metrics(d, m, DT)
            report[f"{kind}|{m}"] = r
            e, dg = r["ttc_abs_error_s"], r["danger"]
            print(f"    {label:<24s} {r['scale_error_percent_median']:5.2f}%      "
                  f"{e['median']:5.2f} s / {e['p90']:5.2f} s            "
                  f"{dg['recall']:.0%} / {dg['precision']:.0%}")
    (OUT / "kitti_heldout_eval.json").write_text(json.dumps(dict(log=log, report=report), indent=1, default=float))
    print(f"\nsaved {OUT / 'kitti_heldout_eval.json'}")


if __name__ == "__main__":
    main()
