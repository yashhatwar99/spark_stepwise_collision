"""
Version A (flow + affine removed), Conv3 taught only by the full model (distillation) on MORE KITTI frames.

Training pairs: the 140 KITTI scene flow training pairs (as before) plus the raw drive's neighbouring-frame pairs from cache/
(build_cache.py). The 20 quiz scenes pick the best round; the 40 held-out scenes are used only for the final Mid error.
Loss: mean |log(Conv3's s) - log(full model's s)| over every pixel. No KITTI truth is used.
Round cap 100, the learning rate shrinks over the 100 rounds, stop after 10 rounds without a better quiz loss.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine/distill_more_frames/train_distill_more.py
Writes weights/conv3.pt, results/summary.json, results/summary.md, results/kitti_cars.csv, results/training_log.json.
Pictures: python lightweight_fpttc/removing_flow_affine/compare_outputs_versions.py --conv3 distill_more_frames
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import conv3_training as T  # noqa: E402

EPOCHS, PATIENCE, LR = 100, 10, 1e-4
CACHE = HERE / "cache"

S = T.Setup()
S.F_ = {"f1": [None, S.ft]}          # only the later frame's fine features are needed from here on (saves memory)
dev = S.dev
raw_sf = np.load(CACHE / "raw_sf.npy", mmap_mode="r")
raw_ft = np.load(CACHE / "raw_ft.npy", mmap_mode="r")
raw_tc = np.load(CACHE / "raw_logteacher.npy", mmap_mode="r")
n_raw = raw_sf.shape[0]
items = [("kitti", i) for i in S.train] + [("raw", j) for j in range(n_raw)]
print(f"training pairs: {len(S.train)} KITTI + {n_raw} raw drive = {len(items)}; quiz {len(S.quiz)}", flush=True)


def loss_item(conv3, item):
    kind, i = item
    if kind == "kitti":
        sf, ft, tl = S.sfs[i].to(dev).float(), S.ft[i:i + 1].to(dev).float(), torch.log(S.d["scale_full"][i].to(dev))
    else:
        sf = torch.from_numpy(np.array(raw_sf[i:i + 1])).to(dev).float()
        ft = torch.from_numpy(np.array(raw_ft[i:i + 1])).to(dev).float()
        tl = torch.from_numpy(np.array(raw_tc[i])).to(dev)
    return (conv3.log_scale(sf, ft)[0, 0] - tl).abs().mean()


conv3 = S.conv3_from_part1()
opt = torch.optim.AdamW(conv3.parameters(), lr=LR, weight_decay=T.WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
rng = np.random.default_rng(0)


@torch.no_grad()
def quiz_loss():
    conv3.eval()
    return float(np.mean([float(loss_item(conv3, ("kitti", i))) for i in S.quiz]))


start = best = quiz_loss()
print(f"before training: quiz distillation loss {start * 1e4:.1f}", flush=True)
state = {k: v.clone() for k, v in conv3.state_dict().items()}
bad, log, t0 = 0, [], time.time()
for ep in range(EPOCHS):
    conv3.train()
    order = rng.permutation(len(items))
    run = 0.0
    for b in range(0, len(order), T.BATCH):
        chunk = order[b:b + T.BATCH]
        loss = sum(loss_item(conv3, items[int(k)]) for k in chunk) / len(chunk)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conv3.parameters(), 1.0)
        opt.step()
        run += float(loss) * len(chunk)
    sched.step()
    q = quiz_loss()
    log.append(dict(epoch=ep + 1, train=run / len(order), quiz=q))
    if q < best - 1e-6:
        best, bad = q, 0
        state = {k: v.clone() for k, v in conv3.state_dict().items()}
    else:
        bad += 1
    print(f"round {ep + 1:3d}   train loss {run / len(order) * 1e4:6.1f}   quiz loss {q * 1e4:6.1f}{'   (best)' if bad == 0 else ''}   {time.time() - t0:.0f}s", flush=True)
    if bad >= PATIENCE:
        print("quiz loss stopped improving, stopping", flush=True)
        break
conv3.load_state_dict(state)
conv3.eval()
info = dict(start_quiz_e4=start * 1e4, best_quiz_e4=best * 1e4, rounds=len(log), train_pairs=len(items), kitti_pairs=len(S.train), raw_pairs=n_raw)
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "training_log.json").write_text(json.dumps(log, indent=1))
result, kc = T.evaluate(conv3, S)
T.write_outputs(HERE, "distillation on more frames", "Version A, Conv3 trained by distillation on more KITTI frames",
                f"Conv3 copies the full model's scale map on {len(items)} pairs ({len(S.train)} KITTI scene flow training pairs + {n_raw} raw drive pairs). "
                "No KITTI truth is used in training.", result, kc, dict(stage1=info), conv3.state_dict())
print("Mid error held-out 40:", result["mid_error"]["held_out_40"], flush=True)
