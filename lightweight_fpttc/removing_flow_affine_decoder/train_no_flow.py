"""
Quick test: does FP-TTC still work when flow + affine is removed and the query is made from the image features?

What it does
  1. Run the CNN and the transformer (frozen, no flow stage) on the 200 KITTI training pairs and save the
     features of both levels. Saved to cache/kitti_features.pt (about 1.6 GB, not committed).
     Check: the finest features of the later frame must equal the ones the first light version saved from the
     full model (this proves the flow stage does not change them).
  2. Train the new head on the training images: the query layer (mode two), the scale encoder (fine-tuned from the
     released weights) and Conv3. Loss = mean |log guess - log true| where KITTI has a true scale (the paper's Eq. 10),
     plus, if --distill > 0, the same distance to the full model's own scale map.
  3. Quiz images pick the best round. The 40 held-out images (numbers divisible by 5) are only used for the final numbers.

Images:  held-out test = number divisible by 5 (40);  quiz = number ending in 1 (20);  training = the rest (140).

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine_decoder/train_no_flow.py --query two
    python lightweight_fpttc/removing_flow_affine_decoder/train_no_flow.py --query one
    python lightweight_fpttc/removing_flow_affine_decoder/train_no_flow.py --query two --distill 0.5 --name two_distill
Needs the cache of the first light version (run lightweight_fpttc/removing_decoder_only/train_light_decoder.py once).
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
sys.path.insert(0, str(HERE.parent / "removing_decoder_only"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402
import eval_kitti as K  # noqa: E402
import evaluate_light as E  # noqa: E402  (Mid error code of the first light version)

C = L.C
CACHE_FEATS = HERE / "cache" / "kitti_features.pt"
CACHE_PART1 = HERE.parent / "removing_decoder_only" / "cache" / "kitti_light.pt"
RES = HERE / "results"
N_IMAGES = 200
BATCH = 4
LR_ENC, LR_NEW, WD = 1e-4, 3e-4, 1e-4


def role(i):
    return "test" if i % 5 == 0 else ("quiz" if i % 10 == 1 else "train")


def build_feature_cache(dev="cuda"):
    print("running the CNN + transformer (no flow stage) on the 200 KITTI pairs ...", flush=True)
    kt = K.Kitti()
    model = C.load_model("fpttc_kitti")
    out = [[] for _ in range(4)]          # f0 level0, f0 level1, f1 level0, f1 level1
    t0 = time.time()
    for i in range(N_IMAGES):
        a, b = L.prepare(kt.image(i, 9), kt.image(i, 10), device=dev)
        m0, m1 = M.features(model, a, b)
        for j, t in enumerate((m0[0], m0[1], m1[0], m1[1])):
            out[j].append(t[0].half().cpu())
        if i % 25 == 0:
            print(f"  image {i}/{N_IMAGES}   {time.time() - t0:.0f}s", flush=True)
    CACHE_FEATS.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(f0=[torch.stack(out[0]), torch.stack(out[1])], f1=[torch.stack(out[2]), torch.stack(out[3])]), CACHE_FEATS)
    print(f"saved {CACHE_FEATS}", flush=True)


def get_feats(F_, i, dev):
    return [x[i:i + 1].to(dev).float() for x in F_["f0"]], [x[i:i + 1].to(dev).float() for x in F_["f1"]]


def image_loss(head, F_, d, i, dev, distill):
    m0, m1 = get_feats(F_, i, dev)
    logs = head.log_scale(m0, m1)                                   # (1, 1, 320, 640)
    gt, mask = d["gt"][i].to(dev), d["mask"][i].to(dev)
    big = F.interpolate(logs, size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
    loss = (big - torch.log(gt.float())).abs()[mask].mean()
    if distill > 0:
        teacher = torch.log(d["scale_full"][i].to(dev))
        loss = loss + distill * (logs[0, 0] - teacher).abs().mean()
    return loss


@torch.no_grad()
def scale_maps(head, F_, idx, dev):
    head.eval()
    out = []
    for i in idx:
        m0, m1 = get_feats(F_, i, dev)
        out.append(head(m0, m1)[0, 0].cpu().numpy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", choices=["one", "two"], default="two")
    ap.add_argument("--distill", type=float, default=0.0, help="weight of the 'copy the full model' term")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--init_conv3", choices=["part1", "fresh"], default="part1")
    ap.add_argument("--name", default=None, help="name of the saved weights, default no_flow_<query>")
    ap.add_argument("--rebuild_cache", action="store_true")
    a = ap.parse_args()
    name = a.name or f"no_flow_{a.query}"
    dev = "cuda"

    if a.rebuild_cache or not CACHE_FEATS.exists():
        build_feature_cache(dev)
    F_ = torch.load(CACHE_FEATS, weights_only=False)
    d = torch.load(CACHE_PART1, weights_only=False)

    # the check: finest features of the later frame equal the ones saved from the full model
    diff = (F_["f1"][1][0].float() - d["ft"][0].float()).abs().max().item()
    print(f"check, no-flow features against the full model's features (image 0): max difference {diff:.4f}", flush=True)
    sf_part1, ft_part1 = d.pop("sf"), d.pop("ft")
    del sf_part1, ft_part1

    train_idx = [i for i in range(N_IMAGES) if role(i) == "train"]
    quiz_idx = [i for i in range(N_IMAGES) if role(i) == "quiz"]
    test_idx = [i for i in range(N_IMAGES) if role(i) == "test"]
    print(f"images: train {len(train_idx)}, quiz {len(quiz_idx)}, test {len(test_idx)}", flush=True)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = C.load_model("fpttc_kitti")
    conv3 = M.part1_conv3(dev) if a.init_conv3 == "part1" else None
    conv3 = conv3 if conv3 is not None else L.LightDecoder()
    head = M.NoFlowHead(model.scalenet.encoder, a.query, conv3).to(dev)

    # per-channel scaling of Conv3's input, from the training images, with the starting head
    s1, s2, n = torch.zeros(2 * L.CH, device=dev), torch.zeros(2 * L.CH, device=dev), 0
    with torch.no_grad():
        head.eval()
        for i in train_idx:
            m0, m1 = get_feats(F_, i, dev)
            x = torch.cat([head.sf(m0, m1), m1[-1]], 1)[0].float()
            s1 += x.sum((1, 2))
            s2 += (x * x).sum((1, 2))
            n += x.shape[1] * x.shape[2]
    mean = s1 / n
    std = (s2 / n - mean ** 2).clamp_min(1e-8).sqrt()
    head.conv3.mean.copy_(mean.view(1, -1, 1, 1))
    head.conv3.std.copy_(std.view(1, -1, 1, 1))

    p = M.parameter_report(model, head)
    print(f"parameters: full {p['full']:,}   this version {p['light']:,}   (flow + affine {p['flow_affine']:,}, new {p['new']:,})", flush=True)

    enc = list(head.encoder.parameters())
    new = [q for n_, q in head.named_parameters() if not n_.startswith("encoder.")]
    opt = torch.optim.AdamW([dict(params=enc, lr=LR_ENC), dict(params=new, lr=LR_NEW)], weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)

    @torch.no_grad()
    def quiz_loss():
        head.eval()
        return float(np.mean([float(image_loss(head, F_, d, i, dev, 0.0)) for i in quiz_idx]))

    start = quiz_loss()
    print(f"before any training: quiz error {start * 1e4:.1f}", flush=True)
    best, best_state, bad, log = start, {k: v.detach().clone() for k, v in head.state_dict().items()}, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        head.train()
        order = rng.permutation(train_idx)
        run = 0.0
        for b in range(0, len(order), BATCH):
            chunk = order[b:b + BATCH]
            opt.zero_grad()
            for i in chunk:                                           # one image at a time: the encoder is heavy on memory
                loss = image_loss(head, F_, d, int(i), dev, a.distill)
                (loss / len(chunk)).backward()
                run += float(loss)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        sched.step()
        tr, qz = run / len(order), quiz_loss()
        log.append(dict(epoch=ep + 1, train=tr, quiz=qz))
        if qz < best - 1e-6:
            best, bad = qz, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            bad += 1
        print(f"round {ep + 1:3d}   train error {tr * 1e4:7.1f}   quiz error {qz * 1e4:7.1f}{'   (best)' if bad == 0 else ''}   {time.time() - t0:.0f}s", flush=True)
        if bad >= a.patience:
            print("quiz error stopped improving, stopping", flush=True)
            break

    head.load_state_dict(best_state)
    M.save_head(head, name, info=dict(quiz_error_e4=best * 1e4, rounds=len(log), query=a.query, distill=a.distill,
                                      init_conv3=a.init_conv3, start_quiz_error_e4=start * 1e4))

    # final numbers: Mid error as in evaluate_light.py (same function), against the full model and the first light version
    maps = scale_maps(head, F_, range(N_IMAGES), dev)
    me = E.mid_errors(maps, d)
    prev = json.loads((HERE.parent / "removing_decoder_only" / "results" / "summary.json").read_text())["mid_error"]
    res = dict(name=name, query=a.query, distill=a.distill, parameters=p, quiz_error_e4=best * 1e4, rounds=len(log), mid_error={})
    for k, sel in (("held_out_40", lambda r: r == "test"), ("other_160", lambda r: r != "test"), ("all_200", lambda r: True)):
        m = me.role.map(sel)
        res["mid_error"][k] = dict(this=float(me[m].mid.mean()), full=prev[k]["full"], decoder_only=prev[k]["light"],
                                   this_fg=float(me[m].fg.mean()), this_bg=float(me[m].bg.mean()))
    RES.mkdir(exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(res, indent=1, default=float))
    (RES / f"{name}_training_log.json").write_text(json.dumps(log, indent=1))
    h = res["mid_error"]["held_out_40"]
    print(f"\nMid error on the 40 held-out images (lower is better):  full model {h['full']:.1f}   decoder removed only {h['decoder_only']:.1f}   "
          f"flow + affine also removed ({name}) {h['this']:.1f}")
    print(f"saved {M.WEIGHTS / (name + '.pt')} and {RES / (name + '.json')}")


if __name__ == "__main__":
    main()
