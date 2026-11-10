"""
Shared code for the Conv3 training experiments on version A (flow + affine removed, scale_conv kept, encoder frozen).

Three ways to teach Conv3 (the scale encoder and everything before it stay frozen, all images are KITTI):
  direct               loss = distance to KITTI's true scale (only where KITTI has it). Done already: "A retrained" in
                       evaluate_versions.py, Mid error 64.6 on the 40 held-out images.
  distill              loss = distance to the FULL model's scale map, at every pixel. No KITTI truth used.
  distill_then_kitti   first "distill", then continue with "direct" from those weights.

Images: held-out test = number divisible by 5 (40);  quiz = number ending in 1 (20);  training = the rest (140).
The quiz images pick the best round (by the same kind of loss that was trained on). The held-out images are only used for the final numbers.
The full model's scale maps are the ones saved in removing_decoder/cache/kitti_light.pt (scale_full).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "removing_decoder"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402
import evaluate_light as E  # noqa: E402
import evaluate_versions as EV  # noqa: E402

C = L.C
N = 200
BATCH, WD = 4, 1e-4
DT = 0.1


class Setup:
    """Cached features, the frozen version A encoder's SF_t for every image, and the targets."""

    def __init__(self, dev="cuda"):
        self.dev = dev
        self.F_ = torch.load(EV.CACHE_FEATS, weights_only=False)
        self.d = torch.load(EV.CACHE_PART1, weights_only=False)
        self.d.pop("sf")
        self.d.pop("ft")
        model = C.load_model("fpttc_kitti")
        self.head = M.build_head(model, "A", None, dev)
        self.sfs = EV.all_sf(self.head, self.F_, dev)
        self.ft = self.F_["f1"][1]
        self.train = [i for i in range(N) if E.role(i) == "train"]
        self.quiz = [i for i in range(N) if E.role(i) == "quiz"]

    def conv3_from_part1(self):
        c = L.LightDecoder().to(self.dev)
        c.load_state_dict(self.head.conv3.state_dict())
        return c

    def logs(self, conv3, i):
        return conv3.log_scale(self.sfs[i].to(self.dev).float(), self.ft[i:i + 1].to(self.dev).float())

    def loss_truth(self, conv3, i):
        gt, mask = self.d["gt"][i].to(self.dev), self.d["mask"][i].to(self.dev)
        big = F.interpolate(self.logs(conv3, i), size=gt.shape, mode="bilinear", align_corners=False)[0, 0]
        return (big - torch.log(gt.float())).abs()[mask].mean()

    def loss_distill(self, conv3, i):
        teacher = torch.log(self.d["scale_full"][i].to(self.dev))
        return (self.logs(conv3, i)[0, 0] - teacher).abs().mean()


def train(conv3, loss_fn, S, lr, epochs=40, patience=8, label=""):
    """Train Conv3 with loss_fn(conv3, i). The quiz images pick the best round using the same loss."""
    opt = torch.optim.AdamW(conv3.parameters(), lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(0)

    @torch.no_grad()
    def qloss():
        conv3.eval()
        return float(np.mean([float(loss_fn(conv3, i)) for i in S.quiz]))

    start = best = qloss()
    state = {k: v.clone() for k, v in conv3.state_dict().items()}
    bad, rounds, log = 0, 0, []
    for ep in range(epochs):
        conv3.train()
        order = rng.permutation(S.train)
        run = 0.0
        for b in range(0, len(order), BATCH):
            chunk = order[b:b + BATCH]
            loss = sum(loss_fn(conv3, int(i)) for i in chunk) / len(chunk)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(conv3.parameters(), 1.0)
            opt.step()
            run += float(loss) * len(chunk)
        sched.step()
        q = qloss()
        rounds = ep + 1
        log.append(dict(epoch=ep + 1, train=run / len(order), quiz=q))
        if q < best - 1e-6:
            best, bad = q, 0
            state = {k: v.clone() for k, v in conv3.state_dict().items()}
        else:
            bad += 1
        if (ep + 1) % 5 == 0:
            print(f"  {label} round {ep + 1}  quiz loss {q * 1e4:.1f}  (best {best * 1e4:.1f})", flush=True)
        if bad >= patience:
            break
    conv3.load_state_dict(state)
    conv3.eval()
    return dict(start_quiz_e4=start * 1e4, best_quiz_e4=best * 1e4, rounds=rounds, log=log)


def evaluate(conv3, S):
    """Mid error (KITTI, three image groups) and per-car numbers on the KITTI cars, as in evaluate_versions.py."""
    maps = EV.maps_from(conv3, S.sfs, S.F_, S.dev)
    me = E.mid_errors(maps, S.d)
    out = dict(mid_error={})
    for name, sel in (("held_out_40", lambda r: r == "test"), ("other_160", lambda r: r != "test"), ("all_200", lambda r: True)):
        m = me.role.map(sel)
        out["mid_error"][name] = dict(all=float(me[m].mid.mean()), cars=float(me[m].fg.mean()), background=float(me[m].bg.mean()))
    rows = []
    for c in S.d["cars"]:
        i = c["image"]
        rows.append(dict(image=i, role=E.role(i), s_true=c["s_true"], s_this=C.car_scale_median(maps[i], c["box"], 0.7)[0]))
    kc = pd.DataFrame(rows)
    out["kitti_cars"] = {name: E.car_stats(kc[sel], "this", DT) for name, sel in (("held_out_40", kc.role == "test"), ("all_200", kc.role.notna()))}
    return out, kc


def reference_numbers():
    """The numbers of the earlier versions, from evaluate_versions.py."""
    v = json.loads((HERE / "results" / "versions_summary.json").read_text())
    m = v["mid_error"]["held_out_40"]
    return dict(full=m["full"], decoder_removed=m["decoder_removed"], A_asis=m["A_asis"], A_direct=m["A_retrained"])


def write_outputs(folder, name, title, how, result, kc, S_info, weights):
    folder = Path(folder)
    (folder / "results").mkdir(exist_ok=True)
    (folder / "weights").mkdir(exist_ok=True)
    torch.save(dict(state=weights, info=S_info), folder / "weights" / "conv3.pt")
    kc.to_csv(folder / "results" / "kitti_cars.csv", index=False)
    ref = reference_numbers()
    result["reference_held_out_40"] = ref
    result["training"] = S_info
    (folder / "results" / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    h = result["mid_error"]["held_out_40"]
    md = [f"# {title}", "", how, "",
          "Mid error on the 40 held-out KITTI images (lower is better):", "",
          "| | Mid error |", "|---|---|",
          f"| Full FP-TTC | {ref['full']:.1f} |", f"| Decoder removed (part 1) | {ref['decoder_removed']:.1f} |",
          f"| Version A, Conv3 as it is | {ref['A_asis']:.1f} |", f"| Version A, Conv3 trained directly on KITTI truth | {ref['A_direct']:.1f} |",
          f"| **Version A, this method ({name})** | **{h['all']:.1f}** |", "",
          f"Cars only {h['cars']:.1f}, background only {h['background']:.1f}.", "",
          "| | all 40 held-out | other 160 | all 200 |", "|---|---|---|---|",
          f"| Mid error, this method | {h['all']:.1f} | {result['mid_error']['other_160']['all']:.1f} | {result['mid_error']['all_200']['all']:.1f} |", ""]
    for k, lab in (("held_out_40", "KITTI cars, held-out 40"), ("all_200", "KITTI cars, all 200")):
        c = result["kitti_cars"][k]
        md.append(f"- {lab}: scale error {c['scale_err_pct']:.2f}%, TTC error (middle) {c['ttc_med']:.2f} s, "
                  f"cars under 3 s found {100 * c['found']:.0f}%, precision {100 * c['precision']:.0f}%")
    (folder / "results" / "summary.md").write_text("\n".join(md) + "\n")
