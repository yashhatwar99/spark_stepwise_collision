"""
Decoder removed ONLY (flow + affine kept), with Conv3 taught by distillation: it copies the full model's scale map.

The model: the full FP-TTC up to the scale encoder (CNN, transformer, flow + affine, scale encoder, all frozen, released weights),
then Conv3 in place of the decoder. Conv3's inputs are the full model's SF_t and x_t, saved by train_light_decoder.py in
removing_decoder_only/cache/kitti_light.pt, with the full model's own scale map for every pair (the teacher).

  loss = mean |log(Conv3's s) - log(full model's s)| over every pixel, on the 140 KITTI training pairs. No KITTI truth is used.
  Conv3 starts from the weights of removing_decoder_only/weights/light_decoder.pt (trained on KITTI truth).
  The 20 quiz pairs pick the best round (same loss). The 40 held-out pairs are used only for the final Mid error against KITTI's truth.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_decoder_only/distillation/train_distill.py
    python lightweight_fpttc/removing_decoder_only/distillation/compare_outputs_distill.py     # KITTI pictures
Writes weights/conv3.pt, results/summary.json, results/summary.md, results/kitti_cars.csv, results/training_log.json.
"""

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LW = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(LW / "removing_flow_affine_decoder"))

import light_model as L  # noqa: E402
import conv3_training as T  # noqa: E402  (training loop and evaluation shared with the version A experiments)

C = L.C


class FullQuerySetup(T.Setup):
    """Same interface as T.Setup, but the inputs are the full model's SF_t and x_t (flow + affine kept)."""

    def __init__(self, dev="cuda"):
        self.dev = dev
        d = torch.load(HERE.parent / "cache" / "kitti_light.pt", weights_only=False)
        self.sfs = [d["sf"][i:i + 1] for i in range(T.N)]
        self.ft = d["ft"]
        self.F_ = {"f1": [None, self.ft]}                     # evaluation reads x_t from here
        self.d = d
        self.train = [i for i in range(T.N) if T.E.role(i) == "train"]
        self.quiz = [i for i in range(T.N) if T.E.role(i) == "quiz"]
        self.head = None

    def conv3_from_part1(self):
        c, _ = L.load_decoder(device=self.dev)
        return c


def main():
    S = FullQuerySetup()
    conv3 = S.conv3_from_part1()
    print(f"decoder removed only, distillation: {len(S.train)} KITTI training pairs, quiz {len(S.quiz)}", flush=True)
    info = T.train(conv3, S.loss_distill, S, lr=1e-4, label="distill")
    print(f"quiz distillation loss {info['start_quiz_e4']:.1f} -> {info['best_quiz_e4']:.1f} after {info['rounds']} rounds", flush=True)
    result, kc = T.evaluate(conv3, S)

    res = HERE / "results"
    (res / "").mkdir(exist_ok=True)
    (HERE / "weights").mkdir(exist_ok=True)
    torch.save(dict(state=conv3.state_dict(), info=dict(stage1=info)), HERE / "weights" / "conv3.pt")
    kc.to_csv(res / "kitti_cars.csv", index=False)
    (res / "training_log.json").write_text(json.dumps(info.pop("log"), indent=1))
    prev = json.loads((HERE.parent / "results" / "summary.json").read_text())
    ref = prev["mid_error"]["held_out_40"]
    h = result["mid_error"]["held_out_40"]
    result["reference_held_out_40"] = dict(full=ref["full"], decoder_removed_direct=ref["light"])
    result["training"] = info
    (res / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    md = ["# Decoder removed only, Conv3 taught by distillation", "",
          "Flow + affine is kept. Conv3 copies the full model's scale map on the 140 KITTI training pairs. No KITTI truth is used in training.", "",
          "Mid error on the 40 held-out KITTI images (lower is better):", "", "| | Mid error |", "|---|---|",
          f"| Full FP-TTC | {ref['full']:.1f} |",
          f"| Decoder removed, Conv3 trained on KITTI truth (removing_decoder_only) | {ref['light']:.1f} |",
          f"| **Decoder removed, Conv3 distilled (this folder)** | **{h['all']:.1f}** |", "",
          f"Cars only {h['cars']:.1f}, background only {h['background']:.1f}. Other 160 images: {result['mid_error']['other_160']['all']:.1f}; all 200: {result['mid_error']['all_200']['all']:.1f}.", ""]
    for k, lab in (("held_out_40", "KITTI cars, held-out 40"), ("all_200", "KITTI cars, all 200")):
        c = result["kitti_cars"][k]
        md.append(f"- {lab}: scale error {c['scale_err_pct']:.2f}%, TTC error (middle) {c['ttc_med']:.2f} s, "
                  f"cars under 3 s found {100 * c['found']:.0f}%, precision {100 * c['precision']:.0f}%")
    md += ["", "Speed and size are those of the decoder-removed model (same architecture, only Conv3's weights differ): see `../results/summary.md`."]
    (res / "summary.md").write_text("\n".join(md) + "\n")
    print("Mid error held-out 40:", h, flush=True)


if __name__ == "__main__":
    main()
