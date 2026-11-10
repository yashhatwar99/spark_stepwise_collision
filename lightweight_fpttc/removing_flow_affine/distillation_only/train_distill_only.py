"""
Version A (flow + affine removed), Conv3 taught ONLY by the full model (distillation): no KITTI truth is used in training.

  loss = mean |log(Conv3's s) - log(full model's s)| over every pixel, on the 140 KITTI training pairs.
  The quiz pairs pick the best round by the same loss. The 40 held-out pairs are used only for the final Mid error,
  which is measured against KITTI's truth, as for every other method.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine/distillation_only/train_distill_only.py
Writes weights/conv3.pt, results/summary.json, results/summary.md, results/kitti_cars.csv in this folder.
Pictures: python lightweight_fpttc/removing_flow_affine/compare_outputs_versions.py --conv3 distillation_only
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import conv3_training as T  # noqa: E402

S = T.Setup()
conv3 = S.conv3_from_part1()
print(f"distillation only: training on {len(S.train)} KITTI pairs, quiz {len(S.quiz)}", flush=True)
info = T.train(conv3, S.loss_distill, S, lr=1e-4, label="distill")
print(f"quiz distillation loss {info['start_quiz_e4']:.1f} -> {info['best_quiz_e4']:.1f} after {info['rounds']} rounds", flush=True)
result, kc = T.evaluate(conv3, S)
T.write_outputs(HERE, "distillation only", "Version A, Conv3 trained by distillation only",
                "Conv3 copies the full model's scale map (every pixel). No KITTI truth is used in training.",
                result, kc, dict(stage1=info), conv3.state_dict())
print("Mid error held-out 40:", result["mid_error"]["held_out_40"], flush=True)
