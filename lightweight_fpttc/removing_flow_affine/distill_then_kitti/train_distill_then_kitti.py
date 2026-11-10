"""
Version A (flow + affine removed), Conv3 taught in two stages:
  stage 1  distillation: copy the full model's scale map at every pixel (no KITTI truth)
  stage 2  then KITTI truth: continue from the stage 1 weights with the paper's loss (distance to KITTI's true log s,
           where KITTI has it)
All on the 140 KITTI training pairs; the 20 quiz pairs pick the best round of each stage; the 40 held-out pairs are used
only for the final Mid error.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine/distill_then_kitti/train_distill_then_kitti.py
Writes weights/conv3.pt, results/summary.json, results/summary.md, results/kitti_cars.csv in this folder.
Pictures: python lightweight_fpttc/removing_flow_affine/compare_outputs_versions.py --conv3 distill_then_kitti
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import conv3_training as T  # noqa: E402

S = T.Setup()
conv3 = S.conv3_from_part1()
print(f"stage 1, distillation: {len(S.train)} KITTI pairs", flush=True)
i1 = T.train(conv3, S.loss_distill, S, lr=1e-4, label="stage 1")
mid1, _ = T.evaluate(conv3, S)
print(f"after stage 1: Mid error held-out 40 = {mid1['mid_error']['held_out_40']['all']:.1f}", flush=True)
print("stage 2, KITTI truth", flush=True)
i2 = T.train(conv3, S.loss_truth, S, lr=1e-4, label="stage 2")
print(f"quiz truth loss {i2['start_quiz_e4']:.1f} -> {i2['best_quiz_e4']:.1f} after {i2['rounds']} rounds", flush=True)
result, kc = T.evaluate(conv3, S)
result["after_stage_1_held_out_40"] = mid1["mid_error"]["held_out_40"]
T.write_outputs(HERE, "distillation, then KITTI truth", "Version A, Conv3 trained by distillation first, then on KITTI truth",
                "Stage 1: copy the full model's scale map. Stage 2: continue with KITTI's true scale. "
                f"Mid error after stage 1 alone: {mid1['mid_error']['held_out_40']['all']:.1f}.",
                result, kc, dict(stage1=i1, stage2=i2), conv3.state_dict())
print("Mid error held-out 40:", result["mid_error"]["held_out_40"], flush=True)
