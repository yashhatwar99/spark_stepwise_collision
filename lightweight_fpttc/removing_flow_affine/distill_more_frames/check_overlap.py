"""
Does the raw drive (data/kitti/.../drive_0009_sync, image_02) contain frames that are the same pictures as any of
KITTI scene flow's 200 training scenes? If a raw frame is (nearly) the same picture as a held-out or quiz scene, it must
not be used for training. Compares small grayscale thumbnails of every raw frame with frames 10 and 11 of every scene.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine/distill_more_frames/check_overlap.py
Writes results/overlap_check.json.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "fpttc_results"))
import eval_kitti as K  # noqa: E402

RAW = ROOT / "data" / "kitti" / "2011_09_26" / "2011_09_26_drive_0009_sync" / "image_02" / "data"


def thumb(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (124, 37), interpolation=cv2.INTER_AREA).astype(np.float32)


kt = K.Kitti()
scene = []
for i in range(200):
    for k in (10, 11):
        scene.append((i, k, thumb(kt.image(i, k))))
files = sorted(RAW.glob("*.png"))
print(f"{len(files)} raw frames, {len(scene)} scene frames (200 scenes x frames 10, 11)")
S = np.stack([t for _, _, t in scene])
best = []
for f in files:
    t = thumb(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB))
    d = np.abs(S - t).mean((1, 2))
    j = int(d.argmin())
    best.append((float(d[j]), f.name, scene[j][0], scene[j][1]))
best.sort()
print("closest raw frame to any scene frame (mean gray difference, 0 = identical):")
for b in best[:8]:
    print(f"  diff {b[0]:.2f}  raw {b[1]}  scene {b[2]:06d}_{b[3]}")
d_all = np.array([b[0] for b in best])
print(f"smallest {d_all.min():.2f}, median {np.median(d_all):.1f}, frames with diff < 3: {int((d_all < 3).sum())}")
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "overlap_check.json").write_text(json.dumps(dict(raw_frames=len(files), closest=best[:20], frames_below_3=int((d_all < 3).sum())), indent=1))
