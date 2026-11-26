"""
Does the scale encoder's output (SF_t) depend on its query at all?

Compares SF_t from the full model (query made by the flow + affine stage; saved by the first light version)
with SF_t from the same encoder when the query is made from the image features, set to zero, or random noise.
Run on 10 KITTI images, features from the cache of train_no_flow.py.

Usage (from the project root, with the project .venv):
    python lightweight_fpttc/removing_flow_affine_decoder/query_matters.py
"""

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "removing_decoder_only"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402

C = L.C
dev = "cuda"
F_ = torch.load(HERE / "cache" / "kitti_features.pt", weights_only=False)
d = torch.load(HERE.parent / "removing_decoder_only" / "cache" / "kitti_light.pt", weights_only=False)
model = C.load_model("fpttc_kitti")
enc = model.scalenet.encoder.eval()


def rel(a, b):
    """mean |a - b| divided by the mean |b|, in percent"""
    return 100 * float((a - b).abs().mean() / b.abs().mean())


rows = {k: [] for k in ("image features of t-1 (query 'one')", "zero query", "random query")}
with torch.no_grad():
    for i in range(10):
        m0 = [x[i:i + 1].to(dev).float() for x in F_["f0"]]
        m1 = [x[i:i + 1].to(dev).float() for x in F_["f1"]]
        ref = d["sf"][i:i + 1].to(dev).float()                                   # SF_t of the full model
        rows["image features of t-1 (query 'one')"].append(rel(enc(m0, m1, -1, ini_query=None), ref))
        z = torch.zeros(1, 128, 80, 160, device=dev)
        rows["zero query"].append(rel(enc(m0, m1, -1, ini_query=z), ref))
        torch.manual_seed(i)
        rows["random query"].append(rel(enc(m0, m1, -1, ini_query=torch.randn_like(z)), ref))
print("difference of SF_t from the full model's SF_t, mean over 10 images (percent of the size of SF_t):")
for k, v in rows.items():
    print(f"  {k:40s} {sum(v) / len(v):6.1f} %")
print("SF_t spread (std over the image):", float(d["sf"][0].float().std()))
