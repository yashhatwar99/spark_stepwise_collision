"""
Follow-up to query_matters.py: if the query does not matter, the layers that make it are not needed.
Replaces the encoder's scale_conv (3 convolutions that turn the query input into the 128-channel query)
by a constant zero query, and compares SF_t with the full model's SF_t.

Usage: python lightweight_fpttc/removing_flow_affine/query_not_needed.py
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "removing_decoder"))

import no_flow_model as M  # noqa: E402
import light_model as L  # noqa: E402

C = L.C
dev = "cuda"


class ZeroQuery(nn.Module):
    """Stands in for scale_conv: a constant all-zero query (only the position code is added by the encoder)."""

    def forward(self, x):
        return torch.zeros(x.shape[0], 128, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)


F_ = torch.load(HERE / "cache" / "kitti_features.pt", weights_only=False)
d = torch.load(HERE.parent / "removing_decoder" / "cache" / "kitti_light.pt", weights_only=False)
model = C.load_model("fpttc_kitti")
enc = model.scalenet.encoder.eval()
enc2 = copy.deepcopy(enc)
enc2.scale_conv = ZeroQuery()
print("encoder parameters:", M.count(enc), " without scale_conv:", M.count(enc2))

def rel(a, b):
    return 100 * float((a - b).abs().mean() / b.abs().mean())

r = []
with torch.no_grad():
    for i in range(10):
        m0 = [x[i:i + 1].to(dev).float() for x in F_["f0"]]
        m1 = [x[i:i + 1].to(dev).float() for x in F_["f1"]]
        ref = d["sf"][i:i + 1].to(dev).float()
        r.append(rel(enc2(m0, m1, -1, ini_query=m0[-1]), ref))
print(f"SF_t without scale_conv differs from the full model's by {sum(r) / len(r):.2f} % (mean of 10 images)")
