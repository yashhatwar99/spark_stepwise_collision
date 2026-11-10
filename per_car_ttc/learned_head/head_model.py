"""
Method 2 building blocks: read FP-TTC's scale encoder output (SF_t) and turn the
part inside a car's box into one scale s with a small trained head.

    frames -> FP-TTC (frozen) -> SF_t (128 x H/4 x W/4)
           -> crop the car's box out of SF_t (RoIAlign, 7 x 7 x 128)
           -> small head -> log s            then   TTC = dt / (1 - s)

FP-TTC is not edited: SF_t is copied with a forward hook on
model.scalenet.encoder. The decoder still runs in this first version; its
output is simply not used for the per-car number.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.ops import roi_align

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                  # the project root
sys.path.insert(0, str(ROOT / "fpttc_results"))

import common as C  # noqa: E402  (loads FP-TTC from fpttc/)

STRIDE = 4                       # SF_t is 1/4 of the model input size
CROP = 7                         # every car is resampled to a 7 x 7 grid
CHANNELS = 128
SIZE = (640, 320)                # the model input size the head is trained for
WEIGHTS_FILE = HERE / "weights" / "learned_head.pt"


class FeatureReader:
    """Run FP-TTC on a frame pair and also return SF_t, captured by a hook."""

    def __init__(self, model):
        self.model = model
        self._feat = None
        model.scalenet.encoder.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._feat = output

    def __call__(self, rgb0, rgb1, size=SIZE):
        """rgb0 earlier, rgb1 later (uint8 RGB). Returns (scale map HxW numpy, SF_t of shape (128, H/4, W/4))."""
        scale, _ = C.run_pair(self.model, rgb0, rgb1, size=size)
        return scale, self._feat[0]


def crop(feat, boxes):
    """feat (C, h, w); boxes (N, 4) = x1, y1, x2, y2 in MODEL-INPUT pixels (same device as feat).
    Returns (N, C, 7, 7): each box resampled to a fixed grid."""
    return roi_align(feat[None].float(), [boxes.float()], output_size=CROP,
                     spatial_scale=1.0 / STRIDE, sampling_ratio=2, aligned=True)


class Head(nn.Module):
    """A few small layers: 7x7x128 crop -> log s."""

    def __init__(self, mid=32, hidden=64, drop=0.2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, CHANNELS, 1, 1))
        self.register_buffer("std", torch.ones(1, CHANNELS, 1, 1))
        self.reduce = nn.Sequential(nn.Conv2d(CHANNELS, mid, 1), nn.ReLU())
        self.mlp = nn.Sequential(nn.Flatten(), nn.Dropout(drop), nn.Linear(mid * CROP * CROP, hidden),
                                 nn.ReLU(), nn.Dropout(drop), nn.Linear(hidden, 1))
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.constant_(self.mlp[-1].bias, math.log(0.95))     # start from a typical car

    def forward(self, crops):
        x = (crops - self.mean) / self.std
        return self.mlp(self.reduce(x)).squeeze(1)               # log s


def save_head(head, path=WEIGHTS_FILE, info=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=head.state_dict(), weights="fpttc_kitti", size=SIZE, info=info or {}), path)


def load_head(path=WEIGHTS_FILE, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    head = Head().to(device)
    head.load_state_dict(ck["state"])
    return head.eval(), ck


@torch.no_grad()
def predict_s(head, feat, boxes):
    """feat (C, h, w) on the model's device; boxes (N, 4) in model-input pixels. Returns s as a numpy array."""
    boxes = torch.as_tensor(boxes, dtype=torch.float32, device=feat.device).reshape(-1, 4)
    return torch.exp(head(crop(feat, boxes))).cpu().numpy()
