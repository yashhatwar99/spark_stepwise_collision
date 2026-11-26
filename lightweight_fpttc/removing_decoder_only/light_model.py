"""
Lightweight FP-TTC, first version: everything stays as in the full FP-TTC (same
backbone, same flow + affine stage, same scale encoder, same released weights,
all frozen) and only the decoder is replaced, as the paper describes for its
light-weight version (Section III-C):

    full:   SF_t -> matching-clue attention -> 2 recurrent decoder blocks -> upsample
    light:  eta_t = Conv3(Concat(SF_t, x_t))      Conv3 = 3 convolution layers

x_t is the image feature map of the later frame (the finest level, 128 numbers on
the 80 x 160 grid). The paper does not say how the result is brought back to full
size, so Conv3's output (at 1/4 size) is stretched x4 with a plain bilinear resize.

Nothing in fpttc/ is edited. The scale encoder's output (SF_t) and the features are
copied with a forward hook; in the light path the hook also stops the model right
after the scale encoder, so the matching-clue attention and the decoder really are
skipped (this is what makes it faster).
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                 # the project root
sys.path.insert(0, str(ROOT / "fpttc_results"))

import common as C  # noqa: E402  (loads FP-TTC from fpttc/)

SIZE = (640, 320)                 # model input size (width, height)
STRIDE = 4                        # SF_t and the finest features are 1/4 of the input size
CH = 128
WEIGHTS_FILE = HERE / "weights" / "light_decoder.pt"


class Stop(Exception):
    """Raised by the hook to end the forward pass right after the scale encoder."""


class LightDecoder(nn.Module):
    """Conv3: three 3x3 convolution layers on Concat(SF_t, x_t); gives log s at 1/4 size, then x4 bilinear."""

    def __init__(self, ch=CH, mid1=128, mid2=64):
        super().__init__()
        self.register_buffer("mean", torch.zeros(1, 2 * ch, 1, 1))
        self.register_buffer("std", torch.ones(1, 2 * ch, 1, 1))
        self.net = nn.Sequential(
            nn.Conv2d(2 * ch, mid1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid1, mid2, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mid2, 1, 3, padding=1))
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)                 # start at s = 1 (nothing changes)

    def log_scale(self, sf, ft):
        """sf, ft: (B, 128, h, w). Returns log s at full input size (B, 1, 4h, 4w)."""
        x = (torch.cat([sf, ft], 1).float() - self.mean) / self.std
        return F.interpolate(self.net(x), scale_factor=STRIDE, mode="bilinear", align_corners=False)

    def forward(self, sf, ft):
        return torch.exp(self.log_scale(sf, ft))


def save_decoder(dec, info=None, path=WEIGHTS_FILE):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=dec.state_dict(), weights="fpttc_kitti", size=SIZE, info=info or {}), path)


def load_decoder(path=WEIGHTS_FILE, device="cuda"):
    ck = torch.load(path, map_location=device, weights_only=False)
    dec = LightDecoder().to(device)
    dec.load_state_dict(ck["state"])
    return dec.eval(), ck


def prepare(rgb0, rgb1, size=SIZE, device="cuda"):
    """Same input preparation as common.run_pair: resize, to float tensors (the model normalises inside)."""
    import cv2
    w, h = size
    a = torch.from_numpy(cv2.resize(rgb0, (w, h))).permute(2, 0, 1).float().unsqueeze(0).to(device)
    b = torch.from_numpy(cv2.resize(rgb1, (w, h))).permute(2, 0, 1).float().unsqueeze(0).to(device)
    return a, b


class Runner:
    """Runs the (frozen) full model and copies what the light decoder needs with a hook.

      full(rgb0, rgb1)   the whole model runs. Returns (scale map as numpy (H, W), SF_t, x_t).
      light(rgb0, rgb1)  the model stops after the scale encoder; Conv3 makes the scale map.
                         Returns the scale map as a tensor (1, 1, H, W).
    """

    def __init__(self, model, decoder=None):
        self.model, self.decoder = model, decoder
        self._cap, self._stop = {}, False
        model.scalenet.encoder.register_forward_hook(self._hook)

    def _hook(self, module, args, out):
        self._cap = dict(sf=out, ft=args[1][-1])          # args = (feature0_list, feature1_list, query_lvl)
        if self._stop:
            raise Stop

    @torch.no_grad()
    def forward_full(self, a, b):
        """a, b: prepared frames (see prepare). The whole model runs. Returns the scale map tensor (1, 1, H, W)."""
        self._stop = False
        return self.model(a, b, **C.RUN_KW)[0]

    @torch.no_grad()
    def forward_light(self, a, b):
        """The model stops after the scale encoder; Conv3 makes the scale map tensor (1, 1, H, W)."""
        self._stop = True
        try:
            self.model(a, b, **C.RUN_KW)
        except Stop:
            pass
        finally:
            self._stop = False
        return self.decoder(self._cap["sf"], self._cap["ft"])

    @torch.no_grad()
    def full(self, rgb0, rgb1, size=SIZE):
        a, b = prepare(rgb0, rgb1, size)
        scale = self.forward_full(a, b)
        return scale[0, 0].cpu().numpy(), self._cap["sf"][0], self._cap["ft"][0]

    @torch.no_grad()
    def light(self, rgb0, rgb1, size=SIZE):
        a, b = prepare(rgb0, rgb1, size)
        return self.forward_light(a, b)

    @torch.no_grad()
    def light_from_captured(self):
        """Light scale map (numpy) from the SF_t and x_t of the latest full() run, without running anything again."""
        return self.decoder(self._cap["sf"], self._cap["ft"])[0, 0].cpu().numpy()


def count(module):
    return sum(p.numel() for p in module.parameters())


def parameter_report(model, decoder):
    """Parameters of the full model and of the light one (same model without the decoder side, plus Conv3)."""
    g = {"backbone CNN": count(model.cnet), "backbone transformer": count(model.featnet),
         "flow + affine (matching)": count(model.corrnet) + count(model.conv_corr),
         "scale encoder": count(model.scalenet.encoder),
         "matching-clue attention": count(model.scalenet.gma), "decoder + upsample": count(model.scalenet.decoder)}
    full = sum(g.values())
    assert full == count(model), "parameter groups do not add up to the model"
    removed = g["matching-clue attention"] + g["decoder + upsample"]
    conv3 = count(decoder)
    return dict(groups=g, full=full, removed=removed, conv3=conv3, light=full - removed + conv3)
