"""
Lightweight FP-TTC, second version: remove flow + affine (the matching stage).

Full FP-TTC:    images -> CNN -> transformer -> [flow + affine] -> query -> scale encoder -> decoder -> s
This version:   images -> CNN -> transformer ------------------> query -> scale encoder -> Conv3    -> s
                                                      (query made from the image features instead)

What was found in the code (fpttc/fpttc/scale_net/feature_net/feature_net.py):
  FeatureNet computes flow_wrap(feature1, flow) but never uses it in the path we run (testing=False,
  both levels), so the transformer features are the SAME with or without the flow stage. The flow stage
  only feeds (a) the query of the scale encoder, through conv_corr, and (b) the decoder side.
  So removing it changes the query and nothing else; the "features are identical" check is in
  train_no_flow.py.

The query (what the scale encoder asks about every position of the 80 x 160 grid):
  "one"  query = scale_conv(F_t-1)             the encoder's own built-in path for "no flow given"
  "two"  query = scale_conv(Q(F_t-1, F_t))     Q = one 3x3 conv on Concat(F_t-1, F_t), as the paper's light model;
                                               it starts as "pass F_t-1 through", so it starts equal to "one"
scale_conv was trained on flow-like numbers, so the scale encoder is fine-tuned together with the new layer.

Nothing in fpttc/ is edited. The scale encoder is deep-copied; the CNN and the transformer stay frozen.
"""

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "removing_decoder_only"))

import light_model as L  # noqa: E402

C = L.C
CH = L.CH
WEIGHTS = HERE / "weights"


@torch.no_grad()
def features(model, a, b):
    """The CNN and the transformer, exactly as in FpTTC.forward but with the flow stage left out.
    a, b: prepared frames (L.prepare). Returns two lists (one per level, coarse to fine) of (1, 128, h, w)."""
    mod = sys.modules[type(model).__module__]
    img0, img1 = mod.normalize_img(a, b)
    f0, f1 = model.extract_feature(img0, img1)
    m0, m1 = [], []
    for k in range(model.num_scales):
        x0, x1 = model.featnet(f0[k], f1[k], k, C.RUN_KW["attn_type"], C.RUN_KW["attn_splits_list"], None)
        m0.append(x0)
        m1.append(x1)
    return m0, m1


class TwoFrameQuery(nn.Module):
    """One 3x3 conv on Concat(F_t-1, F_t) -> 128 channels. Starts as 'F_t-1 unchanged' (identity on the first half)."""

    def __init__(self, ch=CH):
        super().__init__()
        self.conv = nn.Conv2d(2 * ch, ch, 3, padding=1)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            for i in range(ch):
                self.conv.weight[i, i, 1, 1] = 1.0

    def forward(self, f0, f1):
        return self.conv(torch.cat([f0, f1], 1))


class NoFlowHead(nn.Module):
    """Everything after the transformer in the version without flow + affine: query, scale encoder, Conv3."""

    def __init__(self, encoder, mode="two", conv3=None):
        super().__init__()
        assert mode in ("one", "two")
        self.mode = mode
        self.encoder = copy.deepcopy(encoder)             # the released weights, to be fine-tuned
        self.query = TwoFrameQuery() if mode == "two" else None
        self.conv3 = conv3 if conv3 is not None else L.LightDecoder()

    def sf(self, m0, m1):
        ini = self.query(m0[-1], m1[-1]) if self.query is not None else None
        return self.encoder(m0, m1, -1, ini_query=ini)

    def log_scale(self, m0, m1):
        """log s at the model input size (1, 1, 320, 640)."""
        return self.conv3.log_scale(self.sf(m0, m1), m1[-1])

    def forward(self, m0, m1):
        return torch.exp(self.log_scale(m0, m1))


class ConstantQuery(nn.Module):
    """Stands in for scale_conv in version B: returns a stored, fixed query (no parameters, no input used)."""

    def __init__(self, q):
        super().__init__()
        self.register_buffer("q", q)

    def forward(self, x):
        return self.q.clone()          # the encoder adds the position code to it in place, so never hand out the buffer


VERSIONS = {
    "A": "only flow + affine removed: the query is made from the image features by the encoder's own scale_conv (kept)",
    "B": "flow + affine and scale_conv removed: the query is one fixed constant (what scale_conv gives for an all-zero input)",
}


@torch.no_grad()
def build_head(model, version, conv3_state=None, device="cuda"):
    """Version A or B head. The scale encoder keeps the released weights. conv3_state: weights of Conv3 (default: first light version)."""
    assert version in VERSIONS
    conv3 = L.LightDecoder().to(device)
    if conv3_state is None:
        conv3.load_state_dict(L.load_decoder(device=device)[0].state_dict())
    else:
        conv3.load_state_dict(conv3_state)
    head = NoFlowHead(model.scalenet.encoder, "one", conv3).to(device).eval()
    if version == "B":
        enc = head.encoder
        q = enc.scale_conv(torch.zeros(1, CH, 80, 160, device=device))
        enc.scale_conv = ConstantQuery(q)
    return head


def part1_conv3(device="cuda"):
    """The Conv3 trained in the first light version (decoder removed only), or None if it is not there."""
    if not L.WEIGHTS_FILE.exists():
        return None
    return L.load_decoder(device=device)[0]


def save_head(head, name, info=None):
    WEIGHTS.mkdir(exist_ok=True)
    torch.save(dict(state=head.state_dict(), mode=head.mode, info=info or {}), WEIGHTS / f"{name}.pt")


def load_head(model, name, device="cuda"):
    ck = torch.load(WEIGHTS / f"{name}.pt", map_location=device, weights_only=False)
    head = NoFlowHead(model.scalenet.encoder, ck["mode"]).to(device)
    head.load_state_dict(ck["state"])
    return head.eval(), ck


def count(module):
    return sum(p.numel() for p in module.parameters())


def parameter_report(model, head):
    """Parameters of the full model and of this light version (CNN + transformer + scale encoder + query + Conv3)."""
    kept = count(model.cnet) + count(model.featnet) + count(head.encoder)
    new = (count(head.query) if head.query is not None else 0) + count(head.conv3)
    removed = (count(model.corrnet) + count(model.conv_corr) + count(model.scalenet.gma) + count(model.scalenet.decoder))
    return dict(full=count(model), kept=kept, new=new, removed=removed, light=kept + new,
                flow_affine=count(model.corrnet) + count(model.conv_corr))
