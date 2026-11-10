"""
The four-panel picture every per-car method makes when the true TTC is known
(a KITTI pair or a nuScenes frame):

  top left      the method's TTC on each car
  top right     the true TTC on each car
  bottom left   the per-pixel TTC map with the method's boxes and labels
  bottom right  "predicted | true" on each car; green = close, orange = off, red = far off

A car is a dict with box (x1, y1, x2, y2 in the later frame's pixels), ttc_s (None = not
approaching) and, once add_truth() has run, has_truth, true_ttc_s and error_s.
"""

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ttc_utils import C, draw, ttc_map  # noqa: E402

GREEN, ORANGE, RED, GREY = (60, 190, 60), (0, 150, 255), (40, 40, 235), (150, 150, 150)     # BGR


def add_truth(cars, truths, key="ttc_s", error_key="error_s"):
    """Attach the true TTC and how far off the prediction (cars[i][key]) is."""
    for c, t in zip(cars, truths):
        c["has_truth"] = t["has_truth"]
        c["true_ttc_s"] = round(t["true_ttc_s"], 2) if t["true_ttc_s"] is not None else None
        c["error_s"] = None
        if t["has_truth"] and c[key] is not None and c["true_ttc_s"] is not None:
            c["error_s"] = round(c[key] - c["true_ttc_s"], 2)


def closeness_color(pred, true):
    """Green: close to the true TTC. Orange: off. Red: far off."""
    if pred is None and true is None:
        return GREEN
    if pred is None or true is None:
        return RED
    d = abs(pred - true)
    if d <= 0.3 or d <= 0.15 * true:
        return GREEN
    if d <= 1.0 or d <= 0.4 * true:
        return ORANGE
    return RED


def _fmt(t):
    return "not closing" if t is None else (f"{t:.1f}" if t <= 99 else ">99")


def true_labelled(cars):
    out = []
    for c in cars:
        if not c["has_truth"]:
            out.append(dict(box=c["box"], ttc_s=None, label="no truth", color=GREY))
        else:
            out.append(dict(box=c["box"], ttc_s=c["true_ttc_s"]))
    return out


def together_labelled(cars):
    out = []
    for c in cars:
        if not c["has_truth"]:
            out.append(dict(box=c["box"], ttc_s=None, label=f"{_fmt(c['ttc_s'])} | no truth", color=GREY))
        else:
            suffix = " s" if c["true_ttc_s"] is not None else ""
            out.append(dict(box=c["box"], ttc_s=None, label=f"{_fmt(c['ttc_s'])} | {_fmt(c['true_ttc_s'])}{suffix}",
                            color=closeness_color(c["ttc_s"], c["true_ttc_s"])))
    return out


def ttc_panel(ax, curr_bgr, scale, cars, dt, frac=None):
    """The per-pixel TTC map with each car's box and label (and the inner box that is read, if frac is given)."""
    H, W = curr_bgr.shape[:2]
    # display only: the map is stretched back to the photo's size; the car
    # numbers themselves come from the model-size map
    ttc_full = ttc_map(cv2.resize(scale, (W, H)), dt)
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad((0.6, 0.6, 0.6))                       # grey = not approaching
    im = ax.imshow(np.ma.masked_invalid(np.where(np.isfinite(ttc_full), ttc_full, np.nan)), cmap=cmap, vmin=0, vmax=10)
    for c in cars:
        x1, y1, x2, y2 = c["box"]
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=1.0))
        if frac is not None:
            ix1, iy1, ix2, iy2 = C.inner_box((x1, y1, x2, y2), frac)
            ax.add_patch(plt.Rectangle((ix1, iy1), ix2 - ix1, iy2 - iy1, fill=False, ec="white", lw=1.2, ls="--"))
        label = "not closing" if c["ttc_s"] is None else (f"{c['ttc_s']:.1f} s" if c["ttc_s"] <= 99 else ">99 s")
        ax.text(x1, y1 - 3, label, color="white", fontsize=8, va="bottom",
                bbox=dict(fc="black", ec="none", alpha=0.65, pad=1.5))
    return im


def four_panels(curr_bgr, scale, cars, dt, path, where, titles, frac=None):
    """titles: dict with keys a, b, c, d (the four panel titles, in the order of the list above)."""
    H, W = curr_bgr.shape[:2]
    fig, ax = plt.subplots(2, 2, figsize=(17, 2 * (8.0 * H / W) + 1.6))
    ax[0, 0].imshow(cv2.cvtColor(draw(curr_bgr, cars), cv2.COLOR_BGR2RGB))
    ax[0, 0].set_title(titles["a"])
    ax[0, 1].imshow(cv2.cvtColor(draw(curr_bgr, true_labelled(cars)), cv2.COLOR_BGR2RGB))
    ax[0, 1].set_title(titles["b"])
    im = ttc_panel(ax[1, 0], curr_bgr, scale, cars, dt, frac)
    ax[1, 0].set_title(titles["c"])
    ax[1, 1].imshow(cv2.cvtColor(draw(curr_bgr, together_labelled(cars), font=0.55, stagger=True), cv2.COLOR_BGR2RGB))
    ax[1, 1].set_title(titles["d"])
    for a in ax.ravel():
        a.axis("off")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.025, pad=0.01, label="seconds")
    fig.suptitle(f"{where}, dt = {dt:.2f} s", fontsize=11)
    fig.savefig(path, dpi=100, bbox_inches="tight", pil_kwargs={"quality": 90})
    plt.close(fig)
