"""
Chart images for the Canva presentation (dark background, transparent, white text), made from the saved results:

  fpttc_results/results/report_data.json      full model speed, median and flow + affine results
  per_car_ttc/median/output/kitti/all_cars.csv, per_car_ttc/learned_head/output/kitti/all_cars.csv
  per_car_ttc/learned_head/output/nuscenes_eval.json
  lightweight_fpttc/removing_decoder/results/summary.json

Usage (from the project root):  python presentation/make_slide_images.py
Writes presentation/images/*.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "images"
OUT.mkdir(exist_ok=True)

R = json.loads((ROOT / "fpttc_results/results/report_data.json").read_text(encoding="utf8"))
NH = json.loads((ROOT / "per_car_ttc/learned_head/output/nuscenes_eval.json").read_text(encoding="utf8"))
LS = json.loads((ROOT / "lightweight_fpttc/removing_decoder/results/summary.json").read_text(encoding="utf8"))

WHITE, MUTED = "#f2f2f2", "#9aa3b5"
BEIGE, BLUE, ORANGE, GREEN, GREY = "#d9c3a0", "#4a82cf", "#e08a3c", "#6fc08f", "#6b7690"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": WHITE, "axes.labelcolor": WHITE,
                     "xtick.color": MUTED, "ytick.color": WHITE, "axes.edgecolor": "#3a4356", "font.size": 13})


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", name)


def style(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_facecolor("none")
    if grid:
        ax.grid(axis="x", color="#2a3242", lw=0.8)
        ax.set_axisbelow(True)


def hbars(ax, labels, values, colors, fmt, xmax=None, fs=13):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs)
    xm = xmax or max(values) * 1.22
    ax.set_xlim(0, xm)
    for yi, v in zip(y, values):
        ax.text(v + xm * 0.015, yi, fmt(v), va="center", fontsize=fs, fontweight="bold")
    style(ax)


# ---------------------------------------------------------------- A: where the time goes (full model)
mods = R["speed"]["by_module"]
rows = [(k, v["pct"]) for k, v in mods.items() if not k.startswith("(")]
rows.sort(key=lambda r: -r[1])
names = {"backbone CNN": "Backbone CNN", "backbone transformer": "Backbone transformer", "flow / matching": "Flow matching",
         "initial scale (conv_corr)": "Initial scale", "scale encoder": "Scale encoder",
         "matching-clue attention": "Matching-clue attention", "decoder + upsample": "Decoder + upsample"}
fig, ax = plt.subplots(figsize=(8.2, 4.6))
hbars(ax, [names[k] for k, _ in rows], [p for _, p in rows],
      [ORANGE if k in ("decoder + upsample", "matching-clue attention") else BEIGE for k, _ in rows], lambda v: f"{v:.0f}%", xmax=62)
ax.set_xlabel("share of the model's run time", color=MUTED)
save(fig, "a_time_by_part.png")

# ---------------------------------------------------------------- B: ground truth table
fig, ax = plt.subplots(figsize=(8.6, 4.2))
ax.axis("off")
cells = [["Data", "What it gives", "True answer", "Used for"],
         ["KITTI scene flow\n200 pairs", "depth at two times,\nmotion, car masks", "s = distance later\n÷ distance earlier", "scores and training\n40 pairs held out"],
         ["nuScenes\n394 frames, 1,418 cars", "labelled 3D boxes\nand speeds", "TTC = distance\n÷ closing speed", "test on data the\nmodel never saw"]]
tb = ax.table(cellText=cells, cellLoc="left", loc="center", colWidths=[0.25, 0.27, 0.27, 0.23])
tb.auto_set_font_size(False)
tb.set_fontsize(11)
tb.scale(1, 3.1)
for (r, c), cell in tb.get_celld().items():
    cell.set_facecolor("none")
    cell.set_edgecolor("#3a4356")
    cell.get_text().set_color(BEIGE if r == 0 else WHITE)
    cell.get_text().set_fontweight("bold" if r == 0 else "normal")
    cell.PAD = 0.04
save(fig, "b_ground_truth.png")

# ---------------------------------------------------------------- C: median method results
n = R["nuscenes"]["fpttc_kitti|640x320"]["median_box70"]
k = R["kitti_per_car"]["held_out_40|640x320|median_box70"]
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8))
hbars(axs[0], ["nuScenes\n(never seen)", "KITTI\nheld-out"], [n["ttc_med"], k["ttc_med"]], [BEIGE, BEIGE], lambda v: f"{v:.2f} s", xmax=1.3)
axs[0].set_title("TTC error (middle)", color=MUTED, fontsize=13, loc="left")
ax = axs[1]
y = np.arange(2)[::-1]
ax.barh(y + 0.18, [100 * n["recall"], 100 * k["recall"]], height=0.34, color=BEIGE)
ax.barh(y - 0.18, [100 * n["precision"], 100 * k["precision"]], height=0.34, color=BLUE)
for yi, a, b in zip(y, [n["recall"], k["recall"]], [n["precision"], k["precision"]]):
    ax.text(100 * a + 1.5, yi + 0.18, f"{100 * a:.0f}%", va="center", fontsize=12, fontweight="bold")
    ax.text(100 * b + 1.5, yi - 0.18, f"{100 * b:.0f}%", va="center", fontsize=12, fontweight="bold")
ax.set_yticks([])
ax.set_xlim(0, 118)
ax.set_title("cars under 3 s: found / precision", color=MUTED, fontsize=13, loc="left")
style(ax)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BEIGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)], labels=["found", "precision"],
          loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False, fontsize=11, labelcolor=WHITE)
fig.tight_layout()
save(fig, "c_median_results.png")

# ---------------------------------------------------------------- D: learned head against median
m = pd.read_csv(ROOT / "per_car_ttc/median/output/kitti/all_cars.csv")
h = pd.read_csv(ROOT / "per_car_ttc/learned_head/output/kitti/all_cars.csv")
d = h.merge(m[["image", "box", "predicted_ttc_s"]], on=["image", "box"])
e = d[d.has_truth & d.true_ttc_s.notna()]            # each method over the cars where it and the truth have a TTC (as on the KITTI results page)
kitti_med = float((e.median_ttc_s - e.true_ttc_s).abs().dropna().median())
kitti_head = float((e.head_ttc_s - e.true_ttc_s).abs().dropna().median())
nm, nh = n, NH["head"]
panels = [("nuScenes: TTC error (s)", nm["ttc_med"], nh["ttc_abs_error_s"]["median"], lambda v: f"{v:.2f}"),
          ("nuScenes: cars under 3 s found", 100 * nm["recall"], 100 * nh["danger"]["recall"], lambda v: f"{v:.0f}%"),
          ("KITTI: TTC error (s)", kitti_med, kitti_head, lambda v: f"{v:.2f}")]
fig, axs = plt.subplots(3, 1, figsize=(6.4, 5.4))
for ax, (t_, a_, b_, f) in zip(axs, panels):
    hbars(ax, ["Median", "Learned head"], [a_, b_], [BEIGE, BLUE], f, fs=12)
    ax.set_title(t_, color=MUTED, fontsize=12, loc="left")
fig.tight_layout()
save(fig, "d_head_vs_median.png")

# ---------------------------------------------------------------- E: the three methods on nuScenes
z = R["nuscenes"]["fpttc_kitti|640x320"]["zoom_box70"]
trio = [("Scale error (%)", [n["scale_err_pct"], nh["scale_error_percent_median"], z["scale_err_pct"]], lambda v: f"{v:.2f}%"),
        ("TTC error, middle (s)", [n["ttc_med"], nh["ttc_abs_error_s"]["median"], z["ttc_med"]], lambda v: f"{v:.2f} s"),
        ("Cars under 3 s found", [100 * n["recall"], 100 * nh["danger"]["recall"], 100 * z["recall"]], lambda v: f"{v:.0f}%")]
fig, axs = plt.subplots(3, 1, figsize=(6.2, 6.6))
for ax, (t, vals, f) in zip(axs, trio):
    hbars(ax, ["Median", "Learned head", "Flow + affine"], vals, [BEIGE, BLUE, ORANGE], f, fs=12)
    ax.set_title(t, color=MUTED, fontsize=12, loc="left")
fig.tight_layout()
save(fig, "e_three_methods.png")

# ---------------------------------------------------------------- G: parameters by part (full model)
P = LS["parameters"]
gl = list(P["groups"].items()) + [("Conv3 (new)", P["conv3"])]
lab = {"backbone CNN": "Backbone CNN", "backbone transformer": "Backbone transformer", "flow + affine (matching)": "Flow + affine",
       "scale encoder": "Scale encoder", "matching-clue attention": "Matching-clue attention", "decoder + upsample": "Decoder + upsample", "Conv3 (new)": "Conv3 (new)"}
cols = [ORANGE if k in ("matching-clue attention", "decoder + upsample") else (GREEN if k == "Conv3 (new)" else GREY) for k, _ in gl]
fig, ax = plt.subplots(figsize=(5.8, 5.2))
hbars(ax, [lab[k] for k, _ in gl], [v / 1e6 for _, v in gl], cols, lambda v: f"{v:.2f} M", xmax=7.6, fs=12)
ax.set_xlabel("parameters (millions)", color=MUTED)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (GREY, ORANGE, GREEN)], labels=["kept, frozen", "removed", "added, trained"],
          loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, frameon=False, fontsize=10, labelcolor=WHITE)
save(fig, "g_parameters_by_part.png")

print("kitti head/median:", round(kitti_head, 3), round(kitti_med, 3))
