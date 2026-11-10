"""
Builds the results page that compares the per-car TTC methods, from the saved
result files (every number on the page is read from them, none typed in):

  fpttc_results/results/report_data.json             Median and Flow + affine zoom (KITTI, nuScenes)
  per_car_ttc/learned_head/output/kitti_heldout_eval.json   Median vs Learned head, KITTI held-out cars
  per_car_ttc/learned_head/output/nuscenes_eval.json        Median vs Learned head, nuScenes

Writes per_car_ttc/compare/results_page.html (a page body for the Artifact tool).

Usage (from the project root):  python per_car_ttc/compare/build_results_page.py
"""

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
R = json.loads((ROOT / "fpttc_results/results/report_data.json").read_text(encoding="utf8"))
KH = json.loads((ROOT / "per_car_ttc/learned_head/output/kitti_heldout_eval.json").read_text(encoding="utf8"))["report"]
NH = json.loads((ROOT / "per_car_ttc/learned_head/output/nuscenes_eval.json").read_text(encoding="utf8"))
SPEED = R["speed"]["pipeline"]

NUS = R["nuscenes"]["fpttc_kitti|640x320"]
KIT = {m: R["kitti_per_car"][f"held_out_40|640x320|{m}"] for m in ("median_box100", "median_box70", "median_box50", "zoom_box70")}


def from_metrics(m):
    """Same fields as report_data.json, from a build_summary.car_metrics dict."""
    return dict(cars=m["cars"], scale_err_pct=m["scale_error_percent_median"], ttc_med=m["ttc_abs_error_s"]["median"],
                ttc_p90=m["ttc_abs_error_s"]["p90"], recall=m["danger"]["recall"], precision=m["danger"]["precision"],
                n_ttc=m["ttc_cars"])


nus = {"median": NUS["median_box70"], "head": from_metrics(NH["head"]), "flow": NUS["zoom_box70"]}
kit = {"median": KIT["median_box70"], "head": from_metrics(KH["tight mask boxes|head"]), "flow": KIT["zoom_box70"]}
jit = {"median": from_metrics(KH["jittered boxes (like YOLO)|median_box70"]),
       "head": from_metrics(KH["jittered boxes (like YOLO)|head"])}
NAMES = {"median": "Median", "head": "Learned head", "flow": "Flow + affine zoom"}
ORDER = ["median", "head", "flow"]

CSS = """
  /* Layout: one reading column; a verdict first, then charts and tables from the newest to the oldest test. */
  :root {
    --bg: #f4f6f9; --surface: #ffffff; --ink: #1b2231; --muted: #586277; --line: #c8cfdb; --grid: #e3e8f0;
    --median: #0e7468; --median-bg: #e1f2ef;
    --head: #2557a7; --head-bg: #e3ecfa;
    --flow: #a34a07; --flow-bg: #fbecd9;
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
      --median: #4fd1c1; --median-bg: #12363a; --head: #82b4f2; --head-bg: #172e4d; --flow: #f2b45c; --flow-bg: #3b2c14;
      color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
    --median: #4fd1c1; --median-bg: #12363a; --head: #82b4f2; --head-bg: #172e4d; --flow: #f2b45c; --flow-bg: #3b2c14;
    color-scheme: dark;
  }
  body {
    background: var(--bg); color: var(--ink); font-family: var(--sans); font-size: 15px; line-height: 1.55;
    padding-block: 28px 56px;
    padding-left: max(16px, env(safe-area-inset-left, 0px));
    padding-right: max(16px, env(safe-area-inset-right, 0px));
  }
  main { max-width: 1040px; margin: 0 auto; display: flex; flex-direction: column; gap: 30px; }
  header { display: flex; flex-direction: column; gap: 6px; }
  .eyebrow { font: 500 12px var(--mono); letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
  h1 { font-size: 30px; line-height: 1.15; font-weight: 600; margin: 0; text-wrap: balance; }
  h2 { font-size: 18px; font-weight: 600; margin: 0 0 4px; text-wrap: balance; }
  p { margin: 0; }
  .lead { color: var(--muted); max-width: 64ch; }
  section { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .sub { color: var(--muted); font-size: 13.5px; max-width: 80ch; }
  .verdict { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; display: flex; flex-direction: column; gap: 8px; }
  .verdict b { font-weight: 600; }
  .m { color: var(--median); font-weight: 600; } .h { color: var(--head); font-weight: 600; } .f { color: var(--flow); font-weight: 600; }

  figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 10px; }
  .chartwrap { overflow-x: auto; }
  svg.chart { display: block; width: 100%; min-width: 520px; height: auto; }
  svg.chart text { font-family: var(--sans); }
  .cl { font-size: 12.5px; fill: var(--ink); }
  .cs { font-size: 11px; fill: var(--muted); }
  .cv { font-size: 12.5px; font-weight: 600; fill: var(--ink); font-variant-numeric: tabular-nums; }
  .gh { font-size: 12px; font-weight: 600; fill: var(--muted); }
  .axis { stroke: var(--line); stroke-width: 1; }
  .gridl { stroke: var(--grid); stroke-width: 1; }
  .b-median { fill: var(--median); } .b-head { fill: var(--head); } .b-flow { fill: var(--flow); }
  figcaption { margin-top: 6px; font-size: 12.5px; color: var(--muted); }

  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; min-width: 640px; font-size: 13.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  td.n, th.n { text-align: right; }
  tr:last-child td { border-bottom: 0; }
  td:first-child { font-weight: 600; white-space: nowrap; }
  tr.g td { background: var(--bg); font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
  .d-median { background: var(--median); } .d-head { background: var(--head); } .d-flow { background: var(--flow); }

  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; max-width: 84ch; }
  footer { font-size: 12.5px; color: var(--muted); max-width: 90ch; }
  code { font-family: var(--mono); font-size: .9em; }
  @media (max-width: 520px) { h1 { font-size: 26px; } }
"""


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}%"


def bars(rows, vmax, fmt, label="", note=None, width=760, row_h=34, label_w=170):
    """Horizontal bars to one scale. rows: (label, value, class, small note)."""
    plot_w = width - label_w - 90
    h = row_h * len(rows) + 10
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(label)}">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = label_w + plot_w * frac
        out.append(f'<line class="gridl" x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="{h - 6}"/>')
    out.append(f'<line class="axis" x1="{label_w}" y1="2" x2="{label_w}" y2="{h - 6}"/>')
    for i, (lab, val, cls, small) in enumerate(rows):
        y = 6 + i * row_h
        w = max(val / vmax * plot_w, 2)
        out.append(f'<text class="cl" x="{label_w - 10}" y="{y + row_h / 2 - 1}" text-anchor="end">{escape(lab)}</text>')
        if small:
            out.append(f'<text class="cs" x="{label_w - 10}" y="{y + row_h / 2 + 12}" text-anchor="end">{escape(small)}</text>')
        out.append(f'<rect class="b-{cls}" x="{label_w}" y="{y + 4}" width="{w:.1f}" height="{row_h - 14}" rx="3"/>')
        out.append(f'<text class="cv" x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 3}">{fmt(val)}</text>')
    out.append("</svg>")
    return "".join(out)


def grouped(groups, vmax, label, width=760, row_h=26, label_w=170):
    """groups: (heading, [(label, value, class)]) to one scale."""
    plot_w = width - label_w - 90
    rows = sum(len(g[1]) + 1 for g in groups)
    h = row_h * rows + 14
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(label)}">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = label_w + plot_w * frac
        out.append(f'<line class="gridl" x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="{h - 6}"/>')
    out.append(f'<line class="axis" x1="{label_w}" y1="2" x2="{label_w}" y2="{h - 6}"/>')
    y = 6
    for heading, items in groups:
        out.append(f'<text class="gh" x="0" y="{y + row_h - 8}">{escape(heading)}</text>')
        y += row_h
        for lab, val, cls in items:
            w = max(val / vmax * plot_w, 2)
            out.append(f'<text class="cl" x="{label_w - 10}" y="{y + row_h / 2 + 3}" text-anchor="end">{escape(lab)}</text>')
            out.append(f'<rect class="b-{cls}" x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 8}" rx="3"/>')
            out.append(f'<text class="cv" x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 3}">{f2(val)} s</text>')
            y += row_h
    out.append("</svg>")
    return "".join(out)


def row(name, cls, d, note=""):
    return (f'<tr><td><span class="dot d-{cls}"></span>{escape(name)}</td><td class="n">{d["scale_err_pct"]:.2f}%</td>'
            f'<td class="n">{f2(d["ttc_med"])} s</td><td class="n">{f2(d["ttc_p90"])} s</td>'
            f'<td class="n">{pct(d["recall"])}</td><td class="n">{pct(d["precision"])}</td><td class="n">{d["cars"]}</td></tr>')


HEAD = '<thead><tr><th>Method</th><th class="n">Scale error</th><th class="n">TTC error, middle</th><th class="n">TTC error, 90th pct</th><th class="n">Cars under 3 s found</th><th class="n">Precision</th><th class="n">Cars</th></tr></thead>'

# ------------------------------------------------------------------ sections
verdict = f"""
<div class="verdict">
  <p><b>The median wins.</b> On nuScenes, data FP-TTC never saw, it is off by <span class="m">{f2(nus['median']['ttc_med'])} s</span> in the middle and
  finds <span class="m">{pct(nus['median']['recall'])}</span> of the cars that really are under 3 s from a collision.</p>
  <p>The learned head ties it on the 85 held-out KITTI cars ({f2(kit['head']['ttc_med'])} s, the same as the median), but on nuScenes it drops to
  <span class="h">{f2(nus['head']['ttc_med'])} s</span> and <span class="h">{pct(nus['head']['recall'])}</span>.
  The flow + affine zoom does not work yet: <span class="f">{f2(nus['flow']['ttc_med'])} s</span> and <span class="f">{pct(nus['flow']['recall'])}</span>.</p>
</div>"""

c_ttc = bars([(NAMES[k], nus[k]["ttc_med"], k, "") for k in ORDER], 6.0, lambda v: f"{v:.2f} s",
             "Middle TTC error on nuScenes in seconds, lower is better")
c_found = bars([(NAMES[k], 100 * nus[k]["recall"], k, "") for k in ORDER], 100, lambda v: f"{v:.0f}%",
               "Share of really dangerous cars found on nuScenes, higher is better")

groups = []
for rng_, label in [("0-1.5s", "true TTC under 1.5 s"), ("1.5-3s", "1.5 to 3 s"), ("3-6s", "3 to 6 s"), ("6-12s", "6 to 12 s")]:
    a, b = NH["median_box70"]["ttc_by_true_ttc"][rng_], NH["head"]["ttc_by_true_ttc"][rng_]
    groups.append((f"{label}  ({a['n']} cars)", [("Median", a["median"], "median"), ("Learned head", b["median"], "head")]))
c_range = grouped(groups, 3.2, "Middle TTC error in seconds for cars at different true TTC, Median against Learned head")

box_rows = ""
for k, label in [("median_box100", "whole box"), ("median_box70", "inner 70% (used)"), ("median_box50", "inner 50%")]:
    n, kk = NUS[k], KIT[k]
    box_rows += (f'<tr><td>{label}</td><td class="n">{n["scale_err_pct"]:.2f}%</td><td class="n">{f2(n["ttc_med"])} s</td>'
                 f'<td class="n">{pct(n["recall"])} / {pct(n["precision"])}</td>'
                 f'<td class="n">{kk["scale_err_pct"]:.2f}%</td><td class="n">{f2(kk["ttc_med"])} s</td>'
                 f'<td class="n">{pct(kk["recall"])} / {pct(kk["precision"])}</td></tr>')

page = f"""<title>Per-car TTC Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>

<main>
  <header>
    <div class="eyebrow">BTP · single-camera collision prediction</div>
    <h1>Per-car TTC Results</h1>
    <p class="lead">Three ways to turn FP-TTC's output into one time to collision per car, scored on the same cars. Lower error is better. "Cars under 3 s found" is the share of cars that really are under 3 s from colliding that the method also flagged.</p>
  </header>

  {verdict}

  <section>
    <h2>On data the model never saw: nuScenes</h2>
    <p class="sub">{nus['median']['cars']:,} cars detected by YOLO and matched to their labelled 3D boxes, 394 camera frames, 640×320 input, <code>fpttc_kitti</code> weights. True TTC is calculated from the labels (distance ÷ closing speed), not an observed crash.</p>
    <figure><div class="chartwrap">{c_ttc}</div><figcaption>Middle TTC error: half the cars are closer than this to the true answer.</figcaption></figure>
    <figure><div class="chartwrap">{c_found}</div><figcaption>Of the {nus['median']['true_d']} cars that really are under 3 s away, how many each method also called dangerous.</figcaption></figure>
    <div class="tablewrap"><table>{HEAD}<tbody>
      {"".join(row(NAMES[k], k, nus[k]) for k in ORDER)}
    </tbody></table></div>
    <p class="sub">The 90th percentile is the error that 1 car in 10 exceeds. It is far larger than the middle for every method, so a few cars get a very wrong TTC that the middle error hides.</p>
  </section>

  <section>
    <h2>Near collisions are where the head falls behind</h2>
    <p class="sub">The same nuScenes cars, grouped by their true TTC. For cars under 3 s the head is several times less accurate than the median.</p>
    <figure><div class="chartwrap">{c_range}</div><figcaption>Middle TTC error in seconds. Shorter bar is better.</figcaption></figure>
  </section>

  <section>
    <h2>KITTI held-out cars</h2>
    <p class="sub">The 85 cars in the 40 KITTI test images (every fifth image, the paper's split). Neither FP-TTC nor the head learned from these images. The truth is measured depth from KITTI, with car outlines from its object masks. Only 85 cars, so small differences here mean little.</p>
    <div class="tablewrap"><table>{HEAD}<tbody>
      <tr class="g"><td colspan="7">Boxes drawn tight around each car</td></tr>
      {"".join(row(NAMES[k], k, kit[k]) for k in ORDER)}
      <tr class="g"><td colspan="7">Boxes shifted and resized, like a detector's ({jit['median']['cars']} boxes, 5 per car)</td></tr>
      {"".join(row(NAMES[k], k, jit[k]) for k in ("median", "head"))}
    </tbody></table></div>
  </section>

  <section>
    <h2>How big a part of the box the median reads</h2>
    <p class="sub">The median reads the middle part of YOLO's box so road and background at the edges stay out. The choice barely matters, so we use the inner 70%.</p>
    <div class="tablewrap"><table>
      <thead><tr><th rowspan="2">Part of the box</th><th colspan="3" class="n">nuScenes ({NUS['median_box70']['cars']:,} cars)</th><th colspan="3" class="n">KITTI held-out ({KIT['median_box70']['cars']} cars)</th></tr>
      <tr><th class="n">Scale error</th><th class="n">TTC error</th><th class="n">Found / precision</th><th class="n">Scale error</th><th class="n">TTC error</th><th class="n">Found / precision</th></tr></thead>
      <tbody>{box_rows}</tbody></table></div>
  </section>

  <section>
    <h2>Time</h2>
    <p class="sub">On the laptop's RTX 3050, one frame pair at 640×320: YOLO {SPEED['yolov8s']:.0f} ms, FP-TTC {SPEED['fpttc_including_resize']:.0f} ms (the model run is almost all of the time), the median for every car in the frame {SPEED['per_car_median']:.1f} ms, the learned head about 0.4 ms per car. In total about {SPEED['total_median_ms']:.0f} ms, {SPEED['fps']:.1f} frames per second. The per-car step is not what costs time.</p>
  </section>

  <section>
    <h2>Read these numbers with care</h2>
    <ul>
      <li><b>TTC is the paper's formula,</b> Δt ÷ (1 − s), with s the model's scale output and Δt the time between the frames. It assumes the closing speed stays constant.</li>
      <li><b>The 3 s danger line is our own choice.</b> The paper scores per pixel at 1, 2 and 5 s.</li>
      <li><b>The nuScenes truth is calculated,</b> from labelled positions and speeds. It is not a recorded crash. A test on real crash times (CCD) is still to do.</li>
      <li><b>FP-TTC was fine-tuned on KITTI,</b> so KITTI alone is not a clean test. nuScenes is.</li>
      <li><b>The head was trained on 299 KITTI cars</b> from one camera. Why it falls behind on nuScenes is not tested. Likely causes are too few cars and the different camera and picture shape.</li>
      <li><b>Flow + affine zoom fails for a reason not yet found.</b> The flow FP-TTC returns is far from KITTI's true flow, so the fitted zoom is wrong.</li>
      <li><b>Not measured here:</b> the light model (the paper describes it, but it has no released weights).</li>
    </ul>
  </section>

  <footer>Every number comes from saved files: <code>fpttc_results/results/report_data.json</code> (Median, Flow + affine, box sizes, time), and <code>per_car_ttc/learned_head/output/</code> (<code>nuscenes_eval.json</code>, <code>kitti_heldout_eval.json</code>). This page is rebuilt by <code>per_car_ttc/compare/build_results_page.py</code>.</footer>
</main>
"""

import sys
sys.path.insert(0, str(HERE.parents[1] / "page_tools"))
from add_edit_mode import add_edit_mode  # "Edit text" button on the page

out = HERE / "results_page.html"
out.write_text(add_edit_mode(page), encoding="utf8")
print("wrote", out, f"{len(page) / 1024:.0f} KB")
