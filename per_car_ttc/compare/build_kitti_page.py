"""
Builds the KITTI results page for the two per-car methods (Median, Learned head),
only from their outputs on the 200 KITTI training pairs:

  per_car_ttc/median/output/kitti/all_cars.csv
  per_car_ttc/learned_head/output/kitti/all_cars.csv

Every number on the page is computed from those two tables.

Writes per_car_ttc/compare/kitti_page.html (a page body for the Artifact tool).

Usage (from the project root):  python per_car_ttc/compare/build_kitti_page.py
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MED_DIR = ROOT / "per_car_ttc/median/output/kitti"
HEAD_DIR = ROOT / "per_car_ttc/learned_head/output/kitti"

M = pd.read_csv(MED_DIR / "all_cars.csv")
Hd = pd.read_csv(HEAD_DIR / "all_cars.csv")
df = Hd.merge(M[["image", "box", "predicted_ttc_s"]], on=["image", "box"])
assert len(df) == len(M) == len(Hd)
assert np.allclose(df.median_ttc_s.fillna(-1), df.predicted_ttc_s.fillna(-1)), "the two tables disagree on the median method"
df["held_out"] = df.image % 5 == 0
COL = {"median": "median_ttc_s", "head": "head_ttc_s"}
NAMES = {"median": "Median", "head": "Learned head"}


def metrics(d, col):
    e = d[d.has_truth]
    both = e[e[col].notna() & e.true_ttc_s.notna()]
    err = (both[col] - both.true_ttc_s).abs()
    pos_true, pos_pred = e.true_ttc_s < 3, e[col] < 3
    tp, fp, fn = int((pos_true & pos_pred).sum()), int((~pos_true & pos_pred).sum()), int((pos_true & ~pos_pred).sum())
    ct, cp = e.true_ttc_s.notna(), e[col].notna()
    return dict(cars=len(d), truth=len(e), both=len(both), med=float(err.median()), p90=float(err.quantile(0.9)),
                within03=float((err <= 0.3).mean()), within1=float((err <= 1.0).mean()),
                true_d=int(pos_true.sum()), found=tp, false_alarm=fp, missed=fn,
                recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                both_closing=int((ct & cp).sum()), both_not=int((~ct & ~cp).sum()),
                false_closing=int((~ct & cp).sum()), missed_closing=int((ct & ~cp).sum()))


ALL = {k: metrics(df, c) for k, c in COL.items()}
HELD = {k: metrics(df[df.held_out], c) for k, c in COL.items()}
SEEN = {k: metrics(df[~df.held_out], c) for k, c in COL.items()}

# ----------------------------------------------------------------- charts
CSS = """
  /* Layout: one reading column; the answer first, then bars, a scatter plot and the pictures. */
  :root {
    --bg: #f4f6f9; --surface: #ffffff; --ink: #1b2231; --muted: #586277; --line: #c8cfdb; --grid: #e3e8f0;
    --median: #0e7468; --median-bg: #e1f2ef; --head: #2557a7; --head-bg: #e3ecfa; --bad-bg: #fbe4e1;
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
      --median: #4fd1c1; --median-bg: #12363a; --head: #82b4f2; --head-bg: #172e4d; --bad-bg: #4a2323; color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
    --median: #4fd1c1; --median-bg: #12363a; --head: #82b4f2; --head-bg: #172e4d; --bad-bg: #4a2323; color-scheme: dark;
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
  h3 { font-size: 15px; font-weight: 600; margin: 0; }
  p { margin: 0; }
  .lead { color: var(--muted); max-width: 64ch; }
  section { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .sub { color: var(--muted); font-size: 13.5px; max-width: 80ch; }
  .verdict { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; display: flex; flex-direction: column; gap: 8px; }
  .m { color: var(--median); font-weight: 600; } .h { color: var(--head); font-weight: 600; }
  figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 10px; }
  .chartwrap { overflow-x: auto; }
  svg.chart { display: block; width: 100%; min-width: 520px; height: auto; }
  svg.chart text { font-family: var(--sans); }
  .cl { font-size: 12.5px; fill: var(--ink); } .cs { font-size: 11px; fill: var(--muted); }
  .cv { font-size: 12.5px; font-weight: 600; fill: var(--ink); font-variant-numeric: tabular-nums; }
  .gh { font-size: 12px; font-weight: 600; fill: var(--muted); }
  .axis { stroke: var(--line); stroke-width: 1; } .gridl { stroke: var(--grid); stroke-width: 1; }
  .b-median { fill: var(--median); } .b-head { fill: var(--head); }
  .dot-median { fill: var(--median); fill-opacity: .55; } .dot-head { fill: var(--head); fill-opacity: .55; }
  .perfect { stroke: var(--muted); stroke-width: 1.2; stroke-dasharray: 5 4; }
  figcaption { margin-top: 6px; font-size: 12.5px; color: var(--muted); }
  .twin { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 14px; }
  .twin svg.chart { min-width: 0; }
  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; min-width: 600px; font-size: 13.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  td.n, th.n { text-align: right; }
  tr:last-child td { border-bottom: 0; }
  td:first-child { font-weight: 600; white-space: nowrap; }
  tr.g td { background: var(--bg); font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  .dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
  .d-median { background: var(--median); } .d-head { background: var(--head); }
  .example { display: flex; flex-direction: column; gap: 8px; }
  .example img { width: 100%; height: auto; border: 1px solid var(--line); border-radius: 8px; display: block; }
  .tag { font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; }
  .cm { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 16px; display: flex; flex-direction: column; gap: 10px; min-width: 0; }
  .cm-title { font-weight: 600; }
  .cm-grid { display: grid; grid-template-columns: minmax(90px, 1.1fr) 1fr 1fr; gap: 6px; align-items: stretch; }
  .cm-h { font: 500 11px var(--mono); letter-spacing: .03em; text-transform: uppercase; color: var(--muted); text-align: center; align-self: end; }
  .cm-r { font-size: 12.5px; color: var(--muted); align-self: center; }
  .cell { display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 2px; padding: 16px 6px; border-radius: 8px; border: 1px solid var(--line); }
  .cell b { font-size: 24px; font-weight: 600; font-variant-numeric: tabular-nums; }
  .cell span { font-size: 11.5px; color: var(--muted); }
  .ok-median { background: var(--median-bg); } .ok-head { background: var(--head-bg); } .bad { background: var(--bad-bg); }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; max-width: 84ch; }
  footer { font-size: 12.5px; color: var(--muted); max-width: 90ch; }
  code { font-family: var(--mono); font-size: .9em; }
  @media (max-width: 520px) { h1 { font-size: 26px; } }
"""


def f2(x):
    return f"{x:.2f}"


def pct(x):
    return f"{100 * x:.0f}%"


def bars(rows, vmax, fmt, label, width=760, row_h=34, label_w=210):
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


def scatter(col, cls, title, lim=12.0, size=380, pad=44):
    d = df[df.has_truth & df.true_ttc_s.notna() & df[col].notna()]
    inside = d[(d.true_ttc_s <= lim) & (d[col] <= lim)]
    off = int(((d.true_ttc_s <= lim) & (d[col] > lim)).sum())
    inner = size - pad - 14

    def sx(v):
        return pad + v / lim * inner

    def sy(v):
        return size - pad - v / lim * inner

    out = [f'<svg class="chart" viewBox="0 0 {size} {size}" role="img" aria-label="{escape(title)}">']
    for t in range(0, int(lim) + 1, 3):
        out.append(f'<line class="gridl" x1="{sx(t):.1f}" y1="{sy(0):.1f}" x2="{sx(t):.1f}" y2="{sy(lim):.1f}"/>')
        out.append(f'<line class="gridl" x1="{sx(0):.1f}" y1="{sy(t):.1f}" x2="{sx(lim):.1f}" y2="{sy(t):.1f}"/>')
        out.append(f'<text class="cs" x="{sx(t):.1f}" y="{size - pad + 16}" text-anchor="middle">{t}</text>')
        out.append(f'<text class="cs" x="{pad - 8}" y="{sy(t) + 4:.1f}" text-anchor="end">{t}</text>')
    out.append(f'<line class="axis" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(lim)}" y2="{sy(0)}"/>')
    out.append(f'<line class="axis" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(0)}" y2="{sy(lim)}"/>')
    out.append(f'<line class="perfect" x1="{sx(0)}" y1="{sy(0)}" x2="{sx(lim)}" y2="{sy(lim)}"/>')
    for t, p in zip(inside.true_ttc_s, inside[col]):
        out.append(f'<circle class="dot-{cls}" cx="{sx(t):.1f}" cy="{sy(p):.1f}" r="2.6"/>')
    out.append(f'<text class="gh" x="{pad}" y="14">{escape(title)}</text>')
    out.append(f'<text class="cs" x="{(pad + size) / 2:.0f}" y="{size - 8}" text-anchor="middle">true TTC (s)</text>')
    out.append(f'<text class="cs" x="12" y="{size / 2:.0f}" text-anchor="middle" transform="rotate(-90 12 {size / 2:.0f})">predicted TTC (s)</text>')
    out.append("</svg>")
    return "".join(out), len(inside), off


def row(name, cls, m):
    return (f'<tr><td><span class="dot d-{cls}"></span>{escape(name)}</td><td class="n">{f2(m["med"])} s</td>'
            f'<td class="n">{pct(m["within03"])}</td><td class="n">{pct(m["within1"])}</td>'
            f'<td class="n">{pct(m["recall"])}</td><td class="n">{pct(m["precision"])}</td><td class="n">{m["both"]}</td></tr>')


HEAD_ROW = ('<thead><tr><th>Method</th><th class="n">TTC error, middle</th>'
            '<th class="n">Within 0.3 s</th><th class="n">Within 1 s</th><th class="n">Cars under 3 s found</th>'
            '<th class="n">Precision</th><th class="n">Cars compared</th></tr></thead>')

c_all = bars([(f"{NAMES[k]}, all images", ALL[k]["med"], k, "") for k in ("median", "head")] +
             [(f"{NAMES[k]}, held-out images", HELD[k]["med"], k, "") for k in ("median", "head")], 1.0,
             lambda v: f"{v:.2f} s", "Middle TTC error in seconds on KITTI, lower is better")

groups = []
for lo, hi, label in [(0, 1.5, "true TTC under 1.5 s"), (1.5, 3, "1.5 to 3 s"), (3, 6, "3 to 6 s"), (6, 12, "6 to 12 s")]:
    sel = df[df.has_truth & df.true_ttc_s.notna() & df.median_ttc_s.notna() & df.head_ttc_s.notna() & (df.true_ttc_s >= lo) & (df.true_ttc_s < hi)]
    groups.append((f"{label}  ({len(sel)} cars)",
                   [(NAMES[k], float((sel[COL[k]] - sel.true_ttc_s).abs().median()), k) for k in ("median", "head")]))
vmax_range = max(v for _, items in groups for _, v, _ in items) * 1.12
c_range = grouped(groups, vmax_range, "Middle TTC error by true TTC range on KITTI")

sc_m, n_m, off_m = scatter("median_ttc_s", "median", "Median")
sc_h, n_h, off_h = scatter("head_ttc_s", "head", "Learned head")

n_cars, n_truth = len(df), int(df.has_truth.sum())
LONG_WRONG = int((df[df.has_truth & df.true_ttc_s.isna() & df.head_ttc_s.notna()].head_ttc_s > 30).sum())
held_cars = int(df.held_out.sum())


def matrix(name, cls, m):
    """2 x 2 grid: what KITTI says (rows) against what the method says (columns)."""
    app_total = m["both_closing"] + m["missed_closing"]
    not_total = m["false_closing"] + m["both_not"]

    def cell(n, row_total, good):
        share = f"{100 * n / max(row_total, 1):.0f}% of this row"
        return f'<div class="cell {"ok-" + cls if good else "bad"}"><b>{n}</b><span>{share}</span></div>'

    return f"""
      <div class="cm">
        <div class="cm-title"><span class="dot d-{cls}"></span>{escape(name)}</div>
        <div class="cm-grid">
          <div></div><div class="cm-h">method says approaching</div><div class="cm-h">method says not</div>
          <div class="cm-r">KITTI says approaching ({app_total} cars)</div>{cell(m["both_closing"], app_total, True)}{cell(m["missed_closing"], app_total, False)}
          <div class="cm-r">KITTI says not approaching ({not_total} cars)</div>{cell(m["false_closing"], not_total, False)}{cell(m["both_not"], not_total, True)}
        </div>
      </div>"""


page = f"""<title>KITTI Per-car Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>

<main>
  <header>
    <div class="eyebrow">BTP · single-camera collision prediction</div>
    <h1>KITTI Per-car Results</h1>
    <p class="lead">The Median method and the Learned head, run on all 200 KITTI training pairs and compared with the true TTC from KITTI's measured depth. Lower error is better.</p>
  </header>

  <div class="verdict">
    <p><b>The median is closer to the true TTC.</b> Over all images it is off by <span class="m">{f2(ALL['median']['med'])} s</span> in the middle, against
    <span class="h">{f2(ALL['head']['med'])} s</span> for the learned head. On the 40 held-out images, which neither the model nor the head learned from,
    it is <span class="m">{f2(HELD['median']['med'])} s</span> against <span class="h">{f2(HELD['head']['med'])} s</span>.</p>
    <p>The median also finds more of the really dangerous cars (true TTC under 3 s): {pct(ALL['median']['recall'])} against {pct(ALL['head']['recall'])} over all images, and {pct(HELD['median']['recall'])} against {pct(HELD['head']['recall'])} on the held-out ones.</p>
  </div>

  <section>
    <h2>How far from the true TTC</h2>
    <p class="sub">{n_cars} cars found by YOLO in the 200 pairs. {n_truth} have a true answer from KITTI's depth (at least 50 measured pixels inside the box). The error is how many seconds the method's TTC is away from the true one, for cars where both have a TTC.</p>
    <figure><div class="chartwrap">{c_all}</div><figcaption>Middle error: half the cars are closer than this to the true TTC.</figcaption></figure>
    <div class="tablewrap"><table>{HEAD_ROW}<tbody>
      <tr class="g"><td colspan="7">All 200 images</td></tr>
      {row('Median', 'median', ALL['median'])}{row('Learned head', 'head', ALL['head'])}
      <tr class="g"><td colspan="7">The 40 held-out images ({held_cars} cars found)</td></tr>
      {row('Median', 'median', HELD['median'])}{row('Learned head', 'head', HELD['head'])}
      <tr class="g"><td colspan="7">The other 160 images (both learned from them)</td></tr>
      {row('Median', 'median', SEEN['median'])}{row('Learned head', 'head', SEEN['head'])}
    </tbody></table></div>
    <p class="sub">"Cars under 3 s found" counts the cars whose true TTC is under 3 s that the method also called dangerous (our own 3 s line).</p>
  </section>

  <section>
    <h2>Near collisions and far ones</h2>
    <p class="sub">The same cars, grouped by their true TTC. A short TTC is the case that matters for a warning.</p>
    <figure><div class="chartwrap">{c_range}</div><figcaption>Middle TTC error in seconds. Shorter bar is better.</figcaption></figure>
  </section>

  <section>
    <h2>Predicted against true</h2>
    <p class="sub">Each dot is one car. A dot on the dashed line has the right TTC. Cars with a true TTC above 12 s are left out, and so are the few that a method put above 12 s ({off_m} for the median, {off_h} for the head).</p>
    <div class="twin">
      <figure>{sc_m}<figcaption>Median: {n_m} cars.</figcaption></figure>
      <figure>{sc_h}<figcaption>Learned head: {n_h} cars.</figcaption></figure>
    </div>
  </section>

  <section>
    <h2>Is the car approaching at all?</h2>
    <p class="sub">KITTI says some cars are not getting closer. Counting the {n_truth} cars with a true answer:</p>
    <div class="twin">{matrix('Median', 'median', ALL['median'])}{matrix('Learned head', 'head', ALL['head'])}</div>
    <p class="sub">The head wrongly calls {ALL['head']['false_closing']} cars approaching (the median {ALL['median']['false_closing']}) and misses {ALL['head']['missed_closing']} that KITTI says are approaching (the median {ALL['median']['missed_closing']}). {LONG_WRONG} of the head's {ALL['head']['false_closing']} wrong "approaching" calls come with a TTC over 30 s, which a warning system would ignore anyway.</p>
  </section>

  <section>
    <h2>Read these numbers with care</h2>
    <ul>
      <li><b>Boxes come from YOLO,</b> not from KITTI's car outlines. A box can include road or a second car, so the true TTC inside it is a median of what KITTI measured there, not one car's exact value.</li>
      <li><b>Only cars with enough measured pixels have a truth.</b> Far and small cars often do not, so they are left out of the error numbers.</li>
      <li><b>FP-TTC learned from 160 of the 200 pairs.</b> The 40 held-out images (numbers divisible by 5) are the fair test. The head learned from 140 of the other 160 images and used the other 20 to pick its best round.</li>
      <li><b>TTC is the paper's formula,</b> Δt ÷ (1 − s) with Δt = 0.1 s, and assumes constant speed.</li>
      <li><b>No crashes are involved.</b> KITTI is normal driving, so the true TTC is a calculation from measured depth, not a recorded collision.</li>
      <li><b>nuScenes is not on this page.</b> The head fell further behind there.</li>
    </ul>
  </section>

  <footer>Every number is computed from <code>per_car_ttc/median/output/kitti/all_cars.csv</code> and <code>per_car_ttc/learned_head/output/kitti/all_cars.csv</code>. This page is rebuilt by <code>per_car_ttc/compare/build_kitti_page.py</code>.</footer>
</main>
"""

import sys
sys.path.insert(0, str(HERE.parents[1] / "page_tools"))
from add_edit_mode import add_edit_mode  # "Edit text" button on the page

out = HERE / "kitti_page.html"
out.write_text(add_edit_mode(page), encoding="utf8")
print("wrote", out, f"{len(page) / 1024:.0f} KB")
print("ALL", {k: (round(v['med'], 2), round(v['recall'], 2), round(v['precision'], 2)) for k, v in ALL.items()})
print("HELD", {k: (round(v['med'], 2), round(v['recall'], 2), round(v['precision'], 2)) for k, v in HELD.items()})
print("agree", {k: (v['both_closing'], v['both_not'], v['false_closing'], v['missed_closing']) for k, v in ALL.items()})
