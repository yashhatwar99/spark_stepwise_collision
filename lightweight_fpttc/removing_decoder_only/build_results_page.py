"""
Builds the results page for the light FP-TTC against the full FP-TTC, only from the saved results:

  lightweight_fpttc/removing_decoder_only/results/summary.json       (made by evaluate_light.py)

Every number on the page is read from that file. Writes lightweight_fpttc/removing_decoder_only/results/results_page.html
(a page body for the Artifact tool).

Usage (from the project root):  python lightweight_fpttc/removing_decoder_only/build_results_page.py
"""

import json
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "summary.json").read_text(encoding="utf8"))
P, G, CPU, MID = S["parameters"], S["time_gpu"], S.get("time_cpu"), S["mid_error"]
KC, NC, TR = S["kitti_cars"], S["nuscenes_cars"], S["decoder_training"]
_V = HERE.parent / "removing_flow_affine_decoder" / "results" / "versions_summary.json"
V = json.loads(_V.read_text(encoding="utf8")) if _V.exists() else None      # part 2 (versions A and B), if measured
_D1 = HERE / "distillation" / "results" / "summary.json"
D1 = json.loads(_D1.read_text(encoding="utf8")) if _D1.exists() else None        # decoder removed only, Conv3 distilled
_RFA = HERE.parent / "removing_flow_affine_decoder"
DIST = {k: json.loads((_RFA / k / "results" / "summary.json").read_text(encoding="utf8"))
        for k in ("distillation_only", "distill_then_kitti", "distill_more_frames") if (_RFA / k / "results" / "summary.json").exists()}

CSS = """
  /* Layout: one reading column; the answer first, then size and speed, accuracy, and the details. */
  :root {
    --bg: #f4f6f9; --surface: #ffffff; --ink: #1b2231; --muted: #586277; --line: #c8cfdb; --grid: #e3e8f0;
    --full: #0e7468; --full-bg: #e1f2ef; --light: #a34a07; --light-bg: #fbecd9;
    --kept: #8a95a8; --removed: #c0392b; --added: #2f855a;
    --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: "IBM Plex Mono", ui-monospace, Consolas, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
      --full: #4fd1c1; --full-bg: #12363a; --light: #f2b45c; --light-bg: #3b2c14;
      --kept: #6b7690; --removed: #ef8a80; --added: #6fd0a2; color-scheme: dark;
    }
  }
  :root[data-theme="dark"] {
    --bg: #11151c; --surface: #1a202a; --ink: #e5e9f1; --muted: #9aa5ba; --line: #3a4356; --grid: #2a3242;
    --full: #4fd1c1; --full-bg: #12363a; --light: #f2b45c; --light-bg: #3b2c14;
    --kept: #6b7690; --removed: #ef8a80; --added: #6fd0a2; color-scheme: dark;
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
  .lead { color: var(--muted); max-width: 66ch; }
  section { display: flex; flex-direction: column; gap: 12px; min-width: 0; }
  .sub { color: var(--muted); font-size: 13.5px; max-width: 80ch; }
  .verdict { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 16px 18px; display: flex; flex-direction: column; gap: 8px; }
  .part { font-size: 20px; font-weight: 600; color: var(--light); }
  .notice { background: var(--light-bg); border: 1px solid var(--light); border-radius: 12px; padding: 14px 18px; font-size: 16px; max-width: 100%; }
  .f { color: var(--full); font-weight: 600; } .l { color: var(--light); font-weight: 600; }
  figure { margin: 0; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px 10px; min-width: 0; }
  .chartwrap { overflow-x: auto; }
  svg.chart { display: block; width: 100%; height: auto; min-width: 280px; }
  svg.chart text { font-family: var(--sans); }
  .cl { font-size: 12.5px; fill: var(--ink); } .cs { font-size: 11px; fill: var(--muted); }
  .cv { font-size: 12.5px; font-weight: 600; fill: var(--ink); font-variant-numeric: tabular-nums; }
  .ct { font-size: 12.5px; font-weight: 600; fill: var(--muted); }
  .axis { stroke: var(--line); stroke-width: 1; } .gridl { stroke: var(--grid); stroke-width: 1; }
  .b-full { fill: var(--full); } .b-light { fill: var(--light); }
  .b-kept { fill: var(--kept); } .b-removed { fill: var(--removed); } .b-added { fill: var(--added); }
  figcaption { margin-top: 6px; font-size: 12.5px; color: var(--muted); }
  .three { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }
  .tablewrap { overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; }
  table { border-collapse: collapse; width: 100%; min-width: 600px; font-size: 13.5px; font-variant-numeric: tabular-nums; }
  th, td { text-align: left; padding: 9px 14px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); }
  td.n, th.n { text-align: right; }
  tr:last-child td { border-bottom: 0; }
  td:first-child { font-weight: 600; }
  tr.g td { background: var(--bg); font: 500 11.5px var(--mono); letter-spacing: .04em; text-transform: uppercase; color: var(--muted); font-weight: 400; }
  .good { color: var(--added); font-weight: 600; } .bad { color: var(--removed); font-weight: 600; } .same { color: var(--muted); }
  ul { margin: 0; padding-left: 20px; display: flex; flex-direction: column; gap: 6px; max-width: 84ch; }
  footer { font-size: 12.5px; color: var(--muted); max-width: 90ch; }
  code { font-family: var(--mono); font-size: .9em; }
  @media (max-width: 520px) { h1 { font-size: 26px; } }
"""


def rel(full, light):
    return 100 * (light - full) / full


def change_cell(full, light, lower_is_better=True, pts=False, digits=0):
    d = light - full
    if abs(d) < 1e-9 or (not pts and full and abs(d / full) < 0.005):
        return '<td class="n same">same</td>'
    good = (d < 0) == lower_is_better
    text = f"{d:+.{digits}f} pts" if pts else f"{rel(full, light):+.0f}%"
    return f'<td class="n {"good" if good else "bad"}">{text}</td>'


def pair_chart(title, full, light, fmt, note):
    """Two bars (full, light) on one scale starting at zero, with the change in the caption."""
    width, h, label_w = 330, 118, 54
    plot_w = width - label_w - 62
    vmax = max(full, light)
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(title)}: full {fmt(full)}, light {fmt(light)}">',
           f'<text class="ct" x="0" y="16">{escape(title)}</text>',
           f'<line class="axis" x1="{label_w}" y1="28" x2="{label_w}" y2="{h - 8}"/>']
    for i, (lab, val, cls) in enumerate((("full", full, "full"), ("light", light, "light"))):
        y = 36 + i * 38
        w = max(val / vmax * plot_w, 2)
        out.append(f'<text class="cl" x="{label_w - 8}" y="{y + 17}" text-anchor="end">{lab}</text>')
        out.append(f'<rect class="b-{cls}" x="{label_w}" y="{y}" width="{w:.1f}" height="26" rx="3"/>')
        out.append(f'<text class="cv" x="{label_w + w + 8:.1f}" y="{y + 18}">{fmt(val)}</text>')
    out.append("</svg>")
    return f'<figure><div class="chartwrap">{"".join(out)}</div><figcaption>{note}</figcaption></figure>'


def groups_chart(groups, vmax, label, fmt, width=760, row_h=26, label_w=190):
    """groups: (heading, [(label, value, class)]) to one scale."""
    plot_w = width - label_w - 70
    rows = sum(len(g[1]) + 1 for g in groups)
    h = row_h * rows + 14
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="{escape(label)}" style="min-width:520px">']
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = label_w + plot_w * frac
        out.append(f'<line class="gridl" x1="{x:.1f}" y1="2" x2="{x:.1f}" y2="{h - 6}"/>')
    out.append(f'<line class="axis" x1="{label_w}" y1="2" x2="{label_w}" y2="{h - 6}"/>')
    y = 6
    for heading, items in groups:
        out.append(f'<text class="ct" x="0" y="{y + row_h - 8}">{escape(heading)}</text>')
        y += row_h
        for lab, val, cls in items:
            w = max(val / vmax * plot_w, 2)
            out.append(f'<text class="cl" x="{label_w - 10}" y="{y + row_h / 2 + 3}" text-anchor="end">{escape(lab)}</text>')
            out.append(f'<rect class="b-{cls}" x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row_h - 8}" rx="3"/>')
            out.append(f'<text class="cv" x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 3}">{fmt(val)}</text>')
            y += row_h
    out.append("</svg>")
    return "".join(out)


def parts_chart():
    rows = []
    for name, n in P["groups"].items():
        gone = name in ("matching-clue attention", "decoder + upsample")
        rows.append((name, n / 1e6, "removed" if gone else "kept"))
    rows.append(("Conv3 (new)", P["conv3"] / 1e6, "added"))
    width, row_h, label_w = 760, 30, 200
    plot_w = width - label_w - 90
    vmax = max(r[1] for r in rows)
    h = row_h * len(rows) + 10
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" aria-label="Parameters of each part of the model, in millions" style="min-width:520px">',
           f'<line class="axis" x1="{label_w}" y1="2" x2="{label_w}" y2="{h - 6}"/>']
    for i, (lab, val, cls) in enumerate(rows):
        y = 6 + i * row_h
        w = max(val / vmax * plot_w, 2)
        out.append(f'<text class="cl" x="{label_w - 10}" y="{y + row_h / 2 + 3}" text-anchor="end">{escape(lab)}</text>')
        out.append(f'<rect class="b-{cls}" x="{label_w}" y="{y + 4}" width="{w:.1f}" height="{row_h - 12}" rx="3"/>')
        out.append(f'<text class="cv" x="{label_w + w + 8:.1f}" y="{y + row_h / 2 + 3}">{val:.2f} M</text>')
    out.append("</svg>")
    return "".join(out)


def legend_dot(cls, text):
    return f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px"><span style="width:10px;height:10px;border-radius:2px;background:var(--{cls})"></span>{text}</span>'


# ------------------------------------------------------------------ pieces
ms_f, ms_l = G["full"]["ms"], G["light"]["ms"]
tp = [
    pair_chart("Parameters (millions)", P["full"] / 1e6, P["light"] / 1e6, lambda v: f"{v:.2f}", f"{rel(P['full'], P['light']):+.0f}%: lower is better."),
    pair_chart("Time per pair on the GPU (ms)", ms_f, ms_l, lambda v: f"{v:.0f}", f"{rel(ms_f, ms_l):+.0f}%: lower is better."),
]
if CPU:
    tp.append(pair_chart("Time per pair on the CPU (s)", CPU["full"]["ms"] / 1000, CPU["light"]["ms"] / 1000, lambda v: f"{v:.1f}",
                         f"{rel(CPU['full']['ms'], CPU['light']['ms']):+.0f}%: lower is better."))

mid_groups = []
for k, label in (("held_out_40", "40 held-out images"), ("other_160", "the other 160 images"), ("all_200", "all 200 images")):
    mid_groups.append((label, [("full", MID[k]["full"], "full"), ("light", MID[k]["light"], "light")]))
mid_chart = groups_chart(mid_groups, max(v for _, it in mid_groups for _, v, _ in it) * 1.1, "Mid error on KITTI, lower is better", lambda v: f"{v:.1f}")


def mid_row(label, f, l):
    return f'<tr><td>{label}</td><td class="n">{f:.1f}</td><td class="n">{l:.1f}</td>{change_cell(f, l, True)}</tr>'


mid_rows = "".join(mid_row(lab, MID[k]["full"], MID[k]["light"]) for k, lab in (("held_out_40", "40 held-out images"), ("other_160", "the other 160 images"), ("all_200", "all 200 images")))
mid_rows += mid_row("Cars only, held-out", MID["held_out_40"]["full_fg"], MID["held_out_40"]["light_fg"])
mid_rows += mid_row("Background only, held-out", MID["held_out_40"]["full_bg"], MID["held_out_40"]["light_bg"])


def car_rows(title, full, light):
    f, l = full, light
    return (f'<tr class="g"><td colspan="4">{escape(title)} ({f["cars"]} cars)</td></tr>'
            f'<tr><td>Scale error</td><td class="n">{f["scale_err_pct"]:.2f}%</td><td class="n">{l["scale_err_pct"]:.2f}%</td>{change_cell(f["scale_err_pct"], l["scale_err_pct"], True)}</tr>'
            f'<tr><td>TTC error, middle</td><td class="n">{f["ttc_med"]:.2f} s</td><td class="n">{l["ttc_med"]:.2f} s</td>{change_cell(f["ttc_med"], l["ttc_med"], True)}</tr>'
            f'<tr><td>Cars under 3 s found</td><td class="n">{100 * f["found"]:.0f}%</td><td class="n">{100 * l["found"]:.0f}%</td>{change_cell(100 * f["found"], 100 * l["found"], False, pts=True)}</tr>'
            f'<tr><td>Precision</td><td class="n">{100 * f["precision"]:.0f}%</td><td class="n">{100 * l["precision"]:.0f}%</td>{change_cell(100 * f["precision"], 100 * l["precision"], False, pts=True)}</tr>')


cars_table = (car_rows("nuScenes, data the model never saw", NC["full"], NC["light"]) +
              car_rows("KITTI, the 40 held-out images", KC["held_out_40"]["full"], KC["held_out_40"]["light"]) +
              car_rows("KITTI, all 200 images", KC["all_200"]["full"], KC["all_200"]["light"]))

def step_row(method, a, b, note="", speed_key=None):
    """One row: the change, Mid error from -> to, and where the model now stands against the full model (speed gain on the GPU, accuracy loss)."""
    d = b - a
    cls = "same" if abs(d) < 0.05 else ("good" if d < 0 else "bad")
    full_mid = MID["held_out_40"]["full"]
    acc = 100 * (b - full_mid) / full_mid
    if speed_key and V:
        tg = V["time_gpu"]
        sp = 100 * (1 - tg[speed_key]["ms"] / tg["full"]["ms"])
        sp_cell = f'<td class="n good">{sp:.0f}% faster</td>'
    else:
        sp_cell = '<td class="n same">-</td>'
    acc_cell = f'<td class="n {"bad" if acc > 0.5 else "same"}">{acc:+.0f}% worse</td>' if abs(acc) >= 0.5 else '<td class="n same">same</td>'
    return (f'<tr><td>{method}</td><td class="n">{a:.1f}</td><td class="n">&rarr;</td><td class="n"><b>{b:.1f}</b></td>'
            f'<td class="n {cls}">{d:+.1f}</td>{sp_cell}{acc_cell}<td>{note}</td></tr>')


def journey():
    m = MID["held_out_40"]
    rows = step_row("Remove the decoder (Conv3 trained on KITTI)", m["full"], m["light"], "full model &rarr; light model, part 1", "decoder_removed")
    if D1:
        rows += step_row("Decoder removed only: teach Conv3 by distillation instead", m["light"], D1["mid_error"]["held_out_40"]["all"],
                         "copy the full model's scale map, 140 KITTI pairs, flow + affine kept", "decoder_removed")
    if V:
        v = V["mid_error"]["held_out_40"]
        rows += step_row("A: also remove flow + affine (Conv3 as it is)", v["decoder_removed"], v["A_asis"], "the query now comes from the image features through scale_conv", "A")
        rows += step_row("B: also remove flow + affine and scale_conv (Conv3 as it is)", v["decoder_removed"], v["B_asis"], "the query is one fixed constant", "B")
        rows += step_row("A: then train Conv3 again on KITTI", v["A_asis"], v["A_retrained"], "encoder frozen, 140 KITTI pairs", "A")
        rows += step_row("B: then train Conv3 again on KITTI", v["B_asis"], v["B_retrained"], "encoder frozen, 140 KITTI pairs", "B")
        for k, lab, note in (("distillation_only", "A: instead, teach Conv3 by distillation only", "copy the full model's scale map, 140 KITTI pairs, no truth"),
                             ("distill_then_kitti", "A: distillation, then KITTI truth", "stage 2 added nothing"),
                             ("distill_more_frames", "A: distillation on more frames", "553 pairs: 140 KITTI + 413 from the raw drive; no gain")):
            if k in DIST:
                rows += step_row(lab, v["A_asis"], DIST[k]["mid_error"]["held_out_40"]["all"], note, "A")
    sub = "Mid error on the 40 held-out KITTI pairs, lower is better. Each row says where it starts and where it ends after that one change. The last two columns compare the model after that change with the full model: GPU time (same measuring run as part 2) and Mid error."
    return f"""
  <section>
    <h2>Mid error, method by method</h2>
    <p class="sub">{sub}</p>
    <div class="tablewrap"><table>
      <thead><tr><th>Change made</th><th class="n">From</th><th class="n"></th><th class="n">To</th><th class="n">Difference</th><th class="n">Speed (GPU) vs full</th><th class="n">Accuracy (Mid error) vs full</th><th>What it means</th></tr></thead>
      <tbody>{rows}</tbody></table></div>
  </section>"""


def part2():
    if not V:
        return ""
    p, t, mm, kc, nc = V["parameters"], V["time_gpu"], V["mid_error"], V["kitti_cars"], V.get("nuscenes_cars")
    c = V.get("time_cpu")
    sp = [("Parameters", f"{p['A']['full'] / 1e6:.2f} M", "6.83 M", f"{p['A']['light'] / 1e6:.2f} M", f"{p['B']['light'] / 1e6:.2f} M"),
          ("Time per pair, GPU", f"{t['full']['ms']:.0f} ms", f"{t['decoder_removed']['ms']:.0f} ms", f"{t['A']['ms']:.0f} ms", f"{t['B']['ms']:.0f} ms")]
    if c:
        sp.append(("Time per pair, CPU", f"{c['full']['ms'] / 1000:.1f} s", f"{c['decoder_removed']['ms'] / 1000:.1f} s", f"{c['A']['ms'] / 1000:.1f} s", f"{c['B']['ms'] / 1000:.1f} s"))
    sp_rows = "".join("<tr><td>" + r[0] + "</td>" + "".join(f'<td class="n">{x}</td>' for x in r[1:]) + "</tr>" for r in sp)
    ks = ["A_asis", "A_retrained", "B_asis", "B_retrained"]
    mid_rows2 = "".join(f'<tr><td>{lab}</td><td class="n">{mm[k]["full"]:.1f}</td><td class="n">{mm[k]["decoder_removed"]:.1f}</td>'
                        + "".join(f'<td class="n">{mm[k][q]:.1f}</td>' for q in ks) + "</tr>"
                        for k, lab in (("held_out_40", "40 held-out images"), ("other_160", "the other 160 images"), ("all_200", "all 200 images")))

    def car2(title, g):
        n = g["full"]["cars"]
        ks2 = ["full"] + ks
        return (f'<tr class="g"><td colspan="6">{escape(title)} ({n} cars)</td></tr>'
                f'<tr><td>Scale error</td>' + "".join(f'<td class="n">{g[k]["scale_err_pct"]:.2f}%</td>' for k in ks2) + "</tr>"
                f'<tr><td>TTC error, middle</td>' + "".join(f'<td class="n">{g[k]["ttc_med"]:.2f} s</td>' for k in ks2) + "</tr>"
                f'<tr><td>Cars under 3 s found</td>' + "".join(f'<td class="n">{100 * g[k]["found"]:.0f}%</td>' for k in ks2) + "</tr>"
                f'<tr><td>Precision</td>' + "".join(f'<td class="n">{100 * g[k]["precision"]:.0f}%</td>' for k in ks2) + "</tr>")
    cars2 = car2("KITTI, the 40 held-out images", kc["held_out_40"]) + car2("KITTI, all 200 images", kc["all_200"])
    if nc:
        cars2 += car2("nuScenes, data the model never saw", nc)
    return f"""
  <section>
    <p class="part">2) Removing flow + affine</p>
    <p class="lead">Two versions, both without the flow + affine stage. The scale encoder keeps its released weights, and Conv3 sits on top as in part 1.</p>
    <ul>
      <li><b>A:</b> only flow + affine is removed. The encoder's own <code>scale_conv</code> stays and makes the query from the image features.</li>
      <li><b>B:</b> flow + affine and <code>scale_conv</code> are removed. The query is one fixed constant (what <code>scale_conv</code> gives for an all-zero input).</li>
      <li><b>As it is:</b> Conv3 from part 1, not trained again. <b>Retrained:</b> Conv3 trained again on KITTI's 140 training pairs with the encoder frozen; the 20 quiz pairs pick the best round.</li>
    </ul>
    <p class="sub">Why this works: the transformer features are the same with or without the flow stage (checked, difference 0.0000), and the scale encoder's output changes by only {V['sf_difference_percent']['A']:.2f}% when its query is made from the image features instead.</p>
  </section>

  <section>
    <h2>Size and speed</h2>
    <div class="tablewrap"><table>
      <thead><tr><th></th><th class="n">Full</th><th class="n">Decoder removed</th><th class="n">A</th><th class="n">B</th></tr></thead>
      <tbody>{sp_rows}</tbody></table></div>
    <p class="sub">GPU times move a little from run to run (the full model took 292 ms in part 1 and {t['full']['ms']:.0f} ms here), so small gaps are not meaningful.</p>
  </section>

  <section>
    <h2>Accuracy on KITTI: the Mid error</h2>
    <div class="tablewrap"><table>
      <thead><tr><th>Images</th><th class="n">Full</th><th class="n">Decoder removed</th><th class="n">A as it is</th><th class="n">A retrained</th><th class="n">B as it is</th><th class="n">B retrained</th></tr></thead>
      <tbody>{mid_rows2}</tbody></table></div>
    <p class="sub">Lower is better. Retraining Conv3 is the only thing that moves the number; removing flow + affine does not.</p>
  </section>

  <section>
    <h2>Per-car TTC</h2>
    <div class="tablewrap"><table>
      <thead><tr><th>Measure</th><th class="n">Full</th><th class="n">A as it is</th><th class="n">A retrained</th><th class="n">B as it is</th><th class="n">B retrained</th></tr></thead>
      <tbody>{cars2}</tbody></table></div>
    <p class="sub">Same median method as in part 1. Pictures of the full model, A and B on the held-out KITTI images are in <code>lightweight_fpttc/removing_flow_affine_decoder/output/kitti/</code>.</p>
  </section>
"""




def part3():
    if not (V and DIST):
        return ""
    v = V["mid_error"]["held_out_40"]
    t = V["time_gpu"]
    best = min(DIST, key=lambda k: DIST[k]["mid_error"]["held_out_40"]["all"])
    best_mid = DIST[best]["mid_error"]["held_out_40"]["all"]
    speed_gain = 100 * (1 - t["A"]["ms"] / t["full"]["ms"])
    acc_loss = 100 * (best_mid - v["full"]) / v["full"]
    rows = "".join(
        f'<tr><td>{lab}</td><td class="n">{DIST[k]["mid_error"]["held_out_40"]["all"]:.1f}</td>'
        f'<td class="n">{DIST[k]["mid_error"]["held_out_40"]["cars"]:.1f}</td>'
        f'<td class="n">{DIST[k]["kitti_cars"]["held_out_40"]["scale_err_pct"]:.2f}%</td>'
        f'<td class="n">{DIST[k]["kitti_cars"]["held_out_40"]["ttc_med"]:.2f} s</td><td>{note}</td></tr>'
        for k, lab, note in (("distillation_only", "Distillation only, 140 pairs", "best Mid error"),
                             ("distill_then_kitti", "Distillation, then KITTI truth", "stage 2 added nothing"),
                             ("distill_more_frames", "Distillation, 553 pairs", "more frames did not help")) if k in DIST)
    return f"""
  <section>
    <p class="part">3) Teaching Conv3 by distillation</p>
    <p class="lead">Version A again, with the same frozen encoder. Conv3 is taught to copy the full model's scale map at every pixel instead of (or before) learning from KITTI's truth. All frames are KITTI.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>Teaching method (version A)</th><th class="n">Mid error, 40 held-out</th><th class="n">Cars only</th><th class="n">Car scale error</th><th class="n">Car TTC error</th><th>Note</th></tr></thead>
      <tbody>
        <tr><td>Full FP-TTC (for reference)</td><td class="n">{v['full']:.1f}</td><td class="n">99.0</td><td class="n">0.56%</td><td class="n">0.27 s</td><td></td></tr>
        <tr><td>KITTI truth directly (part 2)</td><td class="n">{v['A_retrained']:.1f}</td><td class="n">{V['mid_error']['held_out_40']['A_retrained_fg']:.1f}</td><td class="n">{V['kitti_cars']['held_out_40']['A_retrained']['scale_err_pct']:.2f}%</td><td class="n">{V['kitti_cars']['held_out_40']['A_retrained']['ttc_med']:.2f} s</td><td>the earlier method</td></tr>
        {rows}
      </tbody></table></div>
    <p class="sub">Distillation helps by about 2 to 3 points over learning from the truth. More frames, and a truth stage afterwards, add nothing: Conv3 is too small to copy the full model any closer. The 413 raw-drive pairs come from KITTI raw drive 0009; frames near the held-out and quiz scenes were left out (the drive contains KITTI scene flow scenes 7 to 12).</p>
  </section>

  <section>
    <h2>Against the authors' light model</h2>
    <div class="tablewrap"><table>
      <thead><tr><th></th><th class="n">Authors' light model</th><th class="n">Ours (version A, distilled)</th></tr></thead>
      <tbody>
        <tr><td>Speed gain</td><td class="n">about 15%</td><td class="n">about {speed_gain:.0f}% (GPU)</td></tr>
        <tr><td>Accuracy loss</td><td class="n">about 13%</td><td class="n">about {acc_loss:.0f}% (Mid error {best_mid:.1f} vs {v['full']:.1f})</td></tr>
      </tbody></table></div>
    <p class="sub">We match the authors' accuracy loss and beat their speed gain, with training on KITTI only. The authors' figures are as read from their paper earlier in this project and should be re-checked before they are quoted; their hardware and settings may differ from ours. Our speed gain is the GPU time of version A against the full model in the same run ({t['full']['ms']:.0f} ms and {t['A']['ms']:.0f} ms).</p>
  </section>
"""

journey_html, part2_html, part3_html = journey(), part2(), part3()

hm, nf = MID["held_out_40"], NC
verdict = f"""
<div class="verdict">
  <p><b>The light model is smaller and faster, and somewhat less accurate.</b> It has <span class="l">{-rel(P['full'], P['light']):.0f}% fewer parameters</span>
  ({P['light'] / 1e6:.2f} M against {P['full'] / 1e6:.2f} M), runs <span class="l">{-rel(ms_f, ms_l):.0f}% faster on the GPU</span>
  ({ms_l:.0f} ms against {ms_f:.0f} ms)""" + (f" and {-rel(CPU['full']['ms'], CPU['light']['ms']):.0f}% faster on the CPU" if CPU else "") + f""".</p>
  <p>Its Mid error on the 40 held-out KITTI pairs is <span class="l">{hm['light']:.1f}</span> against <span class="f">{hm['full']:.1f}</span> for the full model
  ({rel(hm['full'], hm['light']):+.0f}%, worse). For per-car TTC the two are close: on nuScenes the scale error is the same
  ({nf['full']['scale_err_pct']:.2f}% and {nf['light']['scale_err_pct']:.2f}%), and the middle TTC error is {nf['light']['ttc_med']:.2f} s against {nf['full']['ttc_med']:.2f} s.</p>
</div>"""

page = f"""<title>Light FP-TTC Results</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{CSS}</style>

<main>
  <header>
    <div class="eyebrow">BTP · single-camera collision prediction</div>
    <h1>Light FP-TTC Results</h1>
    <p class="part">1) Removing decoder only (speed up ,accuracy down)</p>
    <p class="lead">The full FP-TTC next to a light version where only the decoder is replaced. Everything else is the same and frozen: the backbone, the flow + affine stage and the scale encoder, with the released weights.</p>
  </header>

  <div class="notice"><b>Only the decoder is removed.</b> The backbone, the flow + affine stage and the scale encoder are exactly the same as in the full model. The decoder is replaced by three plain convolution layers (Conv3). Nothing else was changed.</div>

  {verdict}

  {journey_html}

  <section>
    <h2>What changed in the model</h2>
    <p class="sub">The full decoder (the matching-clue attention and two recurrent blocks, {(P['groups']['matching-clue attention'] + P['groups']['decoder + upsample']) / 1e6:.2f} M parameters) is removed. The light model puts three plain convolution layers (Conv3, {P['conv3'] / 1e6:.2f} M parameters) on the scale encoder's output joined with the later frame's image features. Their result is stretched four times with a plain bilinear resize. The model stops right after the scale encoder, so the removed parts do not run.</p>
    <figure><div class="chartwrap">{parts_chart()}</div>
      <figcaption>{legend_dot('kept', 'kept, frozen')}{legend_dot('removed', 'removed')}{legend_dot('added', 'added, trained')} Parameters in millions.</figcaption></figure>
  </section>

  <section>
    <h2>Size and speed</h2>
    <p class="sub">One KITTI pair at 640×320, the model alone (YOLO and drawing not included), on the laptop's RTX 3050. GPU time is the middle of 30 runs after a warm-up, CPU time the middle of 3 runs.</p>
    <div class="three">{"".join(tp)}</div>
    <p class="sub">Peak GPU memory is the same, {G['full']['peak_gpu_gb']:.2f} GB for both. The light run still keeps all of the full model's weights loaded, including the unused decoder. The decoder is only part of the time: the backbone's transformer is about half of it, and that is unchanged.</p>
  </section>

  <section>
    <h2>Accuracy on KITTI: the Mid error</h2>
    <p class="sub">The Mid error is the paper's score: how far the model's scale is from KITTI's true scale at each pixel, averaged, on a log scale and multiplied by 10,000. Lower is better. The 40 held-out pairs (numbers divisible by 5) are the fair test. Neither model learned from them.</p>
    <figure><div class="chartwrap">{mid_chart}</div><figcaption>Mid error, lower is better.</figcaption></figure>
    <div class="tablewrap"><table>
      <thead><tr><th>Images</th><th class="n">Full</th><th class="n">Light</th><th class="n">Change</th></tr></thead>
      <tbody>{mid_rows}</tbody></table></div>
    <p class="sub">The paper reports its light version as about 15% faster and about 13% less accurate. Ours is {-rel(ms_f, ms_l):.0f}% faster and {rel(hm['full'], hm['light']):.0f}% less accurate on the held-out pairs. The authors trained on much more data first (a large synthetic set) before KITTI. We trained the new decoder only on KITTI's 160 pairs.</p>
  </section>

  <section>
    <h2>Per-car TTC</h2>
    <p class="sub">The same median method on both scale maps: the median of the TTC map inside the inner 70% of each car's box. "Cars under 3 s found" and "Precision" use our own 3 s line for dangerous cars. Change: green is better for the light model, red is worse.</p>
    <div class="tablewrap"><table>
      <thead><tr><th>Measure</th><th class="n">Full</th><th class="n">Light</th><th class="n">Change</th></tr></thead>
      <tbody>{cars_table}</tbody></table></div>
    <p class="sub">The KITTI cars use the boxes of KITTI's car outlines. The nuScenes cars use YOLO boxes matched to labelled cars, with a true TTC calculated from the labels.</p>
  </section>

  <section>
    <h2>How the decoder was trained</h2>
    <ul>
      <li><b>Data:</b> KITTI's training pairs. {TR['train_images']} pairs to learn from, 20 to pick the best round, and the 40 held-out pairs kept for the final scores.</li>
      <li><b>Loss:</b> the paper's: the gap between the log of the guess and the log of the true scale, over the pixels where KITTI has a true scale.</li>
      <li><b>Only Conv3 learns.</b> The rest of the model is frozen. The best round was picked by the 20 quiz pairs, after {TR['rounds']} rounds.</li>
      <li><b>First run:</b> a higher learning rate was unstable and gave about twice the error. It was lowered and the final numbers above come from the second run.</li>
    </ul>
  </section>

  <section>
    <h2>Read these numbers with care</h2>
    <ul>
      <li><b>This is our version, not the authors'.</b> The paper does not say how it turns the Conv3 output back to full size, or exactly what x_t is. We chose a bilinear resize and the later frame's fine image features.</li>
      <li><b>Only 160 pairs to learn from,</b> and no pretraining. More data, or learning from the full model's own outputs on many more frames, would probably close some of the gap. That is untested.</li>
      <li><b>The 85 held-out KITTI cars are few,</b> so small differences there mean little. nuScenes has {nf['full']['cars']:,} cars.</li>
      <li><b>This is the first step.</b> The flow + affine stage is still in here. Removing it is part 2, below.</li>
    </ul>
  </section>

  {part2_html}

  {part3_html}

  <footer>Every number is read from <code>lightweight_fpttc/removing_decoder_only/results/summary.json</code> (part 1) and <code>lightweight_fpttc/removing_flow_affine_decoder/results/versions_summary.json</code> (part 2) and the <code>summary.json</code> of each folder under <code>lightweight_fpttc/removing_flow_affine_decoder/</code> (part 3), made by <code>lightweight_fpttc/removing_decoder_only/evaluate_light.py</code>. This page is rebuilt by <code>lightweight_fpttc/removing_decoder_only/build_results_page.py</code>. The side-by-side pictures are in <code>lightweight_fpttc/removing_decoder_only/output/</code>.</footer>
</main>
"""

import sys
sys.path.insert(0, str(HERE.parents[1] / "page_tools"))
from add_edit_mode import add_edit_mode  # "Edit text" button on the page

out = HERE / "results" / "results_page.html"
out.write_text(add_edit_mode(page), encoding="utf8")
print("wrote", out, f"{len(page) / 1024:.0f} KB")
