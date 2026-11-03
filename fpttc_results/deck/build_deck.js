// Builds fpttc_results.pptx from results/report_data.json and the figures.
// Run from this folder:  node build_deck.js
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");

const R = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "results", "report_data.json"), "utf8"));
const FIG = (f) => path.join(__dirname, "..", "figures", f);

// ---- palette: deep slate ground, one amber accent for "danger", teal for "learned"
const C = {
  bg: "0F1A24", panel: "172633", line: "2C4256", ink: "F2F5F8", mute: "9DB0C2",
  teal: "4FD1C5", amber: "F6B93B", red: "FF6B5E", green: "7BD88F", paper: "F6F8FA", paperInk: "17212B",
};
const F = { head: "Calibri", body: "Calibri" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5
pres.title = "FP-TTC measured: accuracy, speed and per-car time-to-collision";

const W = 13.33, H = 7.5, M = 0.6;

function base(slide, title, kicker) {
  slide.background = { color: C.bg };
  if (kicker) slide.addText(kicker.toUpperCase(), { x: M, y: 0.35, w: 9, h: 0.3, fontFace: F.head, fontSize: 12, bold: true, color: C.teal, charSpacing: 3, margin: 0 });
  slide.addText(title, { x: M, y: 0.65, w: W - 2 * M, h: 0.9, fontFace: F.head, fontSize: 30, bold: true, color: C.ink, margin: 0, valign: "top" });
}
function foot(slide, text) {
  slide.addText(text, { x: M, y: H - 0.5, w: W - 2 * M, h: 0.3, fontFace: F.body, fontSize: 11, color: C.mute, margin: 0 });
}
function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.8, fontFace: F.head, fontSize: 40, bold: true, color: color || C.ink, margin: 0 });
  slide.addText(label, { x, y: y + 0.82, w, h: 0.6, fontFace: F.body, fontSize: 13, color: C.mute, margin: 0, valign: "top" });
}
const near = (a, b) => `${a}`;

// ======================================================== 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.bg };
  s.addText("FP-TTC, measured", { x: M, y: 2.2, w: 10, h: 1.1, fontFace: F.head, fontSize: 54, bold: true, color: C.ink, margin: 0 });
  s.addText("How accurate, how fast and how usable per car, on a 4 GB laptop GPU", { x: M, y: 3.35, w: 10.5, h: 0.6, fontFace: F.body, fontSize: 22, color: C.mute, margin: 0 });
  s.addText("Single-camera collision prediction  |  BTP  |  October 2026", { x: M, y: 6.6, w: 10, h: 0.35, fontFace: F.body, fontSize: 14, color: C.teal, margin: 0 });
  s.addText("Model: FP-TTC (Li et al., IEEE TCSVT 2024), released weights, nothing retrained", { x: M, y: 4.3, w: 11, h: 0.4, fontFace: F.body, fontSize: 15, color: C.ink, margin: 0 });
}

// ======================================================== 2 what was measured
{
  const s = pres.addSlide();
  base(s, "What we measured, and on what", "Setup");
  const cols = [
    ["Model", ["FP-TTC, released weights", `${(R.speed.params / 1e6).toFixed(1)} M parameters, ${R.speed.files["fpttc_kitti.pth.tar"]} MB file`, "Two weight files: fpttc_kitti (fine-tuned on KITTI) and fpttc_mix", "No retraining, no changes to the network"]],
    ["Data", [`KITTI scene flow: 200 pairs with stereo ground truth. 160 were in the model's training set, ${R.kitti_images.held} were held out`,
      `nuScenes mini: ${R.nuscenes_frames.keyframes} keyframes, ${R.nuscenes_frames.true_cars} annotated vehicles. FP-TTC never saw it`,
      "Ground truth: true scale from stereo and flow (KITTI); true TTC from annotated position and velocity (nuScenes)"]],
    ["Machine", [R.speed.device + ", 4 GB", "PyTorch 2.6, CUDA 12.4", "Deformable attention runs in pure PyTorch, because the authors' mmcv CUDA kernel has no ready build for this setup", "Every timing: GPU synchronised, warm-up first, median of repeated runs"]],
  ];
  cols.forEach((c, i) => {
    const x = M + i * 4.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.9, w: 3.85, h: 4.6, fill: { color: C.panel }, line: { color: C.line, width: 1 } });
    s.addText(c[0], { x: x + 0.25, y: 2.05, w: 3.4, h: 0.45, fontFace: F.head, fontSize: 20, bold: true, color: C.teal, margin: 0 });
    s.addText(c[1].map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < c[1].length - 1 } })),
      { x: x + 0.25, y: 2.6, w: 3.4, h: 3.8, fontFace: F.body, fontSize: 14, color: C.ink, paraSpaceAfter: 8, valign: "top", margin: 0 });
  });
  foot(s, "Accuracy metric: Mid error = mean |log(predicted scale) - log(true scale)| x 10,000. The FP-TTC paper reports 59.35.");
}

// ======================================================== 3 headline accuracy
{
  const s = pres.addSlide();
  base(s, "On unseen KITTI frames the error is in line with the paper", "Accuracy");
  const row = (proto, w, size) => R.kitti_mid.find((r) => r.protocol === proto && r.weights === w && r.size === size);
  const a = row("past_09_10", "fpttc_kitti", "960x288"), b = row("past_09_10", "fpttc_kitti", "640x320");
  const c = row("script_10_11", "fpttc_kitti", "640x320");
  stat(s, M, 1.8, 3.2, a.held_out_40.toFixed(1), "Mid error, 40 held-out images (960x288, past frames)", C.teal);
  stat(s, M, 3.5, 3.2, "59.35", "Reported in the paper (KITTI validation)", C.mute);
  stat(s, M, 5.2, 3.2, a.baseline_held.toFixed(0), "Error of the lazy answer 'nothing changes' (s = 1)", C.amber);

  s.addChart(pres.charts.BAR, [{
    name: "Mid error (x 1e-4)",
    labels: ["Paper", "Held out 40\n960x288", "Held out 40\n640x320", "Seen in training\n(160)", "Repo script\npairing (_10,_11)"],
    values: [59.35, a.held_out_40, b.held_out_40, a.trained_on_160, c.held_out_40],
  }], {
    x: 4.2, y: 1.7, w: 8.6, h: 4.7, barDir: "col", chartColors: [C.teal],
    catAxisLabelColor: C.mute, valAxisLabelColor: C.mute, catAxisLabelFontSize: 12, valAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink,
    dataLabelFontSize: 13, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.0", showLegend: false, valAxisMaxVal: 140,
    showTitle: true, title: "Mid error, lower is better", titleColor: C.ink, titleFontSize: 14,
  });
  foot(s, `The repo's own evaluation script pairs frames _10 and _11, which is not how the model is trained. That pairing gives ${c.held_out_40.toFixed(0)}, about twice as bad. Past-frame pairing (_09,_10) is the one used above.`);
}

// ======================================================== 4 where the error is
{
  const s = pres.addSlide();
  base(s, "Error is lowest far away and highest on nearby cars", "Accuracy by region");
  const d = R.kitti_depth_held;
  s.addChart(pres.charts.BAR, [{
    name: "Mid error (x 1e-4)", labels: ["0-10 m", "10-20 m", "20-40 m", "40+ m"],
    values: [d["0_10"], d["10_20"], d["20_40"], d["40_999"]],
  }], {
    x: M, y: 1.7, w: 6.1, h: 4.6, barDir: "col", chartColors: [C.amber], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink,
    dataLabelFontSize: 13, dataLabelFormatCode: "0.0", showLegend: false, showTitle: true, title: "By distance to the surface (held-out images)", titleColor: C.ink, titleFontSize: 14, valAxisMaxVal: 80,
  });
  const r = R.kitti_mid.find((x) => x.protocol === "past_09_10" && x.weights === "fpttc_kitti" && x.size === "960x288");
  s.addChart(pres.charts.BAR, [{
    name: "Mid error (x 1e-4)", labels: ["Background", "Cars and objects"], values: [r.held_bg, r.held_fg],
  }], {
    x: 6.9, y: 1.7, w: 5.8, h: 4.6, barDir: "col", chartColors: [C.teal], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink,
    dataLabelFontSize: 13, dataLabelFormatCode: "0.0", showLegend: false, showTitle: true, title: "Background versus moving objects", titleColor: C.ink, titleFontSize: 14, valAxisMaxVal: 120,
  });
  foot(s, "Objects are about twice as wrong as background, and they are the part that matters for collision. The per-pixel average hides this.");
}

// ======================================================== 5 KITTI example
{
  const s = pres.addSlide();
  base(s, "One held-out frame: prediction, truth and error", "Example");
  s.addImage({ path: FIG("kitti_example.jpg"), x: M, y: 1.7, w: 12.1, h: 4.9, sizing: { type: "contain", w: 12.1, h: 4.9 } });
  foot(s, "Red means getting closer, blue means moving away. The true scale (bottom left) exists only where stereo and flow are both valid.");
}

// ======================================================== 6 per-car methods
{
  const s = pres.addSlide();
  base(s, "Turning a per-pixel map into one number per car: the median works, the flow zoom does not", "Per-car scale, unseen nuScenes data");
  const n = R.nuscenes["fpttc_kitti|640x352"];
  const names = [["median_box100", "Median, whole box"], ["median_box70", "Median, inner 70%"], ["median_box50", "Median, inner 50%"], ["zoom_box70", "Zoom fit on flow"]];
  s.addChart(pres.charts.BAR, [{
    name: "TTC error (s)", labels: names.map((x) => x[1]), values: names.map((x) => n[x[0]].ttc_med),
  }], {
    x: M, y: 1.8, w: 6.2, h: 4.4, barDir: "bar", chartColors: [C.teal, C.teal, C.teal, C.red], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 13, dataLabelFormatCode: "0.00",
    showLegend: false, showTitle: true, title: "Median TTC error in seconds, lower is better", titleColor: C.ink, titleFontSize: 14, catAxisOrientation: "maxMin",
  });
  const rows = [["Method", "Scale error", "TTC error (median)", "TTC error (90th pct)"]]
    .concat(names.map((x) => [x[1], n[x[0]].scale_err_pct.toFixed(2) + "%", n[x[0]].ttc_med.toFixed(2) + " s", n[x[0]].ttc_p90.toFixed(1) + " s"]));
  const hdr = { bold: true, color: C.teal, fill: { color: C.panel }, fontFace: F.body, fontSize: 13 };
  const cell = { color: C.ink, fill: { color: C.bg }, fontFace: F.body, fontSize: 13 };
  s.addTable(rows.map((r, i) => r.map((t) => ({ text: t, options: i === 0 ? hdr : cell }))),
    { x: 7.0, y: 1.9, w: 5.75, colW: [2.0, 1.05, 1.3, 1.4], border: { type: "solid", color: C.line, pt: 0.75 }, rowH: 0.5 });
  s.addText([
    { text: `${n.median_box70.cars} cars matched to annotations, 394 keyframes. `, options: {} },
    { text: "Median scale error is 0.3%, but a 0.3% scale error becomes about 0.9 s of TTC error, because TTC is very sensitive when the scale is close to 1.", options: {} },
  ], { x: 7.0, y: 4.65, w: 5.75, h: 1.6, fontFace: F.body, fontSize: 14, color: C.mute, margin: 0, valign: "top" });
  foot(s, "The flow zoom fit ends up around 22% wrong. The flow FP-TTC returns is about 210 px away from KITTI's true flow, and we did not find out why.");
}

// ======================================================== 7 danger calls
{
  const s = pres.addSlide();
  base(s, "As a danger alarm (TTC under 3 s), the median method catches 9 in 10 and 4 in 5 alarms are right", "Danger calls on nuScenes");
  const n = R.nuscenes["fpttc_kitti|640x352"].median_box70;
  stat(s, M, 1.8, 3.4, (n.recall * 100).toFixed(0) + "%", `Recall: ${n.tp} of ${n.true_d} truly dangerous cars flagged`, C.green);
  stat(s, M, 3.5, 3.4, (n.precision * 100).toFixed(0) + "%", `Precision: ${n.tp} of ${n.pred_d} alarms were right`, C.amber);
  stat(s, M, 5.2, 3.4, String(n.fn), "Dangerous cars missed", C.red);
  const bt = R.nuscenes_by_ttc;
  const labels = Object.keys(bt);
  s.addChart(pres.charts.BAR, [{ name: "Median TTC error (s)", labels: labels.map((l) => "true TTC " + l.replace("-", " to ")), values: labels.map((l) => +bt[l].median.toFixed(2)) }], {
    x: 4.4, y: 1.7, w: 8.4, h: 4.6, barDir: "col", chartColors: [C.teal], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 13, dataLabelFormatCode: "0.00",
    showLegend: false, showTitle: true, title: "TTC error by how close the collision really is", titleColor: C.ink, titleFontSize: 14,
  });
  foot(s, "Errors are small exactly where it matters (under 3 s: 0.09 s and 0.30 s) and grow for distant, slow cars. Median method, inner 70%, 640x352. The 3 s threshold is our choice.");
}

// ======================================================== 8 nuScenes example
{
  const s = pres.addSlide();
  base(s, "Per-car TTC on a real street, against the annotated truth", "Example");
  s.addImage({ path: FIG("nuscenes_example.jpg"), x: 1.6, y: 1.65, w: 10.1, h: 4.9, sizing: { type: "contain", w: 10.1, h: 4.9 } });
  foot(s, "Each label: predicted TTC, then the true TTC from the annotations in brackets. Red: under 3 s. Amber: under 6 s.");
}

// ======================================================== 9 speed
{
  const s = pres.addSlide();
  base(s, "Speed: about 3.5 pairs per second, and half of the time is one transformer", "Speed");
  const bs = R.speed.by_size.filter((x) => !x.error);
  s.addChart(pres.charts.BAR, [{ name: "ms per frame pair", labels: bs.map((x) => `${x.width}x${x.height}`), values: bs.map((x) => Math.round(x.median_ms)) }], {
    x: M, y: 1.7, w: 6.3, h: 4.7, barDir: "col", chartColors: [C.teal], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute, catAxisLabelFontSize: 10,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 11,
    showLegend: false, showTitle: true, title: "Milliseconds per frame pair, by input size", titleColor: C.ink, titleFontSize: 14,
  });
  const mods = Object.entries(R.speed.by_module).filter(([k]) => !k.startsWith("("));
  s.addChart(pres.charts.BAR, [{ name: "% of time", labels: mods.map(([k]) => k), values: mods.map(([, v]) => +v.pct.toFixed(1)) }], {
    x: 7.1, y: 1.7, w: 5.7, h: 4.7, barDir: "bar", chartColors: [C.amber], catAxisLabelColor: C.mute, valAxisLabelColor: C.mute, catAxisLabelFontSize: 11,
    valGridLine: { color: C.line, size: 0.5 }, catGridLine: { style: "none" }, showValue: true, dataLabelColor: C.ink, dataLabelFontSize: 12, dataLabelFormatCode: "0.0",
    showLegend: false, showTitle: true, title: "Share of time per part (640x320)", titleColor: C.ink, titleFontSize: 14, catAxisOrientation: "maxMin",
  });
  foot(s, "Lightweight version in the paper removes flow, initial scale and clue attention, about a fifth of the time here, but keeps the transformer.");
}

// ======================================================== 10 resources
{
  const s = pres.addSlide();
  base(s, "Memory, precision and hardware limits", "Resources");
  const sp = R.speed;
  const t = sp.by_size.find((x) => x.width === 640 && x.height === 320);
  const big = sp.by_size.filter((x) => !x.error).slice(-1)[0];
  const b4 = sp.batch.find((x) => x.batch === 4), b8 = sp.batch.find((x) => x.batch === 8);
  stat(s, M, 1.8, 3.8, t.peak_gpu_gb.toFixed(2) + " GB", "GPU memory at 640x320, one pair", C.teal);
  stat(s, M, 3.5, 3.8, big.peak_gpu_gb.toFixed(1) + " GB", `GPU memory at ${big.width}x${big.height} (${Math.round(big.median_ms)} ms)`, C.amber);
  stat(s, M, 5.2, 3.8, (sp.cpu_ms / 1000).toFixed(1) + " s", `per pair on the CPU, ${(sp.cpu_ms / t.median_ms).toFixed(0)}x slower than the GPU`, C.mute);
  const items = [
    ["Batching does not help", `2 pairs at once cost the same per pair; 4 pairs use ${b4.peak_gpu_gb.toFixed(1)} GB and get slower; 8 pairs run out of the 4 GB`],
    ["Half precision fails", "Switching to fp16 stops with an overflow error: the model was not built for it, so no easy speed-up there"],
    ["Whole camera-to-car pipeline", `YOLOv8s ${sp.pipeline.yolov8s} ms + FP-TTC ${Math.round(sp.pipeline.fpttc_including_resize)} ms + per-car median under 1 ms = ${Math.round(sp.pipeline.total_median_ms)} ms, about ${sp.pipeline.fps.toFixed(0)} frames per second`],
    ["Where a slower attention costs us", "The attention step runs as plain PyTorch instead of the authors' CUDA kernel, which we did not install; its time share here is probably higher than in the paper"],
  ];
  items.forEach((it, i) => {
    const y = 1.8 + i * 1.18;
    s.addText(it[0], { x: 5.0, y, w: 7.7, h: 0.35, fontFace: F.head, fontSize: 17, bold: true, color: C.teal, margin: 0 });
    s.addText(it[1], { x: 5.0, y: y + 0.38, w: 7.7, h: 0.75, fontFace: F.body, fontSize: 14, color: C.ink, margin: 0, valign: "top" });
  });
  foot(s, "All numbers measured on one run per setting, 15 to 30 repeats each, on an otherwise idle GPU.");
}

// ======================================================== 11 takeaways
{
  const s = pres.addSlide();
  base(s, "What this means for the project", "Takeaways");
  const pts = [
    ["Usable as a TTC sensor", "On data it has never seen, the per-car median is within 0.3% on scale and flags 91% of the cars truly under 3 s."],
    ["Not real time on this laptop", "About 3 frames per second end to end. Fine for evaluation and demos, too slow for a live warning without changes."],
    ["Do not use its flow output", "It does not match KITTI's true flow (about 210 px apart) and the flow-zoom per-car method fails with it. The scale map works."],
    ["Far and slow cars are unreliable", "TTC error grows to about 2 s when the true TTC is 6 to 12 s. Warnings should only trust close, fast-closing cars."],
    ["Open question", "Whether a small trained head on its internal features beats the plain median. Not measured yet."],
  ];
  pts.forEach((p, i) => {
    const y = 1.75 + i * 1.0;
    s.addShape(pres.shapes.OVAL, { x: M, y: y + 0.05, w: 0.42, h: 0.42, fill: { color: i === 4 ? C.amber : C.teal }, line: { color: C.bg, width: 0 } });
    s.addText(String(i + 1), { x: M, y: y + 0.05, w: 0.42, h: 0.42, fontFace: F.head, fontSize: 15, bold: true, color: C.bg, align: "center", valign: "middle", margin: 0 });
    s.addText(p[0], { x: M + 0.65, y, w: 11.6, h: 0.35, fontFace: F.head, fontSize: 18, bold: true, color: C.ink, margin: 0 });
    s.addText(p[1], { x: M + 0.65, y: y + 0.36, w: 11.6, h: 0.55, fontFace: F.body, fontSize: 14, color: C.mute, margin: 0, valign: "top" });
  });
}

pres.writeFile({ fileName: path.join(__dirname, "..", "FP-TTC_results.pptx") }).then((f) => console.log("wrote", f));
