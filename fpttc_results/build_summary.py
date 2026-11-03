"""
Turn the raw measurement files into the tables and numbers used in the
slides and the web page. Reads results/*.csv and results/speed.json, writes
results/summary.json. Prints a readable digest.

Usage (from the project root):  python fpttc_results/build_summary.py
"""

import json

import numpy as np
import pandas as pd

import common as C

R = C.RESULTS_DIR
PAPER = {"fpttc_kitti": 59.35, "fpttc_mix": 62.30}      # Mid error, KITTI-40 validation, from the FP-TTC README
KITTI_DT = 0.1
DANGER_S = 3.0                                          # a car is "dangerous" if its TTC is under this many seconds
METHODS = [("median_box100", "median, whole box"), ("median_box70", "median, inner 70% of box"),
           ("median_box50", "median, inner 50% of box"), ("median_mask", "median, exact car outline (oracle)"),
           ("zoom_box70", "zoom fit on the flow (RANSAC)")]


def ttc(s, dt, past_step=False):
    """TTC in seconds. Default: the paper's Eq. 1, dt / (1 - s), for the model's
    scale output s (next / current distance). past_step=True is for the zoom
    fit, whose s = 1/k is the ratio between the two frames already seen:
    TTC = dt / (k - 1) = dt * s / (1 - s)."""
    s = np.asarray(s, dtype=float)
    dt = np.broadcast_to(np.asarray(dt, dtype=float), s.shape)
    out = np.full(s.shape, np.nan)
    ok = s < 1
    out[ok] = (dt[ok] * s[ok] if past_step else dt[ok]) / (1 - s[ok])
    return out


def stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()), median=float(np.median(x)), p90=float(np.percentile(x, 90)))


def car_metrics(df, method, dt, s_true_col="s_true", ttc_true=None):
    """Errors of one per-car method over the rows of df."""
    s_pred = df[f"s_{method}"].to_numpy(float)
    s_true = df[s_true_col].to_numpy(float)
    ok = np.isfinite(s_pred) & (s_pred > 0) & (s_true > 0)
    log_err = np.abs(np.log(np.where(ok, s_pred, 1)) - np.log(np.where(ok, s_true, 1)))
    log_err[~ok] = np.nan
    dts = np.full(len(df), dt) if np.ndim(dt) == 0 else np.asarray(dt, float)
    tp = ttc(s_pred, dts, past_step=method.startswith("zoom"))
    tt = ttc(s_true, dts) if ttc_true is None else np.asarray(ttc_true, float)
    both = np.isfinite(tp) & np.isfinite(tt)
    abs_t = np.abs(tp - tt)
    rel_t = abs_t / tt
    # danger call: TTC under DANGER_S seconds
    pos_true = np.isfinite(tt) & (tt < DANGER_S)
    pos_pred = np.isfinite(tp) & (tp < DANGER_S)
    tp_n, fp_n, fn_n = int((pos_true & pos_pred).sum()), int((~pos_true & pos_pred).sum()), int((pos_true & ~pos_pred).sum())
    prec = tp_n / (tp_n + fp_n) if tp_n + fp_n else float("nan")
    rec = tp_n / (tp_n + fn_n) if tp_n + fn_n else float("nan")
    out = dict(cars=int(len(df)), cars_with_prediction=int(ok.sum()),
               scale_log_error=stats(log_err), scale_error_percent_median=float(100 * (np.exp(np.nanmedian(log_err)) - 1)) if ok.any() else None,
               ttc_cars=int(both.sum()), ttc_abs_error_s=stats(abs_t[both]), ttc_rel_error=stats(rel_t[both]),
               danger=dict(true_dangerous=int(pos_true.sum()), predicted_dangerous=int(pos_pred.sum()),
                           correct_alarms=tp_n, false_alarms=fp_n, missed=fn_n, precision=prec, recall=rec))
    bins = [(0, 1.5), (1.5, 3), (3, 6), (6, 12)]
    out["ttc_by_true_ttc"] = {f"{lo}-{hi}s": stats(abs_t[both & (tt >= lo) & (tt < hi)]) for lo, hi in bins}
    return out


def main():
    summary = {"paper_mid_error": PAPER, "danger_threshold_s": DANGER_S}

    # ------------------------------------------------------------ KITTI, per image
    d = pd.read_csv(R / "kitti_images.csv")
    d["mid_e4"] = d.mid_err * 1e4
    for c in [c for c in d.columns if c.startswith("mid_err_")]:
        d[c.replace("mid_err", "mid_e4")] = d[c] * 1e4
    d["baseline_e4"] = d.baseline_no_change * 1e4
    key = ["protocol", "split", "weights", "width", "height"]
    cols = [c for c in d.columns if c.startswith("mid_e4") or c in ("baseline_e4", "flow_epe", "flow_fl_all", "seconds")]
    agg = d.groupby(key)[cols].mean().round(3).reset_index()
    agg["images"] = d.groupby(key).size().values
    summary["kitti_image_table"] = agg.to_dict(orient="records")

    # image-weighted mean over all 200 images
    allagg = d.groupby(["protocol", "weights", "width", "height"])[cols].mean().round(3).reset_index()
    summary["kitti_all200_table"] = allagg.to_dict(orient="records")

    # ------------------------------------------------------------ KITTI, per car
    c = pd.read_csv(R / "kitti_cars.csv")
    k = {}
    for (proto, split, wn, w, h), g in c.groupby(["protocol", "split", "weights", "width", "height"]):
        k[f"{proto}|{split}|{wn}|{w}x{h}"] = {m: car_metrics(g, m, KITTI_DT) for m, _ in METHODS}
    summary["kitti_per_car"] = k
    summary["kitti_car_counts"] = {s: int(n) for s, n in c[(c.weights == "fpttc_kitti") & (c.width == 640) & (c.protocol == "past_09_10")].groupby("split").size().items()}

    # ------------------------------------------------------------ nuScenes
    try:
        n = pd.read_csv(R / "nuscenes_cars.csv")
        f = pd.read_csv(R / "nuscenes_frames.csv")
        summary["nuscenes_frames"] = dict(keyframes=int(len(f)), true_cars=int(f.n_true.sum()), found=int(f.n_found.sum()),
                                          detector_recall=float(f.n_found.sum() / max(f.n_true.sum(), 1)),
                                          mean_gap_s=float(f.dt.mean()), gap_min_s=float(f.dt.min()), gap_max_s=float(f.dt.max()),
                                          yolo_ms_mean=float(f.yolo_seconds.mean() * 1000))
        ns = {}
        for (wn, w, h), g in n.groupby(["weights", "width", "height"]):
            approaching = g[np.isfinite(g.ttc_true)]
            ns[f"{wn}|{w}x{h}"] = dict(
                matched_cars=int(len(g)), approaching_cars=int(len(approaching)),
                methods={m: car_metrics(g, m, g.dt.to_numpy(float), ttc_true=g.ttc_true.to_numpy(float)) for m, _ in METHODS if f"s_{m}" in g},
                by_distance={f"{lo}-{hi}m": {m: car_metrics(g[(g.z >= lo) & (g.z < hi)], m, g[(g.z >= lo) & (g.z < hi)].dt.to_numpy(float),
                                                            ttc_true=g[(g.z >= lo) & (g.z < hi)].ttc_true.to_numpy(float))["scale_log_error"]
                                             for m in ("median_box50", "median_box70")}
                             for lo, hi in [(0, 15), (15, 30), (30, 50), (50, 80)] if ((g.z >= lo) & (g.z < hi)).any()})
        summary["nuscenes"] = ns
        summary["nuscenes_truth"] = dict(
            cars=int(len(n[(n.weights == "fpttc_kitti") & (n.width == 640) & (n.height == 352)])),
            approaching_share=float(np.isfinite(n[(n.weights == "fpttc_kitti") & (n.width == 640) & (n.height == 352)].ttc_true).mean()),
            dangerous_under_3s=int((n[(n.weights == "fpttc_kitti") & (n.width == 640) & (n.height == 352)].ttc_true < DANGER_S).sum()))
    except FileNotFoundError:
        summary["nuscenes"] = None

    try:
        summary["speed"] = json.loads((R / "speed.json").read_text())
    except FileNotFoundError:
        summary["speed"] = None

    (R / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("saved summary.json")


if __name__ == "__main__":
    main()
