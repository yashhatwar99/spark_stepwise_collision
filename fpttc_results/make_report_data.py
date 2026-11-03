"""
Condense results/summary.json into results/report_data.json, the one file the
slides (deck/build_deck.js) and the web page (build_site.py) read.

Only the numbers that depend on the TTC formula are rebuilt from summary.json:
the per-car KITTI table, the per-car nuScenes table and the nuScenes error by
true-TTC range. Everything else (Mid error, depth buckets, nuScenes truth,
speed benchmark) does not use the formula and is carried over from the
existing report_data.json.

Usage (from fpttc_results):
    python make_report_data.py
    python make_report_data.py --summary a.json --template b.json --out c.json   # to compare runs
"""

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

KITTI_METHODS = ["median_box100", "median_box70", "median_box50", "median_mask", "zoom_box70"]
KITTI_SIZES = ["640x320", "960x288"]
NUS_CONFIGS = ["fpttc_kitti|640x320", "fpttc_kitti|640x352", "fpttc_kitti|960x544", "fpttc_mix|640x352"]
NUS_METHODS = ["median_box100", "median_box70", "median_box50", "zoom_box70"]
BY_TTC_SOURCE = ("fpttc_kitti|640x352", "median_box70")


def kitti_entry(m):
    return dict(cars=m["cars"], scale_err_pct=round(m["scale_error_percent_median"], 2),
                ttc_med=round(m["ttc_abs_error_s"]["median"], 2), ttc_p90=round(m["ttc_abs_error_s"]["p90"], 2),
                recall=round(m["danger"]["recall"], 2), precision=round(m["danger"]["precision"], 2),
                n_ttc=m["ttc_cars"])


def nus_entry(m):
    d = m["danger"]
    return dict(cars=m["cars"], scale_err_pct=round(m["scale_error_percent_median"], 2),
                ttc_med=round(m["ttc_abs_error_s"]["median"], 2), ttc_p90=round(m["ttc_abs_error_s"]["p90"], 2),
                n_ttc=m["ttc_cars"], recall=round(d["recall"], 2), precision=round(d["precision"], 2),
                tp=d["correct_alarms"], fp=d["false_alarms"], fn=d["missed"],
                true_d=d["true_dangerous"], pred_d=d["predicted_dangerous"])


def build(summary, template):
    out = dict(template)
    kp = {}
    for split in ("held_out_40", "trained_on_160"):
        for size in KITTI_SIZES:
            for meth in KITTI_METHODS:
                m = summary["kitti_per_car"][f"past_09_10|{split}|fpttc_kitti|{size}"][meth]
                kp[f"{split}|{size}|{meth}"] = kitti_entry(m)
    out["kitti_per_car"] = kp
    out["nuscenes"] = {cfg: {meth: nus_entry(summary["nuscenes"][cfg]["methods"][meth]) for meth in NUS_METHODS}
                       for cfg in NUS_CONFIGS}
    cfg, meth = BY_TTC_SOURCE
    out["nuscenes_by_ttc"] = summary["nuscenes"][cfg]["methods"][meth]["ttc_by_true_ttc"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(RES / "summary.json"))
    ap.add_argument("--template", default=str(RES / "report_data.json"))
    ap.add_argument("--out", default=str(RES / "report_data.json"))
    a = ap.parse_args()
    summary = json.loads(Path(a.summary).read_text(encoding="utf8"))
    template = json.loads(Path(a.template).read_text(encoding="utf8"))
    Path(a.out).write_text(json.dumps(build(summary, template), indent=1), encoding="utf8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
