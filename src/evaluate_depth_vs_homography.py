"""
Head-to-head: monocular metric depth vs flat-road homography, measured
against nuScenes 3D ground truth on exactly the same vehicles.

Both methods are given the SAME ground-truth 2D box, so this isolates the
depth-estimation method itself -- detector error is not part of the
comparison. Homography is given nuScenes' real intrinsics/extrinsics rather
than a guessed camera profile, so it is being tested at its best.
"""

import os
import sys

import numpy as np
from PIL import Image
from pyquaternion import Quaternion

sys.path.insert(0, os.path.dirname(__file__))
from depth_3d_box import load_depth_pipeline, depth_map, sample_box_depth  # noqa: E402

from nuscenes.nuscenes import NuScenes  # noqa: E402
from nuscenes.utils.geometry_utils import BoxVisibility, view_points  # noqa: E402

DATAROOT = os.path.join(os.path.dirname(__file__), "..", "data", "nuScenes", "v1.0-mini")
N_SAMPLES = 60
MIN_RANGE, MAX_RANGE = 4.0, 60.0


def measure_size_priors(nusc):
    dims = [nusc.get("sample_annotation", a)["size"]
            for s in nusc.sample for a in s["anns"]
            if nusc.get("sample_annotation", a)["category_name"].startswith("vehicle.car")]
    d = np.array(dims)
    return np.median(d, axis=0), len(d)


def homography_distance(u, v, K, cam_height, R_cam2ego):
    """Flat-road projection: where does the ray through (u, v) hit the ground?"""
    ray = R_cam2ego @ (np.linalg.inv(K) @ np.array([u, v, 1.0]))
    if ray[2] >= -1e-6:
        return None  # at or above the horizon -- no ground intersection
    t = cam_height / (-ray[2])
    return float(np.hypot(ray[0] * t, ray[1] * t))


def main():
    nusc = NuScenes(version="v1.0-mini", dataroot=DATAROOT, verbose=False)

    prior, n_prior = measure_size_priors(nusc)
    print(f"car size prior measured from {n_prior} annotations: "
          f"w={prior[0]:.2f} l={prior[1]:.2f} h={prior[2]:.2f} m")
    print()

    pipe = load_depth_pipeline()
    rows = []

    for si in range(min(N_SAMPLES, len(nusc.sample))):
        sample = nusc.sample[si]
        cam_tok = sample["data"]["CAM_FRONT"]
        sd = nusc.get("sample_data", cam_tok)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        K = np.array(cs["camera_intrinsic"])
        cam_h = cs["translation"][2]
        R = Quaternion(cs["rotation"]).rotation_matrix

        path, boxes, _ = nusc.get_sample_data(cam_tok, box_vis_level=BoxVisibility.ALL)
        cars = [b for b in boxes if b.name.startswith("vehicle.car")
                and MIN_RANGE < b.center[2] < MAX_RANGE]
        if not cars:
            continue

        depth = depth_map(pipe, Image.open(path))

        for b in cars:
            ci = view_points(b.corners(), K, normalize=True)[:2, :]
            x1, x2 = float(ci[0].min()), float(ci[0].max())
            y1, y2 = float(ci[1].min()), float(ci[1].max())

            true_z = float(b.center[2])                      # forward distance to box centre
            true_near = float(b.corners()[2].min())          # forward distance to nearest face

            d_est = sample_box_depth(depth, x1, y1, x2, y2)
            h_est = homography_distance((x1 + x2) / 2.0, y2, K, cam_h, R)
            rows.append((true_z, true_near, d_est, h_est))

        if (si + 1) % 15 == 0:
            print(f"  processed {si + 1} frames, {len(rows)} vehicles so far...")

    print()
    arr_true = np.array([r[0] for r in rows])
    arr_near = np.array([r[1] for r in rows])
    depth_est = np.array([np.nan if r[2] is None else r[2] for r in rows])
    homo_est = np.array([np.nan if r[3] is None else r[3] for r in rows])

    # Depth networks report the nearest visible surface, not the object's
    # centre, so compare each estimate against the quantity it is actually
    # measuring; report the residual bias rather than assuming it away.
    valid_d = np.isfinite(depth_est)
    valid_h = np.isfinite(homo_est)
    bias = np.median(depth_est[valid_d] - arr_near[valid_d])
    print(f"vehicles evaluated: {len(rows)}")
    print(f"depth produced an estimate for {valid_d.sum()}, homography for {valid_h.sum()}")
    print(f"depth median bias vs nearest face: {bias:+.2f} m")
    print()

    err_d = np.abs(depth_est - arr_near)
    err_h = np.abs(homo_est - arr_true)

    def band_table(name, err, valid):
        print(f"--- {name} ---")
        n = valid.sum()
        for lo, hi, lbl in [(0, 1, "< 1 m  excellent"), (1, 3, "1-3 m  usable"),
                            (3, 10, "3-10 m poor"), (10, 1e9, "> 10 m useless")]:
            m = valid & (err >= lo) & (err < hi)
            print(f"   {lbl:18s} {m.sum():3d} / {n}  ({100*m.sum()/n:5.1f}%)")
        print(f"   median abs error : {np.median(err[valid]):6.2f} m")
        print(f"   mean abs error   : {err[valid].mean():6.2f} m")
        print(f"   worst            : {err[valid].max():6.2f} m")
        print()

    band_table("MONOCULAR DEPTH", err_d, valid_d)
    band_table("HOMOGRAPHY (flat road)", err_h, valid_h)

    print("--- median abs error BY RANGE ---")
    print(f"{'range':>10s} {'n':>4s} {'depth':>9s} {'homography':>12s}")
    for lo, hi in [(4, 10), (10, 20), (20, 35), (35, 60)]:
        m = (arr_true >= lo) & (arr_true < hi)
        md = np.median(err_d[m & valid_d]) if (m & valid_d).sum() else float("nan")
        mh = np.median(err_h[m & valid_h]) if (m & valid_h).sum() else float("nan")
        print(f"  {lo:2d}-{hi:2d} m {m.sum():5d} {md:8.2f}m {mh:11.2f}m")


if __name__ == "__main__":
    main()
