"""
How accurate is stereo distance, really?

Scores triangulated depth against the LiDAR sweep captured at the same
instant, per pixel, so there is a known answer everywhere the laser hit
something. Reported by range, because the whole point about stereo is that
its error grows with the square of distance -- a single averaged number
would hide exactly the behaviour that decides whether stereo is usable.

The measured error is also compared against the theoretical prediction
Z^2 / (f*B) per pixel of disparity error. If the measurement tracks the
theory, the implementation is sound and the remaining error is physics, not
a bug -- and the disparity matching accuracy can be read straight off it.

Usage (from the project root):
    python -m src2.evaluate_stereo
    python -m src2.evaluate_stereo --frames 40
"""

import argparse

import numpy as np

from .kitti_source import KittiStereoSource
from .stereo import compute_disparity, depth_from_disparity, make_matcher

BANDS = [(3, 10), (10, 20), (20, 30), (30, 45), (45, 70)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=25)
    parser.add_argument("--step", type=int, default=4)
    args = parser.parse_args()

    source = KittiStereoSource()
    matcher = make_matcher()

    f, b = source.focal_length_px, source.baseline_m
    print(f"drive            : {source.drive}")
    print(f"stereo pairs     : {len(source)}")
    print(f"focal length     : {f:.1f} px")
    print(f"baseline         : {b:.3f} m")
    print(f"f * B            : {f * b:.1f}  (depth = this / disparity)")
    print()

    truths, estimates = [], []
    for i in range(0, min(args.frames * args.step, len(source)), args.step):
        frame = source.frame(i)
        if frame.lidar_points_cam is None:
            continue
        disparity = compute_disparity(frame.left, frame.right, matcher)
        depth = depth_from_disparity(disparity, f, b)

        uv, lidar_depth = source.project_lidar(frame.lidar_points_cam, frame.left.shape)
        cols = uv[0].astype(int)
        rows = uv[1].astype(int)
        stereo_depth = depth[rows, cols]

        good = np.isfinite(stereo_depth)
        truths.append(lidar_depth[good])
        estimates.append(stereo_depth[good])

    truth = np.concatenate(truths)
    estimate = np.concatenate(estimates)
    error = np.abs(estimate - truth)

    print(f"points compared  : {len(truth)}")
    print()
    print("STEREO DEPTH vs LiDAR")
    print(f"{'range':>10s} {'n':>8s} {'median err':>11s} {'as % of range':>14s} "
          f"{'theory (1px)':>13s} {'implied px err':>15s}")
    for lo, hi in BANDS:
        m = (truth >= lo) & (truth < hi)
        if m.sum() < 50:
            continue
        mid = (lo + hi) / 2
        median = np.median(error[m])
        theory = mid ** 2 / (f * b)                  # metres per pixel of disparity error
        print(f"  {lo:2d}-{hi:2d} m {m.sum():8d} {median:10.2f}m "
              f"{100 * median / mid:13.1f}% {theory:12.2f}m {median / theory:14.2f}")
    print()
    print("The last column is how many pixels of disparity error the measured")
    print("error corresponds to. If it stays roughly constant across ranges,")
    print("the matcher has a fixed pixel accuracy and the growing metric error")
    print("is pure geometry -- which is the expected, unavoidable result.")


if __name__ == "__main__":
    main()
