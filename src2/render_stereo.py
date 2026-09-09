"""
Visualise stereo depth: the left image, and the depth triangulated from the
pair, with LiDAR points overlaid so agreement (or disagreement) is visible
rather than merely tabulated.

Usage:
    python -m src2.render_stereo --index 40
"""

import argparse

import cv2
import numpy as np

from . import config
from .kitti_source import KittiStereoSource
from .stereo import compute_disparity, depth_from_disparity, make_matcher

MAX_DISPLAY_M = 50.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=40)
    args = parser.parse_args()

    source = KittiStereoSource()
    frame = source.frame(args.index)
    disparity = compute_disparity(frame.left, frame.right, make_matcher())
    depth = depth_from_disparity(disparity, frame.focal_length_px, frame.baseline_m)

    # Colourise: near = warm, far = cool. Unmatched pixels stay black so a
    # gap in the data never masquerades as a distance.
    shown = np.clip(np.nan_to_num(depth, nan=MAX_DISPLAY_M), 0, MAX_DISPLAY_M)
    normalised = (255 * (1 - shown / MAX_DISPLAY_M)).astype(np.uint8)
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_TURBO)
    coloured[~np.isfinite(depth)] = 0

    # LiDAR truth on top, same colour scale, as small filled dots.
    uv, lidar_depth = source.project_lidar(frame.lidar_points_cam, frame.left.shape)
    for (u, v), d in zip(uv.T, lidar_depth):
        if d > MAX_DISPLAY_M:
            continue
        c = cv2.applyColorMap(
            np.array([[int(255 * (1 - d / MAX_DISPLAY_M))]], np.uint8), cv2.COLORMAP_TURBO
        )[0, 0]
        cv2.circle(coloured, (int(u), int(v)), 2, tuple(int(x) for x in c), -1)
        cv2.circle(coloured, (int(u), int(v)), 2, (255, 255, 255), 1)

    cv2.putText(coloured, "stereo depth (colour) + LiDAR truth (ringed dots) -- matching colour = agreement",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    labelled = frame.left.copy()
    cv2.putText(labelled, f"KITTI {source.drive}  frame {args.index}  baseline {frame.baseline_m:.3f} m",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_DIR / f"stereo_{args.index:04d}.jpg"
    cv2.imwrite(str(out), np.vstack([labelled, coloured]))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
