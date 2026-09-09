"""
Are the predicted future positions actually right?

A trajectory drawing always looks plausible -- the boxes march off in a
straight line whatever the data says. The only honest check is to wait, see
where the vehicle really went, and compare.

Predicted positions are scored against the annotated position of the SAME
vehicle N keyframes later, in the ego frame of that later moment, which is
the quantity a collision decision actually depends on.

Two baselines are reported alongside, because a prediction is only worth
something if it beats the trivial alternatives:
    "stays put"   assume the vehicle does not move relative to us
    "perfect now" use the TRUE current position, then extrapolate
The gap between our prediction and "perfect now" is the cost of our
perception error; the gap to "stays put" is what the prediction adds.

Usage:
    python -m src3.evaluate_prediction --horizon 2
"""

import argparse
from collections import defaultdict

import numpy as np

from src2 import config
from src2.nuscenes_source import NuScenesSource

FRAME_DT = 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=2, help="frames ahead (0.5 s each)")
    parser.add_argument("--history", type=int, default=2,
                        help="frames of history used to measure velocity")
    args = parser.parse_args()

    source = NuScenesSource()
    nusc = source.nusc

    # Per-instance history of (frame index, global position, ego pose).
    seen = defaultdict(dict)
    ego_pose = {}
    cam_rotation = {}
    for i in range(len(source)):
        frame = source.frame(i)
        ego_pose[i] = frame.cam_to_global_t
        cam_rotation[i] = frame.cam_to_global_R
        sample = nusc.get("sample", frame.sample_token)
        _, boxes, _ = nusc.get_sample_data(sample["data"]["CAM_FRONT"])
        for b in boxes:
            if not b.name.startswith("vehicle."):
                continue
            ann = nusc.get("sample_annotation", b.token)
            if int(ann["visibility_token"]) < 3:
                continue
            if not (0 < b.center[2] < 60):
                continue
            global_pos = frame.to_global(np.array(b.center).reshape(3, 1)).ravel()
            seen[ann["instance_token"]][i] = (np.array(b.center), global_pos)

    ours, stays, speeds = [], [], []
    for instance, history in seen.items():
        frames = sorted(history)
        for k, now in enumerate(frames):
            past = now - args.history
            future = now + args.horizon
            if past not in history or future not in history:
                continue
            if (now - past) != args.history or (future - now) != args.horizon:
                continue

            _, past_global = history[past]
            _, now_global = history[now]
            future_cam_true, _ = history[future]

            velocity = (now_global - past_global) / (args.history * FRAME_DT)
            ahead = args.horizon * FRAME_DT
            predicted_global = now_global + velocity * ahead

            # Express both in the ego frame of the FUTURE moment.
            R_future = cam_rotation[future]
            t_future = ego_pose[future]
            predicted_cam = R_future.T @ (predicted_global - t_future)

            truth_xz = np.array([future_cam_true[0], future_cam_true[2]])
            pred_xz = np.array([predicted_cam[0], predicted_cam[2]])
            ours.append(float(np.linalg.norm(pred_xz - truth_xz)))

            # baseline: assume it stays where it is now, in the world
            stay_cam = R_future.T @ (now_global - t_future)
            stays.append(float(np.linalg.norm(np.array([stay_cam[0], stay_cam[2]]) - truth_xz)))
            speeds.append(float(np.hypot(velocity[0], velocity[1])))

    print(f"horizon {args.horizon} frames ({args.horizon * FRAME_DT:.1f} s ahead), "
          f"velocity from {args.history} frames of history")
    print(f"observations: {len(ours)}")
    print()
    o = np.array(ours)
    s = np.array(stays)
    print(f"{'method':>28s} {'median':>9s} {'mean':>8s} {'90th':>8s} {'within 2m':>10s}")
    print(f"{'constant-velocity forecast':>28s} {np.median(o):8.2f}m {o.mean():7.2f}m "
          f"{np.percentile(o, 90):7.2f}m {100 * (o < 2).mean():9.0f}%")
    print(f"{'baseline: assume it stays put':>28s} {np.median(s):8.2f}m {s.mean():7.2f}m "
          f"{np.percentile(s, 90):7.2f}m {100 * (s < 2).mean():9.0f}%")
    print()
    better = 100 * (o < s).mean()
    print(f"forecast beats the do-nothing baseline on {better:.0f}% of observations")
    print()
    # Split by whether the vehicle actually moves. Most vehicles in these
    # scenes are parked, and for a parked car "assume it stays put" is exactly
    # right -- so an overall median is dominated by the easy cases and hides
    # what the forecast is for.
    sp = np.array(speeds)
    print("split by the vehicle's own speed:")
    print(f"{'speed':>12s} {'n':>6s} {'forecast':>10s} {'stays put':>11s}")
    for lo, hi, label in [(0, 0.5, "parked"), (0.5, 3, "0.5-3 m/s"),
                          (3, 8, "3-8 m/s"), (8, 999, "8+ m/s")]:
        m = (sp >= lo) & (sp < hi)
        if m.sum() < 5:
            continue
        print(f"{label:>12s} {m.sum():6d} {np.median(o[m]):9.2f}m {np.median(s[m]):10.2f}m")


if __name__ == "__main__":
    main()
