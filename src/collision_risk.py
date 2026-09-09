"""
Computes Time-To-Collision (TTC) and Closest Point of Approach (CPA) for
each tracked vehicle, from its smoothed relative position/velocity (from
smooth_trajectory.py), and classifies risk from them.

Everything here is relative to ego (see ground_plane.py / kalman_filter.py):
position (x, z) and velocity (vx, vz) describe how the gap between ego and
the other vehicle is changing, not either vehicle's road speed.

TTC: assumes the vehicle stays on its current relative path -- time until
the forward gap (z) reaches zero, given the current closing rate. Only
meaningful for something actually closing in the forward direction.

CPA: more general -- the time at which the full relative distance (not just
forward distance) will be smallest, and how small. Catches closing vehicles
that aren't directly ahead (e.g. a crossing or merging vehicle), which TTC
alone would miss since TTC only looks at forward distance shrinking to 0.

RISK_THRESHOLDS below are heuristic starting points from general driving-
safety literature (TTC < 2s = commonly flagged as high risk, < 4s = caution),
not tuned against this project's own data -- a candidate for calibration
against the CCD dataset's real accident timing once we get there.
"""

from ultralytics import YOLO

from detect_vehicles import INPUT_VIDEO, draw_labeled_box, open_video_io
from smooth_trajectory import CAMERA, track_and_smooth
from track_vehicles import MODEL_WEIGHTS

OUTPUT_VIDEO = "data/risk_output.mp4"

# Colors are BGR (OpenCV convention).
RISK_COLORS = {
    "CRITICAL": (0, 0, 255),   # red
    "WARNING": (0, 200, 255),  # amber
    "SAFE": (0, 255, 0),       # green
}


def compute_ttc(x, z, vx, vz):
    """Time until forward distance z reaches 0, assuming constant relative
    velocity. Returns None if not closing (vz >= 0, i.e. steady or receding)."""
    closing_rate = -vz
    if closing_rate <= 1e-3:
        return None
    return z / closing_rate


def compute_cpa(x, z, vx, vz):
    """Time and distance at the closest point of approach, given current
    relative position (x, z) and relative velocity (vx, vz). Returns
    (t_cpa, distance_at_cpa). t_cpa is clamped to >= 0 -- if the vehicles
    are already moving apart, the closest approach was in the past, so the
    "future" closest distance is just the current distance, now."""
    r_dot_v = x * vx + z * vz
    v_dot_v = vx * vx + vz * vz

    if v_dot_v <= 1e-6:
        return 0.0, (x ** 2 + z ** 2) ** 0.5

    t_cpa = max(0.0, -r_dot_v / v_dot_v)
    cpa_x = x + t_cpa * vx
    cpa_z = z + t_cpa * vz
    return t_cpa, (cpa_x ** 2 + cpa_z ** 2) ** 0.5


def classify_risk(ttc, cpa_time, cpa_distance):
    if (ttc is not None and ttc <= 2.0) or (cpa_distance <= 2.0 and cpa_time <= 3.0):
        return "CRITICAL"
    if (ttc is not None and ttc <= 4.0) or (cpa_distance <= 4.0 and cpa_time <= 5.0):
        return "WARNING"
    return "SAFE"


def main():
    model = YOLO(MODEL_WEIGHTS)

    cap, writer = open_video_io(INPUT_VIDEO, OUTPUT_VIDEO)
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))
    fps = cap.get(5) or 25
    dt = 1.0 / fps

    frame_count = 0
    risk_counts = {"CRITICAL": 0, "WARNING": 0, "SAFE": 0}

    for frame, frame_count, detections in track_and_smooth(model, cap, image_width, image_height, dt, CAMERA):
        for det in detections:
            x, z = det["kf"].position
            vx, vz = det["kf"].velocity

            ttc = compute_ttc(x, z, vx, vz)
            cpa_time, cpa_distance = compute_cpa(x, z, vx, vz)
            risk = classify_risk(ttc, cpa_time, cpa_distance)
            risk_counts[risk] += 1

            ttc_str = f"{ttc:.1f}s" if ttc is not None else "N/A"
            label = [
                f"ID {det['track_id']} {risk}",
                f"TTC: {ttc_str}",
                f"CPA: {cpa_distance:.1f}m @ {cpa_time:.1f}s",
            ]
            draw_labeled_box(
                frame, det["x1"], det["y1"], det["x2"], det["y2"],
                label, color=RISK_COLORS[risk],
            )

            print(f"frame {frame_count}: id={det['track_id']} z={z:.2f}m ttc={ttc_str} "
                  f"cpa_dist={cpa_distance:.2f}m cpa_time={cpa_time:.2f}s risk={risk}")

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_count}")
    print(f"Risk readings: {risk_counts}")
    print(f"Output written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
