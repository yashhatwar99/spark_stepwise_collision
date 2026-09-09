"""
Smooths each tracked vehicle's ground position using a per-vehicle Kalman
filter (constant-acceleration model) instead of raw frame-to-frame
homography measurements, and produces velocity (and acceleration) estimates
that raw positions alone don't give us.

One filter instance per ByteTrack id, created the first time that id gets a
valid ("ok") ground measurement. On frames where the measurement is
"too_close" or "invalid", the filter still runs its predict step (coasting
on the motion model) but skips the update step -- so a vehicle passing
through the ego-hood occlusion zone keeps a continuously extrapolated
position/velocity instead of the signal just cutting out.

track_and_smooth() is the reusable core (tracking + per-vehicle Kalman
filtering) -- other scripts (e.g. the TTC/risk step) build on it instead of
duplicating the tracking loop.
"""

from ultralytics import YOLO

from detect_vehicles import INPUT_VIDEO, VEHICLE_CLASS_IDS, draw_labeled_box, open_video_io
from ground_plane import DASHCAM, measure_ground_position
from kalman_filter import ConstantAccelerationKalmanFilter
from track_vehicles import DETECT_IMGSZ, DEVICE, MIN_CONFIDENCE, MODEL_WEIGHTS, TRACKER_CONFIG

CAMERA = DASHCAM
OUTPUT_VIDEO = "data/smoothed_output.mp4"


def track_and_smooth(model, cap, image_width, image_height, dt, camera=CAMERA, verbose=True):
    """Runs detection+tracking frame by frame, maintaining one Kalman filter
    per track_id for the lifetime of this call. Yields (frame, frame_count,
    detections), where detections is a list of dicts with keys: track_id,
    class_id, x1, y1, x2, y2, status, kf (the ConstantAccelerationKalmanFilter
    for that track, already predicted/updated for this frame)."""
    filters = {}  # track_id -> ConstantAccelerationKalmanFilter
    last_seen_frame = {}  # track_id -> frame_count it was last predicted/updated on
    frame_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        results = model.track(
            frame, persist=True, tracker=TRACKER_CONFIG, device=DEVICE,
            imgsz=DETECT_IMGSZ, conf=MIN_CONFIDENCE, verbose=False,
        )[0]

        detections = []

        if results.boxes.id is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                status, ground_pos = measure_ground_position(x1, y1, x2, y2, image_width, image_height, camera)

                if track_id not in filters:
                    if status != "ok":
                        # No valid measurement yet to start this filter from.
                        if verbose:
                            print(f"frame {frame_count}: id={track_id} waiting for first valid measurement ({status})")
                        continue
                    filters[track_id] = ConstantAccelerationKalmanFilter(ground_pos[0], ground_pos[1], dt)
                    last_seen_frame[track_id] = frame_count

                kf = filters[track_id]
                elapsed_frames = frame_count - last_seen_frame[track_id]
                last_seen_frame[track_id] = frame_count

                if verbose and elapsed_frames > 1:
                    print(f"frame {frame_count}: id={track_id} was missing for "
                          f"{elapsed_frames - 1} frame(s), predicting across the gap")

                kf.predict(elapsed_frames)
                if status == "ok":
                    kf.update(*ground_pos)

                detections.append({
                    "track_id": track_id, "class_id": class_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "status": status, "kf": kf,
                })

        yield frame, frame_count, detections


def main():
    model = YOLO(MODEL_WEIGHTS)

    cap, writer = open_video_io(INPUT_VIDEO, OUTPUT_VIDEO)
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))
    fps = cap.get(5) or 25
    dt = 1.0 / fps

    frame_count = 0
    seen_track_ids = set()

    for frame, frame_count, detections in track_and_smooth(model, cap, image_width, image_height, dt):
        for det in detections:
            seen_track_ids.add(det["track_id"])
            x_m, z_m = det["kf"].position
            vx, vz = det["kf"].velocity
            speed = (vx ** 2 + vz ** 2) ** 0.5

            label = [
                f"ID {det['track_id']}",
                f"Dist: {z_m:.1f} m",
                f"Speed: {speed:.1f} m/s",
            ]
            draw_labeled_box(frame, det["x1"], det["y1"], det["x2"], det["y2"], label)

            print(f"frame {frame_count}: id={det['track_id']} status={det['status']} "
                  f"x={x_m:.2f}m z={z_m:.2f}m vz={vz:.2f}m/s speed={speed:.2f}m/s")

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_count}")
    print(f"Tracked vehicles smoothed: {len(seen_track_ids)}")
    print(f"Output written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
