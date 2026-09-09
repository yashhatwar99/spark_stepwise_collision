"""
Adds real-world distance to the tracked vehicles, using the ground-plane
projection in ground_plane.py. For each tracked vehicle, we take the
bottom-center of its box (where it touches the road) and convert that
pixel to a (X, Z) ground position in meters.
"""

from ultralytics import YOLO

from detect_vehicles import INPUT_VIDEO, VEHICLE_CLASS_IDS, draw_labeled_box, open_video_io
from ground_plane import DASHCAM, measure_ground_position
from track_vehicles import DETECT_IMGSZ, DEVICE, MIN_CONFIDENCE, MODEL_WEIGHTS, TRACKER_CONFIG

CAMERA = DASHCAM

OUTPUT_VIDEO = "data/distance_output.mp4"


def main():
    model = YOLO(MODEL_WEIGHTS)

    cap, writer = open_video_io(INPUT_VIDEO, OUTPUT_VIDEO)
    image_width = int(cap.get(3))
    image_height = int(cap.get(4))

    frame_count = 0
    all_distances = []
    too_close_count = 0
    invalid_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        results = model.track(
            frame, persist=True, tracker=TRACKER_CONFIG, device=DEVICE,
            imgsz=DETECT_IMGSZ, conf=MIN_CONFIDENCE, verbose=False,
        )[0]

        if results.boxes.id is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue

                track_id = int(box.id[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                status, ground_pos = measure_ground_position(x1, y1, x2, y2, image_width, image_height, CAMERA)

                if status == "too_close":
                    too_close_count += 1
                    label = f"ID {track_id} {VEHICLE_CLASS_IDS[class_id]} TOO CLOSE"
                    color = (0, 0, 255)  # red -- stands out from normal green boxes
                elif status == "invalid":
                    invalid_count += 1
                    label = f"ID {track_id} {VEHICLE_CLASS_IDS[class_id]} dist=?"
                    color = (0, 255, 0)
                else:
                    x_m, z_m = ground_pos
                    all_distances.append(z_m)
                    label = f"ID {track_id} {VEHICLE_CLASS_IDS[class_id]} {z_m:.1f}m"
                    color = (0, 255, 0)  # green -- normal, trusted reading

                print(f"frame {frame_count}: id={track_id} {label.split(' ', 2)[-1]}")
                draw_labeled_box(frame, x1, y1, x2, y2, label, color)

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_count}")
    if all_distances:
        print(f"Distance range seen: {min(all_distances):.1f}m to {max(all_distances):.1f}m")
    print(f"{too_close_count} reading(s) were TOO CLOSE to measure (ground-contact point occluded by ego hood)")
    print(f"{invalid_count} reading(s) were invalid (above horizon / no ground-plane intersection)")
    print(f"Output written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
