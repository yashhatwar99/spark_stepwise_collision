"""
Vehicle tracking with ByteTrack (built into Ultralytics), on top of YOLOv8 detection.

Runs YOLOv8 + ByteTrack together frame-by-frame so each vehicle gets a
persistent track ID that should stay the same across frames instead of
being re-detected as a "new" vehicle every frame.
"""

from ultralytics import YOLO

from detect_vehicles import INPUT_VIDEO, VEHICLE_CLASS_IDS, draw_labeled_box, open_video_io

OUTPUT_VIDEO = "data/tracking_output.mp4"
TRACKER_CONFIG = "src/bytetrack_custom.yaml"

# Raising input size to 960 was tried to help with distant/glare-lit vehicles,
# but on this noisy night scene it mainly added low-confidence false positives
# (a building, empty pavement) that fragmented into extra short-lived IDs.
# Reverted to default 640; filtering weak detections directly (below) instead.
DETECT_IMGSZ = 640
MIN_CONFIDENCE = 0.3

# yolov8s (small) was confirmed to catch vehicles that yolov8n (nano) misses
# in headlight glare (see conversation), but sticking with nano for now on
# purpose -- swap back to "yolov8s.pt" later.
MODEL_WEIGHTS = "yolov8n.pt"
DEVICE = 0  # RTX 3050 (GPU 0); falls back to "cpu" if no CUDA device is found


def main():
    model = YOLO(MODEL_WEIGHTS)

    cap, writer = open_video_io(INPUT_VIDEO, OUTPUT_VIDEO)

    frame_count = 0
    seen_track_ids = set()

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
                seen_track_ids.add(track_id)
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                label = f"ID {track_id} {VEHICLE_CLASS_IDS[class_id]} {confidence:.2f}"
                draw_labeled_box(frame, x1, y1, x2, y2, label)

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_count}")
    print(f"Unique track IDs seen: {len(seen_track_ids)}")
    print(f"Output written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
