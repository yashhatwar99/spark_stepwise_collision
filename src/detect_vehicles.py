"""
Vehicle detection with YOLOv8 (pretrained, no training needed).

Reads a video, runs YOLOv8 on each frame, keeps only vehicle classes
(car, motorcycle, bus, truck), draws boxes, and writes an output video
so we can visually check the boxes look right.
"""

import cv2
from ultralytics import YOLO

INPUT_VIDEO = "data/CarCrash/videos/Crash/000004.mp4"
OUTPUT_VIDEO = "data/detections_output.mp4"

# COCO class ids (the classes YOLOv8 is pretrained on) that are vehicles.
VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

DEVICE = 0  # RTX 3050 (GPU 0); falls back to "cpu" if no CUDA device is found


def open_video_io(input_path, output_path):
    """Open a video for reading and create a writer with matching fps/size."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    return cap, writer


def draw_labeled_box(frame, x1, y1, x2, y2, label, color=(0, 255, 0)):
    """label can be a single string, or a list of strings for multiple
    stacked lines. Each line gets a filled background behind the text so
    it stays readable regardless of what's behind it in the frame."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    lines = [label] if isinstance(label, str) else label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    line_height = 24

    # Stack lines upward starting just above the box's top edge.
    baseline_y = y1 - 10
    for line in reversed(lines):
        (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        top_left = (x1, max(0, baseline_y - text_h - 6))
        bottom_right = (x1 + text_w + 8, max(0, baseline_y + 4))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.putText(
            frame, line, (x1 + 4, max(text_h, baseline_y)),
            font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
        )
        baseline_y -= line_height


def main():
    model = YOLO("yolov8n.pt")  # nano model: smallest/fastest, CPU-friendly

    cap, writer = open_video_io(INPUT_VIDEO, OUTPUT_VIDEO)

    frame_count = 0
    vehicle_detections_total = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        results = model(frame, device=DEVICE, verbose=False)[0]

        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id not in VEHICLE_CLASS_IDS:
                continue

            vehicle_detections_total += 1
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = f"{VEHICLE_CLASS_IDS[class_id]} {confidence:.2f}"
            draw_labeled_box(frame, x1, y1, x2, y2, label)

        writer.write(frame)

    cap.release()
    writer.release()

    print(f"Frames processed: {frame_count}")
    print(f"Total vehicle detections across all frames: {vehicle_detections_total}")
    print(f"Output written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
