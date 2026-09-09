"""
Vehicle detection: image in, a list of Detection objects out.

Kept deliberately free of any drawing, file IO or dataset knowledge, so the
same detector can be pointed at nuScenes frames, dashcam video, or simulated
images without change.
"""

from dataclasses import dataclass

from ultralytics import YOLO

from . import config


@dataclass
class Detection:
    """One detected vehicle in image coordinates.

    Pixel coordinates only -- no distance, no 3D. Turning this into a
    real-world position is a separate, later stage, kept separate so a
    detection failure and a distance-estimation failure never get confused
    for one another.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def center(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def bottom_center(self):
        """Where the vehicle meets the road, assuming nothing occludes it.

        This is the point any ground-plane method needs, and the point that
        becomes unreliable when a vehicle is cut off by the image edge or
        hidden behind another -- worth checking before trusting it.
        """
        return (self.x1 + self.x2) / 2.0, self.y2


class VehicleDetector:
    """Thin wrapper over YOLO that yields only the classes we care about."""

    def __init__(self, weights=None, device=None, min_confidence=None, imgsz=None):
        self.weights = str(weights or config.YOLO_WEIGHTS)
        self.device = config.DEVICE if device is None else device
        self.min_confidence = config.MIN_CONFIDENCE if min_confidence is None else min_confidence
        self.imgsz = config.DETECT_IMGSZ if imgsz is None else imgsz
        self.model = YOLO(self.weights)

    def detect(self, image):
        """Detect vehicles in a BGR image array. Returns a list of Detection."""
        result = self.model(
            image,
            device=self.device,
            imgsz=self.imgsz,
            conf=self.min_confidence,
            verbose=False,
        )[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id not in config.VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=float(box.conf),
                    class_id=class_id,
                    class_name=config.VEHICLE_CLASSES[class_id],
                )
            )
        return detections
