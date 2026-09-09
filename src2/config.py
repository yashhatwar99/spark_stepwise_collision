"""
Single source of truth for paths, model choices and thresholds.

Everything tunable lives here rather than being re-declared per script. The
previous codebase drifted precisely because it did not do this: two files
each defined their own `VEHICLE_CLASS_IDS` with different contents, and two
defined `MIN_CONFIDENCE` with different values (0.3 vs 0.4), so results
depended on which script you happened to run.
"""

from pathlib import Path

# --- paths ------------------------------------------------------------
# Anchored to this file, so scripts work regardless of the shell's cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output2"

NUSCENES_DATAROOT = DATA_DIR / "nuScenes" / "v1.0-mini"
NUSCENES_VERSION = "v1.0-mini"

# --- detector ---------------------------------------------------------
# yolov8s (small), chosen on measured evidence rather than by default.
# Benchmarked against nuScenes ground truth (src2/evaluate_detection.py):
# small finds 59.3% of vehicles vs nano's 52.1% at the same confidence, and
# costs nothing to do it -- both run at ~16 ms/frame (the detector is not the
# bottleneck) and both use well under 0.1 GB of VRAM. Nano remains a one-line
# revert if a future stage makes VRAM tight.
YOLO_WEIGHTS = PROJECT_ROOT / "yolov8s.pt"
DEVICE = 0          # CUDA device index; use "cpu" to force CPU
DETECT_IMGSZ = 640  # raising this to 960 previously hurt: more false positives, not better recall
MIN_CONFIDENCE = 0.35

# COCO class ids the detector should keep, with display names. Bicycle is
# excluded deliberately -- this project reasons about vehicle collision
# geometry, and a bicycle's dimensions break the car size priors used later.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# --- camera -----------------------------------------------------------
# Which nuScenes camera to run on. CAM_FRONT is the forward-facing one, so
# it matches the dashcam geometry this project is ultimately about.
CAMERA_CHANNEL = "CAM_FRONT"

# --- drawing ----------------------------------------------------------
BOX_COLOR = (0, 200, 0)         # BGR
TEXT_COLOR = (255, 255, 255)
BOX_THICKNESS = 2
FONT_SCALE = 0.5

# --- lidar / 3D localisation -----------------------------------------
LIDAR_CHANNEL = "LIDAR_TOP"

# Median dimensions of a real car, measured from 7,619 nuScenes annotations
# rather than looked up: w=1.94 l=4.61 h=1.64. Rounded slightly and used as
# the size prior, since a sparse laser sweep sees one or two faces and cannot
# measure a vehicle's true extent.
CAR_SIZE_WLH = (1.95, 4.63, 1.74)

# Depth gap that separates a vehicle from whatever is behind it, in metres.
# Slightly under half a car length, so it splits vehicle-from-background
# without splitting one vehicle in two.
FOREGROUND_DEPTH_GAP = 2.0

# --- tracking ---------------------------------------------------------
# Below this speed a vehicle's direction of travel is unreadable, because
# each position carries ~1 m of error and keyframes are 0.5 s apart, so
# velocity noise is roughly 2 m/s. Measured heading error by speed:
#   1-3 m/s  -> 68.4 deg   (noise dominates; useless)
#   3-8 m/s  -> 13.6 deg
#   8+  m/s  ->  1.9 deg   (82% within 15 deg)
# Raising the threshold trades coverage for quality: 1 m/s gives 636 headings
# at 11.8 deg median, 3 m/s gives 439 at 4.3 deg, 5 m/s gives 302 at 2.5 deg.
MIN_SPEED_FOR_HEADING = 3.0

# Association gate. Wide enough for half a second of traffic motion, tight
# enough not to swap vehicles in adjacent lanes -- and swaps are what produce
# the worst headings, since a track jumping between two cars invents motion
# that neither made. Measured, over 10 scenes:
#   gate 3 m -> 177 headings, 73% within 15 deg, 14% pointing >90 deg wrong
#   gate 4 m -> 252 headings, 71%, 16%
#   gate 6 m -> 439 headings, 63%, 20%
#   gate 8 m -> 553 headings, 54%, 26%   (median doubles: swaps dominate)
# 4 m chosen over the wider gates because for a safety system a confidently
# WRONG heading is worse than a refused one.
TRACK_GATE_M = 4.0
