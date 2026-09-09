"""
Vehicle collision-risk perception, rebuilt.

Structure, and why it differs from the first attempt:
  config.py              all paths/thresholds in one place
  detection.py           image -> Detection objects (2D, pixels only)
  nuscenes_source.py     frames + calibration; ground truth behind a separate
                         call so perception cannot read the answers
  lidar.py               LiDAR sweep -> camera frame, and isolating one vehicle
  localize.py            detection + LiDAR points -> 3D box
  drawing.py             rendering only
  render_3d.py           the visual output
  evaluate_detection.py    scores detection against ground truth
  evaluate_localization.py scores 3D position against ground truth

Run as modules from the project root, e.g.
    python -m src2.render_3d
    python -m src2.evaluate_localization
so imports resolve as a package and no sys.path manipulation is needed.
"""
