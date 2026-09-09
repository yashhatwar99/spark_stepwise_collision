"""
KITTI raw stereo drives: rectified left/right image pairs, their calibration,
and the LiDAR sweep that lets us score the stereo depth against a known answer.

KITTI is used here rather than nuScenes for one reason: nuScenes has six
cameras but no stereo pair. Its cameras sit 0.51 m apart yet aim 55-110
degrees apart, arranged for 360-degree coverage, so nothing in it can be
triangulated. KITTI's two colour cameras face the same way, 0.54 m apart,
which is what stereo needs.

Layout after unzipping (both archives extract into the same 2011_09_26/ tree):
    2011_09_26/
        calib_cam_to_cam.txt        intrinsics + rectification for all 4 cameras
        calib_velo_to_cam.txt       LiDAR -> camera transform
        2011_09_26_drive_XXXX_sync/
            image_02/data/*.png     left  colour, rectified
            image_03/data/*.png     right colour, rectified
            velodyne_points/data/*.bin
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config


def _read_calib(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        try:
            values[key.strip()] = np.array([float(v) for v in rest.split()])
        except ValueError:
            pass          # non-numeric header lines (calib_time etc.)
    return values


@dataclass
class StereoFrame:
    left: np.ndarray            # BGR
    right: np.ndarray           # BGR
    focal_length_px: float
    baseline_m: float
    cx: float
    cy: float
    lidar_points_cam: np.ndarray    # (3, N) in the rectified left-camera frame
    index: int


class KittiStereoSource:
    def __init__(self, root=None, drive=None):
        self.root = Path(root or (config.DATA_DIR / "kitti" / "2011_09_26"))
        self.drive = drive or "2011_09_26_drive_0009_sync"
        self.drive_dir = self.root / self.drive

        cam = _read_calib(self.root / "calib_cam_to_cam.txt")
        velo = _read_calib(self.root / "calib_velo_to_cam.txt")

        # P_rect_0x = [[f, 0, cx, -f*bx], [0, f, cy, 0], [0, 0, 1, 0]]
        p2 = cam["P_rect_02"].reshape(3, 4)
        p3 = cam["P_rect_03"].reshape(3, 4)
        self.focal_length_px = float(p2[0, 0])
        self.cx, self.cy = float(p2[0, 2]), float(p2[1, 2])
        # Each camera's x offset from the rectified origin; their difference
        # is the stereo baseline.
        bx2 = -p2[0, 3] / p2[0, 0]
        bx3 = -p3[0, 3] / p3[0, 0]
        self.baseline_m = float(abs(bx3 - bx2))
        self.P2 = p2

        # LiDAR -> unrectified cam0, then the rectifying rotation.
        r_velo = velo["R"].reshape(3, 3)
        t_velo = velo["T"].reshape(3, 1)
        self.velo_to_cam = np.vstack([np.hstack([r_velo, t_velo]), [0, 0, 0, 1]])
        r_rect = np.eye(4)
        r_rect[:3, :3] = cam["R_rect_00"].reshape(3, 3)
        self.r_rect = r_rect

        self.left_files = sorted((self.drive_dir / "image_02" / "data").glob("*.png"))
        self.right_files = sorted((self.drive_dir / "image_03" / "data").glob("*.png"))
        self.lidar_files = sorted((self.drive_dir / "velodyne_points" / "data").glob("*.bin"))

    def __len__(self):
        return min(len(self.left_files), len(self.right_files))

    def _lidar_in_camera(self, index):
        raw = np.fromfile(self.lidar_files[index], dtype=np.float32).reshape(-1, 4)
        # Drop points behind the sensor before transforming: KITTI's LiDAR is
        # a full 360-degree sweep and rear points would fold onto the image.
        raw = raw[raw[:, 0] > 0.5]
        homogeneous = np.hstack([raw[:, :3], np.ones((len(raw), 1))]).T
        in_cam = self.r_rect @ self.velo_to_cam @ homogeneous
        return in_cam[:3, :]

    def frame(self, index):
        left = cv2.imread(str(self.left_files[index]))
        right = cv2.imread(str(self.right_files[index]))
        if left is None or right is None:
            raise FileNotFoundError(f"could not read stereo pair {index}")
        lidar = self._lidar_in_camera(index) if index < len(self.lidar_files) else None
        return StereoFrame(
            left=left, right=right,
            focal_length_px=self.focal_length_px,
            baseline_m=self.baseline_m,
            cx=self.cx, cy=self.cy,
            lidar_points_cam=lidar,
            index=index,
        )

    def project_lidar(self, points_cam, image_shape):
        """(3, N) camera-frame points -> (pixels_uv, depths) inside the image."""
        infront = points_cam[2, :] > 1.0
        points = points_cam[:, infront]
        homogeneous = np.vstack([points, np.ones((1, points.shape[1]))])
        uv = self.P2 @ homogeneous
        uv = uv[:2] / uv[2]
        h, w = image_shape[:2]
        on_image = (uv[0] >= 0) & (uv[0] < w) & (uv[1] >= 0) & (uv[1] < h)
        return uv[:, on_image], points[2, on_image]
