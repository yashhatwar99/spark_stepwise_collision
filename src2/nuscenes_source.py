"""
Frame source: hands out camera images plus the calibration that goes with
them, from the nuScenes v1.0-mini dataset.

Why nuScenes is the development dataset here: it ships real camera
intrinsics and extrinsics, and LiDAR-verified 3D boxes for every vehicle.
That means any distance or size this project estimates can be scored against
a known answer, instead of being eyeballed. Dashcam footage cannot do that --
it has no ground truth at all.

The known cost of that choice: nuScenes' 10 mini scenes contain ordinary
urban driving and no collisions, so it can validate perception accuracy but
cannot validate collision prediction.

Ground-truth access lives in `ground_truth_boxes()`, deliberately separate
from `frames()`, so that a perception path cannot accidentally read the
answers it is supposed to be predicting.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from . import config
from .lidar import load_lidar_in_camera


@dataclass
class Frame:
    """One camera image plus everything needed to reason about it in 3D."""

    image: np.ndarray          # BGR, as OpenCV expects
    intrinsics: np.ndarray     # 3x3 camera matrix K
    camera_height_m: float     # height of the camera above the road
    sample_index: int
    scene_name: str
    is_night: bool             # depth estimation degrades badly after dark
    sample_token: str
    # LiDAR sweep already moved into this camera's frame, and the same points
    # projected onto its image. None when the source was asked not to load it.
    lidar_points: np.ndarray = None    # (3, N) metres, camera frame
    lidar_pixels: np.ndarray = None    # (2, N) pixels
    # Rotation and translation taking a point from this camera's frame into
    # the global frame. Needed because the camera frame rides on a moving
    # vehicle: a parked car appears to drift backwards through it, so any
    # velocity or heading measured in camera coordinates is ego motion plus
    # the other vehicle's, not the vehicle's own.
    cam_to_global_R: np.ndarray = None
    cam_to_global_t: np.ndarray = None
    timestamp_s: float = 0.0

    def to_global(self, points_cam):
        """(3, N) camera-frame points -> (3, N) global-frame points."""
        return self.cam_to_global_R @ np.atleast_2d(points_cam) + self.cam_to_global_t.reshape(3, 1)

    def direction_to_camera_frame(self, global_direction):
        """A global-frame direction (e.g. a heading) expressed in camera axes."""
        return self.cam_to_global_R.T @ np.asarray(global_direction)


@dataclass
class GroundTruthBox:
    """A LiDAR-verified 3D box, in the camera's coordinate frame.

    Camera frame convention (nuScenes): x right, y down, z forward. So
    `center[2]` is the forward distance -- the quantity a distance estimate
    is trying to recover.
    """

    center: np.ndarray         # (3,) metres, camera frame
    wlh: np.ndarray            # width, length, height in metres
    corners: np.ndarray        # (3, 8) metres, camera frame
    category: str
    # nuScenes visibility bucket, 1-4, meaning roughly 0-40 / 40-60 / 60-80 /
    # 80-100 percent of the object visible across all cameras. Needed for a
    # fair recall figure: nuScenes labels vehicles that are almost entirely
    # hidden, and counting those as detector misses would understate it.
    visibility: int

    @property
    def distance_to_center(self):
        return float(self.center[2])

    @property
    def distance_to_nearest_face(self):
        """Forward distance to the closest part of the vehicle.

        Distinct from the centre distance by roughly half a vehicle length
        (~2.3 m for a car). Depth estimates read the visible surface, so
        this is the fair quantity to score them against; scoring them
        against the centre builds in a systematic bias.
        """
        return float(self.corners[2].min())


class NuScenesSource:
    def __init__(self, dataroot=None, version=None, channel=None, load_lidar=False):
        self.dataroot = str(dataroot or config.NUSCENES_DATAROOT)
        self.version = version or config.NUSCENES_VERSION
        self.channel = channel or config.CAMERA_CHANNEL
        # Loading a sweep costs real time, so callers that only need pixels
        # (detection, drawing) can skip it.
        self.load_lidar = load_lidar
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        self._night_scenes = {
            s["token"] for s in self.nusc.scene if "night" in s["description"].lower()
        }

    def __len__(self):
        return len(self.nusc.sample)

    def _sample_data(self, sample):
        return self.nusc.get("sample_data", sample["data"][self.channel])

    def frame(self, index):
        """Load one frame by sample index."""
        sample = self.nusc.sample[index]
        sd = self._sample_data(sample)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        scene = self.nusc.get("scene", sample["scene_token"])

        image_path = f"{self.dataroot}/{sd['filename']}"
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"could not read frame image: {image_path}")

        # camera -> ego -> global, as one rotation and translation
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        R_cam_ego = Quaternion(cs["rotation"]).rotation_matrix
        t_cam_ego = np.array(cs["translation"])
        R_ego_glob = Quaternion(pose["rotation"]).rotation_matrix
        t_ego_glob = np.array(pose["translation"])
        R_cam_glob = R_ego_glob @ R_cam_ego
        t_cam_glob = R_ego_glob @ t_cam_ego + t_ego_glob

        points = pixels = None
        if self.load_lidar:
            points, pixels = load_lidar_in_camera(
                self.nusc, sample, self.channel, config.LIDAR_CHANNEL
            )

        return Frame(
            image=image,
            intrinsics=np.array(cs["camera_intrinsic"]),
            # nuScenes' ego frame has its origin on the road surface, so the
            # sensor's z translation is directly its height above the road.
            camera_height_m=float(cs["translation"][2]),
            sample_index=index,
            scene_name=scene["name"],
            is_night=sample["scene_token"] in self._night_scenes,
            sample_token=sample["token"],
            lidar_points=points,
            lidar_pixels=pixels,
            cam_to_global_R=R_cam_glob,
            cam_to_global_t=t_cam_glob,
            timestamp_s=sd["timestamp"] / 1e6,
        )

    def frames(self, start=0, stop=None, step=1):
        stop = len(self) if stop is None else stop
        for i in range(start, min(stop, len(self)), step):
            yield self.frame(i)

    def ground_truth_boxes(self, sample_token, categories=("vehicle.",)):
        """LiDAR-verified 3D boxes for a frame, already in the camera frame.

        For evaluation only. `get_sample_data` does the global -> ego ->
        camera transform chain itself, which is why it is used here rather
        than composing the quaternions by hand.
        """
        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][self.channel]
        _, boxes, _ = self.nusc.get_sample_data(cam_token)

        out = []
        for b in boxes:
            if not b.name.startswith(tuple(categories)):
                continue
            ann = self.nusc.get("sample_annotation", b.token)
            out.append(
                GroundTruthBox(
                    center=np.array(b.center),
                    wlh=np.array(b.wlh),
                    corners=b.corners(),
                    category=b.name,
                    visibility=int(ann["visibility_token"]),
                )
            )
        return out

    def project_to_image(self, box, intrinsics, image_shape):
        """2D pixel box enclosing a 3D box, or None if it isn't in front.

        Clipped to the image, since nuScenes labels vehicles that extend past
        the frame edge and an unclipped box would overstate their extent.
        """
        corners = box.corners
        if corners[2].min() <= 0.1:      # at or behind the image plane
            return None
        uv = intrinsics @ corners
        uv = uv[:2] / uv[2]
        h, w = image_shape[:2]
        x1, x2 = float(np.clip(uv[0].min(), 0, w)), float(np.clip(uv[0].max(), 0, w))
        y1, y2 = float(np.clip(uv[1].min(), 0, h)), float(np.clip(uv[1].max(), 0, h))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return None
        return x1, y1, x2, y2
