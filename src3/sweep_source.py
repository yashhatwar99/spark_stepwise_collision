"""
Every front-camera image in a scene, not just the annotated keyframes.

nuScenes annotates only 2 frames per second (the keyframes), but the front
camera records about 10: on scene-0103 there are 229 images, of which 40 are
keyframes. The other images ("sweeps") have no labels, but each one carries
its own timestamp, ego pose and camera calibration -- which is everything the
camera-only pipeline needs. So they can be used for tracking and prediction,
and keyframes remain available for scoring against ground truth.

Why this matters: at 2 fps a car moves about 5 m between frames, and at 10
fps about 1 m. Smaller steps make tracking easier and give many more
positions to average when estimating speed.

Frames come back as the same `Frame` objects NuScenesSource produces, plus
two extra attributes:
    is_key_frame    True on the annotated keyframes
    key_index       the keyframe index (as used by NuScenesSource) this image
                    belongs to, i.e. the most recent keyframe at or before it
"""

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from src2 import config
from src2.nuscenes_source import Frame


class NuScenesSweepSource:
    def __init__(self, dataroot=None, version=None, channel=None):
        self.dataroot = str(dataroot or config.NUSCENES_DATAROOT)
        self.version = version or config.NUSCENES_VERSION
        self.channel = channel or config.CAMERA_CHANNEL
        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        self._key_index = {s["token"]: i for i, s in enumerate(self.nusc.sample)}
        self._night = {s["token"] for s in self.nusc.scene if "night" in s["description"].lower()}

    def images_between_keyframes(self, first_key, last_key):
        """sample_data tokens for every camera image from keyframe `first_key`
        up to and including keyframe `last_key` (NuScenesSource indices).

        Stays inside one scene: the walk stops at the scene's last image.
        """
        start = self.nusc.sample[first_key]
        end = self.nusc.sample[last_key]
        if start["scene_token"] != end["scene_token"]:
            raise ValueError("first_key and last_key are in different scenes; "
                             "tracking across a scene cut would invent motion")
        end_time = self.nusc.get("sample_data", end["data"][self.channel])["timestamp"]

        tokens = []
        token = start["data"][self.channel]
        while token:
            sd = self.nusc.get("sample_data", token)
            if sd["timestamp"] > end_time:
                break
            tokens.append(token)
            token = sd["next"]
        return tokens

    def frame(self, sd_token):
        sd = self.nusc.get("sample_data", sd_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        sample = self.nusc.get("sample", sd["sample_token"])
        scene = self.nusc.get("scene", sample["scene_token"])

        image_path = f"{self.dataroot}/{sd['filename']}"
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"could not read frame image: {image_path}")

        # camera -> ego -> global, using THIS image's own ego pose. Each sweep
        # has one; using the nearest keyframe's pose instead would misplace
        # everything by however far the ego drove in between (~1 m per sweep).
        pose = self.nusc.get("ego_pose", sd["ego_pose_token"])
        R_cam_ego = Quaternion(cs["rotation"]).rotation_matrix
        t_cam_ego = np.array(cs["translation"])
        R_ego_glob = Quaternion(pose["rotation"]).rotation_matrix
        t_ego_glob = np.array(pose["translation"])

        # A sweep's sample_token points at the NEXT keyframe (checked on the
        # data: sweeps between keyframes 47 and 48 are labelled 48). Name each
        # image after the keyframe at or BEFORE it instead, so a keyframe's
        # sweeps sort straight after it.
        if sd["is_key_frame"] or not sample["prev"]:
            key_index = self._key_index[sample["token"]]
        else:
            key_index = self._key_index[sample["prev"]]

        frame = Frame(
            image=image,
            intrinsics=np.array(cs["camera_intrinsic"]),
            camera_height_m=float(cs["translation"][2]),
            sample_index=key_index,
            scene_name=scene["name"],
            is_night=sample["scene_token"] in self._night,
            sample_token=sample["token"],
            cam_to_global_R=R_ego_glob @ R_cam_ego,
            cam_to_global_t=R_ego_glob @ t_cam_ego + t_ego_glob,
            timestamp_s=sd["timestamp"] / 1e6,
        )
        frame.is_key_frame = bool(sd["is_key_frame"])
        frame.key_index = key_index
        return frame
