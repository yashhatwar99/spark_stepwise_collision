"""
Follow vehicles across frames, and read their heading from how they move.

Why this exists: heading is the one part of a 3D box that geometry could not
supply. Fitting a shape to LiDAR points recovers the box's axes to about 20
degrees but then cannot tell the vehicle's side from its rear, so it picks
the wrong axis often enough to land at ~60 degrees error -- no better than
guessing. A moving vehicle, though, points where it is going. Two sightings
give a direction of travel directly, with no shape fitting and no ambiguity,
and it needs only one LiDAR point per frame instead of sixty.

Everything here works in the GLOBAL frame, not the camera frame. The camera
rides on a moving vehicle, so in its coordinates a parked car drifts steadily
backwards; a heading measured there would be ego motion plus the other
vehicle's, and a parked car would be confidently reported as travelling
backwards down the road.

nuScenes keyframes are 0.5 s apart, which is a long time in traffic -- a
vehicle can move 5 m between sightings -- so association is gated generously
and uses each track's own predicted position rather than its last one.
"""

from dataclasses import dataclass, field

import numpy as np

from . import config


@dataclass
class Track:
    track_id: int
    positions: list = field(default_factory=list)     # global (x, y) ground plane
    timestamps: list = field(default_factory=list)
    last_seen: int = 0
    missed: int = 0

    @property
    def position(self):
        return self.positions[-1]

    def velocity(self, window=3):
        """Velocity in m/s, averaged over the last few sightings.

        A single 0.5 s difference is noisy because each position carries
        roughly a metre of error, which becomes 2 m/s of velocity error. A
        short window trades a little lag for a usable number.
        """
        if len(self.positions) < 2:
            return None
        n = min(window + 1, len(self.positions))
        first, last = self.positions[-n], self.positions[-1]
        dt = self.timestamps[-1] - self.timestamps[-n]
        if dt <= 1e-6:
            return None
        return (last - first) / dt

    @property
    def speed(self):
        v = self.velocity()
        return None if v is None else float(np.hypot(v[0], v[1]))

    def heading(self, min_speed=None):
        """Global direction of travel as a unit vector, or None.

        Below `min_speed` the vehicle is parked or crawling and the apparent
        direction is just position noise -- a stationary car with 1 m of
        jitter would be assigned a confident, arbitrary heading. Refusing to
        answer is the honest result; a parked car is also the case where a
        wrong heading matters least.
        """
        min_speed = config.MIN_SPEED_FOR_HEADING if min_speed is None else min_speed
        v = self.velocity()
        if v is None:
            return None
        speed = float(np.hypot(v[0], v[1]))
        if speed < min_speed:
            return None
        return v / speed

    def predict(self, timestamp):
        """Where this track should be now, for association."""
        v = self.velocity()
        if v is None:
            return self.positions[-1]
        return self.positions[-1] + v * (timestamp - self.timestamps[-1])


class Tracker:
    """Greedy nearest-neighbour association in the global ground plane.

    Nearest-neighbour rather than anything cleverer because the association
    problem here is easy: vehicles are metres apart while our position error
    is about a metre. The gate is what does the real work -- it must be wide
    enough for half a second of traffic motion but tight enough that two
    vehicles in adjacent lanes are never swapped.
    """

    def __init__(self, gate_m=None, max_missed=2):
        self.gate_m = config.TRACK_GATE_M if gate_m is None else gate_m
        self.max_missed = max_missed
        self.tracks = {}
        self._next_id = 0

    def update(self, detections_xy, timestamp):
        """`detections_xy` is a list of (x, y) global ground-plane positions.

        Returns a list the same length, giving each detection's track id.
        """
        assignments = [None] * len(detections_xy)
        available = dict(self.tracks)

        # Score every plausible pairing, then take them best-first, so a
        # detection cannot claim a track that another detection matches far
        # more closely.
        candidates = []
        for det_index, position in enumerate(detections_xy):
            for track_id, track in available.items():
                distance = float(np.linalg.norm(np.asarray(position) - track.predict(timestamp)))
                if distance <= self.gate_m:
                    candidates.append((distance, det_index, track_id))
        candidates.sort()

        claimed_tracks, claimed_dets = set(), set()
        for distance, det_index, track_id in candidates:
            if det_index in claimed_dets or track_id in claimed_tracks:
                continue
            claimed_dets.add(det_index)
            claimed_tracks.add(track_id)
            assignments[det_index] = track_id
            track = self.tracks[track_id]
            track.positions.append(np.asarray(detections_xy[det_index], dtype=float))
            track.timestamps.append(timestamp)
            track.missed = 0

        for det_index, position in enumerate(detections_xy):
            if assignments[det_index] is not None:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = Track(
                track_id=track_id,
                positions=[np.asarray(position, dtype=float)],
                timestamps=[timestamp],
            )
            assignments[det_index] = track_id

        for track_id in list(self.tracks):
            if track_id not in claimed_tracks and self.tracks[track_id].timestamps[-1] != timestamp:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]

        return assignments

    def heading_in_camera(self, track_id, frame, min_speed=None):
        """This track's heading as a yaw angle in `frame`'s camera axes.

        Returns None when the vehicle is too slow for its direction of travel
        to mean anything.
        """
        track = self.tracks.get(track_id)
        if track is None:
            return None
        direction = track.heading(min_speed=min_speed)
        if direction is None:
            return None
        global_3d = np.array([direction[0], direction[1], 0.0])
        in_camera = frame.direction_to_camera_frame(global_3d)
        # Camera axes are x right, z forward; box yaw is measured from +z
        # toward +x, matching localize.box_corners.
        return float(np.arctan2(in_camera[0], in_camera[2]))
