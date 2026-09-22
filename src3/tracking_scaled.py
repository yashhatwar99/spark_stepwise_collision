"""
The src2 tracker, with a matching gate that grows with distance.

Why: src2.tracking.Tracker uses one fixed 4 m gate, which was tuned on LiDAR
positions (about 1 m error). Camera-only 3D positions from deep3dbox are much
noisier at range: measured on frames 41-50, a car's estimated position jumps
a median 6.3 m between frames when it is 50-90 m away. Those jumps exceed
4 m, so the tracker loses the car and restarts it, and a restarted track has
no speed and therefore no prediction. That cost half of all predictions.

Fix: gate = max(4 m, 12% of the car's distance). Tested over 82 frames
against the true car identities:

    fixed 4 m gate          : 61% of cars get a speed, 10.6% wrong links
    gate = 12% of distance  : 71% of cars get a speed,  9.0% wrong links

It is the only rule tried that improved both. Wider fixed gates caught more
cars but made more wrong links; matching on 2D box overlap doubled them.

Everything except the gate is inherited unchanged from src2.tracking.
"""

import numpy as np

from src2.tracking import Track, Tracker

GATE_FRACTION_OF_RANGE = 0.12
GATE_FLOOR_M = 4.0


class RangeScaledTracker(Tracker):
    # Which Track type new tracks are created as. A subclass can swap in one
    # that estimates velocity differently (see tracking_timewindow.py).
    track_class = Track

    def __init__(self, fraction=GATE_FRACTION_OF_RANGE, floor_m=GATE_FLOOR_M, max_missed=2):
        super().__init__(gate_m=floor_m, max_missed=max_missed)
        self.fraction = fraction
        self.floor_m = floor_m

    def gate_for(self, range_m):
        """Allowed jump for a car this far away."""
        return max(self.floor_m, self.fraction * range_m)

    def update(self, detections_xy, timestamp, ranges_m):
        """Same as Tracker.update, but each detection gets its own gate.

        `ranges_m` is each detection's distance ahead of the camera, in the
        same order as `detections_xy`.
        """
        assignments = [None] * len(detections_xy)

        candidates = []
        for det_index, position in enumerate(detections_xy):
            gate = self.gate_for(ranges_m[det_index])
            for track_id, track in self.tracks.items():
                distance = float(np.linalg.norm(np.asarray(position) - track.predict(timestamp)))
                if distance <= gate:
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
            self.tracks[track_id] = self.track_class(
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
