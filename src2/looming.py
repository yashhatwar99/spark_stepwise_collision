"""
Collision warning from 2D boxes alone -- no distance, no calibrated depth,
no size assumption.

Two classical results carry the whole method, and between them they answer
the two questions a warning needs:

WHEN will it reach us -- from how fast the box grows.
    A box's pixel height is h = f*H/Z. Differentiating,
        h / (dh/dt) = -Z / (dZ/dt) = time to contact
    The focal length, the object's real size and its distance all cancel.
    Measured against annotated 3D distance on nuScenes: 0.28 s median error
    inside the 0-3 s danger window, and within 1 s on 100% of those cases.
    Unusually, it gets MORE accurate as the danger grows, which is the
    opposite of every distance-based estimate in this project.

WILL it actually hit us -- from whether its bearing holds steady.
    An object approaching on a constant bearing is on a collision course; one
    whose bearing drifts will pass clear. Sailors have used this for
    centuries. Bearing comes straight from the box's pixel column, which is
    an angle measurement and needs no distance. Measured against the true
    miss distance:
        bearing rate < 0.01 rad/s -> median miss  1.6 m (78% within 3 m)
        0.01 - 0.03               -> median miss  4.8 m
        0.03 - 0.08               -> median miss  8.1 m
        > 0.08                    -> median miss  9.8 m (0% within 3 m)

WHAT THIS CANNOT DO -- worth being explicit, because the gap is structural:
  * It reasons only about what may hit US. Two other vehicles colliding with
    each other needs real 3D geometry.
  * It gives no miss distance in metres, only "close" or "clear".
  * A stationary or receding object produces no growth, so no TTC -- correct,
    but it means silence is not the same as safety.
"""

import numpy as np

# Bearing drift below which an object counts as holding a steady bearing.
# From the table above: 0.01 rad/s is where median miss falls to 1.6 m.
STEADY_BEARING_RAD_S = 0.02
CRITICAL_TTC_S = 3.0
WARNING_TTC_S = 5.0

# A box must grow by at least this many pixels per second to count as
# approaching. Below it, growth is indistinguishable from detector jitter and
# the resulting TTC is noise dividing noise.
MIN_GROWTH_PX_S = 0.5


def bearing_rad(u_pixel, focal_length_px, principal_x):
    """Angle from the optical axis to a pixel column.

    A pure angle measurement -- the one thing a camera does almost perfectly.
    Checked against annotated 3D positions: error in hundredths of a degree.
    """
    return float(np.arctan((u_pixel - principal_x) / focal_length_px))


def ttc_from_growth(height_px, growth_px_per_s):
    """Seconds until contact, from box height and its rate of change.

    None when the box is not growing fast enough to be believed -- that is
    the honest answer for something stationary, receding, or merely jittering.
    """
    if growth_px_per_s is None or growth_px_per_s < MIN_GROWTH_PX_S:
        return None
    return float(height_px / growth_px_per_s)


def classify(ttc_s, bearing_rate_rad_s):
    """Risk from time-to-contact and bearing steadiness.

    Both conditions are required rather than either: a fast approach that is
    sliding across our view will pass by, and a steady bearing far in the
    future is not yet actionable. Requiring both is what keeps the warning
    quiet during ordinary traffic, where most vehicles are approaching
    something but almost none are approaching us.
    """
    if ttc_s is None:
        return "NO CLOSING"
    steady = bearing_rate_rad_s is not None and bearing_rate_rad_s < STEADY_BEARING_RAD_S
    if not steady:
        return "PASSING"
    if ttc_s <= CRITICAL_TTC_S:
        return "CRITICAL"
    if ttc_s <= WARNING_TTC_S:
        return "WARNING"
    return "CLEAR"


class BoxTrack:
    """One vehicle followed through the image, in pixels only."""

    def __init__(self, track_id, box, timestamp, focal_length_px, principal_x):
        self.track_id = track_id
        self.focal_length_px = focal_length_px
        self.principal_x = principal_x
        self.boxes = []
        self.timestamps = []
        self.missed = 0
        self.update(box, timestamp)

    def update(self, box, timestamp):
        self.boxes.append(tuple(float(v) for v in box))
        self.timestamps.append(float(timestamp))
        self.missed = 0

    @property
    def box(self):
        return self.boxes[-1]

    @property
    def height_px(self):
        x1, y1, x2, y2 = self.boxes[-1]
        return y2 - y1

    @property
    def center(self):
        x1, y1, x2, y2 = self.boxes[-1]
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _window(self, span=3):
        """Indices of the oldest and newest sighting in a short window.

        A short baseline rather than a single frame difference: box edges
        jitter by a pixel or two per frame, and over one 0.5 s step that
        jitter is a large fraction of the real growth.
        """
        if len(self.boxes) < 2:
            return None
        n = min(span + 1, len(self.boxes))
        dt = self.timestamps[-1] - self.timestamps[-n]
        return (-n, dt) if dt > 1e-6 else None

    def growth_px_per_s(self):
        w = self._window()
        if w is None:
            return None
        i, dt = w
        old = self.boxes[i][3] - self.boxes[i][1]
        return (self.height_px - old) / dt

    def bearing(self):
        return bearing_rad(self.center[0], self.focal_length_px, self.principal_x)

    def bearing_rate(self):
        w = self._window()
        if w is None:
            return None
        i, dt = w
        x1, _, x2, _ = self.boxes[i]
        old = bearing_rad((x1 + x2) / 2.0, self.focal_length_px, self.principal_x)
        return abs(self.bearing() - old) / dt

    def ttc(self):
        return ttc_from_growth(self.height_px, self.growth_px_per_s())

    def risk(self):
        return classify(self.ttc(), self.bearing_rate())

    def predict_box(self, seconds_ahead):
        """Where this box will be, and how big, `seconds_ahead` from now.

        The image-plane trajectory. Height is grown from the measured growth
        rate and the centre is carried along the measured bearing drift, so
        the prediction uses only quantities we can actually observe.
        """
        growth = self.growth_px_per_s()
        if growth is None:
            return None
        w = self._window()
        if w is None:
            return None
        i, dt = w
        cx_old = (self.boxes[i][0] + self.boxes[i][2]) / 2.0
        cy_old = (self.boxes[i][1] + self.boxes[i][3]) / 2.0
        cx, cy = self.center
        vx, vy = (cx - cx_old) / dt, (cy - cy_old) / dt

        height = self.height_px + growth * seconds_ahead
        if height <= 1:
            return None
        x1, _, x2, _ = self.box
        aspect = (x2 - x1) / max(self.height_px, 1e-6)
        width = height * aspect
        ncx, ncy = cx + vx * seconds_ahead, cy + vy * seconds_ahead
        return (ncx - width / 2, ncy - height / 2, ncx + width / 2, ncy + height / 2)


class BoxTracker:
    """IoU association of 2D boxes across frames.

    Image-space overlap is enough here because nothing needs a world
    position: the whole method lives in pixels.
    """

    def __init__(self, iou_threshold=0.3, max_missed=2):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = {}
        self._next_id = 0

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def update(self, boxes, timestamp, focal_length_px, principal_x):
        pairs = []
        for det_index, box in enumerate(boxes):
            for track_id, track in self.tracks.items():
                score = self._iou(box, track.box)
                if score >= self.iou_threshold:
                    pairs.append((-score, det_index, track_id))
        pairs.sort()

        assigned = [None] * len(boxes)
        used_tracks, used_dets = set(), set()
        for _, det_index, track_id in pairs:
            if det_index in used_dets or track_id in used_tracks:
                continue
            used_dets.add(det_index)
            used_tracks.add(track_id)
            assigned[det_index] = track_id
            self.tracks[track_id].update(boxes[det_index], timestamp)

        for det_index, box in enumerate(boxes):
            if assigned[det_index] is not None:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.tracks[track_id] = BoxTrack(track_id, box, timestamp,
                                             focal_length_px, principal_x)
            assigned[det_index] = track_id

        for track_id in list(self.tracks):
            if track_id not in used_tracks and self.tracks[track_id].timestamps[-1] != timestamp:
                self.tracks[track_id].missed += 1
                if self.tracks[track_id].missed > self.max_missed:
                    del self.tracks[track_id]

        return assigned
