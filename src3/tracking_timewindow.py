"""
Tracking for high frame-rate input: velocity measured over TIME, not frames.

The base Track measures velocity from its last 4 positions, using only the
first and last. At 2 fps that spans 1.5 s. At 10 fps it would span only
0.3 s, and a far car's position jitter (about 6 m between frames at 50-90 m)
divided by 0.3 s is a fake speed of up to ~20 m/s. More frames would make
speed WORSE if the window stayed a frame count.

So this track:
  * keeps every position from the last WINDOW_S seconds, and
  * fits a straight line through all of them (least squares), so each
    position's jitter is averaged against the others instead of two noisy
    endpoints deciding the answer; and
  * refuses to give a speed until it has seen the car for MIN_SPAN_S, since a
    fit over a tenth of a second is jitter, not motion.

Matching still uses the range-scaled gate from tracking_scaled.py, because
per-frame jitter does not shrink with frame rate -- only the true motion
between frames does.
"""

import numpy as np

from src2.tracking import Track
from src3.tracking_scaled import RangeScaledTracker

WINDOW_S = 1.5          # same time span the 2 fps pipeline uses
MIN_SPAN_S = 0.5        # below this, a fitted speed is mostly noise


class TimeWindowTrack(Track):
    def velocity(self, window=None):
        """Least-squares velocity over the last WINDOW_S seconds.

        `window` is accepted only for compatibility with Track.velocity and
        is ignored: here the window is a time span.
        """
        if len(self.positions) < 2:
            return None
        t_now = self.timestamps[-1]
        keep = [i for i, t in enumerate(self.timestamps) if t >= t_now - WINDOW_S]
        if len(keep) < 2:
            return None
        t = np.array([self.timestamps[i] for i in keep])
        if t[-1] - t[0] < MIN_SPAN_S:
            return None
        p = np.stack([self.positions[i] for i in keep])      # (n, 2)
        t = t - t.mean()
        # slope of position against time, per axis
        return (t[:, None] * (p - p.mean(axis=0))).sum(axis=0) / (t ** 2).sum()


class TimeWindowTracker(RangeScaledTracker):
    """RangeScaledTracker whose tracks estimate velocity over a time window.

    `max_missed` counts FRAMES, so at 10 fps it is raised: the default of 2
    would drop a car after only 0.2 s without a detection.
    """

    track_class = TimeWindowTrack

    def __init__(self, max_missed=5, **kwargs):
        super().__init__(max_missed=max_missed, **kwargs)
