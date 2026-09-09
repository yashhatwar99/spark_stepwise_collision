"""
A constant-acceleration Kalman filter for smoothing a tracked object's 2D
ground position (X, Z), and estimating its velocity AND acceleration,
frame to frame.

State vector: [x, z, vx, vz, ax, az] (meters, m/s, m/s^2).
Measurement: (x, z) only -- velocity and acceleration are never observed
directly, they emerge from how the filter's own state evolves across updates.

An earlier version of this filter used a constant-velocity model (4-element
state, no acceleration). It lagged badly during the vehicle's braking/
accelerating final approach to a crash -- verified against raw measurements,
where the filter reported ~0.47 m/s while the vehicle's actual closing rate
over that same window was ~1.0-1.2 m/s. A constant-velocity model has no way
to represent "speeding up," so it always trails reality during exactly that
phase -- the one moment collision prediction most needs to be right. Adding
acceleration as its own state lets the filter represent that change
directly, instead of only inferring it indirectly, a little each frame,
from leftover position error (which is what made the old version slow to
catch up, no matter how high its process noise was turned up).

process_noise here represents uncertainty in the (assumed otherwise
constant) acceleration specifically -- how much it's allowed to drift frame
to frame. It is injected only into the acceleration terms of Q; the motion
model (F) is what correctly propagates that into velocity and position each
step, rather than adding flat noise to every state directly (the old
approach, which is part of why it converged slowly).
"""

import numpy as np


class ConstantAccelerationKalmanFilter:
    def __init__(self, initial_x, initial_z, dt, process_noise=4.0, measurement_noise=0.25):
        self.base_dt = dt
        self.process_noise = process_noise

        self.state = np.zeros(6)
        self.state[0] = initial_x
        self.state[1] = initial_z
        # Position is known from the first measurement; velocity and
        # acceleration are completely unknown at the start.
        self.P = np.diag([1.0, 1.0, 10.0, 10.0, 10.0, 10.0])

        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])
        self.R = np.eye(2) * measurement_noise

    def predict(self, elapsed_frames=1):
        """Advance the state by elapsed_frames worth of time (default 1,
        i.e. one normal frame-to-frame step). Pass a larger value when the
        track was missing for several frames and has just reappeared, so
        the motion model advances by the time actually elapsed instead of
        silently assuming only one frame passed."""
        dt = self.base_dt * elapsed_frames
        half_dt2 = 0.5 * dt * dt

        F = np.array([
            [1, 0, dt, 0, half_dt2, 0],
            [0, 1, 0, dt, 0, half_dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        Q = np.zeros((6, 6))
        Q[4, 4] = self.process_noise * elapsed_frames
        Q[5, 5] = self.process_noise * elapsed_frames

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q
        return self.state

    def update(self, measured_x, measured_z):
        z = np.array([measured_x, measured_z])
        residual = z - self.H @ self.state
        innovation_cov = self.H @ self.P @ self.H.T + self.R
        kalman_gain = self.P @ self.H.T @ np.linalg.inv(innovation_cov)

        self.state = self.state + kalman_gain @ residual
        self.P = (np.eye(6) - kalman_gain @ self.H) @ self.P
        return self.state

    @property
    def position(self):
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self):
        return float(self.state[2]), float(self.state[3])

    @property
    def acceleration(self):
        return float(self.state[4]), float(self.state[5])
