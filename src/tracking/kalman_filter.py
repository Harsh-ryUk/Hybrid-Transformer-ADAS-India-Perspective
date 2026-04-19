"""
Kalman Filter for Bounding Box Tracking (L4 ADAS)
Standard linear Kalman filter adapted for bounding box state estimation.

State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
- (cx, cy): bounding box center
- aspect_ratio: width / height
- height: bounding box height
- (vx, vy, va, vh): respective velocities

Measurement vector: [cx, cy, aspect_ratio, height]
"""

import numpy as np
from typing import Optional


class KalmanBoxTracker:
    """
    Kalman filter for tracking a single bounding box in image space.

    Uses constant-velocity model to predict motion between frames.
    """

    # Class-level counter for unique IDs
    _id_counter = 0

    def __init__(self, bbox: np.ndarray):
        """
        Initialize tracker with initial bounding box [x1, y1, x2, y2].
        """
        # Assign unique ID
        self.id = KalmanBoxTracker._id_counter
        KalmanBoxTracker._id_counter += 1

        # State dimension = 8, Measurement dimension = 4
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # position += velocity * dt

        # Measurement matrix (we observe position, not velocity)
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[:4, :4] = np.eye(4)

        # Measurement noise covariance
        self.R = np.eye(self.dim_z)
        self.R[2, 2] *= 10.0   # aspect ratio is less precise
        self.R[3, 3] *= 10.0   # height measurement noise

        # Process noise covariance
        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01  # velocity process noise (small)
        self.Q[6, 6] *= 0.01
        self.Q[7, 7] *= 0.01

        # Initial state covariance
        self.P = np.eye(self.dim_x)
        self.P[4:, 4:] *= 1000.0  # high uncertainty in initial velocity
        self.P *= 10.0

        # Initialize state from first bbox
        measurement = self._bbox_to_z(bbox)
        self.x = np.zeros(self.dim_x)
        self.x[:4] = measurement

        # Track management
        self.hits = 1
        self.hit_streak = 1
        self.age = 1
        self.time_since_update = 0

        # Trajectory history
        self.history = [bbox.copy()]

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, aspect_ratio, height]."""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        aspect = w / max(h, 1e-6)
        return np.array([cx, cy, aspect, h])

    @staticmethod
    def _z_to_bbox(z: np.ndarray) -> np.ndarray:
        """Convert [cx, cy, aspect_ratio, height] to [x1, y1, x2, y2]."""
        w = z[2] * z[3]
        h = z[3]
        x1 = z[0] - w / 2.0
        y1 = z[1] - h / 2.0
        x2 = z[0] + w / 2.0
        y2 = z[1] + h / 2.0
        return np.array([x1, y1, x2, y2])

    def predict(self) -> np.ndarray:
        """
        Advance state by one timestep using motion model.

        Returns:
            Predicted bounding box [x1, y1, x2, y2]
        """
        # Prevent negative aspect ratio / height
        if self.x[6] + self.x[2] <= 0:
            self.x[6] = 0.0
        if self.x[7] + self.x[3] <= 0:
            self.x[7] = 0.0

        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

        self.age += 1
        self.time_since_update += 1

        predicted_bbox = self._z_to_bbox(self.x[:4])
        return predicted_bbox

    def update(self, bbox: np.ndarray):
        """
        Update state with observed bounding box measurement.

        Args:
            bbox: Observed [x1, y1, x2, y2]
        """
        z = self._bbox_to_z(bbox)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update
        y = z - self.H @ self.x  # innovation
        self.x = self.x + K @ y
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P

        # Track management
        self.hits += 1
        self.hit_streak += 1
        self.time_since_update = 0

        self.history.append(bbox.copy())
        # Keep only last 30 history entries
        if len(self.history) > 30:
            self.history.pop(0)

    def get_state(self) -> np.ndarray:
        """Get current bounding box estimate [x1, y1, x2, y2]."""
        return self._z_to_bbox(self.x[:4])

    def get_velocity(self) -> np.ndarray:
        """Get velocity [vx, vy] in pixels/frame."""
        return self.x[4:6]

    @classmethod
    def reset_id_counter(cls):
        """Reset the global ID counter (for testing)."""
        cls._id_counter = 0
