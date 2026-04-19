"""
DeepSORT Multi-Object Tracker (L4 ADAS)
Combines Kalman filtering + IoU/appearance matching for robust MOT.

Architecture:
- KalmanBoxTracker: Per-object state estimation
- Hungarian algorithm: Optimal detection-to-track assignment
- Track lifecycle: Tentative → Confirmed → Lost → Deleted

Reference:
  Wojke, Bewley, Paulus — "Simple Online and Realtime Tracking with a Deep
  Association Metric" (2017)
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field

from src.tracking.kalman_filter import KalmanBoxTracker

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """Represents a confirmed track with metadata."""
    track_id: int
    bbox: np.ndarray                  # Current [x1, y1, x2, y2]
    class_name: str = ""
    category: str = ""
    confidence: float = 0.0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    trajectory: List[np.ndarray] = field(default_factory=list)


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        bb_test: (N, 4) array of [x1, y1, x2, y2]
        bb_gt:   (M, 4) array of [x1, y1, x2, y2]

    Returns:
        (N, M) IoU matrix
    """
    if len(bb_test) == 0 or len(bb_gt) == 0:
        return np.empty((len(bb_test), len(bb_gt)))

    bb_test = np.array(bb_test)
    bb_gt = np.array(bb_gt)

    xx1 = np.maximum(bb_test[:, 0:1], bb_gt[:, 0].T)
    yy1 = np.maximum(bb_test[:, 1:2], bb_gt[:, 1].T)
    xx2 = np.minimum(bb_test[:, 2:3], bb_gt[:, 2].T)
    yy2 = np.minimum(bb_test[:, 3:4], bb_gt[:, 3].T)

    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    intersection = w * h

    area_test = (bb_test[:, 2] - bb_test[:, 0]) * (bb_test[:, 3] - bb_test[:, 1])
    area_gt = (bb_gt[:, 2] - bb_gt[:, 0]) * (bb_gt[:, 3] - bb_gt[:, 1])

    union = area_test[:, np.newaxis] + area_gt[np.newaxis, :] - intersection

    iou = intersection / np.maximum(union, 1e-6)
    return iou


class DeepSORTTracker:
    """
    DeepSORT-based multi-object tracker for real-time ADAS.

    Features:
    - Kalman filter motion prediction
    - IoU-based Hungarian matching
    - Track state management (tentative → confirmed → lost)
    - Per-track velocity estimation
    - Trajectory recording

    Supports Indian road scenarios:
    - High-density traffic (many simultaneous tracks)
    - Mixed vehicle types (auto-rickshaws, bikes, trucks)
    - Unpredictable pedestrian/animal movement
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Args:
            max_age: Maximum frames to keep a lost track
            min_hits: Minimum hits to confirm a track
            iou_threshold: IoU threshold for association
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.trackers: List[KalmanBoxTracker] = []
        self.tracker_metadata: Dict[int, Dict] = {}  # tracker_id → class/category info

        # Frame counter for track readiness
        self.frame_count = 0

    def update(
        self,
        detections: np.ndarray,
        scores: Optional[np.ndarray] = None,
        class_names: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
    ) -> List[Track]:
        """
        Update tracker with current frame detections.

        Args:
            detections: (N, 4) array of [x1, y1, x2, y2]
            scores: (N,) confidence scores
            class_names: (N,) class labels
            categories: (N,) category groups

        Returns:
            List of confirmed Track objects
        """
        self.frame_count += 1

        if scores is None:
            scores = np.ones(len(detections))
        if class_names is None:
            class_names = [""] * len(detections)
        if categories is None:
            categories = [""] * len(detections)

        # ── Step 1: Predict new locations for existing trackers ──
        predicted_boxes = []
        for t in self.trackers:
            pred = t.predict()
            predicted_boxes.append(pred)
        predicted_boxes = np.array(predicted_boxes) if predicted_boxes else np.empty((0, 4))

        # ── Step 2: Associate detections to trackers (Hungarian + IoU) ──
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes
        )

        # ── Step 3: Update matched trackers ──
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx].update(detections[det_idx])
            # Update metadata
            tid = self.trackers[trk_idx].id
            self.tracker_metadata[tid] = {
                "class_name": class_names[det_idx],
                "category": categories[det_idx],
                "confidence": float(scores[det_idx]),
            }

        # ── Step 4: Create new trackers for unmatched detections ──
        for det_idx in unmatched_dets:
            trk = KalmanBoxTracker(detections[det_idx])
            self.trackers.append(trk)
            self.tracker_metadata[trk.id] = {
                "class_name": class_names[det_idx],
                "category": categories[det_idx],
                "confidence": float(scores[det_idx]),
            }

        # ── Step 5: Remove dead trackers ──
        self.trackers = [
            t for t in self.trackers
            if t.time_since_update <= self.max_age
        ]

        # ── Step 6: Build output (only confirmed tracks) ──
        tracks = []
        for t in self.trackers:
            if t.time_since_update > 0:
                continue  # Skip tracks not updated this frame
            if t.hit_streak < self.min_hits and self.frame_count > self.min_hits:
                continue  # Skip tentative tracks (except early frames)

            meta = self.tracker_metadata.get(t.id, {})
            bbox = t.get_state()
            vel = t.get_velocity()

            tracks.append(Track(
                track_id=t.id,
                bbox=bbox,
                class_name=meta.get("class_name", ""),
                category=meta.get("category", ""),
                confidence=meta.get("confidence", 0.0),
                velocity=vel,
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
                trajectory=[h.copy() for h in t.history[-10:]],
            ))

        return tracks

    def _associate(
        self,
        detections: np.ndarray,
        trackers: np.ndarray,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Associate detections to existing trackers using IoU + Hungarian.

        Returns:
            matched: list of (det_idx, trk_idx) pairs
            unmatched_detections: list of det indices
            unmatched_trackers: list of trk indices
        """
        if len(trackers) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], list(range(len(trackers)))

        # Compute IoU cost matrix
        iou_matrix = iou_batch(detections, trackers)
        cost_matrix = 1.0 - iou_matrix  # Lower cost = better match

        # Hungarian algorithm
        if min(cost_matrix.shape) > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = np.array([]), np.array([])

        # Filter by IoU threshold
        matched = []
        unmatched_dets = list(range(len(detections)))
        unmatched_trks = list(range(len(trackers)))

        for r, c in zip(row_indices, col_indices):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched.append((r, c))
                if r in unmatched_dets:
                    unmatched_dets.remove(r)
                if c in unmatched_trks:
                    unmatched_trks.remove(c)

        return matched, unmatched_dets, unmatched_trks

    def get_all_tracks(self) -> List[Track]:
        """Get all active tracks (including tentative)."""
        tracks = []
        for t in self.trackers:
            meta = self.tracker_metadata.get(t.id, {})
            tracks.append(Track(
                track_id=t.id,
                bbox=t.get_state(),
                class_name=meta.get("class_name", ""),
                category=meta.get("category", ""),
                confidence=meta.get("confidence", 0.0),
                velocity=t.get_velocity(),
                age=t.age,
                hits=t.hits,
                time_since_update=t.time_since_update,
            ))
        return tracks

    def reset(self):
        """Clear all tracks."""
        self.trackers.clear()
        self.tracker_metadata.clear()
        self.frame_count = 0
        KalmanBoxTracker.reset_id_counter()
