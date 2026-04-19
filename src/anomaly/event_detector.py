"""
Unusual Event / Anomaly Detector (L4 ADAS)
Detects edge-case scenarios critical for Indian road safety.

Events detected:
1. Wrong-side driving vehicles (moving against traffic flow)
2. Sudden pedestrian/animal crossing into ego lane
3. Animal interference on drivable area
4. Road surface anomalies (potholes, debris)

India-specific importance:
- Wrong-side driving is extremely common on undivided roads
- Stray animals (cows, dogs) frequently occupy roads
- Pedestrians cross at arbitrary points (no crosswalk culture)
- Poor road maintenance = frequent potholes
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyEvent:
    """Single anomaly event detected in the scene."""
    type: str               # wrong_side_vehicle, sudden_crossing, animal_on_road, road_anomaly
    severity: str           # info, warning, critical
    confidence: float
    details: str
    bbox: Optional[List[float]] = None
    track_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return {
            "type": self.type,
            "severity": self.severity,
            "confidence": self.confidence,
            "details": self.details,
            "bbox": self.bbox,
            "track_id": self.track_id,
        }


class AnomalyEventDetector:
    """
    Detects unusual / edge-case events on Indian roads.

    Works with tracked objects (from DeepSORT) and segmentation masks
    to identify dangerous situations that require immediate reaction.

    Designed for Indian conditions where:
    - Undivided roads → frequent wrong-side driving
    - No pedestrian infrastructure → random crossings
    - Stray animals are common → cow/dog/buffalo on road
    - Road quality is poor → potholes everywhere

    Datasets for validation:
    - IDD: http://idd.insaan.iiit.ac.in/ (Indian scenarios)
    - BDD100K: https://bdd-data.berkeley.edu/ (diverse conditions)
    """

    def __init__(
        self,
        wrong_side_velocity_threshold: float = -5.0,
        wrong_side_min_frames: int = 3,
        crossing_velocity_threshold: float = 15.0,
        crossing_proximity_threshold: float = 200.0,
        pothole_contrast_threshold: float = 40.0,
        pothole_min_area: float = 500.0,
    ):
        self.wrong_side_vel_thresh = wrong_side_velocity_threshold
        self.wrong_side_min_frames = wrong_side_min_frames
        self.crossing_vel_thresh = crossing_velocity_threshold
        self.crossing_prox_thresh = crossing_proximity_threshold
        self.pothole_contrast_thresh = pothole_contrast_threshold
        self.pothole_min_area = pothole_min_area

        # Track history for wrong-side detection
        self._wrong_side_counters: Dict[int, int] = {}
        # Previous track positions for velocity estimation
        self._prev_positions: Dict[int, List[float]] = {}

    def detect(
        self,
        tracks: List[Dict],
        drivable_mask: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        frame_width: int = 1280,
        frame_height: int = 720,
    ) -> List[AnomalyEvent]:
        """
        Run all anomaly detectors on current frame data.

        Args:
            tracks: List of tracked objects with bbox, category, velocity, track_id
            drivable_mask: Binary mask of drivable area (255=road)
            frame: BGR frame for pothole detection
            frame_width: Frame width for spatial reasoning
            frame_height: Frame height

        Returns:
            List of detected AnomalyEvent objects
        """
        events = []

        # 1. Wrong-side driving detection
        events.extend(self._detect_wrong_side(tracks, frame_width))

        # 2. Sudden crossing detection
        events.extend(self._detect_sudden_crossing(tracks, frame_width, frame_height))

        # 3. Animal on drivable area
        if drivable_mask is not None:
            events.extend(self._detect_animal_on_road(tracks, drivable_mask))

        # 4. Road surface anomalies (potholes)
        if frame is not None and drivable_mask is not None:
            events.extend(self._detect_road_anomalies(frame, drivable_mask))

        # Update position history
        self._update_position_history(tracks)

        return events

    def _detect_wrong_side(
        self, tracks: List[Dict], frame_width: int
    ) -> List[AnomalyEvent]:
        """
        Detect vehicles moving against expected traffic flow.

        Heuristic: In India, traffic drives on the LEFT side.
        A vehicle on the left half moving TOWARDS the camera (positive vy)
        while being on the wrong side is suspicious.

        We use velocity direction relative to the expected flow.
        If a vehicle has sustained negative vx (moving left→right on the left side
        of the road), it may be wrong-side.
        """
        events = []
        mid_x = frame_width / 2

        for track in tracks:
            category = track.get("category", "")
            if category != "vehicles":
                continue

            track_id = track.get("track_id", -1)
            bbox = track.get("bbox", [0, 0, 0, 0])
            velocity = track.get("velocity", [0, 0])

            cx = (bbox[0] + bbox[2]) / 2
            vx = velocity[0] if len(velocity) > 0 else 0
            vy = velocity[1] if len(velocity) > 1 else 0

            # Check if vehicle is on the "wrong" side with opposing velocity
            # Left side of road (Indian roads): vehicles should move away (vy < 0 = going up)
            # If vehicle is on left side but moving right (vx > 0) + towards camera (vy > 0)
            is_wrong_side = False

            if cx < mid_x:
                # Vehicle on left side — should be moving upward in frame (vy negative)
                # If moving downward strongly, it's wrong-side
                if vy > abs(self.wrong_side_vel_thresh):
                    is_wrong_side = True
            else:
                # Vehicle on right side — should be moving downward (vy positive)
                # If moving upward strongly, it's wrong-side
                if vy < self.wrong_side_vel_thresh:
                    is_wrong_side = True

            if is_wrong_side:
                counter = self._wrong_side_counters.get(track_id, 0) + 1
                self._wrong_side_counters[track_id] = counter

                if counter >= self.wrong_side_min_frames:
                    events.append(AnomalyEvent(
                        type="wrong_side_vehicle",
                        severity="critical",
                        confidence=min(0.95, 0.6 + counter * 0.1),
                        details=f"Vehicle #{track_id} moving against traffic (vx={vx:.1f}, vy={vy:.1f})",
                        bbox=bbox,
                        track_id=track_id,
                    ))
            else:
                self._wrong_side_counters[track_id] = 0

        return events

    def _detect_sudden_crossing(
        self, tracks: List[Dict], frame_width: int, frame_height: int
    ) -> List[AnomalyEvent]:
        """
        Detect pedestrians/animals suddenly crossing into the ego lane.

        Heuristic: Large lateral velocity (|vx|) combined with
        proximity to ego vehicle (large y-position).
        """
        events = []
        ego_y_start = frame_height * 0.6

        for track in tracks:
            category = track.get("category", "")
            if category not in ("vulnerable_road_users", "animals"):
                continue

            track_id = track.get("track_id", -1)
            bbox = track.get("bbox", [0, 0, 0, 0])
            velocity = track.get("velocity", [0, 0])

            # Is the object in the ego danger zone?
            obj_y = bbox[3]  # Bottom of bbox
            if obj_y < ego_y_start:
                continue

            # Check for large lateral velocity (sudden crossing)
            vx = abs(velocity[0]) if len(velocity) > 0 else 0

            if vx > self.crossing_vel_thresh:
                # Check proximity to center (ego path)
                cx = (bbox[0] + bbox[2]) / 2
                center_proximity = abs(cx - frame_width / 2)

                if center_proximity < self.crossing_prox_thresh:
                    events.append(AnomalyEvent(
                        type="sudden_crossing",
                        severity="critical",
                        confidence=min(0.9, 0.5 + vx / 50),
                        details=f"{track.get('class_name', 'object')} crossing with vx={vx:.1f}px/frame",
                        bbox=bbox,
                        track_id=track_id,
                    ))

        return events

    def _detect_animal_on_road(
        self, tracks: List[Dict], drivable_mask: np.ndarray
    ) -> List[AnomalyEvent]:
        """
        Detect animals that are within the drivable area.

        In India, cows, dogs, and buffaloes commonly sit/stand on roads.
        """
        events = []
        h, w = drivable_mask.shape[:2]

        for track in tracks:
            if track.get("category") != "animals":
                continue

            bbox = track.get("bbox", [0, 0, 0, 0])
            track_id = track.get("track_id", -1)

            # Check if bbox center is on drivable area
            cx = int(np.clip((bbox[0] + bbox[2]) / 2, 0, w - 1))
            cy = int(np.clip((bbox[1] + bbox[3]) / 2, 0, h - 1))

            # Check a small region around center
            y_start = max(0, cy - 5)
            y_end = min(h, cy + 5)
            x_start = max(0, cx - 5)
            x_end = min(w, cx + 5)

            roi = drivable_mask[y_start:y_end, x_start:x_end]
            if roi.size > 0 and np.mean(roi) > 128:  # >50% of ROI is drivable
                events.append(AnomalyEvent(
                    type="animal_on_road",
                    severity="warning",
                    confidence=0.8,
                    details=f"{track.get('class_name', 'animal')} on drivable area at ({cx}, {cy})",
                    bbox=bbox,
                    track_id=track_id,
                ))

        return events

    def _detect_road_anomalies(
        self, frame: np.ndarray, drivable_mask: np.ndarray
    ) -> List[AnomalyEvent]:
        """
        Detect potholes and road surface damage using contrast analysis.

        Heuristic: Dark patches on the drivable area that are significantly
        darker than surrounding road surface.
        """
        import cv2

        events = []

        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Only analyze drivable area
            road_only = cv2.bitwise_and(gray, drivable_mask)

            # Calculate mean road brightness
            road_pixels = gray[drivable_mask > 128]
            if len(road_pixels) < 100:
                return events

            mean_brightness = np.mean(road_pixels)

            # Find dark patches (potential potholes)
            dark_threshold = max(20, mean_brightness - self.pothole_contrast_thresh)
            _, dark_mask = cv2.threshold(road_only, dark_threshold, 255, cv2.THRESH_BINARY_INV)

            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
            dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)

            # Only within drivable area
            dark_mask = cv2.bitwise_and(dark_mask, drivable_mask)

            # Find contours
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.pothole_min_area:
                    continue

                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / max(h, 1)

                # Potholes tend to be roughly circular (aspect ratio 0.5–2.0)
                if 0.3 < aspect < 3.0:
                    events.append(AnomalyEvent(
                        type="road_anomaly",
                        severity="warning",
                        confidence=min(0.85, 0.4 + area / 5000),
                        details=f"Pothole at ({x}, {y}), area={area:.0f}px²",
                        bbox=[float(x), float(y), float(x + w), float(y + h)],
                    ))

        except Exception as e:
            logger.error(f"Road anomaly detection failed: {e}")

        return events

    def _update_position_history(self, tracks: List[Dict]):
        """Update position history for velocity-based anomaly detection."""
        current_ids = set()
        for track in tracks:
            tid = track.get("track_id", -1)
            bbox = track.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self._prev_positions[tid] = [cx, cy]
            current_ids.add(tid)

        # Cleanup lost tracks
        stale = [tid for tid in self._prev_positions if tid not in current_ids]
        for tid in stale:
            del self._prev_positions[tid]
            self._wrong_side_counters.pop(tid, None)

    def reset(self):
        """Clear all internal state."""
        self._wrong_side_counters.clear()
        self._prev_positions.clear()
