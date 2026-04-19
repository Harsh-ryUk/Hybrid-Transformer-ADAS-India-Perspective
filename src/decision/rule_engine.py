"""
Rule-Based Decision Engine (L4 ADAS — Phase 1)
Makes real-time driving decisions based on perception + tracking data.

Rules implemented:
1. Emergency brake if obstacle < critical distance
2. Stop at red traffic light
3. Maintain lane discipline (steer correction)
4. Slow down in crowded environments
5. React to anomaly events (wrong-side, animal, sudden crossing)

India-specific considerations:
- Higher density tolerance (Indian traffic is always dense)
- Animal-on-road triggers immediate slow-down
- Auto-rickshaw proximity → wider safety margin
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.decision.control_output import (
    VehicleControl, DecisionOutput, ActionType, SeverityLevel
)

logger = logging.getLogger(__name__)


@dataclass
class SceneContext:
    """
    Aggregated scene understanding from perception + tracking.
    This is the input to the decision engine.
    """
    # Tracked objects with positions and velocities
    tracked_objects: List[Dict] = None  # track_id, bbox, class, category, velocity
    # Traffic signal state
    traffic_signal: str = "Unknown"     # "Red", "Green", "Yellow", "Off", "Unknown"
    # Lane information
    lane_center_offset: float = 0.0     # Pixels from lane center (negative=left)
    lane_detected: bool = False
    drivable_mask: Optional[np.ndarray] = None
    # Anomaly events from event detector
    active_anomalies: List[Dict] = None  # type, severity, details
    # Frame metadata
    frame_height: int = 720
    frame_width: int = 1280
    ego_zone_y_start: float = 0.6       # Bottom 40% = danger zone

    def __post_init__(self):
        if self.tracked_objects is None:
            self.tracked_objects = []
        if self.active_anomalies is None:
            self.active_anomalies = []


class RuleBasedDecisionEngine:
    """
    Phase 1 decision-making: deterministic rules for safe driving.

    Operates on the SceneContext built from perception + tracking outputs.
    Produces VehicleControl commands that can be sent to CARLA or logged.

    Designed for Indian roads:
    - Handles extreme traffic density
    - Respects animal presence as high-priority obstacle
    - Adapts to missing/faded lane markings
    """

    def __init__(
        self,
        emergency_brake_distance: float = 80,
        hard_brake_distance: float = 150,
        soft_brake_distance: float = 250,
        slow_down_distance: float = 400,
        lane_departure_threshold: float = 50,
        crowd_density_threshold: int = 5,
        crowd_speed_factor: float = 0.5,
        max_speed: float = 40.0,
        cruise_speed: float = 30.0,
        slow_speed: float = 15.0,
    ):
        self.emergency_brake_dist = emergency_brake_distance
        self.hard_brake_dist = hard_brake_distance
        self.soft_brake_dist = soft_brake_distance
        self.slow_down_dist = slow_down_distance
        self.lane_departure_threshold = lane_departure_threshold
        self.crowd_density_threshold = crowd_density_threshold
        self.crowd_speed_factor = crowd_speed_factor
        self.max_speed = max_speed
        self.cruise_speed = cruise_speed
        self.slow_speed = slow_speed

    def decide(self, scene: SceneContext) -> DecisionOutput:
        """
        Main decision loop. Evaluates rules in priority order.

        Priority (highest first):
        1. Emergency obstacles
        2. Anomaly events (wrong-side, sudden crossing)
        3. Traffic signal
        4. Crowd density
        5. Lane discipline
        6. Default cruise

        Args:
            scene: Aggregated scene context

        Returns:
            DecisionOutput with action + control + reason
        """
        active_events = []

        # ── Rule 1: Emergency Obstacles ──
        closest_obj, closest_dist = self._find_closest_obstacle(scene)
        if closest_obj is not None:
            if closest_dist < self.emergency_brake_dist:
                return DecisionOutput(
                    action=ActionType.EMERGENCY_STOP,
                    control=VehicleControl(brake=1.0, emergency_stop=True),
                    confidence=0.95,
                    reason=f"EMERGENCY: {closest_obj.get('class_name', 'object')} at {closest_dist:.0f}px",
                    severity=SeverityLevel.EMERGENCY,
                    active_events=["emergency_obstacle"],
                )

            if closest_dist < self.hard_brake_dist:
                brake_intensity = 1.0 - (closest_dist / self.hard_brake_dist)
                return DecisionOutput(
                    action=ActionType.HARD_BRAKE,
                    control=VehicleControl(brake=min(0.9, brake_intensity + 0.3)),
                    confidence=0.9,
                    reason=f"Hard brake: {closest_obj.get('class_name', 'object')} at {closest_dist:.0f}px",
                    severity=SeverityLevel.CRITICAL,
                    active_events=["close_obstacle"],
                )

        # ── Rule 2: Anomaly Events ──
        anomaly_decision = self._handle_anomalies(scene)
        if anomaly_decision is not None:
            return anomaly_decision

        # ── Rule 3: Traffic Signal ──
        signal_decision = self._handle_traffic_signal(scene)
        if signal_decision is not None:
            return signal_decision

        # ── Rule 4: Crowd Density ──
        ego_zone_objects = self._count_ego_zone_objects(scene)
        if ego_zone_objects >= self.crowd_density_threshold:
            active_events.append("crowded_environment")
            speed_factor = self.crowd_speed_factor
            return DecisionOutput(
                action=ActionType.SLOW_DOWN,
                control=VehicleControl(
                    throttle=0.2 * speed_factor,
                    brake=0.1,
                ),
                confidence=0.85,
                reason=f"Crowded: {ego_zone_objects} objects in ego zone",
                severity=SeverityLevel.WARNING,
                active_events=active_events,
            )

        # ── Rule 5: Soft Brake for Medium-Distance Obstacles ──
        if closest_obj is not None and closest_dist < self.soft_brake_dist:
            category = closest_obj.get("category", "")
            # Animals and pedestrians get extra caution
            if category in ("animals", "vulnerable_road_users"):
                return DecisionOutput(
                    action=ActionType.SLOW_DOWN,
                    control=VehicleControl(throttle=0.1, brake=0.3),
                    confidence=0.85,
                    reason=f"Caution: {closest_obj.get('class_name', 'VRU')} ahead at {closest_dist:.0f}px",
                    severity=SeverityLevel.WARNING,
                    active_events=["vru_ahead"],
                )

            return DecisionOutput(
                action=ActionType.SLOW_DOWN,
                control=VehicleControl(throttle=0.2, brake=0.15),
                confidence=0.8,
                reason=f"Obstacle ahead at {closest_dist:.0f}px",
                severity=SeverityLevel.INFO,
                active_events=["obstacle_ahead"],
            )

        # ── Rule 6: Lane Discipline ──
        steering = self._compute_lane_steering(scene)

        # ── Rule 7: Default Cruise ──
        return DecisionOutput(
            action=ActionType.CRUISE,
            control=VehicleControl(
                throttle=0.5,
                steering=steering,
            ),
            confidence=0.9,
            reason="Clear road — cruising",
            severity=SeverityLevel.INFO,
            active_events=active_events,
        )

    def _find_closest_obstacle(self, scene: SceneContext) -> Tuple[Optional[Dict], float]:
        """Find the closest object in the ego zone (bottom portion of frame)."""
        ego_y_start = scene.frame_height * scene.ego_zone_y_start
        closest = None
        closest_dist = float("inf")

        for obj in scene.tracked_objects:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            # Bottom-center of bbox
            obj_y = bbox[3]  # y2 (bottom of bbox = closest point to ego)
            obj_cx = (bbox[0] + bbox[2]) / 2

            if obj_y < ego_y_start:
                continue  # Object is above ego zone

            # Distance proxy: distance from bottom of frame
            dist = scene.frame_height - obj_y

            # Lateral proximity factor (objects in center are more dangerous)
            center_x = scene.frame_width / 2
            lateral_offset = abs(obj_cx - center_x) / center_x
            # Adjust distance — closer to center = more dangerous
            adjusted_dist = dist * (1.0 + lateral_offset * 0.3)

            if adjusted_dist < closest_dist:
                closest_dist = adjusted_dist
                closest = obj

        return closest, closest_dist

    def _handle_anomalies(self, scene: SceneContext) -> Optional[DecisionOutput]:
        """Handle anomaly events with appropriate responses."""
        for anomaly in scene.active_anomalies:
            event_type = anomaly.get("type", "")
            severity = anomaly.get("severity", "warning")

            if event_type == "wrong_side_vehicle":
                return DecisionOutput(
                    action=ActionType.HARD_BRAKE,
                    control=VehicleControl(brake=0.8, steering=0.3),  # Steer right + brake
                    confidence=0.85,
                    reason="WRONG-SIDE vehicle detected — braking + steering right",
                    severity=SeverityLevel.CRITICAL,
                    active_events=["wrong_side_vehicle"],
                )

            if event_type == "sudden_crossing":
                return DecisionOutput(
                    action=ActionType.EMERGENCY_STOP,
                    control=VehicleControl(brake=1.0, emergency_stop=True),
                    confidence=0.9,
                    reason="Sudden pedestrian/animal crossing — emergency stop",
                    severity=SeverityLevel.EMERGENCY,
                    active_events=["sudden_crossing"],
                )

            if event_type == "animal_on_road":
                return DecisionOutput(
                    action=ActionType.SLOW_DOWN,
                    control=VehicleControl(throttle=0.1, brake=0.4),
                    confidence=0.85,
                    reason=f"Animal on drivable area: {anomaly.get('details', '')}",
                    severity=SeverityLevel.WARNING,
                    active_events=["animal_on_road"],
                )

            if event_type == "road_anomaly":
                return DecisionOutput(
                    action=ActionType.SLOW_DOWN,
                    control=VehicleControl(throttle=0.15, brake=0.2),
                    confidence=0.75,
                    reason=f"Road surface anomaly: {anomaly.get('details', 'pothole')}",
                    severity=SeverityLevel.WARNING,
                    active_events=["road_anomaly"],
                )

        return None

    def _handle_traffic_signal(self, scene: SceneContext) -> Optional[DecisionOutput]:
        """React to traffic signal state."""
        signal = scene.traffic_signal

        if signal == "Red":
            return DecisionOutput(
                action=ActionType.STOP_AT_SIGNAL,
                control=VehicleControl(brake=0.7),
                confidence=0.95,
                reason="Red traffic light — stopping",
                severity=SeverityLevel.INFO,
                active_events=["red_signal"],
            )

        if signal == "Yellow":
            return DecisionOutput(
                action=ActionType.SLOW_DOWN,
                control=VehicleControl(throttle=0.1, brake=0.3),
                confidence=0.85,
                reason="Yellow traffic light — preparing to stop",
                severity=SeverityLevel.INFO,
                active_events=["yellow_signal"],
            )

        return None  # Green or Unknown → continue

    def _count_ego_zone_objects(self, scene: SceneContext) -> int:
        """Count objects in the ego danger zone."""
        ego_y_start = scene.frame_height * scene.ego_zone_y_start
        count = 0
        for obj in scene.tracked_objects:
            bbox = obj.get("bbox", [0, 0, 0, 0])
            if bbox[3] > ego_y_start:  # Object's bottom is in ego zone
                count += 1
        return count

    def _compute_lane_steering(self, scene: SceneContext) -> float:
        """Compute steering correction for lane keeping."""
        if not scene.lane_detected:
            return 0.0  # No lane info — go straight

        offset = scene.lane_center_offset
        if abs(offset) < self.lane_departure_threshold:
            return 0.0  # Within tolerance

        # Proportional steering correction
        max_steer = 0.3
        steering = np.clip(
            -offset / (self.lane_departure_threshold * 3),
            -max_steer,
            max_steer,
        )
        return float(steering)
