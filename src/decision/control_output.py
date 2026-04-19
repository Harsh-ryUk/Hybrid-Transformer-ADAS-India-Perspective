"""
Vehicle Control Output Data Classes (L4 ADAS)
Defines the control commands output by the decision module.
Compatible with CARLA simulator and ROS2 interfaces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import time


class ActionType(Enum):
    """High-level driving action."""
    CRUISE = "cruise"
    ACCELERATE = "accelerate"
    SLOW_DOWN = "slow_down"
    BRAKE = "brake"
    HARD_BRAKE = "hard_brake"
    EMERGENCY_STOP = "emergency_stop"
    STEER_LEFT = "steer_left"
    STEER_RIGHT = "steer_right"
    LANE_KEEP = "lane_keep"
    STOP_AT_SIGNAL = "stop_at_signal"


class SeverityLevel(Enum):
    """Severity of an event or decision."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class VehicleControl:
    """
    Low-level vehicle control command.
    Values are normalized [0, 1] or [-1, 1] for steering.
    """
    steering: float = 0.0       # [-1.0 (full left) to 1.0 (full right)]
    throttle: float = 0.0       # [0.0 to 1.0]
    brake: float = 0.0          # [0.0 to 1.0]
    emergency_stop: bool = False
    hand_brake: bool = False

    def to_carla(self):
        """Convert to CARLA-compatible control dict."""
        return {
            "steer": self.steering,
            "throttle": self.throttle,
            "brake": self.brake,
            "hand_brake": self.hand_brake,
            "reverse": False,
        }

    def to_dict(self):
        return {
            "steering": round(self.steering, 3),
            "throttle": round(self.throttle, 3),
            "brake": round(self.brake, 3),
            "emergency_stop": self.emergency_stop,
        }


@dataclass
class DecisionOutput:
    """
    Complete decision output from the rule engine.
    """
    action: ActionType = ActionType.CRUISE
    control: VehicleControl = field(default_factory=VehicleControl)
    confidence: float = 1.0
    reason: str = ""
    severity: SeverityLevel = SeverityLevel.INFO
    active_events: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def is_emergency(self) -> bool:
        return self.severity == SeverityLevel.EMERGENCY or self.control.emergency_stop

    def to_dict(self):
        return {
            "action": self.action.value,
            "control": self.control.to_dict(),
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "severity": self.severity.value,
            "active_events": self.active_events,
        }
