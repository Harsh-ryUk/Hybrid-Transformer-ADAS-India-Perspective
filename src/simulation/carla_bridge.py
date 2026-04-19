"""
CARLA Simulation Bridge (L4 ADAS)
Connects the ADAS perception pipeline to CARLA simulator.

Architecture:
  CARLA Server ←→ Bridge ←→ Perception Pipeline ←→ Decision Module → Control Output → CARLA Vehicle

Features:
- Spawns ego vehicle with camera sensors
- Feeds camera frames into the perception pipeline
- Applies control outputs back to the CARLA vehicle
- Spawns NPC traffic (vehicles + pedestrians)
- Graceful degradation when CARLA is not installed

Reference:
  CARLA Simulator: https://carla.org/
"""

import logging
import time
import sys
import numpy as np
from typing import Optional, Dict, Callable

from src.simulation.sensor_manager import SensorManager, CameraConfig

logger = logging.getLogger(__name__)

# Check CARLA availability
CARLA_AVAILABLE = False
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    logger.info("CARLA Python API not found — simulation bridge will run in stub mode")


class CARLABridge:
    """
    Bridge between the L4 ADAS pipeline and CARLA simulator.

    Usage:
        bridge = CARLABridge()
        bridge.connect()
        bridge.setup_scene()

        while bridge.is_running:
            frame = bridge.get_frame()
            if frame is not None:
                # ... run perception pipeline ...
                bridge.apply_control(steering, throttle, brake)

        bridge.cleanup()

    If CARLA is not installed, all methods are no-ops.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        town: str = "Town03",
        weather: str = "ClearNoon",
        ego_vehicle_filter: str = "vehicle.tesla.model3",
        camera_config: Optional[CameraConfig] = None,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.town = town
        self.weather = weather
        self.ego_vehicle_filter = ego_vehicle_filter

        # CARLA objects
        self._client = None
        self._world = None
        self._ego_vehicle = None
        self._traffic_manager = None
        self._npc_vehicles = []
        self._npc_walkers = []

        # Sensor manager
        self.sensor_manager = SensorManager(camera_config or CameraConfig())

        self._connected = False
        self._running = False

    @property
    def is_available(self) -> bool:
        return CARLA_AVAILABLE

    @property
    def is_running(self) -> bool:
        return self._running

    def connect(self) -> bool:
        """
        Connect to CARLA server.

        Returns:
            True if connection succeeded, False otherwise
        """
        if not CARLA_AVAILABLE:
            logger.warning("CARLA not available — running in stub mode")
            return False

        try:
            self._client = carla.Client(self.host, self.port)
            self._client.set_timeout(self.timeout)

            # Load town
            logger.info(f"Loading {self.town}...")
            self._world = self._client.load_world(self.town)

            # Set weather
            self._set_weather(self.weather)

            # Set synchronous mode for deterministic simulation
            settings = self._world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self._world.apply_settings(settings)

            # Traffic manager
            self._traffic_manager = self._client.get_trafficmanager(8000)
            self._traffic_manager.set_synchronous_mode(True)

            self._connected = True
            logger.info(f"Connected to CARLA at {self.host}:{self.port}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to CARLA: {e}")
            return False

    def setup_scene(
        self,
        num_vehicles: int = 30,
        num_pedestrians: int = 20,
    ) -> bool:
        """
        Setup the simulation scene with ego vehicle and NPC traffic.

        Args:
            num_vehicles: Number of NPC vehicles to spawn
            num_pedestrians: Number of NPC pedestrians

        Returns:
            True if setup succeeded
        """
        if not self._connected:
            logger.warning("Not connected to CARLA")
            return False

        try:
            # ─── Spawn Ego Vehicle ───
            blueprint_library = self._world.get_blueprint_library()
            ego_bp = blueprint_library.filter(self.ego_vehicle_filter)[0]
            ego_bp.set_attribute("role_name", "ego")

            spawn_points = self._world.get_map().get_spawn_points()
            if not spawn_points:
                logger.error("No spawn points available")
                return False

            self._ego_vehicle = self._world.spawn_actor(ego_bp, spawn_points[0])
            logger.info(f"Ego vehicle spawned: {ego_bp.id}")

            # ─── Attach Sensors ───
            self.sensor_manager.attach_to_vehicle(self._world, self._ego_vehicle)

            # ─── Spawn NPC Traffic ───
            self._spawn_npc_vehicles(num_vehicles, spawn_points[1:])
            self._spawn_npc_walkers(num_pedestrians)

            self._running = True
            logger.info(f"Scene ready: {num_vehicles} vehicles, {num_pedestrians} pedestrians")
            return True

        except Exception as e:
            logger.error(f"Scene setup failed: {e}")
            return False

    def _spawn_npc_vehicles(self, count: int, spawn_points):
        """Spawn NPC vehicles with autopilot."""
        if not CARLA_AVAILABLE:
            return

        bp_library = self._world.get_blueprint_library()
        vehicle_bps = bp_library.filter("vehicle.*")

        import random
        random.shuffle(spawn_points)

        for i in range(min(count, len(spawn_points))):
            bp = random.choice(vehicle_bps)
            try:
                npc = self._world.spawn_actor(bp, spawn_points[i])
                npc.set_autopilot(True, self._traffic_manager.get_port())
                self._npc_vehicles.append(npc)
            except:
                pass

        logger.info(f"Spawned {len(self._npc_vehicles)} NPC vehicles")

    def _spawn_npc_walkers(self, count: int):
        """Spawn NPC pedestrians with AI controllers."""
        if not CARLA_AVAILABLE:
            return

        bp_library = self._world.get_blueprint_library()
        walker_bps = bp_library.filter("walker.pedestrian.*")
        controller_bp = bp_library.find("controller.ai.walker")

        import random

        for _ in range(count):
            try:
                spawn_point = carla.Transform()
                loc = self._world.get_random_location_from_navigation()
                if loc is not None:
                    spawn_point.location = loc

                bp = random.choice(walker_bps)
                walker = self._world.spawn_actor(bp, spawn_point)

                controller = self._world.spawn_actor(controller_bp, carla.Transform(), walker)
                controller.start()
                controller.go_to_location(self._world.get_random_location_from_navigation())
                controller.set_max_speed(1.4 + random.random())

                self._npc_walkers.extend([walker, controller])
            except:
                pass

        logger.info(f"Spawned {len(self._npc_walkers) // 2} NPC pedestrians")

    def _set_weather(self, preset: str):
        """Set weather conditions."""
        if not CARLA_AVAILABLE:
            return

        presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
        }

        weather = presets.get(preset, carla.WeatherParameters.ClearNoon)
        self._world.set_weather(weather)
        logger.info(f"Weather set to: {preset}")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get the latest camera frame from the simulation.

        Returns:
            BGR image or None if no frame available
        """
        if not self._running:
            return None

        # Tick the simulation forward
        try:
            self._world.tick()
        except:
            pass

        return self.sensor_manager.get_rgb_frame(timeout)

    def get_segmentation_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest semantic segmentation frame (ground truth)."""
        if not self._running:
            return None
        return self.sensor_manager.get_seg_frame(timeout)

    def apply_control(
        self,
        steering: float = 0.0,
        throttle: float = 0.5,
        brake: float = 0.0,
        hand_brake: bool = False,
        reverse: bool = False,
    ):
        """
        Apply vehicle control commands to the ego vehicle.

        Args:
            steering: [-1.0, 1.0] left to right
            throttle: [0.0, 1.0]
            brake: [0.0, 1.0]
            hand_brake: emergency brake
            reverse: reverse gear
        """
        if not self._running or self._ego_vehicle is None:
            return

        try:
            control = carla.VehicleControl(
                steer=np.clip(steering, -1.0, 1.0),
                throttle=np.clip(throttle, 0.0, 1.0),
                brake=np.clip(brake, 0.0, 1.0),
                hand_brake=hand_brake,
                reverse=reverse,
            )
            self._ego_vehicle.apply_control(control)
        except Exception as e:
            logger.error(f"Failed to apply control: {e}")

    def apply_decision_output(self, decision):
        """
        Apply a DecisionOutput directly.

        Args:
            decision: DecisionOutput from the rule engine
        """
        ctrl = decision.control
        self.apply_control(
            steering=ctrl.steering,
            throttle=ctrl.throttle,
            brake=ctrl.brake,
            hand_brake=ctrl.hand_brake,
        )

    def get_ego_transform(self) -> Optional[Dict]:
        """Get ego vehicle position and rotation."""
        if not self._running or self._ego_vehicle is None:
            return None

        try:
            t = self._ego_vehicle.get_transform()
            v = self._ego_vehicle.get_velocity()
            return {
                "location": {"x": t.location.x, "y": t.location.y, "z": t.location.z},
                "rotation": {"pitch": t.rotation.pitch, "yaw": t.rotation.yaw, "roll": t.rotation.roll},
                "speed_kmh": 3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2),
            }
        except:
            return None

    def cleanup(self):
        """Destroy all actors and disconnect."""
        self._running = False

        # Destroy sensors
        self.sensor_manager.destroy()

        # Destroy NPC walkers (controllers first)
        for actor in reversed(self._npc_walkers):
            try:
                actor.stop() if hasattr(actor, 'stop') else None
                actor.destroy()
            except:
                pass

        # Destroy NPC vehicles
        for v in self._npc_vehicles:
            try:
                v.destroy()
            except:
                pass

        # Destroy ego
        if self._ego_vehicle is not None:
            try:
                self._ego_vehicle.destroy()
            except:
                pass

        self._npc_vehicles.clear()
        self._npc_walkers.clear()
        self._ego_vehicle = None

        # Reset synchronous mode
        if self._world is not None:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                self._world.apply_settings(settings)
            except:
                pass

        self._connected = False
        logger.info("CARLA bridge cleaned up.")

    def __del__(self):
        self.cleanup()
