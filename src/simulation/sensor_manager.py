"""
CARLA Sensor Manager (L4 ADAS Simulation)
Manages camera and other sensors attached to the ego vehicle in CARLA.

Handles:
- RGB camera sensor lifecycle
- Semantic segmentation camera (ground truth)
- Frame buffer queuing for real-time processing
- Camera intrinsics configuration
"""

import logging
import queue
import numpy as np
from typing import Dict, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for a CARLA camera sensor."""
    width: int = 1280
    height: int = 720
    fov: float = 110.0
    position: tuple = (1.5, 0.0, 2.4)    # x, y, z relative to vehicle
    rotation: tuple = (0.0, 0.0, 0.0)    # pitch, yaw, roll


class SensorManager:
    """
    Manages CARLA sensor lifecycle and frame buffering.

    Sensors:
    - RGB camera: feeds frames into the perception pipeline
    - Semantic segmentation camera: provides ground truth for evaluation

    Frame Buffer:
    - Thread-safe queue with configurable max size
    - Drops oldest frames if queue is full (real-time processing)
    """

    def __init__(self, camera_config: Optional[CameraConfig] = None, buffer_size: int = 5):
        self.camera_config = camera_config or CameraConfig()
        self.buffer_size = buffer_size

        # Frame buffers (thread-safe)
        self.rgb_buffer = queue.Queue(maxsize=buffer_size)
        self.seg_buffer = queue.Queue(maxsize=buffer_size)

        # CARLA references (set during attach)
        self._rgb_sensor = None
        self._seg_sensor = None
        self._attached = False

    def attach_to_vehicle(self, world, vehicle):
        """
        Attach camera sensors to a CARLA vehicle.

        Args:
            world: carla.World
            vehicle: carla.Vehicle (ego vehicle)
        """
        try:
            import carla

            blueprint_library = world.get_blueprint_library()

            # ─── RGB Camera ───
            rgb_bp = blueprint_library.find("sensor.camera.rgb")
            rgb_bp.set_attribute("image_size_x", str(self.camera_config.width))
            rgb_bp.set_attribute("image_size_y", str(self.camera_config.height))
            rgb_bp.set_attribute("fov", str(self.camera_config.fov))

            pos = self.camera_config.position
            rot = self.camera_config.rotation
            transform = carla.Transform(
                carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
            )

            self._rgb_sensor = world.spawn_actor(rgb_bp, transform, attach_to=vehicle)
            self._rgb_sensor.listen(self._rgb_callback)

            # ─── Semantic Segmentation Camera ───
            seg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
            seg_bp.set_attribute("image_size_x", str(self.camera_config.width))
            seg_bp.set_attribute("image_size_y", str(self.camera_config.height))
            seg_bp.set_attribute("fov", str(self.camera_config.fov))

            self._seg_sensor = world.spawn_actor(seg_bp, transform, attach_to=vehicle)
            self._seg_sensor.listen(self._seg_callback)

            self._attached = True
            logger.info(f"Sensors attached: RGB({self.camera_config.width}x{self.camera_config.height}), "
                       f"SegCam, FOV={self.camera_config.fov}")

        except ImportError:
            logger.error("CARLA Python API not available — cannot attach sensors")
        except Exception as e:
            logger.error(f"Failed to attach sensors: {e}")

    def _rgb_callback(self, image):
        """Process incoming RGB camera frame."""
        try:
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))  # BGRA
            bgr = array[:, :, :3]  # Drop alpha

            # Non-blocking put — drop oldest if full
            if self.rgb_buffer.full():
                try:
                    self.rgb_buffer.get_nowait()
                except queue.Empty:
                    pass
            self.rgb_buffer.put(bgr.copy())

        except Exception as e:
            logger.error(f"RGB callback error: {e}")

    def _seg_callback(self, image):
        """Process incoming semantic segmentation frame."""
        try:
            import carla
            image.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            seg = array[:, :, :3]

            if self.seg_buffer.full():
                try:
                    self.seg_buffer.get_nowait()
                except queue.Empty:
                    pass
            self.seg_buffer.put(seg.copy())

        except Exception as e:
            logger.error(f"Segmentation callback error: {e}")

    def get_rgb_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest RGB frame (blocking with timeout)."""
        try:
            return self.rgb_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_seg_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Get the latest segmentation frame."""
        try:
            return self.seg_buffer.get(timeout=timeout)
        except queue.Empty:
            return None

    def destroy(self):
        """Destroy all sensors and clear buffers."""
        if self._rgb_sensor is not None:
            try:
                self._rgb_sensor.stop()
                self._rgb_sensor.destroy()
            except:
                pass
        if self._seg_sensor is not None:
            try:
                self._seg_sensor.stop()
                self._seg_sensor.destroy()
            except:
                pass

        self._rgb_sensor = None
        self._seg_sensor = None
        self._attached = False

        # Clear buffers
        while not self.rgb_buffer.empty():
            self.rgb_buffer.get_nowait()
        while not self.seg_buffer.empty():
            self.seg_buffer.get_nowait()

        logger.info("All sensors destroyed and buffers cleared.")

    @property
    def is_attached(self) -> bool:
        return self._attached
