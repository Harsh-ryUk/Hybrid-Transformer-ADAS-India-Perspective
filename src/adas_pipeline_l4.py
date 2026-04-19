"""
ADAS Level 4 Pipeline — India-Focused (Master Orchestrator)
Integrates all modules into a unified real-time processing pipeline.

Pipeline Flow:
  Frame → Detect → Track → Segment → Analyze Events → Decide → Control

Modules:
- Perception: IndiaObjectDetector (YOLOv8) + OWLv2 (zero-shot)
- Segmentation: SegFormerLaneDetector (drivable area + lanes)
- Tracking: DeepSORTTracker (Kalman + Hungarian)
- Anomaly: AnomalyEventDetector (wrong-side, crossing, animal, pothole)
- Decision: RuleBasedDecisionEngine → VehicleControl
- Evaluation: SystemProfiler (per-stage latency)
- Simulation: CARLABridge (optional)

Indian Datasets Used:
- IDD: http://idd.insaan.iiit.ac.in/
- BDD100K: https://bdd-data.berkeley.edu/
- Mapillary Vistas: https://www.mapillary.com/dataset/vistas
"""

import logging
import time
import yaml
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple

# ─── Module Imports ───
from src.perception.india_detector import IndiaObjectDetector, DetectionResult
from src.perception.owl_detector import OWLv2Detector
from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
from src.tracking.deep_sort_tracker import DeepSORTTracker
from src.anomaly.event_detector import AnomalyEventDetector
from src.decision.rule_engine import RuleBasedDecisionEngine, SceneContext
from src.decision.control_output import DecisionOutput, VehicleControl
from src.traffic_control.signal_classifier import TrafficSignalClassifier
from src.evaluation.metrics import LatencyProfile, SystemProfiler

logger = logging.getLogger(__name__)


class ADASPipelineL4:
    """
    Level 4 ADAS Pipeline for Indian Road Conditions.

    Combines all perception, tracking, decision, and anomaly modules
    into a single real-time processing loop.

    Architecture:
    ```
    Camera → IndiaDetector → DeepSORT → SegFormer → AnomalyDetector → RuleEngine → Control
               ↓                                        ↑
           OWLv2 (async)                          TrafficClassifier
    ```

    Usage:
        pipeline = ADASPipelineL4(config_path="config.yaml")
        pipeline.process_video("data/samples/indian_road.mp4")

        # Or with CARLA:
        pipeline.run_simulation()
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize all pipeline components.

        Args:
            config_path: Path to YAML config file
            device: 'cuda' or 'cpu'
        """
        self.config = self._load_config(config_path)
        self.device = device

        logger.info("═" * 60)
        logger.info("  ADAS Level 4 Pipeline — India-Focused")
        logger.info("═" * 60)

        # ─── Perception ───
        det_cfg = self.config.get("detection", {})
        self.detector = IndiaObjectDetector(
            model_path=det_cfg.get("model_path", "yolov8n.pt"),
            conf_thres=det_cfg.get("confidence_threshold", 0.35),
            iou_thres=det_cfg.get("iou_threshold", 0.45),
            device=device,
            category_thresholds=det_cfg.get("category_thresholds"),
        )

        zs_cfg = self.config.get("zero_shot", {})
        self.owl_detector = None
        if zs_cfg.get("enabled", False):
            self.owl_detector = OWLv2Detector(
                model_name=zs_cfg.get("model_name", "google/owlv2-base-patch16-ensemble"),
                text_queries=zs_cfg.get("text_queries"),
                confidence_threshold=zs_cfg.get("confidence_threshold", 0.15),
                device=device,
                run_every_n_frames=zs_cfg.get("run_every_n_frames", 10),
            )

        # ─── Segmentation ───
        seg_cfg = self.config.get("segmentation", {})
        self.lane_detector = SegFormerLaneDetector(
            model_name=seg_cfg.get("model_name", "nvidia/segformer-b0-finetuned-ade-512-512"),
            device=device,
        )

        # ─── Traffic Signal ───
        self.signal_classifier = TrafficSignalClassifier()

        # ─── Tracking ───
        trk_cfg = self.config.get("tracking", {})
        self.tracker = DeepSORTTracker(
            max_age=trk_cfg.get("max_age", 30),
            min_hits=trk_cfg.get("min_hits", 3),
            iou_threshold=trk_cfg.get("iou_threshold", 0.3),
        )

        # ─── Anomaly Detection ───
        anom_cfg = self.config.get("anomaly", {})
        self.anomaly_detector = AnomalyEventDetector(
            wrong_side_velocity_threshold=anom_cfg.get("wrong_side_velocity_threshold", -5.0),
            wrong_side_min_frames=anom_cfg.get("wrong_side_min_frames", 3),
            crossing_velocity_threshold=anom_cfg.get("crossing_velocity_threshold", 15.0),
            crossing_proximity_threshold=anom_cfg.get("crossing_proximity_threshold", 200.0),
            pothole_contrast_threshold=anom_cfg.get("pothole_contrast_threshold", 40.0),
            pothole_min_area=anom_cfg.get("pothole_min_area", 500.0),
        )

        # ─── Decision Engine ───
        dec_cfg = self.config.get("decision", {})
        self.decision_engine = RuleBasedDecisionEngine(
            emergency_brake_distance=dec_cfg.get("emergency_brake_distance", 80),
            hard_brake_distance=dec_cfg.get("hard_brake_distance", 150),
            soft_brake_distance=dec_cfg.get("soft_brake_distance", 250),
            slow_down_distance=dec_cfg.get("slow_down_distance", 400),
            lane_departure_threshold=dec_cfg.get("lane_departure_threshold", 50),
            crowd_density_threshold=dec_cfg.get("crowd_density_threshold", 5),
            max_speed=dec_cfg.get("max_speed", 40.0),
            cruise_speed=dec_cfg.get("cruise_speed", 30.0),
            slow_speed=dec_cfg.get("slow_speed", 15.0),
        )

        # ─── Profiler ───
        self.profiler = SystemProfiler()

        # ─── State ───
        self.frame_count = 0

        logger.info("All modules initialized successfully.")

    def _load_config(self, path: Optional[str]) -> Dict:
        """Load YAML config or return defaults."""
        if path:
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Config load failed ({e}), using defaults")
        return {}

    def process_frame(
        self, frame: np.ndarray, frame_id: int = 0
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single frame through the full L4 pipeline.

        Args:
            frame: BGR image (numpy array)
            frame_id: Frame sequence number

        Returns:
            (visualized_frame, metrics_dict)
        """
        self.frame_count += 1
        h, w = frame.shape[:2]
        latency = LatencyProfile()
        t_total_start = time.time()

        # ═════════════════════════════════════════════════════════════
        # Stage 1: Object Detection (YOLOv8 — India-aware)
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        detection_result = self.detector.detect(frame)
        latency.detection_ms = (time.time() - t0) * 1000

        # Optional: OWLv2 zero-shot detection (runs periodically)
        owl_detections = []
        if self.owl_detector is not None:
            owl_detections = self.owl_detector.detect(frame)

        # ═════════════════════════════════════════════════════════════
        # Stage 2: Multi-Object Tracking (DeepSORT)
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        boxes, scores, class_ids = detection_result.get_boxes_scores_classes()
        class_names = [d.class_name for d in detection_result.detections]
        categories = [d.category for d in detection_result.detections]

        tracks = self.tracker.update(
            detections=boxes,
            scores=scores,
            class_names=class_names,
            categories=categories,
        )
        latency.tracking_ms = (time.time() - t0) * 1000

        # ═════════════════════════════════════════════════════════════
        # Stage 3: Lane & Drivable Area Segmentation (SegFormer)
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        # Detect seasonal condition for adaptive preprocessing
        condition = self._detect_condition(frame)
        lane_result = self.lane_detector.detect(frame, condition=condition)
        lane_mask = lane_result.get("lane_mask", np.zeros((h, w), dtype=np.uint8))
        latency.segmentation_ms = (time.time() - t0) * 1000

        # ═════════════════════════════════════════════════════════════
        # Stage 4: Traffic Signal Classification
        # ═════════════════════════════════════════════════════════════
        traffic_signal = "Unknown"
        for det in detection_result.detections:
            if det.class_name == "traffic_light":
                bbox = [int(v) for v in det.bbox]
                traffic_signal = self.signal_classifier.classify(frame, tuple(bbox))
                break

        # ═════════════════════════════════════════════════════════════
        # Stage 5: Anomaly Detection
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        track_dicts = [
            {
                "track_id": t.track_id,
                "bbox": t.bbox.tolist() if isinstance(t.bbox, np.ndarray) else t.bbox,
                "class_name": t.class_name,
                "category": t.category,
                "velocity": t.velocity.tolist() if isinstance(t.velocity, np.ndarray) else t.velocity,
            }
            for t in tracks
        ]

        anomaly_events = self.anomaly_detector.detect(
            tracks=track_dicts,
            drivable_mask=lane_mask,
            frame=frame,
            frame_width=w,
            frame_height=h,
        )
        latency.anomaly_ms = (time.time() - t0) * 1000

        # ═════════════════════════════════════════════════════════════
        # Stage 6: Decision Making
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        scene = SceneContext(
            tracked_objects=track_dicts,
            traffic_signal=traffic_signal,
            lane_center_offset=self._compute_lane_offset(lane_result, w),
            lane_detected=len(lane_result.get("lane_points", [])) > 0,
            drivable_mask=lane_mask,
            active_anomalies=[e.to_dict() for e in anomaly_events],
            frame_height=h,
            frame_width=w,
        )

        decision = self.decision_engine.decide(scene)
        latency.decision_ms = (time.time() - t0) * 1000

        # ═════════════════════════════════════════════════════════════
        # Stage 7: Visualization
        # ═════════════════════════════════════════════════════════════
        t0 = time.time()
        viz_frame = self._visualize(
            frame, detection_result, tracks, lane_result,
            anomaly_events, decision, owl_detections, traffic_signal, latency
        )
        latency.visualization_ms = (time.time() - t0) * 1000

        latency.total_ms = (time.time() - t_total_start) * 1000
        self.profiler.add_frame(latency)

        # ─── Metrics ───
        metrics = {
            "frame_id": self.frame_count,
            "detections": detection_result.num_detections,
            "tracks": len(tracks),
            "anomalies": len(anomaly_events),
            "decision": decision.to_dict(),
            "traffic_signal": traffic_signal,
            "latency": latency.to_dict(),
            "owl_detections": len(owl_detections),
        }

        return viz_frame, metrics

    def _detect_condition(self, frame: np.ndarray) -> str:
        """Simple brightness-based condition detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 60:
            return "Night"
        elif mean_brightness < 100:
            return "Monsoon"
        return "Normal"

    def _compute_lane_offset(self, lane_result: Dict, frame_width: int) -> float:
        """Compute ego offset from lane center."""
        lane_points = lane_result.get("lane_points", [])
        if len(lane_points) < 2:
            return 0.0

        # Average x-position of left and right lanes
        left_pts = lane_points[0] if len(lane_points) > 0 else []
        right_pts = lane_points[1] if len(lane_points) > 1 else []

        if not left_pts or not right_pts:
            return 0.0

        left_x = np.mean([p[0] for p in left_pts])
        right_x = np.mean([p[0] for p in right_pts])
        lane_center = (left_x + right_x) / 2
        frame_center = frame_width / 2

        return frame_center - lane_center

    def _visualize(
        self, frame, detections, tracks, lane_result,
        anomalies, decision, owl_detections, traffic_signal, latency
    ) -> np.ndarray:
        """Render all pipeline outputs onto the frame."""
        viz = frame.copy()
        h, w = viz.shape[:2]

        # ─── Lane overlay ───
        lane_mask = lane_result.get("lane_mask", None)
        if lane_mask is not None and lane_mask.any():
            overlay = viz.copy()
            overlay[lane_mask > 0] = [0, 120, 0]  # Dark green for drivable area
            cv2.addWeighted(overlay, 0.3, viz, 0.7, 0, viz)

        # Draw lane polynomials
        for pts in lane_result.get("lane_points", []):
            if len(pts) > 1:
                pts_array = np.array(pts, dtype=np.int32)
                cv2.polylines(viz, [pts_array], False, (0, 255, 255), 2)

        # ─── Tracked objects ───
        for track in tracks:
            bbox = track.bbox
            if isinstance(bbox, np.ndarray):
                bbox = bbox.astype(int)
            else:
                bbox = [int(v) for v in bbox]

            # Color by category
            colors = {
                "vehicles": (255, 200, 0),
                "vulnerable_road_users": (0, 100, 255),
                "animals": (0, 0, 255),
                "traffic_infrastructure": (255, 255, 0),
            }
            color = colors.get(track.category, (200, 200, 200))

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)

            # Label with track ID
            label = f"#{track.track_id} {track.class_name} {track.confidence:.0%}"
            cv2.putText(viz, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Draw velocity vector
            vel = track.velocity
            if isinstance(vel, np.ndarray) and np.linalg.norm(vel) > 1:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                vx, vy = int(vel[0] * 5), int(vel[1] * 5)
                cv2.arrowedLine(viz, (cx, cy), (cx + vx, cy + vy), (0, 255, 0), 2)

        # ─── OWLv2 detections (dashed boxes) ───
        for owl_det in owl_detections:
            bbox = [int(v) for v in owl_det.bbox]
            x1, y1, x2, y2 = bbox
            # Dashed rectangle
            for i in range(x1, x2, 10):
                cv2.line(viz, (i, y1), (min(i+5, x2), y1), (255, 0, 255), 2)
                cv2.line(viz, (i, y2), (min(i+5, x2), y2), (255, 0, 255), 2)
            label = f"[OWL] {owl_det.label} {owl_det.confidence:.0%}"
            cv2.putText(viz, label, (x1, y1 - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # ─── Anomaly overlays ───
        for event in anomalies:
            if event.bbox:
                bbox = [int(v) for v in event.bbox]
                severity_colors = {
                    "critical": (0, 0, 255),
                    "warning": (0, 165, 255),
                    "info": (255, 255, 0),
                }
                color = severity_colors.get(event.severity, (200, 200, 200))
                cv2.rectangle(viz, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
                cv2.putText(viz, f"⚠ {event.type}", (bbox[0], bbox[1] - 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ─── HUD Dashboard ───
        self._draw_hud(viz, decision, traffic_signal, latency, len(tracks), len(anomalies))

        return viz

    def _draw_hud(self, frame, decision, traffic_signal, latency, num_tracks, num_anomalies):
        """Draw heads-up display with key info."""
        h, w = frame.shape[:2]

        # Semi-transparent panel at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Row 1: Action + FPS
        action_colors = {
            "cruise": (0, 255, 0),
            "slow_down": (0, 200, 255),
            "brake": (0, 100, 255),
            "hard_brake": (0, 0, 255),
            "emergency_stop": (0, 0, 255),
            "stop_at_signal": (0, 0, 200),
        }
        action_color = action_colors.get(decision.action.value, (200, 200, 200))
        cv2.putText(frame, f"ACTION: {decision.action.value.upper()}", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, action_color, 2)

        fps_text = f"FPS: {latency.fps:.0f}"
        cv2.putText(frame, fps_text, (w - 120, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Row 2: Decision reason
        cv2.putText(frame, decision.reason[:60], (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Row 3: Stats
        stats = f"Tracks: {num_tracks}  Signal: {traffic_signal}  Anomalies: {num_anomalies}"
        cv2.putText(frame, stats, (10, 72),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

        # Control bar at bottom
        bar_y = h - 30
        ctrl = decision.control
        # Steering indicator
        steer_x = int(w / 2 + ctrl.steering * w / 4)
        cv2.line(frame, (w // 2, bar_y), (steer_x, bar_y), (0, 255, 255), 3)
        cv2.circle(frame, (steer_x, bar_y), 6, (0, 255, 255), -1)

        # Brake/throttle bars
        brake_w = int(ctrl.brake * 100)
        throttle_w = int(ctrl.throttle * 100)
        cv2.rectangle(frame, (10, bar_y - 5), (10 + brake_w, bar_y + 5), (0, 0, 255), -1)
        cv2.putText(frame, "BRK", (10, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        cv2.rectangle(frame, (130, bar_y - 5), (130 + throttle_w, bar_y + 5), (0, 255, 0), -1)
        cv2.putText(frame, "THR", (130, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    def process_video(
        self,
        source,
        display: bool = True,
        save_output: bool = False,
        output_path: str = "output_l4.mp4",
    ):
        """
        Process a video file or camera stream.

        Args:
            source: Video path, camera index (0), or CARLA frame generator
            display: Show live window
            save_output: Save output video
            output_path: Output video path
        """
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            logger.error(f"Could not open video source: {source}")
            return

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

        logger.info(f"Processing video: {source} ({width}x{height} @ {fps_video}fps)")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.info("Video stream ended.")
                    break

                viz_frame, metrics = self.process_frame(frame, self.frame_count)

                if writer:
                    writer.write(viz_frame)

                if display:
                    cv2.imshow("ADAS L4 — India", viz_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27 or key == ord("q"):
                        break

                # Log periodic stats
                if self.frame_count % 100 == 0:
                    summary = self.profiler.get_summary()
                    logger.info(f"Frame {self.frame_count}: {summary.get('avg_fps', 0):.1f} FPS avg")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

            # Final stats
            summary = self.profiler.get_summary()
            logger.info(f"Pipeline complete. {self.frame_count} frames processed.")
            logger.info(f"Average: {summary}")

    def run_simulation(self):
        """
        Run the pipeline inside CARLA simulator.
        Requires CARLA server running on localhost:2000.
        """
        from src.simulation.carla_bridge import CARLABridge

        carla_cfg = self.config.get("carla", {})
        bridge = CARLABridge(
            host=carla_cfg.get("host", "localhost"),
            port=carla_cfg.get("port", 2000),
            town=carla_cfg.get("town", "Town03"),
            weather=carla_cfg.get("weather", "ClearNoon"),
        )

        if not bridge.connect():
            logger.error("Failed to connect to CARLA. Is the server running?")
            return

        traffic_cfg = carla_cfg.get("traffic", {})
        bridge.setup_scene(
            num_vehicles=traffic_cfg.get("num_vehicles", 30),
            num_pedestrians=traffic_cfg.get("num_pedestrians", 20),
        )

        logger.info("CARLA simulation running. Press Ctrl+C to stop.")

        try:
            while bridge.is_running:
                frame = bridge.get_frame(timeout=2.0)
                if frame is None:
                    continue

                viz_frame, metrics = self.process_frame(frame)

                # Apply decision to CARLA vehicle
                decision = DecisionOutput(**{
                    k: v for k, v in metrics["decision"].items()
                    if k in ("control",)
                })
                ctrl = metrics["decision"]["control"]
                bridge.apply_control(
                    steering=ctrl["steering"],
                    throttle=ctrl["throttle"],
                    brake=ctrl["brake"],
                )

                # Display
                cv2.imshow("ADAS L4 — CARLA", viz_frame)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break

        except KeyboardInterrupt:
            logger.info("Simulation stopped.")
        finally:
            bridge.cleanup()
            cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="ADAS Level 4 Pipeline — India-Focused")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or file path)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config YAML path")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--headless", action="store_true", help="Save output without display")
    parser.add_argument("--carla", action="store_true", help="Run with CARLA simulator")
    parser.add_argument("--output", type=str, default="output_l4.mp4", help="Output video path")

    args = parser.parse_args()

    pipeline = ADASPipelineL4(config_path=args.config, device=args.device)

    if args.carla:
        pipeline.run_simulation()
    else:
        pipeline.process_video(
            args.source,
            display=not args.headless,
            save_output=args.headless,
            output_path=args.output,
        )
