"""
Test Suite for ADAS Level 4 Pipeline (India-Focused)
Covers: detection, tracking, decision, anomaly, evaluation, and integration.
"""

import unittest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tracking.kalman_filter import KalmanBoxTracker
from src.tracking.deep_sort_tracker import DeepSORTTracker, Track, iou_batch
from src.decision.control_output import VehicleControl, DecisionOutput, ActionType, SeverityLevel
from src.decision.rule_engine import RuleBasedDecisionEngine, SceneContext
from src.anomaly.event_detector import AnomalyEventDetector, AnomalyEvent
from src.evaluation.metrics import (
    DetectionMetrics, SegmentationMetrics, TrackingMetrics,
    SystemProfiler, LatencyProfile, generate_evaluation_report
)


class TestKalmanFilter(unittest.TestCase):
    """Test Kalman filter for bounding box tracking."""

    def setUp(self):
        KalmanBoxTracker.reset_id_counter()

    def test_initialization(self):
        bbox = np.array([10.0, 20.0, 50.0, 80.0])
        tracker = KalmanBoxTracker(bbox)

        self.assertEqual(tracker.id, 0)
        self.assertEqual(tracker.hits, 1)
        state = tracker.get_state()
        np.testing.assert_array_almost_equal(state, bbox, decimal=0)

    def test_prediction_moves_state(self):
        bbox = np.array([100.0, 100.0, 200.0, 200.0])
        tracker = KalmanBoxTracker(bbox)

        pred = tracker.predict()
        self.assertEqual(tracker.age, 2)
        self.assertEqual(tracker.time_since_update, 1)
        # Without update, prediction should be near initial position
        self.assertAlmostEqual(pred[0], 100.0, places=-1)

    def test_update_reduces_uncertainty(self):
        bbox = np.array([10.0, 10.0, 50.0, 50.0])
        tracker = KalmanBoxTracker(bbox)

        tracker.predict()
        tracker.update(np.array([15.0, 15.0, 55.0, 55.0]))

        self.assertEqual(tracker.time_since_update, 0)
        self.assertEqual(tracker.hits, 2)

    def test_velocity_estimation(self):
        bbox = np.array([100.0, 100.0, 150.0, 150.0])
        tracker = KalmanBoxTracker(bbox)

        # Simulate moving object
        for i in range(5):
            tracker.predict()
            tracker.update(np.array([100.0 + i * 10, 100.0, 150.0 + i * 10, 150.0]))

        vel = tracker.get_velocity()
        # Should have positive x-velocity
        self.assertGreater(vel[0], 0)

    def test_unique_ids(self):
        KalmanBoxTracker.reset_id_counter()
        t1 = KalmanBoxTracker(np.array([0, 0, 10, 10]))
        t2 = KalmanBoxTracker(np.array([20, 20, 30, 30]))
        self.assertEqual(t1.id, 0)
        self.assertEqual(t2.id, 1)


class TestDeepSORTTracker(unittest.TestCase):
    """Test DeepSORT multi-object tracking."""

    def setUp(self):
        KalmanBoxTracker.reset_id_counter()
        self.tracker = DeepSORTTracker(max_age=5, min_hits=2, iou_threshold=0.3)

    def test_iou_batch(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[5, 5, 15, 15]])
        iou = iou_batch(a, b)
        self.assertTrue(0 < iou[0, 0] < 1)

    def test_iou_no_overlap(self):
        a = np.array([[0, 0, 10, 10]])
        b = np.array([[20, 20, 30, 30]])
        iou = iou_batch(a, b)
        self.assertAlmostEqual(iou[0, 0], 0.0)

    def test_track_creation(self):
        dets = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
        scores = np.array([0.9, 0.85])

        # First frame — tracks created but not yet confirmed
        tracks = self.tracker.update(dets, scores)
        # min_hits=2 so tracks are tentative on first frame
        # But frame_count <= min_hits, so they appear
        self.assertEqual(len(tracks), 2)

    def test_track_persistence(self):
        """Same detection across frames should maintain track ID."""
        det = np.array([[100, 100, 200, 200]])

        tracks1 = self.tracker.update(det)
        tid1 = tracks1[0].track_id if tracks1 else None

        tracks2 = self.tracker.update(det)
        tid2 = tracks2[0].track_id if tracks2 else None

        # Same detection → same track ID
        if tid1 is not None and tid2 is not None:
            self.assertEqual(tid1, tid2)

    def test_track_removal(self):
        """Track should be removed after max_age frames without update."""
        det = np.array([[100, 100, 200, 200]])
        self.tracker.update(det)

        # Feed empty detections for max_age+1 frames
        for _ in range(6):
            self.tracker.update(np.array([]).reshape(0, 4))

        all_tracks = self.tracker.get_all_tracks()
        self.assertEqual(len(all_tracks), 0)

    def test_reset(self):
        det = np.array([[100, 100, 200, 200]])
        self.tracker.update(det)
        self.tracker.reset()
        self.assertEqual(len(self.tracker.trackers), 0)


class TestDecisionEngine(unittest.TestCase):
    """Test rule-based decision making."""

    def setUp(self):
        self.engine = RuleBasedDecisionEngine(
            emergency_brake_distance=80,
            hard_brake_distance=150,
            crowd_density_threshold=3,
        )

    def test_cruise_on_clear_road(self):
        scene = SceneContext(frame_height=720, frame_width=1280)
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.CRUISE)
        self.assertGreater(decision.control.throttle, 0)

    def test_emergency_brake_close_obstacle(self):
        scene = SceneContext(
            tracked_objects=[{
                "bbox": [600, 680, 700, 710],  # Very close (bottom of frame)
                "class_name": "car",
                "category": "vehicles",
            }],
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertIn(decision.action, (ActionType.EMERGENCY_STOP, ActionType.HARD_BRAKE))
        self.assertGreater(decision.control.brake, 0.5)

    def test_stop_at_red_light(self):
        scene = SceneContext(
            traffic_signal="Red",
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.STOP_AT_SIGNAL)

    def test_slow_in_crowd(self):
        # Position objects in ego zone but far enough to not trigger hard_brake
        objects = [
            {"bbox": [i * 200 + 50, 440, i * 200 + 130, 500], "class_name": "person", "category": "vulnerable_road_users"}
            for i in range(5)
        ]
        scene = SceneContext(
            tracked_objects=objects,
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.SLOW_DOWN)

    def test_anomaly_wrong_side(self):
        scene = SceneContext(
            active_anomalies=[{"type": "wrong_side_vehicle", "severity": "critical"}],
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.HARD_BRAKE)

    def test_anomaly_sudden_crossing(self):
        scene = SceneContext(
            active_anomalies=[{"type": "sudden_crossing", "severity": "critical"}],
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.EMERGENCY_STOP)

    def test_animal_on_road_slows(self):
        scene = SceneContext(
            active_anomalies=[{"type": "animal_on_road", "severity": "warning", "details": "cow"}],
            frame_height=720,
            frame_width=1280,
        )
        decision = self.engine.decide(scene)
        self.assertEqual(decision.action, ActionType.SLOW_DOWN)


class TestAnomalyDetector(unittest.TestCase):
    """Test anomaly event detection."""

    def setUp(self):
        self.detector = AnomalyEventDetector(
            wrong_side_velocity_threshold=-5.0,
            wrong_side_min_frames=2,
            crossing_velocity_threshold=10.0,
            crossing_proximity_threshold=300,
        )

    def test_no_anomaly_normal_traffic(self):
        tracks = [{
            "track_id": 0,
            "bbox": [100, 200, 200, 300],
            "category": "vehicles",
            "class_name": "car",
            "velocity": [0, -2],  # Moving away normally
        }]
        events = self.detector.detect(tracks, frame_width=1280, frame_height=720)
        wrong_side = [e for e in events if e.type == "wrong_side_vehicle"]
        self.assertEqual(len(wrong_side), 0)

    def test_wrong_side_detection(self):
        """Vehicle on left side moving strongly downward → wrong side."""
        tracks = [{
            "track_id": 1,
            "bbox": [100, 200, 200, 300],
            "category": "vehicles",
            "class_name": "car",
            "velocity": [0, 10],  # Moving toward camera on left side
        }]

        # Need multiple frames to confirm
        for _ in range(3):
            events = self.detector.detect(tracks, frame_width=1280, frame_height=720)

        wrong_side = [e for e in events if e.type == "wrong_side_vehicle"]
        self.assertGreater(len(wrong_side), 0)

    def test_sudden_crossing_detection(self):
        tracks = [{
            "track_id": 2,
            "bbox": [600, 600, 650, 700],
            "category": "vulnerable_road_users",
            "class_name": "person",
            "velocity": [20, 0],  # High lateral velocity
        }]
        events = self.detector.detect(tracks, frame_width=1280, frame_height=720)
        crossings = [e for e in events if e.type == "sudden_crossing"]
        self.assertGreater(len(crossings), 0)

    def test_animal_on_drivable_area(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[300:600, 200:1000] = 255  # Road area

        tracks = [{
            "track_id": 3,
            "bbox": [400, 400, 500, 500],
            "category": "animals",
            "class_name": "cow",
            "velocity": [0, 0],
        }]
        events = self.detector.detect(tracks, drivable_mask=mask, frame_width=1280, frame_height=720)
        animal_events = [e for e in events if e.type == "animal_on_road"]
        self.assertGreater(len(animal_events), 0)

    def test_reset(self):
        self.detector.reset()
        self.assertEqual(len(self.detector._wrong_side_counters), 0)


class TestControlOutput(unittest.TestCase):
    """Test control output data classes."""

    def test_vehicle_control_defaults(self):
        ctrl = VehicleControl()
        self.assertEqual(ctrl.steering, 0.0)
        self.assertEqual(ctrl.throttle, 0.0)
        self.assertFalse(ctrl.emergency_stop)

    def test_vehicle_control_to_carla(self):
        ctrl = VehicleControl(steering=0.5, throttle=0.8, brake=0.1)
        carla_dict = ctrl.to_carla()
        self.assertEqual(carla_dict["steer"], 0.5)
        self.assertFalse(carla_dict["reverse"])

    def test_decision_output_emergency(self):
        decision = DecisionOutput(
            action=ActionType.EMERGENCY_STOP,
            control=VehicleControl(brake=1.0, emergency_stop=True),
            severity=SeverityLevel.EMERGENCY,
        )
        self.assertTrue(decision.is_emergency())

    def test_decision_to_dict(self):
        decision = DecisionOutput(
            action=ActionType.CRUISE,
            reason="All clear",
        )
        d = decision.to_dict()
        self.assertEqual(d["action"], "cruise")
        self.assertEqual(d["reason"], "All clear")


class TestEvaluationMetrics(unittest.TestCase):
    """Test evaluation metrics computation."""

    def test_detection_perfect_score(self):
        metrics = DetectionMetrics(iou_threshold=0.5)
        pred_boxes = np.array([[10, 10, 50, 50]])
        pred_scores = np.array([0.9])
        pred_classes = np.array([0])
        gt_boxes = np.array([[10, 10, 50, 50]])
        gt_classes = np.array([0])

        metrics.add_frame(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes)
        result = metrics.compute_map()
        self.assertGreater(result["mAP"], 0.9)

    def test_detection_no_detections(self):
        metrics = DetectionMetrics()
        metrics.add_frame(
            np.array([]).reshape(0, 4), np.array([]), np.array([]),
            np.array([[10, 10, 50, 50]]), np.array([0])
        )
        result = metrics.compute_map()
        self.assertEqual(result["mAP"], 0.0)

    def test_segmentation_iou(self):
        metrics = SegmentationMetrics(num_classes=2)
        pred = np.ones((100, 100), dtype=np.uint8)
        gt = np.ones((100, 100), dtype=np.uint8)
        metrics.add_frame(pred, gt)
        result = metrics.compute_iou()
        self.assertGreater(result["mIoU"], 0.4)

    def test_latency_profile(self):
        profile = LatencyProfile(
            detection_ms=10, segmentation_ms=15, tracking_ms=5,
            total_ms=35
        )
        self.assertAlmostEqual(profile.fps, 1000.0 / 35, places=1)

    def test_system_profiler(self):
        profiler = SystemProfiler(window_size=10)
        for i in range(5):
            profiler.add_frame(LatencyProfile(total_ms=20 + i))
        summary = profiler.get_summary()
        self.assertEqual(summary["frames_profiled"], 5)
        self.assertGreater(summary["avg_fps"], 0)

    def test_report_generation(self):
        det = DetectionMetrics()
        report = generate_evaluation_report(detection_metrics=det)
        self.assertIn("detection", report)
        self.assertIn("timestamp", report)


class TestConfigLoading(unittest.TestCase):
    """Test YAML config loading."""

    def test_config_file_exists(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )
        self.assertTrue(os.path.exists(config_path))

    def test_config_loads(self):
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.assertIn("detection", cfg)
        self.assertIn("tracking", cfg)
        self.assertIn("decision", cfg)
        self.assertIn("anomaly", cfg)
        self.assertIn("datasets", cfg)

    def test_indian_datasets_present(self):
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        datasets = cfg["datasets"]
        self.assertIn("idd", datasets)
        self.assertIn("bdd100k", datasets)
        self.assertIn("mapillary", datasets)
        self.assertIn("custom", datasets)

        # Verify IDD URL
        self.assertIn("idd.insaan.iiit.ac.in", datasets["idd"]["url"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
