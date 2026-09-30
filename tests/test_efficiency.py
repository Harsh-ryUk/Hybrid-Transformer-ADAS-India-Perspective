"""Offline regression checks for bounded state and inference scheduling."""
import threading
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from src.evaluation.metrics import SegmentationMetrics
from src.perception.owl_detector import OWLv2Detector
from src.tracking.deep_sort_tracker import DeepSORTTracker


def test_confusion_histogram_matches_label_pair_counts():
    rng = np.random.default_rng(4)
    gt = rng.integers(-1, 32, (30, 40))
    pred = rng.integers(-1, 32, (30, 40))
    metrics = SegmentationMetrics(num_classes=30)
    metrics.add_frame(pred[:, ::2], gt[:, ::2])
    expected = np.array([[np.count_nonzero((gt[:, ::2] == i) & (pred[:, ::2] == j))
                          for j in range(30)] for i in range(30)])
    np.testing.assert_array_equal(metrics._confusion, expected)


def test_expired_tracks_release_metadata_and_outputs_are_isolated():
    tracker = DeepSORTTracker(max_age=1, min_hits=1)
    box = np.array([[10., 10., 30., 30.]])
    tracks = tracker.update(box)
    tracks[0].trajectory[0][:] = -100
    np.testing.assert_array_equal(tracker.trackers[0].history[0], box[0])
    tracker.update(np.empty((0, 4)))
    tracker.update(np.empty((0, 4)))
    assert not tracker.trackers
    assert not tracker.tracker_metadata


def test_background_worker_drops_requests_and_copies_input():
    detector = OWLv2Detector(background=True, run_every_n_frames=1)
    started, release = threading.Event(), threading.Event()
    observed = []

    def run(frame):
        started.set()
        assert release.wait(5)
        observed.append(frame.copy())
        return ["completed"]

    frame = np.zeros((10, 10, 3), np.uint8)
    with patch.object(detector, "_run", side_effect=run) as inference:
        try:
            assert detector.detect(frame) == []
            assert started.wait(5)
            worker = detector._worker
            frame[:] = 255
            assert detector.detect(frame) == []
            assert detector._worker is worker
            assert inference.call_count == 1
        finally:
            release.set()
            detector._worker.join(5)
    assert not observed[0].any()
    assert detector._last_result == ["completed"]


def test_invalid_detection_interval_is_rejected():
    with pytest.raises(ValueError):
        OWLv2Detector(run_every_n_frames=0)


def test_detectors_keep_their_own_model_when_loader_key_changes():
    import src.perception.owl_detector as module
    models = [MagicMock(), MagicMock()]
    for model in models:
        model.to.return_value = model
        model.eval.return_value = model
    with patch.object(module, '_owlv2_loaded', False), patch.object(module, '_model_key', None), \
            patch.object(module, '_processor', None), patch.object(module, '_model', None), \
            patch('transformers.Owlv2Processor.from_pretrained'), \
            patch('transformers.Owlv2ForObjectDetection.from_pretrained', side_effect=models):
        first = OWLv2Detector(model_name='first-model', device='cpu')
        second = OWLv2Detector(model_name='second-model', device='cpu')
        first._ensure_loaded()
        second._ensure_loaded()
        assert first._model is models[0]
        assert second._model is models[1]


def test_road_mask_excludes_people_and_respects_custom_labels():
    from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
    detector = SegFormerLaneDetector.__new__(SegFormerLaneDetector)
    detector._kernel = np.ones((5, 5), np.uint8)
    detector.preprocess = lambda frame, condition: None
    logits = torch.zeros((1, 13, 20, 20))
    logits[:, 12] = 10  # person, not road
    detector.model = lambda **kwargs: type("Output", (), {"logits": logits})()
    frame = np.zeros((80, 80, 3), np.uint8)
    detector.road_class_ids = (6,)
    assert cv2.countNonZero(detector.detect(frame)["lane_mask"]) == 0
    detector.road_class_ids = (12,)
    assert cv2.countNonZero(detector.detect(frame)["lane_mask"]) > 0
