"""Road-mask contracts and conservative surface-candidate behaviour."""
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
import torch
from src.anomaly.event_detector import AnomalyEventDetector
from src.decision.rule_engine import SceneContext, RuleBasedDecisionEngine
from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
from src.utils.runtime import validate_config


def stub_segmentation(logits, roi=(0, 0, 1, 1)):
    detector = SegFormerLaneDetector.__new__(SegFormerLaneDetector)
    detector.road_class_ids = (6,)
    detector.frame_roi = roi
    detector.roi_top_fraction = 0.45
    detector.min_road_coverage = 0.02
    detector.preprocess = lambda frame, condition: frame
    detector.model = lambda **kwargs: SimpleNamespace(logits=logits)
    return detector


def test_source_detail_reaches_preprocessing_without_fixed_16_9_squeeze():
    detector = stub_segmentation(torch.zeros(1, 7, 8, 8))
    shapes = []
    detector.preprocess = lambda frame, condition: shapes.append(frame.shape)
    result = detector.detect(np.zeros((1280, 720, 3), np.uint8))
    assert shapes == [(1280, 720, 3)]
    assert result['road_status'] == 'unknown'
    assert not result['road_observed']
    assert not result['lane_markings_detected']
    assert result['lane_points'] == []


def test_crop_mask_maps_back_without_filling_borders_or_upper_scene():
    logits = torch.zeros(1, 7, 16, 16)
    logits[:, 6] = 10
    detector = stub_segmentation(logits, (0.1, 0.2, 0.9, 0.8))
    result = detector.detect(np.zeros((100, 100, 3), np.uint8))
    mask = result['lane_mask']
    assert result['road_observed']
    assert not np.any(mask[:47])
    assert not np.any(mask[80:])
    assert not np.any(mask[:, :10])
    assert not np.any(mask[:, 90:])
    assert np.any(mask[60:75, 20:80])


def test_missing_road_never_yields_clear_road_or_acceleration():
    decision = RuleBasedDecisionEngine().decide(SceneContext(road_observed=False))
    assert decision.action.value != 'cruise'
    assert decision.control.throttle == 0
    assert 'unknown' in decision.reason.lower()


def surface_scene():
    frame = np.full((240, 320, 3), 150, np.uint8)
    cv2.circle(frame, (160, 160), 15, (20, 20, 20), -1)
    mask = np.zeros((240, 320), np.uint8)
    mask[80:230, 20:300] = 255
    return frame, mask


@pytest.mark.parametrize('box', [(80, 90, 130, 145), (0, 0, 35, 50), (165, 130, 200, 180), (0, 0, 200, 180)])
def test_bounded_surface_filter_is_pixel_identical_to_full_frame(box):
    rng = np.random.default_rng(42)
    gray = rng.integers(0, 256, (180, 200), dtype=np.uint8)
    support = np.zeros_like(gray)
    x0, y0, x1, y1 = box
    support[y0:y1, x0:x1] = 255
    support[100:105, 110:120] = 0
    contrast = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    expected = ((contrast >= 40) & (support > 0)).astype(np.uint8) * 255
    expected = cv2.morphologyEx(expected, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    np.testing.assert_array_equal(AnomalyEventDetector._surface_dark_mask(gray, support, 40), expected)


def test_bounded_surface_filter_handles_empty_support():
    gray = np.full((100, 100), 150, np.uint8)
    assert not np.any(AnomalyEventDetector._surface_dark_mask(gray, np.zeros_like(gray), 40))


def test_dark_patch_requires_persistence_and_is_not_confirmed_pothole():
    detector = AnomalyEventDetector(pothole_min_area=100, surface_min_frames=3)
    frame, mask = surface_scene()
    assert not detector._detect_road_anomalies(frame, mask)
    assert not detector._detect_road_anomalies(frame, mask)
    events = detector._detect_road_anomalies(frame, mask)
    assert len(events) == 1
    assert 'not a confirmed pothole' in events[0].details
    assert events[0].confidence == 0.35
    detector.reset()
    assert not detector._surface_candidates


def test_objects_mask_edges_and_missing_road_are_excluded():
    detector = AnomalyEventDetector(pothole_min_area=100, surface_min_frames=1)
    frame, mask = surface_scene()
    assert not detector._detect_road_anomalies(frame, mask, [{'bbox': [140, 140, 180, 180]}])
    assert not detector._detect_road_anomalies(frame, np.zeros_like(mask))
    mask[150:] = 0
    assert not detector._detect_road_anomalies(frame, mask)


def test_one_frame_flash_does_not_carry_confirmation_across_blank_frames():
    detector = AnomalyEventDetector(pothole_min_area=100, surface_min_frames=2)
    frame, mask = surface_scene()
    assert not detector._detect_road_anomalies(frame, mask)
    assert not detector._detect_road_anomalies(np.full_like(frame, 150), mask)
    assert not detector._detect_road_anomalies(frame, mask)


def test_wrong_side_heuristic_can_be_disabled_for_moving_dashcam():
    detector = AnomalyEventDetector(wrong_side_enabled=False)
    tracks = [{'track_id': 1, 'bbox': [10, 100, 40, 160], 'category': 'vehicles', 'velocity': [0, 50]}]
    for _ in range(5):
        assert not detector.detect(tracks)


def test_surface_persistence_requires_positive_integer():
    with pytest.raises(ValueError):
        AnomalyEventDetector(surface_min_frames=0)


@pytest.mark.parametrize('config', [
    {'segmentation': {'frame_roi': [0, 1, 1, 0]}},
    {'segmentation': {'frame_roi': [0, 0, 1.2, 1]}},
    {'segmentation': {'roi_top_fraction': 1}},
    {'segmentation': {'min_road_coverage': -1}},
    {'anomaly': {'surface_min_frames': 0}},
    {'anomaly': {'wrong_side_enabled': 'false'}},
])
def test_quality_config_invalid_values_fail_early(config):
    with pytest.raises(ValueError):
        validate_config(config)
