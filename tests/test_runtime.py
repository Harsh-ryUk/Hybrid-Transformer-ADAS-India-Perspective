"""Contract checks for configs, preprocessing and the complete orchestration."""
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from src.utils.runtime import resolve_device, validate_config


@pytest.mark.parametrize('config', [[], {'system': 'cpu'}, {'zero_shot': {'run_every_n_frames': 0}},
                                    {'segmentation': {'input_size': [0, 144]}},
                                    {'detection': {'confidence_threshold': 1.1}}])
def test_bad_config_fails_before_loading_models(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_device_fallback_is_shared():
    with patch('torch.cuda.is_available', return_value=False):
        assert resolve_device('cuda') == 'cpu'
        assert resolve_device('auto') == 'cpu'
    with pytest.raises(ValueError):
        resolve_device('mps')


def test_segformer_preprocessing_matches_rgb_reference():
    from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
    from transformers import SegformerImageProcessor
    processor = SegformerImageProcessor(size={'height': 32, 'width': 32})
    detector = SegFormerLaneDetector.__new__(SegFormerLaneDetector)
    detector.device = torch.device('cpu')
    detector.input_size = (32, 32)
    detector._mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1) * 255
    detector._std = torch.tensor(processor.image_std).view(1, 3, 1, 1) * 255
    frame = np.random.default_rng(8).integers(0, 256, (16, 24, 3), dtype=np.uint8)
    expected = processor(images=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), return_tensors='pt').pixel_values
    torch.testing.assert_close(detector.preprocess(frame), expected, rtol=1e-5, atol=1e-6)


def make_pipeline():
    from src.adas_pipeline_l4 import ADASPipelineL4
    from src.perception.india_detector import DetectionResult
    with patch('src.adas_pipeline_l4.IndiaObjectDetector') as detection, patch('src.adas_pipeline_l4.SegFormerLaneDetector') as segmentation:
        detection.return_value.detect.return_value = DetectionResult()
        segmentation.return_value.detect.side_effect = lambda frame, **kwargs: {
            'lane_mask': np.zeros(frame.shape[:2], np.uint8), 'lane_points': []}
        return ADASPipelineL4(device='cpu')


def test_headless_processing_skips_render_and_accounts_for_stages():
    pipeline = make_pipeline()
    frame = np.full((120, 160, 3), 100, np.uint8)
    with patch.object(pipeline, '_visualize', side_effect=AssertionError('render called')):
        output, metrics = pipeline.process_frame(frame, render=False)
    assert output is frame
    assert not metrics['rendered']
    assert not metrics['zero_shot']['enabled']
    assert 'active_anomalies' in metrics
    latency = metrics['latency']
    assert sum(value for key, value in latency.items() if key.endswith('_ms') and key != 'total_ms') <= latency['total_ms'] + 0.1
    pipeline.close()


@pytest.mark.parametrize('frame', [None, np.zeros((10, 10)), np.zeros((0, 10, 3), np.uint8),
                                   np.zeros((10, 10, 4), np.uint8)])
def test_invalid_frames_are_rejected(frame):
    with pytest.raises(ValueError):
        make_pipeline().process_frame(frame)


def test_missing_config_does_not_silently_use_defaults(tmp_path):
    from src.adas_pipeline_l4 import ADASPipelineL4
    with pytest.raises(FileNotFoundError):
        ADASPipelineL4(str(tmp_path / 'missing.yaml'), device='cpu')


def test_custom_detection_names_are_preserved():
    from src.perception.india_detector import IndiaObjectDetector
    with patch('src.perception.india_detector.YOLO') as yolo:
        yolo.return_value.names = {0: 'autorickshaw', 1: 'person'}
        boxes = type('Boxes', (), {'xyxy': torch.tensor([[0, 0, 10, 10]]),
                                   'conf': torch.tensor([0.9]), 'cls': torch.tensor([1])})()
        yolo.return_value.return_value = [type('Result', (), {'boxes': boxes})()]
        result = IndiaObjectDetector(device='cpu').detect(np.zeros((20, 20, 3), np.uint8))
    assert result.detections[0].class_name == 'person'
    assert result.detections[0].category == 'vulnerable_road_users'
