"""Runtime validation shared by model wrappers and the pipeline."""
import logging
import torch


def resolve_device(requested):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.getLogger(__name__).warning('CUDA unavailable; using CPU for all modules')
        return 'cpu'
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda (CUDA index allowed)')
    return str(device)


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping')
    for section in ('system', 'detection', 'zero_shot', 'segmentation', 'tracking', 'anomaly', 'decision'):
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f'{section} must be a mapping')
    for section in ('detection', 'segmentation'):
        size = config.get(section, {}).get('input_size')
        if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2
                                 or any(type(v) is not int or v <= 0 for v in size)):
            raise ValueError(f'{section}.input_size must be [positive width, positive height]')
    interval = config.get('zero_shot', {}).get('run_every_n_frames', 10)
    roi = config.get('segmentation', {}).get('frame_roi', [0, 0, 1, 1])
    if not isinstance(roi, (list, tuple)) or len(roi) != 4 or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in roi) or not (roi[0] < roi[2] and roi[1] < roi[3]):
        raise ValueError('segmentation.frame_roi must be normalized [left, top, right, bottom]')
    for key, default in [('roi_top_fraction', 0.45), ('min_road_coverage', 0.02)]:
        value = config.get('segmentation', {}).get(key, default)
        if type(value) not in (int, float) or not 0 <= value <= 1 or (key == 'roi_top_fraction' and value == 1):
            raise ValueError(f'Invalid segmentation.{key}')
    persistence = config.get('anomaly', {}).get('surface_min_frames', 3)
    if type(persistence) is not int or persistence < 1:
        raise ValueError('anomaly.surface_min_frames must be a positive integer')
    if type(config.get('anomaly', {}).get('wrong_side_enabled', False)) is not bool:
        raise ValueError('anomaly.wrong_side_enabled must be a bool')
    if type(interval) is not int or interval < 1:
        raise ValueError('zero_shot.run_every_n_frames must be a positive integer')
    for key in ('confidence_threshold', 'iou_threshold'):
        value = config.get('detection', {}).get(key, 0.35)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'detection.{key} must be in [0, 1]')
    return config
