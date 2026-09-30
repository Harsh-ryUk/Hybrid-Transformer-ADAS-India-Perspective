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
    if type(interval) is not int or interval < 1:
        raise ValueError('zero_shot.run_every_n_frames must be a positive integer')
    for key in ('confidence_threshold', 'iou_threshold'):
        value = config.get('detection', {}).get(key, 0.35)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'detection.{key} must be in [0, 1]')
    return config
