"""Measure real models, complete frame latency, tail latency and process memory.

python scripts/benchmark_pipeline.py --profile all --source sample --frames 30 --repeats 2
The sample is a repeated letterboxed image, never a road-video accuracy benchmark.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILES = ('full512', 'full256', 'core256')


def benchmark_device(requested):
    """Never quietly publish CPU execution under a requested CUDA label."""
    import torch
    if requested == 'auto':
        requested = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Benchmark supports CPU or CUDA only')
    if device.type == 'cuda':
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError(f'Requested CUDA device is unavailable: {requested}')
        return f'cuda:{index}'
    return 'cpu'


def synchronize(device):
    import torch
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(values):
    import numpy as np
    return {key: round(float(value), 3) for key, value in {
        'mean': np.mean(values), 'p50': np.percentile(values, 50),
        'p95': np.percentile(values, 95), 'p99': np.percentile(values, 99),
        'max': max(values),
    }.items()}


def configure_profile(config, profile):
    """The configured profile preserves model choices and image preprocessing."""
    if profile != 'configured':
        config['segmentation']['input_size'] = [512, 512] if profile == 'full512' else [256, 144]
        config['zero_shot']['enabled'] = profile != 'core256'
        config['zero_shot']['background'] = False
    elif config.get('zero_shot', {}).get('enabled', True) and config['zero_shot'].get('background', False):
        raise ValueError('Configured benchmark requires foreground zero-shot inference or zero-shot disabled')
    return config


def source_frames(source, width, height, native_resolution=False):
    import cv2
    import numpy as np
    if source == 'sample':
        import ultralytics
        path = Path(ultralytics.__file__).parent / 'assets' / 'bus.jpg'
    else:
        path = Path(source).resolve()
    if not path.is_file():
        raise ValueError(f'Input does not exist: {path}')
    metadata = {'path': str(path), 'sha256': digest(path),
                'requested_resolution': None if native_resolution else [width, height],
                'native_resolution': native_resolution}
    image = cv2.imread(str(path)) if path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp') else None
    if image is not None:
        metadata['kind'] = 'repeated_sample_image' if source == 'sample' else 'repeated_user_image'
        ih, iw = image.shape[:2]
        metadata['original_resolution'] = [iw, ih]
        if native_resolution:
            width, height = iw, ih
        scale = min(width / iw, height / ih)
        resized = cv2.resize(image, (max(1, round(iw * scale)), max(1, round(ih * scale))))
        canvas = np.full((height, width, 3), 114, np.uint8)
        y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        metadata['processed_resolution'] = [width, height]

        def images():
            while True:
                yield canvas.copy()
        return images(), metadata
    metadata['kind'] = 'video'
    probe = cv2.VideoCapture(str(path))
    try:
        if not probe.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        metadata['original_resolution'] = [int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                           int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        metadata['source_fps'] = float(probe.get(cv2.CAP_PROP_FPS))
        metadata['reported_frame_count'] = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        metadata['processed_resolution'] = metadata['original_resolution'] if native_resolution else [width, height]
    finally:
        probe.release()

    def video():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f'Cannot decode video: {path}')
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    return
                yield frame if native_resolution else cv2.resize(frame, (width, height))
        finally:
            cap.release()
    return video(), metadata


def run(args):
    import cv2
    import torch
    import yaml
    from src.adas_pipeline_l4 import ADASPipelineL4

    device = benchmark_device(args.device)
    cuda = device.startswith('cuda')
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    config = yaml.safe_load(Path(args.config).read_text())
    config['system']['device'] = device
    config['system']['profile_synchronize'] = cuda
    configure_profile(config, args.profile)
    # Resolve local weights relative to the repository, not the caller's CWD.
    weights = Path(config['detection']['model_path'])
    if not weights.is_absolute() and (ROOT / weights).is_file():
        config['detection']['model_path'] = str(ROOT / weights)

    rows, repeats, actions = [], [], Counter()
    startup = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='adas-benchmark-') as temp:
        config_path = Path(temp) / 'config.yaml'
        config_path.write_text(yaml.safe_dump(config))
        with ADASPipelineL4(str(config_path), device=device) as pipeline:
            synchronize(device)
            initialization_ms = (time.perf_counter() - startup) * 1000
            source_metadata = None
            for repeat in range(args.repeats):
                frames, source_metadata = source_frames(args.source, args.width, args.height, args.native_resolution)
                try:
                    warmup = time.perf_counter()
                    for _ in range(args.warmup):
                        pipeline.process_frame(next(frames), render=not args.no_render)
                    synchronize(device)
                    warmup_ms = (time.perf_counter() - warmup) * 1000
                    # Ultralytics' first CPU predictor setup changes PyTorch's global
                    # thread count. Enforce the requested budget after lazy setup.
                    torch.set_num_threads(args.threads)
                    if cuda:
                        torch.cuda.reset_peak_memory_stats(device)
                    owl_before = pipeline.owl_detector.status()['inference_count'] if pipeline.owl_detector else 0
                    wall_start, cpu_start = time.perf_counter(), time.process_time()
                    for index in range(args.frames):
                        frame_start = time.perf_counter()
                        frame = next(frames)
                        read_ms = (time.perf_counter() - frame_start) * 1000
                        viz, metrics = pipeline.process_frame(frame, index, render=not args.no_render)
                        synchronize(device)
                        row = {'repeat': repeat, 'frame': index, 'read_ms': read_ms,
                               'wall_ms': (time.perf_counter() - frame_start) * 1000,
                               **{key: value for key, value in metrics['latency'].items() if key.endswith('_ms')},
                               'detections': metrics['detections'], 'tracks': metrics['tracks'],
                               'anomalies': metrics['anomalies'], 'action': metrics['decision']['action'],
                               'traffic_signal': metrics['traffic_signal']}
                        row['road_mask_coverage'] = metrics['road_mask_coverage']
                        row['road_status'] = metrics.get('road_status', 'unknown')
                        row['source_frame'] = args.warmup + index
                        row['torch_threads'] = torch.get_num_threads()
                        if row['torch_threads'] != args.threads:
                            raise RuntimeError('A model changed the measured CPU thread budget')
                        rows.append(row)
                        actions[row['action']] += 1
                        if (index + 1) % 10 == 0:
                            print(f'{args.profile}: repeat {repeat + 1}, frame {index + 1}/{args.frames}', flush=True)
                    synchronize(device)
                    wall_seconds = time.perf_counter() - wall_start
                    cpu_seconds = time.process_time() - cpu_start
                    status = pipeline.owl_detector.status() if pipeline.owl_detector else {'enabled': False}
                    if pipeline.owl_detector and (status['last_error'] or status['inference_count'] == owl_before):
                        raise RuntimeError(f'Full benchmark did not successfully run OWLv2: {status}')
                    repeats.append({'repeat': repeat, 'warmup_ms': round(warmup_ms, 2),
                                    'wall_seconds': round(wall_seconds, 4),
                                    'throughput_fps': round(args.frames / wall_seconds, 3),
                                    'cpu_time_seconds': round(cpu_seconds, 4), 'zero_shot': status,
                                    'gpu_peak_allocated_mib': round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 2) if cuda else None,
                                    'gpu_peak_reserved_mib': round(torch.cuda.max_memory_reserved(device) / 1024 ** 2, 2) if cuda else None})
                    print(f'{args.profile} repeat {repeat + 1}: {args.frames / wall_seconds:.2f} FPS', flush=True)
                    if not args.no_render:
                        preview = Path(args.output).with_suffix('.png')
                        preview.parent.mkdir(parents=True, exist_ok=True)
                        if not cv2.imwrite(str(preview), viz):
                            raise RuntimeError(f'Could not save preview: {preview}')
                except StopIteration as error:
                    raise ValueError('Source ended before warmup + measurement; provide a longer video or fewer frames') from error
                finally:
                    frames.close()

    def command(*parts):
        try:
            return subprocess.check_output(parts, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    elapsed = sum(repeat['wall_seconds'] for repeat in repeats)
    stages = {key: distribution([row[key] for row in rows]) for key in rows[0] if key.endswith('_ms')}
    report = {
        'schema_version': 3, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'profile': args.profile, 'source': source_metadata, 'configuration': config,
        'configuration_sha256': hashlib.sha256(yaml.safe_dump(config).encode()).hexdigest(),
        'source_files_sha256': {str(path.relative_to(ROOT)): digest(path) for path in
                               [ROOT / 'scripts/benchmark_pipeline.py', *sorted((ROOT / 'src').rglob('*.py'))]},
        'models': {
            'yolo_weights_sha256': digest(config['detection']['model_path']) if Path(config['detection']['model_path']).is_file() else None,
            'segformer_revision': getattr(pipeline.lane_detector.model.config, '_commit_hash', None),
            'owlv2_revision': getattr(pipeline.owl_detector._model.config, '_commit_hash', None) if pipeline.owl_detector else None,
        },
        'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                        'cuda_runtime': torch.version.cuda,
                        'gpu': {'name': torch.cuda.get_device_name(device),
                                'capability': list(torch.cuda.get_device_capability(device)),
                                'total_memory_mib': round(torch.cuda.get_device_properties(device).total_memory / 1024 ** 2, 2)} if cuda else None,
                        'cpu': command('sysctl', '-n', 'machdep.cpu.brand_string') or platform.processor(),
                        'memory_bytes': command('sysctl', '-n', 'hw.memsize'), 'logical_cpus': os.cpu_count(),
                        'load_average': list(os.getloadavg()) if hasattr(os, 'getloadavg') else None,
                        'torch_threads': torch.get_num_threads(), 'opencv_threads': cv2.getNumThreads(),
                        'requested_torch_threads': args.threads,
                        'packages': {name: importlib.metadata.version(name) for name in
                                     ('torch', 'transformers', 'ultralytics', 'numpy', 'scipy', 'PyYAML')},
                        'opencv_version': cv2.__version__, 'python': platform.python_version(),
                        'git_commit': command('git', 'rev-parse', 'HEAD'), 'git_dirty': bool(command('git', 'status', '--porcelain'))},
        'protocol': {'frames_per_repeat': args.frames, 'repeats': args.repeats,
                     'warmup_frames_per_repeat': args.warmup, 'render': not args.no_render,
                     'device': device, 'cuda_stage_synchronization': cuda,
                     'zero_shot_interval': config.get('zero_shot', {}).get('run_every_n_frames', 10),
                     'state_between_repeats': 'retained; input video reopened and warmup repeated',
                     'native_resolution': args.native_resolution},
        'initialization_ms': round(initialization_ms, 2),
        'throughput_fps': round(len(rows) / elapsed, 3), 'latency_ms': stages,
        'process_peak_rss_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                     (1024 ** 2 if sys.platform == 'darwin' else 1024), 2),
        'gpu_peak_allocated_mib': max(r['gpu_peak_allocated_mib'] for r in repeats) if cuda else None,
        'gpu_peak_reserved_mib': max(r['gpu_peak_reserved_mib'] for r in repeats) if cuda else None,
        'actions': dict(actions), 'repeats': repeats, 'frames': rows,
        'road_observation_frames': dict(Counter(row['road_status'] for row in rows)),
        'accuracy': {'mAP': None, 'road_IoU': None, 'MOTA': None, 'reason': 'No ground-truth annotations in this latency workload'},
        'limitations': ['No ground truth: mask coverage and observation counts are not accuracy scores',
                        'Repeated image workloads do not measure real driving robustness; short video workloads do not establish generalization',
                        'Tail percentiles are descriptive for this sample count, not service guarantees',
                        'Profile changes can alter predictions; core256 disables supplemental zero-shot inference',
                        'CUDA stage boundaries synchronize completed work; instrumentation prevents overlap and adds overhead',
                        'No real-time scheduling deadline or vehicle safety certification is established'],
    }
    source_fps = source_metadata.get('source_fps', 0)
    if source_fps > 0:
        budget = 1000 / source_fps
        report['source_frame_budget'] = {
            'source_fps': source_fps, 'budget_ms': budget,
            'measured_frames_with_wall_time_within_budget': sum(row['wall_ms'] <= budget for row in rows),
            'measured_frames': len(rows),
            'note': 'Offline service-time comparison only; live queue delay and dropped frames were not measured',
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Report: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=(*PROFILES, 'configured', 'all'), default='all')
    parser.add_argument('--source', default='sample', help='sample, image path, or video path')
    parser.add_argument('--device', default='cpu', help='cpu, cuda, cuda:0 or auto; explicit unavailable CUDA fails')
    parser.add_argument('--config', default=str(ROOT / 'config.yaml'))
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--no-render', action='store_true')
    parser.add_argument('--native-resolution', action='store_true', help='Preserve decoded dimensions, especially portrait video')
    parser.add_argument('--output', default=str(ROOT / 'runs' / 'benchmark' / 'result.json'))
    args = parser.parse_args()
    if min(args.frames, args.repeats, args.threads, args.width, args.height) < 1 or args.warmup < 0:
        parser.error('Frame count, repeats, threads and dimensions must be positive; warmup must be nonnegative')
    logging.basicConfig(level=logging.WARNING)
    if args.profile == 'all':
        for profile in PROFILES:
            output = Path(args.output).with_name(profile + '.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--profile', profile,
                       '--source', args.source, '--device', args.device, '--config', args.config, '--output', str(output)]
            for name in ('frames', 'warmup', 'repeats', 'threads', 'width', 'height'):
                command.extend(['--' + name, str(getattr(args, name))])
            if args.no_render:
                command.append('--no-render')
            if args.native_resolution:
                command.append('--native-resolution')
            subprocess.run(command, check=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
