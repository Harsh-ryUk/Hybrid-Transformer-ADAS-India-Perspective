"""Offline GPU contracts and integrity of preserved real-model benchmark evidence."""
import csv
import json
import math
import tarfile
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.benchmark_pipeline import benchmark_device, synchronize

ROOT = Path(__file__).resolve().parents[1]


def test_explicit_cuda_never_silently_falls_back():
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(ValueError, match='unavailable'):
            benchmark_device('cuda')
        assert benchmark_device('auto') == 'cpu'


def test_available_cuda_is_labelled_with_device_index():
    with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
        assert benchmark_device('cuda') == 'cuda:0'
        assert benchmark_device('auto') == 'cuda:0'
        with pytest.raises(ValueError, match='unavailable'):
            benchmark_device('cuda:1')
    with pytest.raises(ValueError):
        benchmark_device('mps')


def test_frame_synchronization_is_cuda_only():
    with patch('torch.cuda.synchronize') as sync:
        synchronize('cpu')
        sync.assert_not_called()
        synchronize('cuda:0')
        sync.assert_called_once_with('cuda:0')


def test_pipeline_clock_can_synchronize_completed_cuda_work():
    from src.adas_pipeline_l4 import ADASPipelineL4
    pipeline = ADASPipelineL4.__new__(ADASPipelineL4)
    pipeline.profile_synchronize = True
    pipeline.device = 'cuda:0'
    with patch('torch.cuda.synchronize') as sync:
        assert pipeline._profile_clock() > 0
        sync.assert_called_once_with('cuda:0')
        sync.reset_mock()
        pipeline.device = 'cpu'
        pipeline._profile_clock()
        sync.assert_not_called()
        pipeline.device = 'cuda:0'
        pipeline.profile_synchronize = False
        pipeline._profile_clock()
        sync.assert_not_called()


def test_colab_notebook_has_valid_unexecuted_python_cells():
    notebook = json.loads((ROOT / 'notebooks/colab_benchmark.ipynb').read_text())
    assert notebook['nbformat'] == 4
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            assert cell['execution_count'] is None
            assert not cell['outputs']
            compile(''.join(cell['source']), cell['id'], 'exec')


def test_preserved_cpu_source_matches_recorded_hashes():
    import hashlib
    directory = ROOT / 'benchmarks/mac_m1_cpu'
    with tarfile.open(directory / 'measured_source.tar.gz', 'r:gz') as source:
        for profile in ('full512', 'full256', 'core256'):
            report = json.loads((directory / (profile + '.json')).read_text())
            for path, expected in report['source_files_sha256'].items():
                assert hashlib.sha256(source.extractfile(path).read()).hexdigest() == expected


@pytest.mark.parametrize('profile', ['full512', 'full256', 'core256'])
def test_published_t4_evidence_matches_raw_trace(profile):
    import numpy as np

    directory = ROOT / 'benchmarks/colab_t4_sample_20260930T202529Z'
    report = json.loads((directory / (profile + '.json')).read_text())
    with (directory / (profile + '.csv')).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert report['schema_version'] == 2
    assert report['profile'] == profile
    assert report['environment']['gpu']['name'] == 'Tesla T4'
    assert report['environment']['git_dirty'] is False
    assert report['environment']['git_commit'] == (directory / 'commit.txt').read_text().strip()
    assert not (directory / 'git-status.txt').read_text().strip()
    assert report['protocol'] == {
        'frames_per_repeat': 100, 'repeats': 3, 'warmup_frames_per_repeat': 20,
        'render': True, 'device': 'cuda:0', 'cuda_stage_synchronization': True,
        'zero_shot_interval': 10,
    }
    assert report['source']['requested_resolution'] == [1280, 720]
    assert report['environment']['torch_threads'] == 2
    assert len(report['frames']) == len(rows) == 300
    assert {(frame['repeat'], frame['frame']) for frame in report['frames']} == {
        (repeat, frame) for repeat in range(3) for frame in range(100)
    }
    for frame, row in zip(report['frames'], rows):
        assert frame.keys() == row.keys()
        for key, value in frame.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value)
                assert float(row[key]) == value
            else:
                assert row[key] == value
        assert frame['torch_threads'] == 2
        assert frame['wall_ms'] > 0
    for stage, summary in report['latency_ms'].items():
        values = [frame[stage] for frame in report['frames']]
        assert summary['mean'] == pytest.approx(np.mean(values), abs=0.00051)
        for name, percentile in [('p50', 50), ('p95', 95), ('p99', 99)]:
            assert summary[name] == pytest.approx(np.percentile(values, percentile), abs=0.00051)
    elapsed = sum(repeat['wall_seconds'] for repeat in report['repeats'])
    assert report['throughput_fps'] == pytest.approx(300 / elapsed, abs=0.001)
    for key in ['gpu_peak_allocated_mib', 'gpu_peak_reserved_mib']:
        assert report[key] == max(repeat[key] for repeat in report['repeats'])
        assert report[key] > 0
    assert report['gpu_peak_reserved_mib'] >= report['gpu_peak_allocated_mib']
    assert all(report['accuracy'][key] is None for key in ['mAP', 'road_IoU', 'MOTA'])
    cpu = json.loads((ROOT / 'benchmarks/mac_m1_cpu' / (profile + '.json')).read_text())
    assert report['source']['sha256'] == cpu['source']['sha256']
    assert report['models'] == cpu['models']
    previous_completions = 0
    for repeat in report['repeats']:
        status = repeat['zero_shot']
        if profile != 'core256':
            assert status['last_error'] is None
            assert status['enabled'] and status['loaded']
            assert not status['background'] and not status['worker_busy']
            assert status['inference_count'] - previous_completions == 12
            previous_completions = status['inference_count']
            frames = [frame for frame in report['frames'] if frame['repeat'] == repeat['repeat']]
            assert sum(frame['zero_shot_ms'] > 100 for frame in frames) == 10
        else:
            assert not status['enabled']
    if profile == 'full512':
        assert all(frame['road_mask_coverage'] == 0 for frame in report['frames'])
