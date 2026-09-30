"""GPU contracts are mocked; a physical CUDA benchmark is still required."""
import json
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

