"""Check archived measurements without loading models or needing CUDA."""
import hashlib
import json
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks/colab_t4_video_20261001'


def report(profile):
    return json.loads((ROOT / profile / 'configured.json').read_text())


def test_video_reports_are_paired_and_observed_trace_fields_match():
    baseline, optimized = report('baseline'), report('optimized')
    for field in ('source', 'configuration', 'models', 'protocol'):
        assert baseline[field] == optimized[field]
    assert len(baseline['frames']) == len(optimized['frames']) == 1782
    fields = ('repeat', 'frame', 'source_frame', 'detections', 'tracks',
              'anomalies', 'action', 'traffic_signal', 'road_mask_coverage', 'road_status')
    for a, b in zip(baseline['frames'], optimized['frames']):
        assert {key: a[key] for key in fields} == {key: b[key] for key in fields}
    assert baseline['accuracy']['road_IoU'] is None
    assert optimized['source_frame_budget']['measured_frames_with_wall_time_within_budget'] == 0


def test_video_aggregate_throughput_matches_repeat_wall_times():
    for profile in ('baseline', 'optimized'):
        result = report(profile)
        expected = len(result['frames']) / sum(r['wall_seconds'] for r in result['repeats'])
        assert result['throughput_fps'] == pytest.approx(expected, abs=0.0006)
        assert len(result['repeats']) == 3
        assert all(row['torch_threads'] == 2 for row in result['frames'])


def test_video_snapshot_contains_every_reported_python_source():
    for profile in ('baseline', 'optimized'):
        result = report(profile)
        with tarfile.open(ROOT / profile / 'measured_source.tar.gz') as archive:
            for path, expected in result['source_files_sha256'].items():
                if Path(path).name.startswith('._'):
                    continue  # AppleDouble packaging metadata is not executable source.
                actual = hashlib.sha256(archive.extractfile(path).read()).hexdigest()
                assert actual == expected, path
