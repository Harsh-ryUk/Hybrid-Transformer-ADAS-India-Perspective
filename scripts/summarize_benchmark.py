"""Build a shareable report and chart from benchmark_pipeline JSON results."""
import argparse
import json
from pathlib import Path


def summarize(directory):
    reports = [json.loads((directory / f'{name}.json').read_text()) for name in ('full512', 'full256', 'core256')]
    first = reports[0]
    protocol = first['protocol']
    for report in reports[1:]:
        if report['protocol'] != protocol or report['source']['sha256'] != first['source']['sha256']:
            raise ValueError('Reports must use the same protocol, device and input source')
    device = protocol['device']
    hardware = (first['environment'].get('gpu') or {}).get('name') or first['environment']['cpu']
    width, height = first['source']['requested_resolution']
    measured_frames = protocol['frames_per_repeat'] * protocol['repeats']
    workload = ('Repeated letterboxed image' if first['source']['kind'].startswith('repeated') else 'Decoded video')
    lines = [f'# RoadSense India — {device} benchmark', '',
             'Real YOLOv8n, SegFormer-B0 and (for full profiles) OWLv2 inference. No mock model timings.', '',
             f"Hardware: {hardware}; device: {device}; PyTorch host threads: {first['environment']['torch_threads']}; input: {width}×{height}.", '',
             f"Workload: {workload}: {Path(first['source']['path']).name}. "
             f"Each profile uses a separate process, {protocol['repeats']} repeats of {protocol['frames_per_repeat']} measured frames, and {protocol['warmup_frames_per_repeat']} warmup frames per repeat. "
             f"Frames include input copy/decode, perception, tracking, segmentation, signal classification, anomalies and decisions; rendering: {protocol['render']}. "
             'Image loading/letterboxing, preview export and model initialization are excluded from warm throughput and reported separately where applicable.', '',
             '| Profile | OWLv2 | SegFormer input | FPS | p50 frame ms | p95 frame ms | p99 frame ms | Peak process RSS MiB |',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for report in reports:
        lat = report['latency_ms']['wall_ms']
        size = report['configuration']['segmentation']['input_size']
        interval = report['configuration']['zero_shot']['run_every_n_frames']
        lines.append(f"| {report['profile']} | {f'every {interval} frames' if report['configuration']['zero_shot']['enabled'] else 'disabled'} | {size[0]}×{size[1]} | {report['throughput_fps']:.2f} | {lat['p50']:.1f} | {lat['p95']:.1f} | {lat['p99']:.1f} | {report['process_peak_rss_mb']:.0f} |")
    if device.startswith('cuda'):
        lines.extend(['', 'CUDA timing synchronizes stage boundaries and frame completion. This includes synchronization overhead and prevents cross-stage overlap.', '',
                      '| Profile | Peak PyTorch allocated MiB | Peak PyTorch reserved MiB |', '|---|---:|---:|'])
        for report in reports:
            lines.append(f"| {report['profile']} | {report['gpu_peak_allocated_mib']:.1f} | {report['gpu_peak_reserved_mib']:.1f} |")
    lines.extend(['', '![Measured throughput and tail latency](comparison.png)', '', '## Where time goes', '',
                  '| Mean stage latency, ms | full512 | full256 | core256 |', '|---|---:|---:|---:|'])
    for stage in ('detection_ms', 'zero_shot_ms', 'segmentation_ms', 'tracking_ms', 'signal_ms', 'anomaly_ms', 'decision_ms', 'visualization_ms'):
        values = [report['latency_ms'][stage]['mean'] for report in reports]
        lines.append(f"| {stage} | {' | '.join(f'{value:.2f}' for value in values)} |")
    lines.extend(['', '## Interpretation', '',
                  'Compare stage means and periodic OWLv2 frames in the raw traces. The zero-shot mean is amortized over skipped frames; '
                  'tail latency exposes occasional slow frames. A resolution change can alter masks. '
                  'core256 is a different operating profile, not a speedup with identical outputs.', '',
                  f"The full512 workload had {sum(row.get('road_mask_coverage', 0) == 0 for row in reports[0]['frames'])}/{measured_frames} frames with an empty road mask; compact profiles can produce different masks and fitted boundaries. "
                  'This is evidence of sensitivity to preprocessing/resolution, not evidence that either profile is accurate. '
                  'Runtime coverage depends on the scenes present; near-zero stage time does not validate that feature. '
                  'The CSV traffic_signal field, when present, records recognized/unknown states, not ground truth. '
                  'Scenario tests cover rules separately, but moving-road footage with ground truth is needed for mAP, road IoU and MOTA. '
                  'Those accuracy metrics are explicitly null in the JSON reports.', '',
                  'Process RSS includes Python, libraries and models. It is a process peak, not GPU memory or steady-state resident memory. '
                  'CUDA peaks are PyTorch allocator statistics after warmup, not total driver/device usage or initialization peaks. '
                  f'Host load is recorded in each JSON and can change timings; p99 from {measured_frames} frames is descriptive, not an SLA.', '',
                  '## Reproduce', '', '```bash',
                  'pip install -r requirements-dev.txt',
                  f"python scripts/benchmark_pipeline.py --device {device} --profile all --source "
                  f"{'sample' if first['source']['kind'] == 'repeated_sample_image' else repr(first['source']['path'])} "
                  f"--frames {protocol['frames_per_repeat']} --warmup {protocol['warmup_frames_per_repeat']} --repeats {protocol['repeats']} "
                  f"--threads {first['environment']['requested_torch_threads']} --width {width} --height {height} "
                  f"{'--no-render ' if not protocol['render'] else ''}--output runs/benchmark/result.json",
                  'python scripts/summarize_benchmark.py --directory runs/benchmark', '```', '',
                  'Models download on the first run. After caching them, set HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for an offline repeat. '
                  'For headless cost use --no-render; for real footage use --source /path/to/video.mp4 with enough frames for warmup and measurement.', '',
                  '## Evidence', '',
                  '[full512 JSON](full512.json) · [full256 JSON](full256.json) · [core256 JSON](core256.json)', '',
                  '[full512 raw CSV](full512.csv) · [full256 raw CSV](full256.csv) · [core256 raw CSV](core256.csv)', '',
                  'Each JSON records source/config/code hashes, checkpoint revisions, versions, cold initialization, warmup, per-repeat FPS, '
                  'OWL completion/error status and all measured frame samples.'])
    (directory / 'README.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    colors = ['#475569', '#0d9488', '#2563eb']
    labels = [report['profile'] for report in reports]
    fps = [report['throughput_fps'] for report in reports]
    p95 = [report['latency_ms']['wall_ms']['p95'] / 1000 for report in reports]
    axes[0].bar(labels, fps, color=colors, width=0.6)
    axes[0].set_ylabel('Measured frames / second ↑')
    axes[0].set_ylim(0, max(fps) * 1.25)
    axes[1].bar(labels, p95, color=colors, width=0.6)
    axes[1].set_ylabel('p95 frame latency, seconds ↓')
    axes[1].set_ylim(0, max(p95) * 1.25)
    for axis, values in zip(axes, (fps, p95)):
        for index, value in enumerate(values):
            axis.text(index, value, f'{value:.2f}', ha='center', va='bottom')
        axis.grid(axis='y', alpha=0.15)
        axis.set_axisbelow(True)
    fig.suptitle(f'RoadSense India • Real-model {device} benchmark', fontsize=15, fontweight='bold')
    fig.supxlabel(f"{hardware} · {width}×{height} · {workload.lower()} · {measured_frames} measured frames/profile\nfull profiles include OWLv2; core256 disables it. Accuracy not measured.", fontsize=9)
    fig.savefig(directory / 'comparison.png', dpi=180)
    plt.close(fig)
    print(directory / 'README.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('runs/benchmark'))
    summarize(parser.parse_args().directory)
