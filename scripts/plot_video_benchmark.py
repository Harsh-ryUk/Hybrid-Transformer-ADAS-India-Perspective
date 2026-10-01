"""Render paired video measurements, without inference or invented accuracy."""
import argparse
import json
from pathlib import Path


def paired_reports(directory):
    reports = [json.loads((directory / revision / 'configured.json').read_text())
               for revision in ('baseline', 'optimized')]
    for key in ('source', 'configuration', 'models', 'protocol'):
        if reports[0][key] != reports[1][key]:
            raise ValueError(f'Cannot plot an unpaired comparison: {key}')
    if len(reports[0]['frames']) != len(reports[1]['frames']):
        raise ValueError('Measured frame counts differ')
    return reports


def plot(directory):
    reports = paired_reports(directory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig = plt.figure(figsize=(12, 7.8), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.35])
    colors = ['#596579', '#087f8c']
    names = ['Baseline', 'Optimized']
    for cell, values, title, ylabel in (
        (grid[0, 0], [r['throughput_fps'] for r in reports], 'Throughput · higher is better', 'Frames / second'),
        (grid[0, 1], [r['latency_ms']['wall_ms']['p95'] for r in reports], 'Tail latency · lower is better', 'p95 end-to-end latency (ms)'),
    ):
        ax = fig.add_subplot(cell)
        ax.bar(names, values, color=colors, width=0.48)
        ax.set_ylim(0, max(values) * 1.25)
        ax.set_title(title, loc='left', pad=12)
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=0.15)
        ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.2f}', (i, value), xytext=(0, 6),
                        textcoords='offset points', ha='center', va='bottom')

    stages = [('detection_ms', 'Detection'), ('segmentation_ms', 'Road segmentation'),
              ('anomaly_ms', 'Surface / anomaly stage'), ('visualization_ms', 'Rendering')]
    ax = fig.add_subplot(grid[1, :])
    positions = np.arange(len(stages))
    for i, result in enumerate(reports):
        values = [result['latency_ms'][key]['mean'] for key, _ in stages]
        bars = ax.barh(positions + (i - 0.5) * 0.34, values, height=0.3,
                       label=names[i], color=colors[i])
        ax.bar_label(bars, fmt='%.2f', padding=5)
    ax.set_yticks(positions, [name for _, name in stages])
    ax.invert_yaxis()
    maximum = max(r['latency_ms'][key]['mean'] for r in reports for key, _ in stages)
    ax.set_xlim(0, maximum * 1.16)
    ax.set_xlabel('Mean stage time (ms) · synchronized instrumentation')
    ax.set_title('Where the optimization saves time', loc='left', pad=12)
    ax.grid(axis='x', alpha=0.15)
    ax.set_axisbelow(True)
    ax.legend(loc='lower right', frameon=False)
    gain = (reports[1]['throughput_fps'] / reports[0]['throughput_fps'] - 1) * 100
    fig.suptitle(f'RoadSense India · Tesla T4 moving-video benchmark\n{gain:.1f}% higher throughput; unchanged recorded trace fields', fontsize=16)
    frames = len(reports[0]['frames'])
    fig.supxlabel(f'720×1280 native video · {frames:,} measured frames/revision · 3 repeats · decode + rendering\n'
                  'OWL disabled in both · offline single-stream replay · accuracy and real-time safety not established', fontsize=10)
    output = directory / 'comparison.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('benchmarks/colab_t4_video_20261001'))
    plot(parser.parse_args().directory)
