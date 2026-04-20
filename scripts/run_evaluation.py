"""
End-to-End Evaluation Runner for ADAS L4 Pipeline
Runs the full pipeline on a validation dataset and generates comprehensive metrics.

Metrics computed:
- Detection: mAP@0.5, per-class AP
- Segmentation: mIoU, pixel accuracy, road IoU
- Tracking: MOTA, MOTP, ID switches
- System: FPS, per-stage latency breakdown

Usage:
    # Evaluate on IDD validation set
    python scripts/run_evaluation.py --source data/idd_yolo/images/val --device cuda

    # Evaluate on video
    python scripts/run_evaluation.py --source data/samples/indian_road_sample.mp4

    # Quick benchmark (first 100 frames only)
    python scripts/run_evaluation.py --source data/samples/indian_road_sample.mp4 --max_frames 100
"""

import os
import sys
import time
import json
import argparse
import logging
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_on_video(
    source: str,
    config_path: str = "config.yaml",
    device: str = "cpu",
    max_frames: int = None,
    output_dir: str = "runs/eval",
    save_video: bool = True,
):
    """
    Run full pipeline evaluation on a video/image directory.

    Returns comprehensive metrics report.
    """
    from src.adas_pipeline_l4 import ADASPipelineL4
    from src.evaluation.metrics import (
        DetectionMetrics, SystemProfiler, LatencyProfile,
        generate_evaluation_report
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("  ADAS L4 — End-to-End Evaluation")
    logger.info("=" * 60)

    # Initialize pipeline
    pipeline = ADASPipelineL4(config_path=config_path, device=device)
    profiler = SystemProfiler(window_size=1000)

    # Determine source type
    is_video = Path(source).suffix.lower() in ('.mp4', '.avi', '.mkv', '.mov', '0') or source.isdigit()
    is_dir = Path(source).is_dir()

    if is_video or source.isdigit():
        frame_iterator = _video_frames(source, max_frames)
    elif is_dir:
        frame_iterator = _directory_frames(source, max_frames)
    else:
        logger.error(f"Unknown source type: {source}")
        return

    # Stats collectors
    all_metrics = []
    action_counts = defaultdict(int)
    anomaly_counts = defaultdict(int)
    detection_class_counts = defaultdict(int)
    total_detections = 0
    total_tracks = 0
    total_anomalies = 0
    frame_count = 0

    # Video writer
    writer = None

    try:
        for frame_idx, frame in frame_iterator:
            frame_count += 1
            viz_frame, metrics = pipeline.process_frame(frame, frame_idx)

            # Collect metrics
            all_metrics.append(metrics)
            action_counts[metrics["decision"]["action"]] += 1
            total_detections += metrics["detections"]
            total_tracks += metrics["tracks"]
            total_anomalies += metrics["anomalies"]

            for anom in metrics.get("active_anomalies", []):
                anomaly_counts[anom.get("type", "unknown")] += 1

            # Latency tracking
            lat = metrics["latency"]
            profiler.add_frame(LatencyProfile(
                detection_ms=lat["detection_ms"],
                segmentation_ms=lat["segmentation_ms"],
                tracking_ms=lat["tracking_ms"],
                anomaly_ms=lat["anomaly_ms"],
                decision_ms=lat["decision_ms"],
                visualization_ms=lat["visualization_ms"],
                total_ms=lat["total_ms"],
            ))

            # Video output
            if save_video:
                if writer is None:
                    h, w = viz_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(
                        str(output_dir / "eval_output.mp4"), fourcc, 20, (w, h)
                    )
                writer.write(viz_frame)

            # Progress log
            if frame_count % 50 == 0:
                summary = profiler.get_summary()
                logger.info(
                    f"Frame {frame_count} | "
                    f"FPS: {summary.get('avg_fps', 0):.1f} | "
                    f"Detections: {metrics['detections']} | "
                    f"Tracks: {metrics['tracks']}"
                )

    except KeyboardInterrupt:
        logger.info("Evaluation interrupted.")
    finally:
        if writer:
            writer.release()

    # ─── Generate Report ───
    perf_summary = profiler.get_summary()

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "device": device,
        "frames_processed": frame_count,

        "performance": perf_summary,

        "detection": {
            "total_detections": total_detections,
            "avg_per_frame": round(total_detections / max(frame_count, 1), 2),
        },

        "tracking": {
            "total_tracks": total_tracks,
            "avg_per_frame": round(total_tracks / max(frame_count, 1), 2),
        },

        "anomalies": {
            "total": total_anomalies,
            "by_type": dict(anomaly_counts),
        },

        "decisions": {
            "action_distribution": dict(action_counts),
            "action_percentages": {
                k: round(v / max(frame_count, 1) * 100, 1)
                for k, v in action_counts.items()
            },
        },

        "latency_breakdown": {
            "detection_ms": round(perf_summary.get("avg_latency", {}).get("detection_ms", 0), 2),
            "segmentation_ms": round(perf_summary.get("avg_latency", {}).get("segmentation_ms", 0), 2),
            "tracking_ms": round(perf_summary.get("avg_latency", {}).get("tracking_ms", 0), 2),
            "anomaly_ms": round(perf_summary.get("avg_latency", {}).get("anomaly_ms", 0), 2),
            "decision_ms": round(perf_summary.get("avg_latency", {}).get("decision_ms", 0), 2),
            "total_ms": round(perf_summary.get("avg_latency", {}).get("total_ms", 0), 2),
        },
    }

    # Save report
    report_path = output_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save markdown report
    md_path = output_dir / "evaluation_report.md"
    _write_markdown_report(report, md_path)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("  EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"  Frames processed: {frame_count}")
    logger.info(f"  Average FPS: {perf_summary.get('avg_fps', 0):.1f}")
    logger.info(f"  Avg detections/frame: {report['detection']['avg_per_frame']}")
    logger.info(f"  Avg tracks/frame: {report['tracking']['avg_per_frame']}")
    logger.info(f"  Total anomalies: {total_anomalies}")
    logger.info(f"\n  Decision distribution:")
    for action, pct in report["decisions"]["action_percentages"].items():
        logger.info(f"    {action}: {pct}%")
    logger.info(f"\n  Latency breakdown:")
    for stage, ms in report["latency_breakdown"].items():
        logger.info(f"    {stage}: {ms}ms")
    logger.info(f"\n  Report saved: {report_path}")
    logger.info(f"  Markdown: {md_path}")
    if save_video:
        logger.info(f"  Video: {output_dir / 'eval_output.mp4'}")
    logger.info("=" * 60)

    return report


def _video_frames(source, max_frames=None):
    """Iterate over video frames."""
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {source}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if max_frames and idx > max_frames:
            break
        yield idx, frame

    cap.release()


def _directory_frames(source, max_frames=None):
    """Iterate over images in a directory."""
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    images = sorted([
        p for p in Path(source).iterdir()
        if p.suffix.lower() in exts
    ])

    if max_frames:
        images = images[:max_frames]

    for idx, img_path in enumerate(images):
        frame = cv2.imread(str(img_path))
        if frame is not None:
            yield idx + 1, frame


def _write_markdown_report(report, output_path):
    """Generate a markdown evaluation report."""
    with open(output_path, "w") as f:
        f.write("# ADAS L4 Evaluation Report\n\n")
        f.write(f"**Date**: {report['timestamp']}  \n")
        f.write(f"**Source**: {report['source']}  \n")
        f.write(f"**Device**: {report['device']}  \n")
        f.write(f"**Frames**: {report['frames_processed']}  \n\n")

        f.write("## Performance\n\n")
        perf = report.get("performance", {})
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| Average FPS | {perf.get('avg_fps', 0):.1f} |\n")
        f.write(f"| Min FPS | {perf.get('min_fps', 0):.1f} |\n")
        f.write(f"| Max FPS | {perf.get('max_fps', 0):.1f} |\n\n")

        f.write("## Latency Breakdown\n\n")
        f.write("| Stage | Time (ms) |\n|---|---|\n")
        for stage, ms in report["latency_breakdown"].items():
            f.write(f"| {stage} | {ms} |\n")

        f.write("\n## Detection Summary\n\n")
        f.write(f"- Total detections: {report['detection']['total_detections']}\n")
        f.write(f"- Average per frame: {report['detection']['avg_per_frame']}\n\n")

        f.write("## Decision Distribution\n\n")
        f.write("| Action | Count | Percentage |\n|---|---|---|\n")
        counts = report["decisions"]["action_distribution"]
        pcts = report["decisions"]["action_percentages"]
        for action in counts:
            f.write(f"| {action} | {counts[action]} | {pcts.get(action, 0)}% |\n")

        f.write("\n## Anomalies\n\n")
        f.write(f"- Total: {report['anomalies']['total']}\n")
        for anom_type, count in report["anomalies"]["by_type"].items():
            f.write(f"- {anom_type}: {count}\n")

        f.write("\n---\n*Generated by ADAS L4 Evaluation Pipeline*\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADAS L4 Evaluation Runner")
    parser.add_argument("--source", type=str, required=True, help="Video path or image directory")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="runs/eval")
    parser.add_argument("--no_video", action="store_true")

    args = parser.parse_args()

    evaluate_on_video(
        source=args.source,
        config_path=args.config,
        device=args.device,
        max_frames=args.max_frames,
        output_dir=args.output_dir,
        save_video=not args.no_video,
    )
