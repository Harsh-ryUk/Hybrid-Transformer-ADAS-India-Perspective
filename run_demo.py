"""
ADAS L4 Demo — Process Indian dashcam video and save annotated output.
Runs the full pipeline: Detection → Tracking → Segmentation → Anomaly → Decision.
Processes only the first N frames for quick demo.
"""

import sys
import os
import time
import logging
import cv2
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_demo(source_path: str, output_path: str, max_frames: int = 120, device: str = "cpu"):
    """
    Run the full L4 ADAS pipeline on a video and save annotated output.

    Args:
        source_path: Input video path
        output_path: Output annotated video path
        max_frames: Maximum frames to process
        device: 'cuda' or 'cpu'
    """
    logger.info("=" * 60)
    logger.info("  ADAS L4 DEMO — Indian Road Dashcam")
    logger.info("=" * 60)

    # Import pipeline
    from src.adas_pipeline_l4 import ADASPipelineL4

    # Initialize pipeline
    logger.info(f"Initializing pipeline on {device}...")
    pipeline = ADASPipelineL4(config_path="config.yaml", device=device)

    # Open video
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        logger.error(f"Could not open: {source_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Input: {source_path} ({width}x{height} @ {fps:.0f}fps, {total} frames)")
    logger.info(f"Processing first {min(max_frames, total)} frames...")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_detection_time = 0

    try:
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            t0 = time.time()

            # Process through full L4 pipeline
            viz_frame, metrics = pipeline.process_frame(frame, frame_count)

            elapsed = (time.time() - t0) * 1000
            total_detection_time += elapsed

            # Write annotated frame
            writer.write(viz_frame)

            # Log progress every 20 frames
            if frame_count % 20 == 0:
                avg_ms = total_detection_time / frame_count
                logger.info(
                    f"Frame {frame_count}/{min(max_frames, total)} | "
                    f"{avg_ms:.1f}ms avg | "
                    f"Detections: {metrics['detections']} | "
                    f"Tracks: {metrics['tracks']} | "
                    f"Action: {metrics['decision']['action']} | "
                    f"Anomalies: {metrics['anomalies']}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cap.release()
        writer.release()

    # Summary
    avg_ms = total_detection_time / max(frame_count, 1)
    avg_fps = 1000 / max(avg_ms, 1)

    logger.info("=" * 60)
    logger.info(f"  DEMO COMPLETE")
    logger.info(f"  Frames processed: {frame_count}")
    logger.info(f"  Average latency: {avg_ms:.1f}ms ({avg_fps:.1f} FPS)")
    logger.info(f"  Output saved to: {output_path}")
    logger.info("=" * 60)

    # Get system profiler summary
    summary = pipeline.profiler.get_summary()
    if summary:
        logger.info(f"Profiler: {summary}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ADAS L4 Demo")
    parser.add_argument("--source", type=str, default="data/samples/indian_road_sample.mp4")
    parser.add_argument("--output", type=str, default="data/samples/l4_demo_output.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    run_demo(args.source, args.output, args.frames, args.device)
