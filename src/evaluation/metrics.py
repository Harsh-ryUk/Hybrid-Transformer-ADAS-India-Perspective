"""
Evaluation Metrics Framework (L4 ADAS)
Comprehensive metrics for detection, segmentation, tracking, and decision performance.

Metrics:
- Detection: mAP, precision, recall per class
- Segmentation: IoU, pixel accuracy
- Tracking: MOTA, MOTP, ID switches
- System: FPS, per-stage latency, decision latency
"""

import logging
import time
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class LatencyProfile:
    """Per-stage timing for a single frame."""
    detection_ms: float = 0.0
    segmentation_ms: float = 0.0
    tracking_ms: float = 0.0
    anomaly_ms: float = 0.0
    decision_ms: float = 0.0
    visualization_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def fps(self) -> float:
        return 1000.0 / max(self.total_ms, 0.001)

    def to_dict(self):
        return {
            "detection_ms": round(self.detection_ms, 2),
            "segmentation_ms": round(self.segmentation_ms, 2),
            "tracking_ms": round(self.tracking_ms, 2),
            "anomaly_ms": round(self.anomaly_ms, 2),
            "decision_ms": round(self.decision_ms, 2),
            "visualization_ms": round(self.visualization_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "fps": round(self.fps, 1),
        }


class DetectionMetrics:
    """
    Compute detection metrics: precision, recall, mAP.

    Datasets for evaluation:
    - IDD: http://idd.insaan.iiit.ac.in/
    - BDD100K: https://bdd-data.berkeley.edu/
    - Mapillary Vistas: https://www.mapillary.com/dataset/vistas
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._predictions = []  # (class, confidence, is_tp)
        self._gt_counts = {}    # class → total ground truth count

    def add_frame(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_classes: np.ndarray,
        gt_boxes: np.ndarray,
        gt_classes: np.ndarray,
    ):
        """Add predictions and ground truth for one frame."""
        # Count GT per class
        for cls in gt_classes:
            cls = str(cls)
            self._gt_counts[cls] = self._gt_counts.get(cls, 0) + 1

        if len(pred_boxes) == 0 or len(gt_boxes) == 0:
            # All predictions are FP if no GT
            for i in range(len(pred_boxes)):
                self._predictions.append((str(pred_classes[i]), float(pred_scores[i]), False))
            return

        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(pred_boxes, gt_boxes)

        # Greedy matching
        matched_gt = set()
        # Sort predictions by confidence (descending)
        sorted_indices = np.argsort(-pred_scores)

        for pred_idx in sorted_indices:
            pred_cls = str(pred_classes[pred_idx])
            best_iou = 0.0
            best_gt = -1

            for gt_idx in range(len(gt_boxes)):
                if gt_idx in matched_gt:
                    continue
                if str(gt_classes[gt_idx]) != pred_cls:
                    continue
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt = gt_idx

            is_tp = best_iou >= self.iou_threshold and best_gt != -1
            if is_tp:
                matched_gt.add(best_gt)

            self._predictions.append((pred_cls, float(pred_scores[pred_idx]), is_tp))

    def compute_map(self) -> Dict[str, float]:
        """
        Compute mean Average Precision (mAP) across all classes.

        Returns:
            Dictionary with per-class AP and overall mAP
        """
        # Group by class
        class_preds = {}
        for cls, score, is_tp in self._predictions:
            if cls not in class_preds:
                class_preds[cls] = []
            class_preds[cls].append((score, is_tp))

        results = {}
        aps = []

        for cls in class_preds:
            # Sort by score descending
            sorted_preds = sorted(class_preds[cls], key=lambda x: -x[0])

            tp = 0
            fp = 0
            precisions = []
            recalls = []
            total_gt = self._gt_counts.get(cls, 1)

            for score, is_tp in sorted_preds:
                if is_tp:
                    tp += 1
                else:
                    fp += 1

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / total_gt if total_gt > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

            # Compute AP using 11-point interpolation
            ap = self._compute_ap_11point(precisions, recalls)
            results[f"AP_{cls}"] = round(ap, 4)
            aps.append(ap)

        results["mAP"] = round(np.mean(aps) if aps else 0.0, 4)
        return results

    @staticmethod
    def _compute_ap_11point(precisions, recalls):
        """11-point interpolation for AP."""
        if not precisions:
            return 0.0

        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            if precisions_at_recall:
                ap += max(precisions_at_recall)
        return ap / 11.0

    @staticmethod
    def _compute_iou_matrix(boxes_a, boxes_b):
        """Compute NxM IoU matrix."""
        xa1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0].T)
        ya1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1].T)
        xa2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2].T)
        ya2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3].T)

        inter = np.maximum(0, xa2 - xa1) * np.maximum(0, ya2 - ya1)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter

        return inter / np.maximum(union, 1e-6)

    def reset(self):
        self._predictions.clear()
        self._gt_counts.clear()


class SegmentationMetrics:
    """Compute segmentation IoU and pixel accuracy."""

    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self._confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def add_frame(self, pred_mask: np.ndarray, gt_mask: np.ndarray):
        """Add prediction and GT mask for one frame."""
        pred_flat = pred_mask.flatten()
        gt_flat = gt_mask.flatten()

        for i in range(self.num_classes):
            for j in range(self.num_classes):
                self._confusion[i, j] += np.sum((gt_flat == i) & (pred_flat == j))

    def compute_iou(self) -> Dict[str, float]:
        """Compute per-class IoU and mean IoU."""
        results = {}
        ious = []

        for i in range(self.num_classes):
            intersection = self._confusion[i, i]
            union = self._confusion[i, :].sum() + self._confusion[:, i].sum() - intersection

            iou = intersection / max(union, 1)
            results[f"IoU_class_{i}"] = round(iou, 4)
            ious.append(iou)

        results["mIoU"] = round(np.mean(ious), 4)
        results["pixel_accuracy"] = round(
            np.diag(self._confusion).sum() / max(self._confusion.sum(), 1), 4
        )
        return results

    def reset(self):
        self._confusion.fill(0)


class TrackingMetrics:
    """
    Compute MOTA and MOTP for multi-object tracking evaluation.

    MOTA = 1 - (FN + FP + IDSW) / GT
    MOTP = Average IoU of matched tracks
    """

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self._total_gt = 0
        self._total_fn = 0
        self._total_fp = 0
        self._total_idsw = 0
        self._total_matched_iou = 0.0
        self._total_matches = 0
        self._prev_matches = {}  # gt_id → track_id

    def add_frame(
        self,
        pred_tracks: List[Dict],   # track_id, bbox
        gt_objects: List[Dict],     # gt_id, bbox
    ):
        """Add tracked predictions and ground truth for one frame."""
        self._total_gt += len(gt_objects)

        if len(pred_tracks) == 0:
            self._total_fn += len(gt_objects)
            return
        if len(gt_objects) == 0:
            self._total_fp += len(pred_tracks)
            return

        # Compute IoU matrix
        pred_boxes = np.array([t["bbox"] for t in pred_tracks])
        gt_boxes = np.array([g["bbox"] for g in gt_objects])

        iou_matrix = DetectionMetrics._compute_iou_matrix(pred_boxes, gt_boxes)

        # Hungarian matching
        from scipy.optimize import linear_sum_assignment
        cost = 1.0 - iou_matrix
        row_ids, col_ids = linear_sum_assignment(cost)

        matched_gt = set()
        matched_pred = set()
        current_matches = {}

        for r, c in zip(row_ids, col_ids):
            if iou_matrix[r, c] >= self.iou_threshold:
                matched_pred.add(r)
                matched_gt.add(c)

                self._total_matched_iou += iou_matrix[r, c]
                self._total_matches += 1

                # Check for ID switch
                gt_id = gt_objects[c].get("gt_id", c)
                track_id = pred_tracks[r].get("track_id", r)
                current_matches[gt_id] = track_id

                if gt_id in self._prev_matches and self._prev_matches[gt_id] != track_id:
                    self._total_idsw += 1

        self._prev_matches = current_matches
        self._total_fn += len(gt_objects) - len(matched_gt)
        self._total_fp += len(pred_tracks) - len(matched_pred)

    def compute(self) -> Dict[str, float]:
        """Compute MOTA and MOTP."""
        gt = max(self._total_gt, 1)
        mota = 1.0 - (self._total_fn + self._total_fp + self._total_idsw) / gt
        motp = self._total_matched_iou / max(self._total_matches, 1)

        return {
            "MOTA": round(mota, 4),
            "MOTP": round(motp, 4),
            "total_gt": self._total_gt,
            "FN": self._total_fn,
            "FP": self._total_fp,
            "ID_switches": self._total_idsw,
            "total_matches": self._total_matches,
        }

    def reset(self):
        self._total_gt = 0
        self._total_fn = 0
        self._total_fp = 0
        self._total_idsw = 0
        self._total_matched_iou = 0.0
        self._total_matches = 0
        self._prev_matches.clear()


class SystemProfiler:
    """Track FPS and per-stage latency over time."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size
        self._latencies: List[LatencyProfile] = []

    def add_frame(self, latency: LatencyProfile):
        self._latencies.append(latency)
        if len(self._latencies) > self.window_size:
            self._latencies.pop(0)

    def get_summary(self) -> Dict:
        """Get average metrics over the window."""
        if not self._latencies:
            return {}

        n = len(self._latencies)
        avg = LatencyProfile(
            detection_ms=sum(l.detection_ms for l in self._latencies) / n,
            segmentation_ms=sum(l.segmentation_ms for l in self._latencies) / n,
            tracking_ms=sum(l.tracking_ms for l in self._latencies) / n,
            anomaly_ms=sum(l.anomaly_ms for l in self._latencies) / n,
            decision_ms=sum(l.decision_ms for l in self._latencies) / n,
            visualization_ms=sum(l.visualization_ms for l in self._latencies) / n,
            total_ms=sum(l.total_ms for l in self._latencies) / n,
        )

        return {
            "avg_latency": avg.to_dict(),
            "avg_fps": round(avg.fps, 1),
            "frames_profiled": n,
            "min_fps": round(min(l.fps for l in self._latencies), 1),
            "max_fps": round(max(l.fps for l in self._latencies), 1),
        }

    def reset(self):
        self._latencies.clear()


def generate_evaluation_report(
    detection_metrics: Optional[DetectionMetrics] = None,
    segmentation_metrics: Optional[SegmentationMetrics] = None,
    tracking_metrics: Optional[TrackingMetrics] = None,
    system_profiler: Optional[SystemProfiler] = None,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Generate a comprehensive evaluation report.

    Args:
        detection_metrics: DetectionMetrics instance
        segmentation_metrics: SegmentationMetrics instance
        tracking_metrics: TrackingMetrics instance
        system_profiler: SystemProfiler instance
        output_path: Optional path to save JSON report

    Returns:
        Dictionary with all metrics
    """
    report = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if detection_metrics:
        report["detection"] = detection_metrics.compute_map()
    if segmentation_metrics:
        report["segmentation"] = segmentation_metrics.compute_iou()
    if tracking_metrics:
        report["tracking"] = tracking_metrics.compute()
    if system_profiler:
        report["system"] = system_profiler.get_summary()

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Evaluation report saved to {output_path}")

    return report
