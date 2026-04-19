"""
India-Aware Object Detector (L4 ADAS)
Multi-class YOLOv8 detection with India-specific categories.
Trained on IDD / BDD100K / COCO with class remapping for Indian roads.
"""

import logging
import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# India-Specific Class Taxonomy
# ─────────────────────────────────────────────────────────────────────────────

# COCO class IDs → India road categories
INDIA_CLASS_MAP = {
    # ── Vehicles ──
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    # ── Vulnerable Road Users ──
    0: "person",
    1: "bicycle",
    # ── Animals (COCO animal classes) ──
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    # ── Traffic Infrastructure ──
    9: "traffic_light",
    11: "stop_sign",
}

# Category grouping for decision-making
CATEGORY_GROUPS = {
    "vehicles": {"car", "motorcycle", "bus", "truck"},
    "vulnerable_road_users": {"person", "bicycle"},
    "animals": {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra"},
    "traffic_infrastructure": {"traffic_light", "stop_sign"},
}

# IDD (Indian Driving Dataset) class names for fine-tuned models
IDD_CLASSES = [
    "road", "drivable_fallback", "sidewalk", "non-drivable_fallback",
    "person", "rider", "motorcycle", "bicycle", "autorickshaw",
    "car", "truck", "bus", "vehicle_fallback", "curb",
    "wall", "fence", "guard_rail", "billboard", "traffic_sign",
    "traffic_light", "pole", "obs-str-bar-fallback", "building",
    "bridge", "vegetation", "sky", "misc", "unlabeled",
    "animal", "train",
]


@dataclass
class Detection:
    """Single detection result."""
    bbox: List[float]            # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    category: str                # vehicles, vulnerable_road_users, animals, traffic
    center: Tuple[float, float]  # (cx, cy)
    area: float                  # bbox area in pixels


@dataclass
class DetectionResult:
    """Batch detection output for a frame."""
    detections: List[Detection] = field(default_factory=list)
    frame_shape: Tuple[int, int] = (0, 0)
    processing_time_ms: float = 0.0
    model_name: str = ""

    @property
    def num_detections(self) -> int:
        return len(self.detections)

    def get_by_category(self, category: str) -> List[Detection]:
        return [d for d in self.detections if d.category == category]

    def get_boxes_scores_classes(self):
        """Legacy interface for backward compat."""
        if not self.detections:
            return np.array([]), np.array([]), np.array([])
        boxes = np.array([d.bbox for d in self.detections])
        scores = np.array([d.confidence for d in self.detections])
        class_ids = np.array([d.class_id for d in self.detections])
        return boxes, scores, class_ids


class IndiaObjectDetector:
    """
    YOLOv8-based object detector with India-specific class awareness.

    Supports:
    - COCO pretrained (default — maps to India categories)
    - IDD fine-tuned (when custom weights provided)
    - BDD100K fine-tuned
    - Per-category confidence thresholds

    Datasets:
    - IDD: http://idd.insaan.iiit.ac.in/
    - BDD100K: https://bdd-data.berkeley.edu/
    - Mapillary Vistas: https://www.mapillary.com/dataset/vistas
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        device: str = "cuda",
        category_thresholds: Optional[Dict[str, float]] = None,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.model_path = model_path

        # Per-category thresholds (safety-critical classes get lower thresholds)
        self.category_thresholds = category_thresholds or {
            "vehicles": 0.40,
            "vulnerable_road_users": 0.25,  # Lower — must not miss pedestrians
            "animals": 0.20,                # Lower — cows/dogs very common in India
            "traffic_infrastructure": 0.35,
        }

        logger.info(f"Loading India Object Detector: {model_path}")
        self.model = YOLO(model_path)
        self.model_names = self.model.names

        # Detect if this is an IDD fine-tuned model
        self.is_idd_model = self._check_idd_model()
        if self.is_idd_model:
            logger.info("IDD fine-tuned model detected — using IDD class mapping")

    def _check_idd_model(self) -> bool:
        """Check if model uses IDD class names."""
        model_classes = set(str(v).lower() for v in self.model_names.values())
        idd_markers = {"autorickshaw", "rider", "drivable_fallback"}
        return bool(model_classes & idd_markers)

    def _get_category(self, class_name: str) -> str:
        """Map a class name to its category group."""
        class_lower = class_name.lower()
        for category, members in CATEGORY_GROUPS.items():
            if class_lower in members:
                return category
        # IDD-specific mappings
        if class_lower in ("autorickshaw", "rider", "vehicle_fallback"):
            return "vehicles"
        if class_lower == "animal":
            return "animals"
        return "other"

    def _passes_category_threshold(self, confidence: float, category: str) -> bool:
        """Check if detection passes per-category confidence threshold."""
        threshold = self.category_thresholds.get(category, self.conf_thres)
        return confidence >= threshold

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Detect objects in a frame with India-aware class mapping.

        Args:
            frame: BGR image (numpy array)

        Returns:
            DetectionResult with structured detections
        """
        t0 = time.time()
        h, w = frame.shape[:2]

        # Run YOLO inference
        results = self.model(
            frame,
            conf=min(self.category_thresholds.values()),  # Use lowest threshold
            iou=self.iou_thres,
            verbose=False,
            device=self.device,
        )

        result = results[0]
        boxes_raw = result.boxes.xyxy.cpu().numpy()
        scores_raw = result.boxes.conf.cpu().numpy()
        class_ids_raw = result.boxes.cls.cpu().numpy().astype(int)

        detections = []
        for bbox, score, cid in zip(boxes_raw, scores_raw, class_ids_raw):
            # Get class name
            if self.is_idd_model and cid < len(IDD_CLASSES):
                class_name = IDD_CLASSES[cid]
            elif cid in INDIA_CLASS_MAP:
                class_name = INDIA_CLASS_MAP[cid]
            else:
                class_name = self.model_names.get(cid, f"class_{cid}")

            category = self._get_category(class_name)

            # Apply per-category thresholding
            if not self._passes_category_threshold(float(score), category):
                continue

            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            detections.append(Detection(
                bbox=[float(x1), float(y1), float(x2), float(y2)],
                confidence=float(score),
                class_id=int(cid),
                class_name=class_name,
                category=category,
                center=(float(cx), float(cy)),
                area=float(area),
            ))

        processing_time = (time.time() - t0) * 1000

        return DetectionResult(
            detections=detections,
            frame_shape=(h, w),
            processing_time_ms=processing_time,
            model_name=self.model_path,
        )

    def detect_legacy(self, frame: np.ndarray):
        """Backward-compatible interface: returns (boxes, scores, class_ids)."""
        result = self.detect(frame)
        return result.get_boxes_scores_classes()

    def get_class_names(self) -> Dict:
        return self.model_names
