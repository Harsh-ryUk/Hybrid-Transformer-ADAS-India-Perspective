"""Semantic road regions, not trained lane-marking segmentation."""

import logging
import time
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from typing import Dict, List, Tuple, Optional, Any
from src.utils.runtime import resolve_device

logger = logging.getLogger(__name__)

# Legacy compact network size. Capture frames are no longer pre-shrunk to it.
CAPTURE_SIZE = (256, 144)

class SegFormerLaneDetector:
    """
    SegFormer-based Lane Detector for ADAS v2.0.
    
    Features:
    - Pretrained SegFormer-B0
    - Adaptive preprocessing (CLAHE)
    - Polynomial fitting
    - Confidence scoring
    """

    def __init__(
        self,
        model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
        device: str = "cuda",
        input_size: Optional[Tuple[int, int]] = None,
        road_class_ids: Optional[List[int]] = None,
        frame_roi: Optional[List[float]] = None,
        roi_top_fraction: float = 0.45,
        min_road_coverage: float = 0.02,
    ):
        """
        Initialize the SegFormer Model.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda' or 'cpu'.
            input_size: Network (width, height); resized directly from the source ROI.
            road_class_ids: Model-specific drivable labels; defaults to ADE20K road (6).
            frame_roi: Normalized [left, top, right, bottom] crop; exclude video borders/hood explicitly.
        """
        self.device = torch.device(resolve_device(device))
        self.frame_roi = tuple(frame_roi) if frame_roi is not None else (0, 0, 1, 1)
        if len(self.frame_roi) != 4 or not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in self.frame_roi) or not (self.frame_roi[0] < self.frame_roi[2] and self.frame_roi[1] < self.frame_roi[3]):
            raise ValueError('frame_roi must be normalized [left, top, right, bottom]')
        if not 0 <= roi_top_fraction < 1 or not 0 <= min_road_coverage <= 1:
            raise ValueError('Invalid road ROI/coverage threshold')
        self.roi_top_fraction = roi_top_fraction
        self.min_road_coverage = min_road_coverage
        logger.info(f"Initializing SegFormer ({model_name}) on {self.device}...")

        try:
            processor = SegformerImageProcessor.from_pretrained(model_name)
            self.input_size = tuple(input_size) if input_size else (processor.size["width"], processor.size["height"])
            if len(self.input_size) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.input_size):
                raise ValueError("input_size must contain two positive integers (width, height)")
            self.road_class_ids = tuple(road_class_ids) if road_class_ids is not None else (6,)
            # Normalisation is done on-device from 0-255 input, so fold the /255 into mean/std.
            self._mean = torch.tensor(processor.image_mean, device=self.device).view(1, 3, 1, 1) * 255
            self._std = torch.tensor(processor.image_std, device=self.device).view(1, 3, 1, 1) * 255
            self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
            
            if not self.road_class_ids or any(type(i) is not int or not 0 <= i < self.model.config.num_labels for i in self.road_class_ids):
                raise ValueError('road_class_ids must reference valid model output labels')
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("SegFormer Initialized Successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load SegFormer: {e}")
            raise e

        # Adaptive Preprocessors
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._kernel = np.ones((5, 5), np.uint8)

    def preprocess(self, frame: np.ndarray, condition: str = "Normal") -> torch.Tensor:
        """
        Adaptive preprocessing based on seasonal conditions.
        Resize with PIL on the CPU, then normalise on the model's device.
        """
        # 1. CLAHE for Monsoon/Night/Faded
        if condition in ["Monsoon", "Night", "Faded"]:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 2. Resize to model input (PIL, exactly what the HF processor does) + normalise on-device
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if (frame.shape[1], frame.shape[0]) != self.input_size:
            frame = np.array(Image.fromarray(frame).resize(self.input_size, Image.BILINEAR))
        x = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)[None].float()
        return (x - self._mean) / self._std

    def detect(self, frame: np.ndarray, condition: str = "Normal") -> Dict[str, Any]:
        """
        Run inference and post-processing.
        """
        t0 = time.time()
        h, w = frame.shape[:2]
        
        roi = getattr(self, 'frame_roi', (0, 0, 1, 1))
        x0, y0, x1, y1 = int(roi[0] * w), int(roi[1] * h), int(roi[2] * w), int(roi[3] * h)
        crop = frame[y0:y1, x0:x1]
        if not crop.size:
            raise ValueError('frame_roi is empty at this frame resolution')

        # Inference
        with torch.inference_mode():
            logits = self.model(pixel_values=self.preprocess(crop, condition)).logits
            # Interpolate class logits before argmax, not a coarse binary mask afterward.
            target = (min(384, crop.shape[0]), min(384, crop.shape[1]))
            logits = torch.nn.functional.interpolate(logits, size=target, mode='bilinear', align_corners=False)
            # Build the binary road mask on-device so only 1 byte/px crosses to the CPU.
            pred = logits.argmax(dim=1)[0]
            road = torch.zeros_like(pred, dtype=torch.bool)
            for class_id in self.road_class_ids:
                road |= pred == class_id
            road_small = road.to(torch.uint8).mul_(255).cpu().numpy()

        top = int(road_small.shape[0] * getattr(self, 'roi_top_fraction', 0.45))
        road_small[:top] = 0
        # Close small holes, but never dilate a predicted road into nearby objects.
        road_small = cv2.morphologyEx(road_small, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(road_small)
        candidates = [i for i in range(1, count)
                      if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] > road_small.shape[0] * 0.6]
        if candidates:
            selected = max(candidates, key=lambda i: stats[i, cv2.CC_STAT_AREA])
            road_small = np.where(labels == selected, 255, 0).astype(np.uint8)
        else:
            road_small[:] = 0
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        lane_mask[y0:y1, x0:x1] = cv2.resize(road_small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)

        # Debug: Check if we see anything
        road_pixels = cv2.countNonZero(lane_mask)
        if road_pixels < 100 and not getattr(self, "_low_road_logged", False):
            logger.warning(f"Low road pixels detected: {road_pixels}. Unique classes: {pred.unique().tolist()}")
        self._low_road_logged = road_pixels < 100

        confidence = cv2.countNonZero(lane_mask) / max((y1 - y0) * (x1 - x0), 1)
        valid = confidence >= getattr(self, 'min_road_coverage', 0.02)
        if not valid:
            lane_mask[:] = 0

        t1 = time.time()
        
        return {
            "lane_points": [],  # Road boundaries are not lane markings or a steering target.
            "lane_mask": lane_mask, 
            "lane_confidence": float(confidence),
            "detection_method": "segformer",
            "polynomial_coeffs": [[], []],
            "road_observed": valid,
            "road_status": 'observed' if valid else 'unknown',
            "lane_markings_detected": False,
            "frame_roi_pixels": [x0, y0, x1, y1],
            "seasonal_condition": condition,
            "processing_time_ms": (t1 - t0) * 1000
        }

    def fit_polynomial(
        self, mask: np.ndarray, x_offset: int = 0, width: Optional[int] = None
    ) -> Tuple[List[float], List[List[int]]]:
        """
        Fits 2nd degree polynomial to non-zero points.
        `mask` may be a column slice of a larger image: pass its left edge as
        `x_offset` and the full image width as `width`.
        """
        y_idxs, x_idxs = np.nonzero(mask)
        x_idxs = x_idxs + x_offset

        # Need enough points to fit
        if len(y_idxs) < 10 or len(np.unique(y_idxs)) < 3:
            return [], []

        try:
            # Fit x = ay^2 + by + c
            coeffs = np.polyfit(y_idxs, x_idxs, 2)
            
            # Generate points for plot (only within the y-range of detected points)
            min_y, max_y = np.min(y_idxs), np.max(y_idxs)
            plot_y = np.linspace(min_y, max_y, num=50) # fewer points for speed
            
            fit_x = coeffs[0]*plot_y**2 + coeffs[1]*plot_y + coeffs[2]
            
            # Pack points
            pts = []
            h, w = mask.shape[0], width or mask.shape[1]
            for y, x in zip(plot_y, fit_x):
                if 0 <= x < w and 0 <= y < h:
                    pts.append([int(x), int(y)])
            
            return coeffs.tolist(), pts
        except:
            return [], []



    def calculate_confidence(self, logits):
        """Mean probability of the predicted class."""
        probs = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1)[0]
        return max_probs.mean().item()

    def export_to_tensorrt(self, output_path: str):
        """Placeholder for TensorRT export logic."""
        logger.info(f"Exporting current model to {output_path} (INT8)...")
        # Actual TRT export needs torch2trt or ONNX conversion
        pass
