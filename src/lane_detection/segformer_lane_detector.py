"""
SegFormer Lane Detection Module (v2.0)
Implements state-of-the-art semantic segmentation for lane markings using SegFormer-B0.
"""

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

# Every frame is first shrunk to this (w, h); it is the only detail the model ever sees.
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
    ):
        """
        Initialize the SegFormer Model.

        Args:
            model_name: HuggingFace model identifier.
            device: 'cuda' or 'cpu'.
            input_size: (width, height) fed to the model. None = the model's own
                preprocessor size (512x512). Use CAPTURE_SIZE to skip the upsample
                and run ~7x fewer pixels through the network.
            road_class_ids: Model-specific drivable labels; defaults to ADE20K road (6).
        """
        self.device = torch.device(resolve_device(device))
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
            
            # Map output to 2 classes (Background=0, Lane=1)
            # Note: The pretrained model has 150 classes. We will take the class indices 
            # corresponding to 'road', 'lane' etc. usually found in ADE20k.
            # For this 'production' simulation, we will use a binary mask derived from logic,
            # or ideally fine-tune. Since we can't fine-tune instantly, we will use a logic 
            # to extract likely road/lane classes or assume the user provided fine-tuned weights.
            # Assuming fine-tuned binary output for v2.0 spec implies `num_labels=2`.
            # Here we wrap it compatibility.
            
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
        
        # Optimization: 256x144 is a good balance for CPU (16:9 aspect)
        small_frame = cv2.resize(frame, CAPTURE_SIZE)

        # Inference
        with torch.inference_mode():
            logits = self.model(pixel_values=self.preprocess(small_frame, condition)).logits
            # Build the binary road mask on-device so only 1 byte/px crosses to the CPU.
            pred = logits.argmax(dim=1)[0]
            road = torch.zeros_like(pred, dtype=torch.bool)
            for class_id in self.road_class_ids:
                road |= pred == class_id
            road_small = road.to(torch.uint8).mul_(255).cpu().numpy()

        # Resize MASK to original (Nearest neighbor for speed)
        lane_mask = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)

        # Debug: Check if we see anything
        road_pixels = cv2.countNonZero(lane_mask)
        if road_pixels < 100 and not getattr(self, "_low_road_logged", False):
            logger.warning(f"Low road pixels detected: {road_pixels}. Unique classes: {pred.unique().tolist()}")
        self._low_road_logged = road_pixels < 100

        # Region of Interest Filter (Remove sky/horizon noise): keep bottom 50%
        lane_mask[:int(h * 0.5)] = 0

        # Dilate mask to close gaps (important for dashed lines/poor segmentation)
        lane_mask = cv2.dilate(lane_mask, self._kernel, iterations=1)

        # Find Edges of the Road (The Lanes)
        edges = cv2.Canny(lane_mask, 100, 200)

        # Split and Fit (Left/Right) on views of `edges` — no per-side mask copies.
        # We assume the camera is roughly centered.
        midpoint = w // 2
        left_coeffs, left_pts = self.fit_polynomial(edges[:, :midpoint], width=w)
        right_coeffs, right_pts = self.fit_polynomial(edges[:, midpoint:], x_offset=midpoint, width=w)
        
        combined_pts = []
        if left_pts: combined_pts.append(left_pts)
        if right_pts: combined_pts.append(right_pts)

        # Fraction of the lower image covered by the road mask; not a calibrated probability.
        confidence = cv2.countNonZero(lane_mask[h // 2:]) / max((h - h // 2) * w, 1)

        t1 = time.time()
        
        return {
            "lane_points": combined_pts, # List of Lists [[x,y]..]
            "lane_mask": lane_mask, 
            "lane_confidence": float(confidence),
            "detection_method": "segformer",
            "polynomial_coeffs": [left_coeffs, right_coeffs],
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
