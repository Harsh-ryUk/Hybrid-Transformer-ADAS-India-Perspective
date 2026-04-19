"""
OWLv2 Zero-Shot Detector for Indian Roads (L4 ADAS)
Detects rare/India-specific objects via text prompts without fine-tuning.
Uses: google/owlv2-base-patch16-ensemble

Handles objects that COCO-trained YOLO will miss:
- Auto-rickshaws, handcarts, cycle-rickshaws
- Cows, buffaloes, stray dogs on roads
- Overloaded trucks, tractors
"""

import logging
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy
_owlv2_loaded = False
_processor = None
_model = None


@dataclass
class ZeroShotDetection:
    """Single zero-shot detection result."""
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float
    label: str                  # Text query that matched
    center: tuple


def _load_owlv2(model_name: str, device: str):
    """Lazy-load OWLv2 model (heavy — ~1GB)."""
    global _owlv2_loaded, _processor, _model
    if _owlv2_loaded:
        return

    try:
        import torch
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        logger.info(f"Loading OWLv2: {model_name} on {device}...")
        _processor = Owlv2Processor.from_pretrained(model_name)
        _model = Owlv2ForObjectDetection.from_pretrained(model_name)
        _model.to(device)
        _model.eval()
        _owlv2_loaded = True
        logger.info("OWLv2 loaded successfully.")
    except ImportError:
        logger.warning("transformers not installed — OWLv2 disabled")
    except Exception as e:
        logger.error(f"Failed to load OWLv2: {e}")


class OWLv2Detector:
    """
    Zero-shot object detector using OWLv2 for India-specific road objects.

    This is a secondary detector used alongside YOLO to catch objects
    that COCO-trained models miss entirely:
    - Auto-rickshaws (3-wheelers)
    - Cows/buffaloes wandering on roads
    - Handcarts, cycle-rickshaws
    - Overloaded vehicles

    Reference datasets for validation:
    - IDD: http://idd.insaan.iiit.ac.in/
    - Mapillary Vistas: https://www.mapillary.com/dataset/vistas
    """

    DEFAULT_QUERIES = [
        "auto-rickshaw",
        "cow on road",
        "stray dog on road",
        "overloaded truck",
        "handcart on road",
        "tractor on road",
        "cycle-rickshaw",
        "buffalo on road",
    ]

    def __init__(
        self,
        model_name: str = "google/owlv2-base-patch16-ensemble",
        text_queries: Optional[List[str]] = None,
        confidence_threshold: float = 0.15,
        device: str = "cuda",
        run_every_n_frames: int = 10,
    ):
        self.model_name = model_name
        self.text_queries = text_queries or self.DEFAULT_QUERIES
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.run_every_n_frames = run_every_n_frames
        self._frame_counter = 0
        self._last_result: List[ZeroShotDetection] = []
        self._initialized = False

    def _ensure_loaded(self):
        if not self._initialized:
            _load_owlv2(self.model_name, self.device)
            self._initialized = True

    def detect(self, frame: np.ndarray, force: bool = False) -> List[ZeroShotDetection]:
        """
        Run zero-shot detection on frame.

        Since OWLv2 is heavy, it only runs every N frames.
        On skipped frames, returns cached results.

        Args:
            frame: BGR image
            force: Force detection regardless of frame counter

        Returns:
            List of ZeroShotDetection
        """
        self._frame_counter += 1

        # Skip frames for performance (OWLv2 is slow)
        if not force and self._frame_counter % self.run_every_n_frames != 0:
            return self._last_result

        self._ensure_loaded()

        if not _owlv2_loaded:
            return []

        try:
            import torch
            from PIL import Image
            import cv2

            t0 = time.time()

            # Convert BGR → RGB → PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)

            # Process
            inputs = _processor(
                text=self.text_queries,
                images=pil_image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = _model(**inputs)

            # Post-process
            target_sizes = torch.tensor([frame.shape[:2]], device=self.device)
            results = _processor.post_process_object_detection(
                outputs,
                threshold=self.confidence_threshold,
                target_sizes=target_sizes,
            )

            detections = []
            if len(results) > 0:
                result = results[0]
                boxes = result["boxes"].cpu().numpy()
                scores = result["scores"].cpu().numpy()
                labels = result["labels"].cpu().numpy()

                for bbox, score, label_idx in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = bbox
                    label_text = self.text_queries[label_idx] if label_idx < len(self.text_queries) else f"unknown_{label_idx}"

                    detections.append(ZeroShotDetection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=float(score),
                        label=label_text,
                        center=((x1 + x2) / 2, (y1 + y2) / 2),
                    ))

            elapsed = (time.time() - t0) * 1000
            logger.debug(f"OWLv2: {len(detections)} detections in {elapsed:.1f}ms")

            self._last_result = detections
            return detections

        except Exception as e:
            logger.error(f"OWLv2 inference failed: {e}")
            return self._last_result
