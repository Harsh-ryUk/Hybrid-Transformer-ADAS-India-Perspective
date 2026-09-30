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
import threading
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy
_owlv2_loaded = False
_processor = None
_model = None
_model_key = None
_load_lock = threading.Lock()


@dataclass
class ZeroShotDetection:
    """Single zero-shot detection result."""
    bbox: List[float]           # [x1, y1, x2, y2]
    confidence: float
    label: str                  # Text query that matched
    center: tuple


def _load_owlv2(model_name: str, device: str):
    """Lazy-load OWLv2 model (heavy — ~1GB)."""
    global _owlv2_loaded, _processor, _model, _model_key
    with _load_lock:
        if _owlv2_loaded and _model_key in (None, (model_name, device)):
            return _processor, _model
        try:
            from transformers import Owlv2Processor, Owlv2ForObjectDetection
            logger.info(f"Loading OWLv2: {model_name} on {device}...")
            processor = Owlv2Processor.from_pretrained(model_name)
            model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device).eval()
            _processor, _model = processor, model
            _model_key = (model_name, device)
            _owlv2_loaded = True
            return processor, model
        except Exception as e:
            logger.error(f"Failed to load OWLv2: {e}")
            return None, None


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
        query_thresholds: Optional[Dict[str, float]] = None,
        device: str = "cuda",
        run_every_n_frames: int = 10,
        background: bool = False,
    ):
        self.model_name = model_name
        self.text_queries = text_queries or self.DEFAULT_QUERIES
        self.confidence_threshold = confidence_threshold
        self.query_thresholds = query_thresholds or {}
        self.device = device
        if run_every_n_frames < 1:
            raise ValueError("run_every_n_frames must be at least 1")
        self.run_every_n_frames = run_every_n_frames
        self.background = background
        self._frame_counter = 0
        self._last_result: List[ZeroShotDetection] = []
        self._initialized = False
        self._worker: Optional[threading.Thread] = None
        self._inference_count = 0
        self._last_completed_at = None
        self._last_error = None
        self._processor = self._model = None
        self._last_source_at = None

    def _ensure_loaded(self):
        if not self._initialized:
            self._processor, self._model = _load_owlv2(self.model_name, self.device)
            self._initialized = True

    def detect(self, frame: np.ndarray, force: bool = False) -> List[ZeroShotDetection]:
        """
        Run zero-shot detection on frame.

        Since OWLv2 is heavy, it only runs every N frames.
        On skipped frames, returns cached results.

        With background=True the run happens on a worker thread so the caller never
        stalls: this call returns the previous result and the new one lands when done.
        A run is dropped (not queued) if the previous one is still going.

        Args:
            frame: BGR image
            force: Force detection regardless of frame counter (always synchronous)

        Returns:
            List of ZeroShotDetection
        """
        self._frame_counter += 1

        # Skip frames for performance (OWLv2 is slow)
        if not force and self._frame_counter % self.run_every_n_frames != 0:
            return self._last_result

        if self.background and not force:
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._run_and_store, args=(frame.copy(),), daemon=True)
                self._worker.start()
            return self._last_result

        if self._worker is not None and self._worker.is_alive():
            self._worker.join()
        return self._run(frame)

    def _run_and_store(self, frame: np.ndarray):
        self._last_result = self._run(frame)

    def _run(self, frame: np.ndarray) -> List[ZeroShotDetection]:
        self._ensure_loaded()

        if self._model is None:
            self._last_error = "model unavailable"
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
            source_at = time.monotonic()
            inputs = self._processor(
                text=self.text_queries,
                images=pil_image,
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self._model(**inputs)

            # Post-process
            target_sizes = torch.tensor([frame.shape[:2]], device=self.device)
            base_threshold = self.confidence_threshold
            if self.query_thresholds:
                base_threshold = min(base_threshold, min(self.query_thresholds.values()))

            results = self._processor.post_process_object_detection(
                outputs,
                threshold=base_threshold,
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

                    # Filter based on query-specific threshold if it exists, otherwise use global confidence_threshold
                    req_threshold = self.query_thresholds.get(label_text, self.confidence_threshold)
                    if score < req_threshold:
                        continue

                    detections.append(ZeroShotDetection(
                        bbox=[float(x1), float(y1), float(x2), float(y2)],
                        confidence=float(score),
                        label=label_text,
                        center=((x1 + x2) / 2, (y1 + y2) / 2),
                    ))

            elapsed = (time.time() - t0) * 1000
            logger.debug(f"OWLv2: {len(detections)} detections in {elapsed:.1f}ms")

            self._last_result = detections
            self._inference_count += 1
            self._last_completed_at = time.monotonic()
            self._last_source_at = source_at
            self._last_error = None
            return detections

        except Exception as e:
            self._last_error = str(e)
            logger.error(f"OWLv2 inference failed: {e}")
            return self._last_result

    def status(self):
        return {
            "enabled": True, "background": self.background,
            "loaded": self._model is not None,
            "inference_count": self._inference_count,
            "worker_busy": self._worker is not None and self._worker.is_alive(),
            "result_age_seconds": None if self._last_completed_at is None else round(time.monotonic() - self._last_completed_at, 3),
            "last_error": self._last_error,
            "result_source_age_seconds": None if self._last_source_at is None else round(time.monotonic() - self._last_source_at, 3),
        }

    def close(self):
        if self._worker is not None:
            self._worker.join()
