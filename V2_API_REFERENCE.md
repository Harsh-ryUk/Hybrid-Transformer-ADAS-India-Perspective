# 📚 ADAS v2.0 API Reference

## 1. `ADASPipelineV2`
**Path**: `src/adas_pipeline_v2.py`
The main orchestrator for the v2.0 system.

### `__init__(self, detection_model_path='yolov8n.pt', lane_model_name='...', device='cuda')`
Initializes the pipeline with SegFormer and YOLOv11.
- **detection_model_path**: Path to YOLO weights (pt/engine).
- **lane_model_name**: HuggingFace ID for SegFormer.
- **device**: 'cuda' or 'cpu'.

### `process_frame(self, frame) -> (np.ndarray, dict)`
Process a single BGR frame.
- **Returns**:
    - `viz_frame`: Annotated image.
    - `metrics`: Dictionary containing FPS, latency breakdown, and lane confidence.

### `detect_seasonal_condition(self, frame) -> str`
Detects if condition is 'Normal', 'Night', 'Monsoon', or 'Faded'.

---

## 2. `SegFormerLaneDetector`
**Path**: `src/lane_detection/segformer_lane_detector.py`
Transformer-based lane segmentation.

### `detect(self, frame, condition='Normal') -> dict`
- **condition**: Affects preprocessing (e.g., CLAHE for 'Monsoon').
- **Returns**:
    - `lane_points`: List of [x, y] coordinates.
    - `polynomial_coeffs`: [a, b, c] for $ax^2 + bx + c$.
    - `lane_confidence`: float (0-1).

---

## 3. `TemporalConsistencyManager`
**Path**: `src/utils/temporal_fusion.py`
Reduces false positives using Optical Flow.

### `process_frame(self, current_frame, current_detections) -> (dict, dict)`
- **current_detections**: Dict with `boxes` (N,4), `scores`, `classes`.
- **Returns**: Enhanced detections with boosted confidence for tracked objects.

---

## 4. `ModelOptimizer`
**Path**: `src/utils/model_optimization.py`

### `optimize_segformer_for_jetson(model_name, output_name) -> dict`
Exports SegFormer to ONNX and provides the `trtexec` command for INT8 conversion.
