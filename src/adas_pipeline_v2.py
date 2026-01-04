"""
ADAS Pipeline v2.0
Production-grade orchestration of YOLOv11 + SegFormer + Temporal Fusion.
"""

import time
import cv2
import torch
import numpy as np
from typing import Dict, Any, Tuple, List

from src.road_damage.detector import RoadDamageDetector
from src.lane_detection.segformer_lane_detector import SegFormerLaneDetector
from src.utils.temporal_fusion import TemporalConsistencyManager
from src.utils.visualization import draw_dashboard, draw_lanes, draw_damage

class ADASPipelineV2:
    def __init__(self, 
                 detection_model_path='yolov8n.pt', # Using v8n as v11 placeholder
                 lane_model_name='nvidia/segformer-b0-finetuned-ade-512-512',
                 device='cuda'):
        
        self.device = device
        
        # 1. Damage Detector (YOLO)
        # Note: Ideally YOLOv11, but wrapper uses Ultralytics generic class
        self.damage_detector = RoadDamageDetector(model_path=detection_model_path)
        
        # 2. Lane Detector (SegFormer)
        self.lane_detector = SegFormerLaneDetector(model_name=lane_model_name, device=device)
        
        # 3. Temporal Fusion
        self.temporal_manager = TemporalConsistencyManager()
        
        # 4. State
        self.is_monsoon = False

    def detect_seasonal_condition(self, frame) -> str:
        """
        Simple heuristic: Check brightness/contrast or date.
        For now, returns 'Normal' or 'Monsoon' based on avg brightness.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_mean = hsv[:,:,2].mean()
        
        if v_mean < 80:
            return "Night"
        # Could check blue bias for rain, etc.
        return "Normal"

    def process_frame(self, frame: np.ndarray, frame_id: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Full v2.0 Pipeline Step
        """
        t0 = time.time()
        
        # A. Pre-check Condition
        condition = self.detect_seasonal_condition(frame)
        
        # B. Parallel Inference (Simulated sequential for Python GIL)
        # 1. Detect Damage (Every 3rd Frame - Optimization)
        # Use Temporal Tracking for intermediate frames
        t_det_start = time.time()
        
        if frame_id % 3 == 0:
            # Full Inference
            boxes, scores, classes = self.damage_detector.detect(frame)
            current_detections = {
                'boxes': boxes,
                'scores': scores,
                'classes': classes
            }
            # Update Tracker with new detections
            enhanced_detections, _ = self.temporal_manager.process_frame(frame, current_detections)
            self.det_cached = False
        else:
            # Tracking Only (Optical Flow)
            # Pass None to signal tracking mode
            enhanced_detections, _ = self.temporal_manager.process_frame(frame, current_detections=None)
            self.det_cached = True
            
        t_det_end = time.time()
        t_fus_end = time.time() # Fusion is part of tracking now
        
        # 3. Detect Lanes (SegFormer) - FRAME SKIPPING (Every 5th frame)
        t_lane_start = time.time()
        
        if frame_id % 5 == 0 or not hasattr(self, 'last_lane_result') or self.last_lane_result is None:
            self.last_lane_result = self.lane_detector.detect(frame, condition=condition)
            self.lane_cached = False
        else:
            self.lane_cached = True
            
        lane_result = self.last_lane_result
        t_lane_end = time.time()
        
        # C. Visualization
        t_viz_start = time.time()
        
        viz = frame.copy()
        
        # Draw Segmentation Mask (Blue Tinge)
        if 'lane_mask' in lane_result and lane_result['lane_mask'] is not None:
            mask = lane_result['lane_mask']
            # Ensure mask is 3-channel for blending if it's 1-channel
            if mask.ndim == 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Create a blue overlay
            color_mask = np.zeros_like(viz)
            color_mask[:, :, 0] = 255 # Blue channel
            color_mask[:, :, 1] = 50  # Green hint
            
            # Apply mask to color
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=mask[:,:,0]) # Use one channel of mask for bitwise_and
            
            # Blend
            viz = cv2.addWeighted(viz, 1.0, color_mask, 0.4, 0)
        
        # Draw polynomial points if available
        # lane_points is now a List of Lists (Left, Right)
        for poly_pts in lane_result['lane_points']:
            if len(poly_pts) > 0:
                pts = np.array(poly_pts, dtype=np.int32)
                cv2.polylines(viz, [pts], isClosed=False, color=(0, 255, 0), thickness=3)
            
        # Draw Damage
        # Unpack enhanced results
        e_boxes = enhanced_detections['boxes']
        e_scores = enhanced_detections['scores']
        e_classes = enhanced_detections['classes']
        
        viz = draw_damage(viz, e_boxes, e_scores, e_classes, self.damage_detector.get_class_names())
        
        t_viz_end = time.time()
        
        t1 = time.time()
        total_latency = (t1 - t0) * 1000
        
        # Pack Metrics
        metrics = {
            "fps": 1000.0 / total_latency if total_latency > 0 else 0,
            "latency_total": total_latency,
            "latency_breakdown": {
                "detection": (t_det_end - t_det_start)*1000 if not self.det_cached else 0,
                "fusion": (t_det_end - t_det_start)*1000 if self.det_cached else 0, # Tracking counts as fusion time
                "lane": (t_lane_end - t_lane_start)*1000 if not self.lane_cached else 0,
                "viz": (t_viz_end - t_viz_start)*1000
            },
            "condition": condition,
            "lane_conf": lane_result['lane_confidence']
        }
        
        # Draw Dashboard
        status_text = f"V2.0 ({condition})"
        if self.lane_cached: status_text += " [L-Cached]"
        if self.det_cached: status_text += " [D-Track]"
        viz = draw_dashboard(viz, metrics['fps'], metrics['latency_total'], system_status=status_text)
        
        return viz, metrics

    def process_video(self, source, display=True, save_output=False, output_path="output_demo.mp4"):
        """
        Process a video file or camera stream.
        """
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            print(f"Error: Could not open source {source}")
            return

        # Setup Video Writer if enabled
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Assuming resized resolution
            writer = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 360))
            print(f"Saving output to {output_path}...")

        print(f"Starting V2 Processing on {source}...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize for consistent processing if needed
            frame = cv2.resize(frame, (640, 360))
            
            # Process
            if frame_count % 30 == 0:
                print(f"Processing Frame {frame_count}...")
            
            viz, metrics = self.process_frame(frame, frame_id=frame_count)
            frame_count += 1
            
            # Write to file
            if writer:
                writer.write(viz)

            # Display
            if display:
                cv2.imshow("ADAS System V2.0 (SegFormer + YOLOv11)", viz)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if writer:
            writer.release()
            print(f"Video saved to {output_path}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ADAS v2.0 Pipeline")
    parser.add_argument("--source", type=str, default="0", help="Video source (0, filename)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--headless", action="store_true", help="Run without UI (Colab mode)")
    
    args = parser.parse_args()
    
    pipeline = ADASPipelineV2(device=args.device)
    # If headless, we implicitly want to save output and NOT display
    pipeline.process_video(args.source, display=not args.headless, save_output=args.headless)
