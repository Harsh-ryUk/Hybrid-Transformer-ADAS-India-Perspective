"""
Temporal Fusion Module (v2.0)
Implements temporal consistency using Optical Flow to reduce false positives.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple

class TemporalConsistencyManager:
    """
    Manages temporal consistency of detections across frames.
    """
    
    def __init__(self, memory_frames=3, confidence_boost=0.15):
        """
        Args:
            memory_frames: Number of previous frames to consider
            confidence_boost: Additional confidence added for consistent detections
        """
        self.memory_frames = memory_frames
        self.confidence_boost = confidence_boost
        
        # History format: List of {'boxes': [], 'scores': [], 'classes': []}
        self.history = []
        self.prev_gray = None
        
        # Tracking IDs
        self.next_id = 0
        self.active_tracks = {} # {id: {'hits': int, 'last_box': [], 'class': int}}

    def process_frame(self, current_frame, current_detections):
        """
        Main entry point for temporal fusion.
        
        Args:
            current_frame: BGR image
            current_detections: Dict {boxes, scores, classes} in normalized format
        
        Returns:
            enhanced_detections: Dict with boosted scores
            tracking_info: data for visualization
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Initialize logic on first frame
        if self.prev_gray is None:
            self.prev_gray = gray
            self.history.append(current_detections)
            # Initialize tracking IDs for all current
            tracking_ids = []
            for i in range(len(current_detections['boxes'])):
                self.active_tracks[self.next_id] = {
                    'hits': 1, 
                    'last_box': current_detections['boxes'][i],
                    'class': current_detections['classes'][i]
                }
                tracking_ids.append(self.next_id)
                self.next_id += 1
            
            enhanced = current_detections.copy()
            enhanced['tracking_ids'] = tracking_ids
            enhanced['consecutive_frames'] = [1]*len(tracking_ids)
            return enhanced, {}

        # 2. Compute Optical Flow (Dense DIS for robustness or Sparse for speed)
        # Using Sparse LK on grid points or just box centers is faster
        # Here we assume we want to predict where *previous* boxes moved to.
        
        # Predict locations of previous detections in current frame
        predicted_boxes = self.predict_locations(self.history[-1], self.prev_gray, gray)
        
        # TRACKING ONLY MODE (Optimization)
        if current_detections is None:
            # Propagate detections forward
            propagated_dets = {
                'boxes': predicted_boxes,
                'scores': self.history[-1]['scores'], 
                'classes': self.history[-1]['classes'],
                'tracking_ids': self.history[-1]['tracking_ids'], # Maintain IDs
                'consecutive_frames': self.history[-1]['consecutive_frames']
            }
            # Update history with predicted
            self.history.append(propagated_dets)
            if len(self.history) > self.memory_frames:
                self.history.pop(0)
            self.prev_gray = gray
            return propagated_dets, {}
        
        # 3. Fuse Detections
        enhanced_detections = self.fuse_detections(current_detections, predicted_boxes)
        
        # Update History
        self.history.append(enhanced_detections)
        if len(self.history) > self.memory_frames:
            self.history.pop(0)
            
        self.prev_gray = gray
        
        return enhanced_detections, {}

    def predict_locations(self, prev_detections, prev_gray, curr_gray):
        """
        Predict where previous detections should appear in current frame using Optical Flow.
        """
        boxes = prev_detections['boxes']
        if len(boxes) == 0:
            return []
            
        predicted_boxes = []
        
        # Calculate flow for the center of each box
        # We can use sparse flow (calcOpticalFlowPyrLK) on the centers
        pts_prev = []
        for box in boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            pts_prev.append([[cx, cy]])
            
        if not pts_prev:
            return []
            
        pts_prev = np.array(pts_prev, dtype=np.float32)
        
        # Calculate Optical Flow
        pts_next, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts_prev, None)
        
        for i, (new_pt, old_pt) in enumerate(zip(pts_next, pts_prev)):
            if status[i] == 1:
                dx = new_pt[0][0] - old_pt[0][0]
                dy = new_pt[0][1] - old_pt[0][1]
                
                old_box = boxes[i]
                # Shift box
                new_box = [
                    old_box[0] + dx,
                    old_box[1] + dy,
                    old_box[2] + dx,
                    old_box[3] + dy
                ]
                predicted_boxes.append(new_box)
            else:
                predicted_boxes.append(boxes[i]) # Fallback: No move
                
        return predicted_boxes

    def fuse_detections(self, current_detections, predicted_locations):
        """
        Merge predictions with current detections and boost confidence.
        """
        boxes = current_detections['boxes']
        scores = current_detections['scores']
        classes = current_detections['classes']
        
        new_scores = []
        consecutive_frames = []
        tracking_ids = []
        
        matched_indices = set()
        
        # Simple IoU/Distance Matching
        # For each current box, check if it matches a predicted box
        for i, box in enumerate(boxes):
            best_match = -1
            max_iou = 0.0
            
            for j, pred_box in enumerate(predicted_locations):
                iou = self.calculate_iou(box, pred_box)
                if iou > 0.5: # Threshold
                    if iou > max_iou:
                        max_iou = iou
                        best_match = j
            
            if best_match != -1:
                # Match found! Boost confidence
                boosted = min(1.0, scores[i] + self.confidence_boost)
                new_scores.append(boosted)
                consecutive_frames.append(2) # At least 2 frames
                matched_indices.add(best_match)
                
                # Inherit ID logic (simplified for this demo)
                # In real tracking, we'd map previous IDs. 
                # Here we simulate consistency boosting.
                tracking_ids.append(self.next_id) 
                self.next_id += 1
            else:
                # New detection
                new_scores.append(scores[i])
                consecutive_frames.append(1)
                tracking_ids.append(self.next_id)
                self.next_id += 1
                
        return {
            "boxes": boxes,
            "scores": np.array(new_scores),
            "classes": classes,
            "tracking_ids": tracking_ids,
            "consecutive_frames": consecutive_frames
        }

    def calculate_iou(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
