import cv2
import numpy as np
from typing import Tuple

class TrafficSignalClassifier:
    """
    Classifies traffic light color (Red, Green, Yellow) using computer vision heuristics.
    Since YOLO only detects 'traffic light' (Class 9), this module parses the specific signal.
    """
    
    def __init__(self):
        # HSV Ranges for Signal Colors
        # Red has two ranges in HSV (wrap around 0-180)
        self.red_lower1 = np.array([0, 70, 50])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([170, 70, 50])
        self.red_upper2 = np.array([180, 255, 255])
        
        self.yellow_lower = np.array([15, 70, 50])
        self.yellow_upper = np.array([35, 255, 255])
        
        self.green_lower = np.array([40, 70, 50])
        self.green_upper = np.array([90, 255, 255])

    def classify(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> str:
        """
        Crop the traffic light and determine color.
        """
        x1, y1, x2, y2 = bbox
        # Crop with boundary checks
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "Unknown"
            
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Count non-zero pixels for each color mask
        mask_r1 = cv2.inRange(hsv_roi, self.red_lower1, self.red_upper1)
        mask_r2 = cv2.inRange(hsv_roi, self.red_lower2, self.red_upper2)
        mask_red = cv2.bitwise_or(mask_r1, mask_r2)
        
        mask_yellow = cv2.inRange(hsv_roi, self.yellow_lower, self.yellow_upper)
        mask_green = cv2.inRange(hsv_roi, self.green_lower, self.green_upper)
        
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Determine dominant color (must be above threshold)
        threshold = 10 # Min pixels
        
        scores = {"Red": red_pixels, "Yellow": yellow_pixels, "Green": green_pixels}
        max_color = max(scores, key=scores.get)
        
        if scores[max_color] < threshold:
            return "Off" # Or unknown
            
        return max_color
