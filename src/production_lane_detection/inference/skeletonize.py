import cv2
import numpy as np
from skimage.morphology import skeletonize

class LanePostProcessor:
    """
    Extracts precise lane lines from Segmentation Masks.
    Technique: Morphological Skeletonization (Thinning).
    """
    
    def __init__(self, lane_class_id: int = 1):
        self.lane_class_id = lane_class_id
        
    def skeletonize_lanes(self, mask: np.ndarray) -> np.ndarray:
        """
        Converts thick segmentation paths into 1-pixel centerlines.
        """
        # 1. Binarize
        binary_mask = (mask == self.lane_class_id).astype(np.uint8)
        
        # 2. Morphological Closing (Fill gaps in dashed lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Skeletonize (Scikit-Image is slightly slower but accurate)
        # For production speed, we can specific cv2 thinning, but skeletonize is robust.
        skeleton = skeletonize(closed_mask) 
        
        return skeleton.astype(np.uint8) * 255

    def extract_lane_points(self, skeleton: np.ndarray):
        """
        Separates points into Left/Right clusters.
        Assumption: Camera is centered.
        """
        h, w = skeleton.shape
        midpoint = w // 2
        
        # Get all non-zero points
        y_idxs, x_idxs = np.nonzero(skeleton)
        
        left_lane = {'x': [], 'y': []}
        right_lane = {'x': [], 'y': []}
        
        for y, x in zip(y_idxs, x_idxs):
            if y < h * 0.5: continue # Ignore horizon/sky
            
            if x < midpoint:
                left_lane['x'].append(x)
                left_lane['y'].append(y)
            else:
                right_lane['x'].append(x)
                right_lane['y'].append(y)
                
        return left_lane, right_lane
