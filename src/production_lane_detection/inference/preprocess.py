import cv2
import numpy as np
import torch
from typing import Tuple

class Preprocessor:
    """
    Production Preprocessing Pipeline:
    1. Resize (640x360)
    2. CLAHE (Adaptive Histogram Eq)
    3. Normalization (ImageNet)
    4. NCHW Conversion
    """
    
    def __init__(self, input_size: Tuple[int, int] = (640, 360), use_clahe: bool = False):
        self.input_size = input_size
        self.use_clahe = use_clahe
        
        # ImageNet Mean/Std
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # CLAHE (For night driving / shadows)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def run(self, frame: np.ndarray) -> np.ndarray:
        # 1. Resize
        frame_resized = cv2.resize(frame, self.input_size)
        
        # 2. CLAHE (Optional - Good for Indian Roads)
        if self.use_clahe:
            lab = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self.clahe.apply(l)
            lab = cv2.merge((l2, a, b))
            frame_resized = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
        # 3. Convert to Float32 & Normalize
        img = frame_resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        
        # 4. HWC -> CHW (Channels First)
        img = img.transpose(2, 0, 1)
        
        # 5. Batch Dimension (1, C, H, W)
        img = np.expand_dims(img, axis=0)
        
        return img.astype(np.float32)
