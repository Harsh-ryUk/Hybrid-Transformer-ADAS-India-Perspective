import cv2
import numpy as np
import os
from ultralytics import YOLO

class RoadDamageDetector:
    def __init__(self, model_path="yolov8n.pt", conf_thres=0.4, iou_thres=0.45):
        """
        Initialize the YOLOv8 detector.
        If model_path is not found, weights will be downloaded automatically by Ultralytics.
        """
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model_path = model_path
        
        print(f"[INFO] Loading Road Damage Model: {model_path}")
        self.model = YOLO(model_path)
        
        # Determine if we are using ONNX
        self.is_onnx = model_path.endswith(".onnx")
        
        # Default classes for generic YOLO (COCO)
        # 0: person, ... 
        # For a dedicated Road Damage dataset, these would be ['pothole', 'crack', etc.]
        # We will use the model's own names
        self.names = self.model.names

    def detect(self, frame):
        """
        Performs inference on a single frame.
        Returns: 
            boxes: list of [x1, y1, x2, y2]
            scores: list of float
            class_ids: list of int
        """
        # Run inference
        results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        
        # Parse results
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        return boxes, scores, class_ids

    def get_class_names(self):
        return self.names
