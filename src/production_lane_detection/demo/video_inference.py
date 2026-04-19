import cv2
import argparse
import time
import numpy as np
from src.production_lane_detection.inference.preprocess import Preprocessor
from src.production_lane_detection.inference.trt_infer import TRTInference
from src.production_lane_detection.inference.skeletonize import LanePostProcessor
from src.production_lane_detection.inference.curve_fit import CurveFitter
from src.production_lane_detection.inference.temporal_filter import TemporalFilter

def run_demo(video_path, model_path):
    # 1. Init Modules
    preprocessor = Preprocessor(input_size=(640, 360), use_clahe=True)
    engine = TRTInference(model_path)
    postprocessor = LanePostProcessor(lane_class_id=1) # 1 = Drivable/Lane depending on model training
    fitter = CurveFitter()
    smoother = TemporalFilter()
    
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        t0 = time.time()
        
        # 2. Preprocess
        tensor = preprocessor.run(frame)
        
        # 3. Infer
        mask = engine.infer(tensor)
        
        # 4. Skeletonize
        skeleton = postprocessor.skeletonize_lanes(mask)
        
        # 5. Extract Points
        left_l, right_l = postprocessor.extract_lane_points(skeleton)
        
        # 6. Fit Curves (RANSAC)
        h, w = frame.shape[:2]
        left_fit = fitter.fit(left_l['x'], left_l['y'], h, w)
        right_fit = fitter.fit(right_l['x'], right_l['y'], h, w)
        
        # 7. Viz
        viz = cv2.resize(frame, (640, 360))
        
        # Draw Skeleton (Red)
        viz[skeleton > 0] = [0, 0, 255]
        
        # Draw Fits (Green)
        if left_fit:
            cv2.polylines(viz, [left_fit['points']], False, (0, 255, 0), 3)
        if right_fit:
            cv2.polylines(viz, [right_fit['points']], False, (0, 255, 0), 3)
            
        fps = 1.0 / (time.time() - t0)
        cv2.putText(viz, f"FPS: {fps:.1f} | TRT-Ready", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Production Lane System", viz)
        if cv2.waitKey(1) == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/samples/indian_road_sample.mp4")
    parser.add_argument("--model", default="src/production_lane_detection/models/segformer_b0.onnx")
    args = parser.parse_args()
    
    run_demo(args.source, args.model)
