import cv2
import argparse
import sys
import time
import os

from src.road_damage.detector import RoadDamageDetector
from src.lane_detection.classical_lane import ClassicalLaneDetector
from src.utils.fps import FPSMeter
from src.utils.visualization import draw_dashboard, draw_lanes, draw_damage

def main():
    parser = argparse.ArgumentParser(description="ADAS Road Damage & Lane Detection")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to file)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLOv8 road damage model")
    parser.add_argument("--no-show", action="store_true", help="Run without display (headless)")
    args = parser.parse_args()

    # Initialize Detectors
    try:
        damage_detector = RoadDamageDetector(model_path=args.model)
        lane_detector = ClassicalLaneDetector()
    except Exception as e:
        print(f"[ERROR] Failed to initialize detectors: {e}")
        return

    # Initialize Video Source
    source = args.source
    if source.isdigit():
        source = int(source)
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        return

    # FPS Meter
    fps_meter = FPSMeter()
    
    # Window
    window_name = "ADAS System (Road Damage + Lane)"
    if not args.no_show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("[INFO] Starting ADAS Loop...")
    
    try:
        while True:
            t_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("[INFO] Video stream ended.")
                break

            # 1. Road Damage Detection
            # Returns: boxes (x1,y1,x2,y2), scores, class_ids
            boxes, scores, class_ids = damage_detector.detect(frame)

            # 2. Lane Detection
            # Returns: left_line, right_line
            left_line, right_line = lane_detector.process(frame)

            # 3. Visualization
            # Draw lanes first (background)
            frame_viz = draw_lanes(frame.copy(), left_line, right_line)
            # Draw damage boxes (foreground)
            frame_viz = draw_damage(frame_viz, boxes, scores, class_ids, damage_detector.get_class_names())
            
            # 4. Metrics & Validation
            fps_meter.tick()
            fps = fps_meter.get_fps()
            latency = fps_meter.get_latency_ms()
            
            # Draw Dashboard
            frame_viz = draw_dashboard(frame_viz, fps, latency)

            # Show Frame
            if not args.no_show:
                cv2.imshow(window_name, frame_viz)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):  # ESC or q
                    break
    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete.")

if __name__ == "__main__":
    main()
