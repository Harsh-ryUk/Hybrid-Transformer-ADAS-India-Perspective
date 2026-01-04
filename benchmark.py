import time
import cv2
import glob
import os
import argparse
import numpy as np
from src.road_damage.detector import RoadDamageDetector
from src.lane_detection.classical_lane import ClassicalLaneDetector

def run_benchmark(source_dir, model_path="yolov8n.pt"):
    print(f"Starting Benchmark on: {source_dir}")
    print("-" * 50)
    
    # Load Models
    damage_detector = RoadDamageDetector(model_path=model_path)
    lane_detector = ClassicalLaneDetector()
    
    # Get Images
    types = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for files in types:
        image_paths.extend(glob.glob(os.path.join(source_dir, files)))
    
    if not image_paths:
        print("[ERROR] No images found in directory.")
        return

    print(f"Processing {len(image_paths)} images...")
    
    latencies = []
    
    # Warmup
    print("Warming up...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        damage_detector.detect(dummy)
    
    # Run Loop
    for img_path in image_paths:
        frame = cv2.imread(img_path)
        if frame is None: continue
        
        t0 = time.time()
        
        # Inference Pipeline
        damage_detector.detect(frame)
        lane_detector.process(frame)
        
        t1 = time.time()
        latencies.append((t1 - t0) * 1000) # ms
        
    # Stats
    avg_latency = np.mean(latencies)
    p99_latency = np.percentile(latencies, 99)
    fps = 1000.0 / avg_latency
    
    print("-" * 50)
    print(f"BENCHMARK RESULTS ({len(image_paths)} samples)")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"99th % Latency : {p99_latency:.2f} ms")
    print(f"Throughput     : {fps:.2f} FPS")
    print("-" * 50)
    
    # Save to CSV
    with open("benchmark_results.csv", "w") as f:
        f.write("metric,value\n")
        f.write(f"avg_latency_ms,{avg_latency:.2f}\n")
        f.write(f"p99_latency_ms,{p99_latency:.2f}\n")
        f.write(f"throughput_fps,{fps:.2f}\n")
    print("Results saved to benchmark_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory of images to benchmark")
    args = parser.parse_args()
    
    run_benchmark(args.dir)
