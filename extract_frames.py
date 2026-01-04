import cv2
import os

video_path = "data/samples/test_drive.mp4"
output_dir = "data/samples/frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
count = 0
while count < 20: # Just get 20 frames for a quick test
    ret, frame = cap.read()
    if not ret: break
    cv2.imwrite(f"{output_dir}/frame_{count:03d}.jpg", frame)
    count += 1
cap.release()
print(f"Extracted {count} frames to {output_dir}")
