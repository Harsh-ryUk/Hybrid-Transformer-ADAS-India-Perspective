import os
import urllib.request
import ssl
import sys

# Allow unverified SSL context for legacy environments
ssl._create_default_https_context = ssl._create_unverified_context

# List of candidate URLs (Priority: Dashboard > Highway > CCTV)
URLS = [
    # 1. Real Dashcam Footage (City/Road)
    "https://github.com/jgOhYeah/dashcam-viewer/raw/main/src/demo-files/DCIM/100video/20240112181119_180.MP4",
    "https://github.com/jgOhYeah/dashcam-viewer/raw/master/src/demo-files/DCIM/100video/20240112181119_180.MP4",
    # 2. Intel ADAS Sample (Driver/Road)
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
    # 3. Fallback (The one we had)
    "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
] 

OUTPUT_DIR = "data/samples"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "indian_road_sample.mp4")

def download_file():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        
    print(f"⬇️ Attempting to download Dashboard Video...")
    
    success = False
    for url in URLS:
        print(f"   Trying: {url}")
        try:
            urllib.request.urlretrieve(url, OUTPUT_FILE)
            
            # Verify download
            if os.path.exists(OUTPUT_FILE):
                 size = os.path.getsize(OUTPUT_FILE)
                 if size > 5000: # At least 5KB
                     print(f"✅ Success! Downloaded {size/1024:.2f} KB.")
                     success = True
                     break
                 else:
                     print("   ⚠️ File too small/corrupt. Trying next...")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            
    if not success:
        print("❌ All download sources failed.")
        sys.exit(1)

if __name__ == "__main__":
    download_file()
