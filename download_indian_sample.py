import os
import urllib.request
import ssl
import sys

# Allow unverified SSL context for legacy environments
ssl._create_default_https_context = ssl._create_unverified_context

# Reliable URL (Intel IoT DevKit Sample)
URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4" 

OUTPUT_DIR = "data/samples"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "indian_road_sample.mp4")

def download_file():
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
        
    print(f"⬇️ Downloading sample video from: {URL}")
    print(f"   Target: {OUTPUT_FILE}")
    
    try:
        urllib.request.urlretrieve(URL, OUTPUT_FILE)
        
        # Verify download
        if os.path.exists(OUTPUT_FILE):
             size = os.path.getsize(OUTPUT_FILE)
             if size > 1000:
                 print(f"✅ Success! Downloaded {size/1024:.2f} KB.")
             else:
                 print("❌ Error: File downloaded but is empty/corrupt.")
                 sys.exit(1)
        else:
             print("❌ Error: File does not exist after download.")
             sys.exit(1)

    except Exception as e:
        print(f"❌ Failed to download: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_file()
