import os
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Reliable Pexels or similar direct video link for walking people/traffic
# This is a Pexels video of Indian Traffic (if available) or similar busy street
# Fallback to a clear reliable URL
URL = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4" 

OUTPUT_DIR = "data/samples"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "indian_road_sample.mp4")

def download_file():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Downloading sample video to {OUTPUT_FILE}...")
    try:
        urllib.request.urlretrieve(URL, OUTPUT_FILE)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")

if __name__ == "__main__":
    download_file()
