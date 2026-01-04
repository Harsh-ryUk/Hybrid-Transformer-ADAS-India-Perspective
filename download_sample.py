import urllib.request
import os

url = "https://github.com/udacity/CarND-LaneLines-P1/raw/master/test_videos/solidWhiteRight.mp4"
output = "data/samples/test_drive.mp4"

print(f"Downloading {url} to {output}...")
try:
    urllib.request.urlretrieve(url, output)
    print("Download complete.")
except Exception as e:
    print(f"Error: {e}")
