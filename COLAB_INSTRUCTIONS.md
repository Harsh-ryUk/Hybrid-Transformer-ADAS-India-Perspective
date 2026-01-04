# ☁️ Running ADAS v2.0 on Google Colab

Use this guide to run the project on Google Colab if your local machine is crashing.

## Step 1: Prepare Files
1.  Locate the file `ADAS_Bundle.zip` in your project folder (`.../scratch/ADAS-Road-Damage-Lane-Detection`).
2.  Open [Google Colab](https://colab.research.google.com/).
3.  Click **New Notebook**.
4.  In the left sidebar (Files icon), drag and drop `ADAS_Bundle.zip` to upload it.

## Step 2: Run the Code
Copy and paste the following code blocks into cells and run them in order.

### Cell 1: Setup
```python
# 1. Unzip the project
!unzip -o ADAS_Bundle.zip

# 2. Install dependencies (Quiet mode)
!pip install ultralytics transformers onnxruntime-gpu opendatasets -q
!apt-get install libgl1 -y # Required for OpenCV

print("Setup Complete!")
```

### Cell 2: Run Variable Setup
```python
# Select your video source
# 'data/samples/indian_road_sample.mp4' is included in the zip if you downloaded it locally first.
VIDEO_SOURCE = "data/samples/indian_road_sample.mp4" 
```

### Cell 3: Execute ADAS Pipeline
```python
# Run the pipeline in Headless mode (Saves to output_demo.mp4)
!python -m src.adas_pipeline_v2 --source {VIDEO_SOURCE} --headless --device cuda

print("Processing Complete. Check output_demo.mp4")
```

### Cell 4: View Output
```python
# Display the video in Colab
from IPython.display import HTML
from base64 import b64encode

mp4 = open('output_demo.mp4','rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML(f"""
<video width=640 controls>
      <source src="{data_url}" type="video/mp4">
</video>
""")
```

## Troubleshooting
- **GPU**: Go to `Runtime > Change runtime type` and select **T4 GPU** for faster performance.
- **Memory**: If it crashes on Colab (rare), try restarting the runtime.
