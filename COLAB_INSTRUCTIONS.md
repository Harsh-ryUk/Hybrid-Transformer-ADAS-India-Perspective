# ☁️ Run ADAS v2.0 on Google Colab (One-Click)

Since the project is hosted on GitHub, you can pull it directly into Colab.

## 🚀 Quick Start
**Step 1:** Open [Google Colab](https://colab.research.google.com/).
**Step 2:** Click **New Notebook**.
**Step 3:** Change Runtime to GPU (`Runtime` > `Change runtime type` > `T4 GPU`).

## 📜 Copy-Paste Code

### Cell 1: Setup & Install
```python
# 0. Clean old repo (if exists) to ensure fresh code
!rm -rf Hybrid-Transformer-ADAS-India-Perspective

# 1. Clone the Repository
!git clone https://github.com/Harsh-ryUk/Hybrid-Transformer-ADAS-India-Perspective.git
%cd Hybrid-Transformer-ADAS-India-Perspective

# 2. Install Dependencies (Quietly)
!pip install ultralytics transformers onnxruntime-gpu opendatasets -q
!apt-get install libgl1 -y # OpenCV requirement

print("✅ Setup Complete! Downloading Sample Video...")
!python download_indian_sample.py
```

### Cell 2: Run the Demo (Headless Mode)
This runs the pipeline on the sample video (`data/samples/indian_road_sample.mp4`) and saves the output.
```python
# Run Application
!python -m src.adas_pipeline_v2 --source data/samples/indian_road_sample.mp4 --headless --device cuda
```

### Cell 3: Watch the Result
```python
# Display the video right here in the browser
from IPython.display import HTML
from base64 import b64encode

# Path to the output video (created by step 2)
output_path = "output_demo.mp4" 

mp4 = open(output_path,'rb').read()
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
