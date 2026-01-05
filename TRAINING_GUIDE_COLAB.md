# 🎓 Industry-Grade Model Training Guide (Google Colab)

**Goal**: Transform the V2 Prototype into a Production Model (99% Accuracy).
**Platform**: Google Colab Pro (or T4 GPU).

---

## Phase 1: The "Industry Standard" Setup
Real ADAS models are not trained on your laptop. They are trained on Cloud GPUs with large datasets stored in Object Storage (Google Drive / S3).

### Step 1.1: Mount Google Drive
In Colab, you don't want to re-upload 10GB of data every time.
```python
from google.colab import drive
drive.mount('/content/drive')
# Result: Your Drive is now at /content/drive/MyDrive/
```

### Step 1.2: Persistent Workspace
Create a folder in Drive for your project so checkpoints are saved forever.
```python
import os
os.makedirs('/content/drive/MyDrive/ADAS_Training/datasets', exist_ok=True)
%cd /content/drive/MyDrive/ADAS_Training
```

---

## Phase 2: Data Engineering (The Real Work)
Industry models are good because of **Data Quality**, not just code.

### Step 2.1: Download "The Big Three"
You must manually download these (license terms) and upload them to `ADAS_Training/datasets/`.

1.  **IDD (Indian Driving Dataset)**: [Download](http://idd.insaan.iiit.ac.in/)
    *   *Why?* Contains Auto-rickshaws, Cows, and unstructured lanes.
2.  **RDD-2022 (India Subset)**: [Download](https://github.com/sekilab/RoadDamageDetector/)
    *   *Why?* Real Indian potholes and cracks.
3.  **Your Own Failures (Active Learning)**:
    *   *Crucial Step*: Take the `output_demo.mp4` where the model failed.
    *   Extract frames: `ffmpeg -i video.mp4 -vf fps=1 data/my_failures/img_%d.jpg`
    *   Label them using [Roboflow](https://roboflow.com/) or [CVAT](https://github.com/opencv/cvat).
    *   **This is how Tesla improves.** They find failures, label them, and re-train.

---

## Phase 3: Fine-Tuning Strategy (Transfer Learning)

Don't train from scratch. Use **Transfer Learning** to save GPU hours.

### 3.1 Training YOLOv11 (Road Damage)
```python
from ultralytics import YOLO

# 1. Load Pre-trained (COCO) - already knows what a "Car" is
model = YOLO('yolov8n.pt') 

# 2. Train on Indian Data (IDD + RDD)
# freeze=10 -> Freezes the first 10 layers (Feature Extractor)
# We only train the "Head" to understand Potholes vs cracks.
results = model.train(
    data='idd_config.yaml', 
    epochs=50, 
    imgsz=640, 
    batch=16,
    freeze=10, 
    project='/content/drive/MyDrive/ADAS_Models'
)
```

### 3.2 Training SegFormer (Lanes)
SegFormer training is heavier. Use HuggingFace `Trainer`.
```python
from transformers import SegformerForSemanticSegmentation, TrainingArguments, Trainer

# Load Pre-trained ADE20k model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0", 
    num_labels=2, # Road vs Background
    ignore_mismatched_sizes=True
)

# Unfreeze ONLY the Decoder (Head)
for name, param in model.segformer.encoder.named_parameters():
    param.requires_grad = False 
    
# Train
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='/content/drive/MyDrive/SegFormer_India',
        learning_rate=2e-4,
        per_device_train_batch_size=8,
        num_train_epochs=10,
    ),
    train_dataset=indian_lane_dataset, # Your loaded IDD dataset
)
trainer.train()
```

---

## Phase 4: Validating for Production (The "Edge Test")

Once trained, an Industry Engineer validates on **Unseen Data**.

1.  **Hold-out Set**: Keep 10% of your data separate. Never train on it.
2.  **Confusion Matrix**: Don't just look at Accuracy. Look at:
    *   **False Positives**: Did it detect a shadow as a pothole? (Annoying driver experience)
    *   **False Negatives**: Did it miss a pedestrian? (Fatal safety issue)
3.  **Edge Export**: Finally, export to TensorRT for the car.
    ```python
    model.export(format='engine', device=0, half=True) # FP16 for Jetson
    ```

## 🚀 Summary Checklist
- [ ] Mount Google Drive.
- [ ] Upload IDD/RDD Datasets.
- [ ] Label your own "Hard Case" images using Roboflow.
- [ ] Run `train.py` with `freeze=10` (Transfer Learning).
- [ ] Export `.engine` file for Jetson.
