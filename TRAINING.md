# 🇮🇳 Comprehensive Guide: Training for Indian Roads

To build a robust ADAS system for India, you must train on data that captures **unstructured traffic**, **faded road markings**, **diverse weather (monsoon/dust)**, and **specific road damage**.

We have prepared the system to use the "Big Three" Indian datasets.

## 1. The Datasets
You need to download these manually (due to size/license) and place them in the `datasets/` folder.

| Dataset | Type | Why it's critical | Size | Link |
| :--- | :--- | :--- | :--- | :--- |
| **RDD2022 (India Subset)** | Damage | Contains specific Indian potholes, cracks, and poor asphalt conditions. | ~2 GB | [Download](https://github.com/sekilab/RoadDamageDetector/) |
| **IDD (Indian Driving Dataset)** | Traffic/Objects | The standard for Indian traffic. Includes **Auto-rickshaws, Animals, and unstructured lanes**. Captured in Hyderabad/Bangalore. | ~30 GB | [Download](http://idd.insaan.iiit.ac.in/) |
| **Mapillary Vistas (India)** | Diverse/Street | High-res street view images, often includes diverse lighting/weather. | Varies | [Download](https://www.mapillary.com/dataset/vistas) |

### 📂 Required Folder Structure
```text
ADAS-Road-Damage-Lane-Detection/
├── datasets/ (Create this sibling folder)
│   ├── RDD2022_India/
│   └── IDD_Detection/
└── ADAS-.. (This project)
```

## 2. Training Configurations
We have pre-written the configuration files for you.

### Scenario A: Detecting Potholes (Road Safety)
Use the **RDD** config.
```bash
python train.py --data data/dataset.yaml --epochs 100 --name indian_pothole_v1
```

### Scenario B: Detecting Traffic/Autos (Traffic Awareness)
Use the **IDD** config.
```bash
python train.py --data data/idd_config.yaml --epochs 100 --name indian_traffic_v1
```

### Scenario C: Combined Model (Advanced)
To detect *everything* (Potholes + Cars + Lanes), you should merge the datasets.
1. Create a `data/combined_config.yaml`.
2. Add both image directories to the `train:` list in `combined_config.yaml`.
3. Train with `--epochs 300` for best results.

## 3. Handling Weather & Climate
IDD contains specific metadata for weather.
- **Dust/Haze**: Common in the dataset.
- **Rain/Monsoon**: IDD has a "Rainy" subset. Ensure you include the `IDD Amodal` or specific weather splits if available in your download.

**Pro Tip**: To improve robustness against **Night** and **Rain**, apply *Data Augmentation* in `train.py`.
YOLOv8 automatically applies mild augmentation, but for Indian conditions, verify `hsv_h`, `hsv_s`, and `mosaic` parameters are enabled (default is yes).
