"""
IDD (Indian Driving Dataset) Preparation Script
Downloads and converts IDD to YOLO format for fine-tuning.

Dataset: http://idd.insaan.iiit.ac.in/
Classes: 30 (including auto-rickshaw, animal, rider, etc.)

Usage:
    python scripts/prepare_idd.py --idd_root /path/to/IDD_Detection --output_dir data/idd_yolo
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── IDD Class Mapping ───
# IDD has 30 classes. We map them to our ADAS class taxonomy.
IDD_CLASSES = [
    "person",           # 0
    "rider",            # 1
    "car",              # 2
    "truck",            # 3
    "bus",              # 4
    "train",            # 5
    "motorcycle",       # 6
    "bicycle",          # 7
    "autorickshaw",     # 8
    "animal",           # 9
    "traffic light",    # 10
    "traffic sign",     # 11
    "vehicle fallback", # 12
    "caravan",          # 13
    "trailer",          # 14
    "on rails",         # 15
]

# Map IDD class index → YOLO class index (our taxonomy)
# Our classes: 0=person, 1=car, 2=motorcycle, 3=bicycle, 4=bus, 5=truck,
#              6=auto-rickshaw, 7=animal, 8=traffic_light, 9=stop_sign, 10=rider
IDD_TO_YOLO = {
    0: 0,    # person → person
    1: 10,   # rider → rider
    2: 1,    # car → car
    3: 5,    # truck → truck
    4: 4,    # bus → bus
    6: 2,    # motorcycle → motorcycle
    7: 3,    # bicycle → bicycle
    8: 6,    # autorickshaw → auto-rickshaw
    9: 7,    # animal → animal
    10: 8,   # traffic light → traffic_light
    11: 9,   # traffic sign → stop_sign
    12: 1,   # vehicle fallback → car
}

YOLO_CLASS_NAMES = [
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "auto-rickshaw", "animal", "traffic_light", "stop_sign", "rider"
]


def convert_idd_to_yolo(idd_root: str, output_dir: str, splits=("train", "val")):
    """
    Convert IDD Detection format to YOLO format.

    IDD format: XML annotations (PASCAL VOC style)
    YOLO format: .txt files with (class_id, cx, cy, w, h) normalized

    Args:
        idd_root: Root path of IDD_Detection dataset
        output_dir: Output directory for YOLO format
        splits: Which splits to convert
    """
    import xml.etree.ElementTree as ET

    idd_root = Path(idd_root)
    output_dir = Path(output_dir)

    stats = defaultdict(int)

    for split in splits:
        # IDD structure: {root}/Annotations/{split}/, {root}/JPEGImages/{split}/
        ann_dir = idd_root / "Annotations" / split
        img_dir = idd_root / "JPEGImages" / split

        if not ann_dir.exists():
            # Try alternate structure
            ann_dir = idd_root / split / "Annotations"
            img_dir = idd_root / split / "JPEGImages"

        if not ann_dir.exists():
            logger.warning(f"Annotation directory not found for split '{split}': {ann_dir}")
            logger.info("Expected IDD structure:")
            logger.info(f"  {idd_root}/Annotations/{split}/")
            logger.info(f"  {idd_root}/JPEGImages/{split}/")
            continue

        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        xml_files = list(ann_dir.rglob("*.xml"))
        logger.info(f"[{split}] Found {len(xml_files)} annotation files")

        converted = 0
        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                # Get image dimensions
                size = root.find("size")
                if size is None:
                    continue
                img_w = int(size.find("width").text)
                img_h = int(size.find("height").text)

                if img_w == 0 or img_h == 0:
                    continue

                # Convert annotations
                yolo_lines = []
                for obj in root.findall("object"):
                    name = obj.find("name").text.lower().strip()

                    # Find IDD class index
                    idd_idx = None
                    for i, cls_name in enumerate(IDD_CLASSES):
                        if cls_name.lower() == name:
                            idd_idx = i
                            break

                    if idd_idx is None or idd_idx not in IDD_TO_YOLO:
                        continue

                    yolo_cls = IDD_TO_YOLO[idd_idx]
                    stats[YOLO_CLASS_NAMES[yolo_cls]] += 1

                    bbox = obj.find("bndbox")
                    xmin = float(bbox.find("xmin").text)
                    ymin = float(bbox.find("ymin").text)
                    xmax = float(bbox.find("xmax").text)
                    ymax = float(bbox.find("ymax").text)

                    # Convert to YOLO format (normalized center x, y, w, h)
                    cx = ((xmin + xmax) / 2) / img_w
                    cy = ((ymin + ymax) / 2) / img_h
                    w = (xmax - xmin) / img_w
                    h = (ymax - ymin) / img_h

                    # Clamp
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))

                    yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

                if yolo_lines:
                    # Write label file
                    label_name = xml_file.stem + ".txt"
                    with open(out_lbl_dir / label_name, "w") as f:
                        f.write("\n".join(yolo_lines))

                    # Copy/link image
                    img_name = xml_file.stem + ".jpg"
                    src_img = img_dir / xml_file.relative_to(ann_dir).parent / img_name

                    if not src_img.exists():
                        # Try PNG
                        src_img = src_img.with_suffix(".png")

                    if src_img.exists():
                        dst_img = out_img_dir / img_name
                        if not dst_img.exists():
                            shutil.copy2(src_img, dst_img)
                        converted += 1

            except Exception as e:
                logger.debug(f"Error processing {xml_file}: {e}")
                continue

        logger.info(f"[{split}] Converted {converted} images")

    # Write YOLO dataset config
    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"# IDD Dataset — YOLO Format\n")
        f.write(f"# Converted from Indian Driving Dataset\n")
        f.write(f"# Source: http://idd.insaan.iiit.ac.in/\n\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(YOLO_CLASS_NAMES)}\n")
        f.write(f"names: {YOLO_CLASS_NAMES}\n")

    # Print stats
    logger.info("\n─── Conversion Statistics ───")
    for cls_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls_name}: {count}")
    logger.info(f"  Total annotations: {sum(stats.values())}")
    logger.info(f"\nDataset YAML saved to: {dataset_yaml}")

    return str(dataset_yaml)


def download_idd_instructions():
    """Print instructions for downloading IDD dataset."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           Indian Driving Dataset (IDD) — Download Guide         ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: http://idd.insaan.iiit.ac.in/                       ║
║  2. Register for an account (academic email recommended)         ║
║  3. Download "IDD Detection" dataset                             ║
║  4. Extract to: data/IDD_Detection/                              ║
║                                                                  ║
║  Expected structure after extraction:                            ║
║    data/IDD_Detection/                                           ║
║    ├── Annotations/                                              ║
║    │   ├── train/                                                ║
║    │   └── val/                                                  ║
║    ├── JPEGImages/                                               ║
║    │   ├── train/                                                ║
║    │   └── val/                                                  ║
║    └── ImageSets/                                                ║
║                                                                  ║
║  After download, run:                                            ║
║    python scripts/prepare_idd.py \\                               ║
║        --idd_root data/IDD_Detection \\                           ║
║        --output_dir data/idd_yolo                                ║
║                                                                  ║
║  Alternative: Use IDD Segmentation for SegFormer fine-tuning     ║
║    Download "IDD Segmentation" and run:                          ║
║    python scripts/prepare_idd.py --segmentation \\                ║
║        --idd_root data/IDD_Segmentation                          ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


def prepare_idd_segmentation(idd_seg_root: str, output_dir: str):
    """
    Prepare IDD Segmentation dataset for SegFormer fine-tuning.

    Converts IDD segmentation labels to binary road/non-road masks.
    """
    import cv2
    import numpy as np

    idd_seg_root = Path(idd_seg_root)
    output_dir = Path(output_dir)

    # IDD Segmentation label IDs for road-related classes
    ROAD_LABEL_IDS = [0, 1, 2]  # road, parking, drivable fallback

    for split in ("train", "val"):
        img_dir = idd_seg_root / "leftImg8bit" / split
        lbl_dir = idd_seg_root / "gtFine" / split

        if not img_dir.exists():
            logger.warning(f"Image directory not found: {img_dir}")
            continue

        out_img = output_dir / "images" / split
        out_mask = output_dir / "masks" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_mask.mkdir(parents=True, exist_ok=True)

        images = list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpg"))
        logger.info(f"[{split}] Found {len(images)} images")

        for img_path in images:
            # Find corresponding label
            rel = img_path.relative_to(img_dir)
            lbl_name = img_path.stem.replace("_leftImg8bit", "_gtFine_labelids") + ".png"
            lbl_path = lbl_dir / rel.parent / lbl_name

            if not lbl_path.exists():
                continue

            # Create binary road mask
            label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
            if label is None:
                continue

            road_mask = np.zeros_like(label)
            for lid in ROAD_LABEL_IDS:
                road_mask[label == lid] = 255

            # Save
            shutil.copy2(img_path, out_img / img_path.name)
            cv2.imwrite(str(out_mask / img_path.name), road_mask)

        logger.info(f"[{split}] Saved to {out_img}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IDD dataset for ADAS L4 training")
    parser.add_argument("--idd_root", type=str, help="Path to IDD_Detection root directory")
    parser.add_argument("--output_dir", type=str, default="data/idd_yolo", help="Output directory")
    parser.add_argument("--segmentation", action="store_true", help="Prepare segmentation dataset instead")
    parser.add_argument("--info", action="store_true", help="Print download instructions")

    args = parser.parse_args()

    if args.info or not args.idd_root:
        download_idd_instructions()
    elif args.segmentation:
        prepare_idd_segmentation(args.idd_root, args.output_dir)
    else:
        convert_idd_to_yolo(args.idd_root, args.output_dir)
