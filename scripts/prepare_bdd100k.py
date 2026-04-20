"""
BDD100K Dataset Preparation Script
Downloads and converts BDD100K to YOLO format for diverse driving conditions.

Dataset: https://bdd-data.berkeley.edu/
Purpose: Multi-weather, multi-time training data to complement IDD.

Usage:
    python scripts/prepare_bdd100k.py --bdd_root /path/to/bdd100k --output_dir data/bdd100k_yolo
"""

import os
import json
import shutil
import argparse
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# BDD100K class mapping → our ADAS taxonomy
BDD_TO_YOLO = {
    "person": 0,
    "pedestrian": 0,
    "car": 1,
    "motorcycle": 2,
    "bicycle": 3,
    "bus": 4,
    "truck": 5,
    "traffic light": 8,
    "traffic sign": 9,
    "rider": 10,
    "train": None,  # Skip
    "other vehicle": 1,  # Map to car
    "other person": 0,  # Map to person
    "trailer": 5,  # Map to truck
}

YOLO_CLASS_NAMES = [
    "person", "car", "motorcycle", "bicycle", "bus", "truck",
    "auto-rickshaw", "animal", "traffic_light", "stop_sign", "rider"
]


def convert_bdd100k_to_yolo(bdd_root: str, output_dir: str):
    """
    Convert BDD100K detection labels to YOLO format.

    BDD100K uses JSON annotation format with one JSON per split.
    """
    bdd_root = Path(bdd_root)
    output_dir = Path(output_dir)

    stats = defaultdict(int)

    for split in ("train", "val"):
        # BDD100K structure:
        # {root}/images/100k/{split}/
        # {root}/labels/det_20/{split}/ (per-image JSON) OR
        # {root}/labels/bdd100k_labels_images_{split}.json (single file)
        img_dir = bdd_root / "images" / "100k" / split
        if not img_dir.exists():
            img_dir = bdd_root / "images" / split

        # Try single JSON file first
        label_file = bdd_root / "labels" / f"bdd100k_labels_images_{split}.json"
        if not label_file.exists():
            label_file = bdd_root / "labels" / f"det_{split}.json"
        if not label_file.exists():
            # Try per-image labels directory
            label_dir = bdd_root / "labels" / "det_20" / split
            if label_dir.exists():
                _convert_per_image_labels(label_dir, img_dir, output_dir, split, stats)
                continue
            else:
                logger.warning(f"No labels found for split '{split}'")
                logger.info(f"Expected: {label_file}")
                continue

        out_img_dir = output_dir / "images" / split
        out_lbl_dir = output_dir / "labels" / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{split}] Loading labels from {label_file}...")
        with open(label_file, "r") as f:
            data = json.load(f)

        converted = 0
        for item in data:
            img_name = item.get("name", "")
            labels = item.get("labels", [])

            if not labels:
                continue

            # Get image dimensions (BDD100K is 1280x720)
            img_w, img_h = 1280, 720

            yolo_lines = []
            for label in labels:
                category = label.get("category", "").lower()
                if category not in BDD_TO_YOLO or BDD_TO_YOLO[category] is None:
                    continue

                yolo_cls = BDD_TO_YOLO[category]
                stats[YOLO_CLASS_NAMES[yolo_cls]] += 1

                box2d = label.get("box2d", {})
                if not box2d:
                    continue

                x1 = float(box2d.get("x1", 0))
                y1 = float(box2d.get("y1", 0))
                x2 = float(box2d.get("x2", 0))
                y2 = float(box2d.get("y2", 0))

                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h

                cx = max(0, min(1, cx))
                cy = max(0, min(1, cy))
                w = max(0, min(1, w))
                h = max(0, min(1, h))

                yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if yolo_lines:
                label_name = Path(img_name).stem + ".txt"
                with open(out_lbl_dir / label_name, "w") as f:
                    f.write("\n".join(yolo_lines))

                # Copy image if available
                src_img = img_dir / img_name
                if src_img.exists():
                    dst_img = out_img_dir / img_name
                    if not dst_img.exists():
                        shutil.copy2(src_img, dst_img)

                converted += 1

        logger.info(f"[{split}] Converted {converted} images")

    # Write dataset YAML
    dataset_yaml = output_dir / "dataset.yaml"
    with open(dataset_yaml, "w") as f:
        f.write(f"# BDD100K Dataset — YOLO Format\n")
        f.write(f"# Source: https://bdd-data.berkeley.edu/\n\n")
        f.write(f"path: {output_dir.resolve()}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n\n")
        f.write(f"nc: {len(YOLO_CLASS_NAMES)}\n")
        f.write(f"names: {YOLO_CLASS_NAMES}\n")

    logger.info("\n─── Conversion Statistics ───")
    for cls_name, count in sorted(stats.items(), key=lambda x: -x[1]):
        logger.info(f"  {cls_name}: {count}")
    logger.info(f"  Total: {sum(stats.values())}")
    logger.info(f"\nDataset YAML: {dataset_yaml}")

    return str(dataset_yaml)


def _convert_per_image_labels(label_dir, img_dir, output_dir, split, stats):
    """Handle per-image JSON format (newer BDD100K releases)."""
    out_img_dir = output_dir / "images" / split
    out_lbl_dir = output_dir / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(label_dir.glob("*.json"))
    logger.info(f"[{split}] Found {len(json_files)} per-image label files")

    converted = 0
    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        img_w, img_h = 1280, 720
        yolo_lines = []

        labels = data if isinstance(data, list) else data.get("labels", [])
        for label in labels:
            category = label.get("category", "").lower()
            if category not in BDD_TO_YOLO or BDD_TO_YOLO[category] is None:
                continue

            yolo_cls = BDD_TO_YOLO[category]
            stats[YOLO_CLASS_NAMES[yolo_cls]] += 1

            box2d = label.get("box2d", {})
            if not box2d:
                continue

            x1, y1 = float(box2d["x1"]), float(box2d["y1"])
            x2, y2 = float(box2d["x2"]), float(box2d["y2"])

            cx = max(0, min(1, ((x1 + x2) / 2) / img_w))
            cy = max(0, min(1, ((y1 + y2) / 2) / img_h))
            w = max(0, min(1, (x2 - x1) / img_w))
            h = max(0, min(1, (y2 - y1) / img_h))

            yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            with open(out_lbl_dir / (jf.stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines))

            img_name = jf.stem + ".jpg"
            src = img_dir / img_name
            if src.exists() and not (out_img_dir / img_name).exists():
                shutil.copy2(src, out_img_dir / img_name)
            converted += 1

    logger.info(f"[{split}] Converted {converted} images")


def download_instructions():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║            BDD100K Dataset — Download Guide                     ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  1. Go to: https://bdd-data.berkeley.edu/                       ║
║  2. Register for a free account                                  ║
║  3. Download:                                                    ║
║     - "100K Images" (training + validation)                      ║
║     - "Detection 2020 Labels"                                    ║
║  4. Extract to: data/bdd100k/                                    ║
║                                                                  ║
║  Expected structure:                                             ║
║    data/bdd100k/                                                 ║
║    ├── images/100k/                                              ║
║    │   ├── train/                                                ║
║    │   └── val/                                                  ║
║    └── labels/                                                   ║
║        └── bdd100k_labels_images_train.json                      ║
║                                                                  ║
║  After download:                                                 ║
║    python scripts/prepare_bdd100k.py \                           ║
║        --bdd_root data/bdd100k \                                 ║
║        --output_dir data/bdd100k_yolo                            ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BDD100K for ADAS L4 training")
    parser.add_argument("--bdd_root", type=str, help="Path to bdd100k root")
    parser.add_argument("--output_dir", type=str, default="data/bdd100k_yolo")
    parser.add_argument("--info", action="store_true", help="Print download instructions")

    args = parser.parse_args()

    if args.info or not args.bdd_root:
        download_instructions()
    else:
        convert_bdd100k_to_yolo(args.bdd_root, args.output_dir)
