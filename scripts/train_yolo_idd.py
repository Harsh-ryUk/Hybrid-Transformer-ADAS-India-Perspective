"""
YOLOv8 Fine-Tuning on Indian Driving Dataset (IDD)
Trains the India-aware object detector on real Indian road data.

Supports:
- IDD (primary — Indian roads)
- BDD100K (secondary — weather diversity)
- Custom dashcam data (annotated via CVAT)

Usage:
    # Fine-tune on IDD
    python scripts/train_yolo_idd.py --data data/idd_yolo/dataset.yaml --epochs 100

    # Fine-tune on BDD100K
    python scripts/train_yolo_idd.py --data data/bdd100k_yolo/dataset.yaml --epochs 50

    # Resume training
    python scripts/train_yolo_idd.py --resume runs/detect/idd_train/weights/last.pt

Requirements:
    - GPU with ≥6GB VRAM (RTX 3060+ recommended)
    - IDD dataset prepared via scripts/prepare_idd.py
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_yolo(
    data_yaml: str,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    project: str = "runs/detect",
    name: str = "idd_train",
    device: str = "",
    resume: str = None,
    patience: int = 20,
    lr0: float = 0.01,
    augment: bool = True,
    freeze_backbone: int = 0,
):
    """
    Fine-tune YOLOv8 on Indian driving dataset.

    Training strategy:
    1. Start from COCO pretrained YOLOv8n (already knows cars, people, etc.)
    2. Fine-tune on IDD to learn India-specific classes (autorickshaw, animal)
    3. Use strong augmentations for robustness (monsoon, dust, night)
    4. Early stopping with patience=20 to prevent overfitting

    Augmentation strategy for Indian roads:
    - Heavy mosaic (0.8) — handles crowded scenes
    - HSV augmentation — handles varying lighting (sun glare, night)
    - Scale variation — handles near/far objects on chaotic roads
    - Mix-up (0.1) — improves generalization
    """
    from ultralytics import YOLO

    if resume:
        logger.info(f"Resuming training from: {resume}")
        model_obj = YOLO(resume)
        model_obj.train(resume=True)
        return

    logger.info("=" * 60)
    logger.info("  YOLOv8 Fine-Tuning — Indian Driving Dataset")
    logger.info("=" * 60)
    logger.info(f"  Base model: {model}")
    logger.info(f"  Dataset: {data_yaml}")
    logger.info(f"  Epochs: {epochs}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Image size: {img_size}")
    logger.info(f"  Device: {device or 'auto'}")
    logger.info("=" * 60)

    # Load pretrained model
    model_obj = YOLO(model)

    # India-specific augmentation settings
    augmentation_params = {}
    if augment:
        augmentation_params = {
            "hsv_h": 0.015,     # Hue variation
            "hsv_s": 0.7,       # Saturation variation (handles dust/fog)
            "hsv_v": 0.4,       # Value variation (handles shadows/glare)
            "degrees": 0.0,     # No rotation (driving videos are level)
            "translate": 0.1,
            "scale": 0.5,       # Scale variation for near/far objects
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,      # No vertical flip (unnatural)
            "fliplr": 0.5,      # Horizontal flip OK
            "mosaic": 0.8,      # Strong mosaic for dense scenes
            "mixup": 0.1,       # Mild mixup for generalization
            "copy_paste": 0.1,  # Copy-paste augmentation
        }

    # Train
    results = model_obj.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=project,
        name=name,
        device=device if device else None,
        patience=patience,
        lr0=lr0,
        lrf=0.01,            # Final LR = lr0 * lrf
        warmup_epochs=3,
        warmup_momentum=0.8,
        weight_decay=0.0005,
        optimizer="AdamW",
        freeze=freeze_backbone,
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Augmentations
        **augmentation_params,
    )

    # Validate
    logger.info("\n─── Running Validation ───")
    val_results = model_obj.val()
    logger.info(f"mAP@0.5: {val_results.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95: {val_results.box.map:.4f}")

    # Export to ONNX for deployment
    logger.info("\n─── Exporting to ONNX ───")
    model_obj.export(format="onnx", imgsz=img_size)

    best_path = Path(project) / name / "weights" / "best.pt"
    logger.info(f"\n✅ Training complete!")
    logger.info(f"Best weights: {best_path}")
    logger.info(f"\nTo use the fine-tuned model in the L4 pipeline:")
    logger.info(f"  Update config.yaml → detection.model_path: '{best_path}'")

    return results


def train_multi_dataset(
    idd_yaml: str = "data/idd_yolo/dataset.yaml",
    bdd_yaml: str = "data/bdd100k_yolo/dataset.yaml",
    custom_yaml: str = None,
    epochs_idd: int = 80,
    epochs_bdd: int = 30,
    epochs_custom: int = 20,
):
    """
    Multi-stage training on multiple Indian driving datasets.

    Strategy:
    1. Pre-train on BDD100K (large, diverse, multi-weather)
    2. Fine-tune on IDD (Indian-specific classes)
    3. Final fine-tune on custom dashcam data (most specific)
    """
    from ultralytics import YOLO

    logger.info("=" * 60)
    logger.info("  Multi-Dataset Training Pipeline")
    logger.info("=" * 60)

    current_model = "yolov8n.pt"

    # Stage 1: BDD100K (if available)
    if os.path.exists(bdd_yaml):
        logger.info("\n──── Stage 1: BDD100K Pre-training ────")
        model = YOLO(current_model)
        model.train(
            data=bdd_yaml,
            epochs=epochs_bdd,
            batch=16,
            imgsz=640,
            project="runs/detect",
            name="bdd100k_pretrain",
            patience=15,
        )
        current_model = "runs/detect/bdd100k_pretrain/weights/best.pt"
    else:
        logger.info(f"BDD100K not found at {bdd_yaml}, skipping Stage 1")

    # Stage 2: IDD Fine-tuning
    if os.path.exists(idd_yaml):
        logger.info("\n──── Stage 2: IDD Fine-tuning ────")
        model = YOLO(current_model)
        model.train(
            data=idd_yaml,
            epochs=epochs_idd,
            batch=16,
            imgsz=640,
            project="runs/detect",
            name="idd_finetune",
            patience=20,
            lr0=0.005,  # Lower LR for fine-tuning
        )
        current_model = "runs/detect/idd_finetune/weights/best.pt"
    else:
        logger.info(f"IDD not found at {idd_yaml}, skipping Stage 2")

    # Stage 3: Custom data fine-tuning
    if custom_yaml and os.path.exists(custom_yaml):
        logger.info("\n──── Stage 3: Custom Data Fine-tuning ────")
        model = YOLO(current_model)
        model.train(
            data=custom_yaml,
            epochs=epochs_custom,
            batch=8,
            imgsz=640,
            project="runs/detect",
            name="custom_finetune",
            patience=10,
            lr0=0.001,  # Even lower LR for final stage
        )
        current_model = "runs/detect/custom_finetune/weights/best.pt"

    logger.info(f"\n✅ Multi-dataset training complete!")
    logger.info(f"Final model: {current_model}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Indian driving data")
    parser.add_argument("--data", type=str, default="data/idd_yolo/dataset.yaml")
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--project", type=str, default="runs/detect")
    parser.add_argument("--name", type=str, default="idd_train")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N backbone layers")
    parser.add_argument("--multi", action="store_true", help="Multi-dataset training pipeline")

    args = parser.parse_args()

    if args.multi:
        train_multi_dataset()
    elif args.resume:
        train_yolo(data_yaml=args.data, resume=args.resume)
    else:
        train_yolo(
            data_yaml=args.data,
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            device=args.device,
            project=args.project,
            name=args.name,
            lr0=args.lr,
            patience=args.patience,
            freeze_backbone=args.freeze,
        )
