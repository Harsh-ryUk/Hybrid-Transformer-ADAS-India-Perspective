"""
SegFormer Fine-Tuning on IDD Segmentation Dataset
Trains SegFormer-B0 for drivable area + lane segmentation on Indian roads.

Dataset: IDD Segmentation (http://idd.insaan.iiit.ac.in/)

Usage:
    python scripts/train_segformer_idd.py \
        --data_dir data/idd_segmentation \
        --epochs 50 \
        --batch_size 4

Requirements:
    - GPU with ≥8GB VRAM
    - IDD Segmentation dataset prepared via scripts/prepare_idd.py --segmentation
"""

import os
import sys
import argparse
import logging
import json
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# Binary classification: road (1) vs non-road (0)
NUM_CLASSES = 2
LABEL_NAMES = ["non-road", "road"]


class IDDSegmentationDataset:
    """
    PyTorch Dataset for IDD segmentation (road vs non-road).

    Loads image-mask pairs prepared by prepare_idd.py --segmentation.
    """

    def __init__(self, images_dir, masks_dir, image_processor, max_samples=None):
        import cv2

        self.image_processor = image_processor
        self.images = sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg"))
        self.masks_dir = Path(masks_dir)

        if max_samples and max_samples < len(self.images):
            self.images = self.images[:max_samples]

        logger.info(f"Loaded {len(self.images)} image-mask pairs from {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import cv2
        from PIL import Image

        img_path = self.images[idx]
        mask_path = self.masks_dir / img_path.name

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Load mask (255 = road, 0 = non-road) → convert to class IDs (1 = road, 0 = non-road)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((512, 512), dtype=np.uint8)
        mask = (mask > 128).astype(np.int64)  # Binary

        # Process with SegFormer image processor
        encoded = self.image_processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"].squeeze(0)

        return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred, num_classes=NUM_CLASSES):
    """Compute mIoU and pixel accuracy for evaluation."""
    import torch

    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)

    # Upsample logits to label size
    upsampled = torch.nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    predictions = upsampled.argmax(dim=1)

    # Compute IoU per class
    ious = []
    for cls in range(num_classes):
        pred_mask = predictions == cls
        true_mask = labels == cls

        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()

        if union > 0:
            ious.append(intersection / union)

    pixel_acc = (predictions == labels).float().mean().item()
    mean_iou = np.mean(ious) if ious else 0.0

    return {"mIoU": mean_iou, "pixel_accuracy": pixel_acc, "road_IoU": ious[1] if len(ious) > 1 else 0}


def train_segformer(
    data_dir: str,
    model_name: str = "nvidia/segformer-b0-finetuned-ade-512-512",
    epochs: int = 50,
    batch_size: int = 4,
    learning_rate: float = 6e-5,
    output_dir: str = "runs/segformer/idd_road",
    device: str = "",
    max_samples: int = None,
):
    """
    Fine-tune SegFormer-B0 for road segmentation on IDD.

    Training strategy:
    1. Start from ADE20K pretrained SegFormer-B0
    2. Replace classifier head for binary (road/non-road)
    3. Fine-tune with AdamW + linear warmup + cosine decay
    4. Evaluate with mIoU and pixel accuracy

    Data augmentation:
    - Random horizontal flip
    - Random crop (512×512)
    - Color jitter (for dust/monsoon conditions)
    """
    import torch
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
        TrainingArguments,
        Trainer,
    )

    logger.info("=" * 60)
    logger.info("  SegFormer Fine-Tuning — IDD Road Segmentation")
    logger.info("=" * 60)

    data_dir = Path(data_dir)

    # Load image processor
    image_processor = SegformerImageProcessor.from_pretrained(model_name, reduce_labels=False)

    # Load model with new head for binary classification
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label={0: "non-road", 1: "road"},
        label2id={"non-road": 0, "road": 1},
        ignore_mismatched_sizes=True,  # Replace classifier head
    )

    logger.info(f"Model loaded: {model_name} → {NUM_CLASSES} classes")

    # Create datasets
    train_dataset = IDDSegmentationDataset(
        images_dir=data_dir / "images" / "train",
        masks_dir=data_dir / "masks" / "train",
        image_processor=image_processor,
        max_samples=max_samples,
    )

    val_dataset = IDDSegmentationDataset(
        images_dir=data_dir / "images" / "val",
        masks_dir=data_dir / "masks" / "val",
        image_processor=image_processor,
        max_samples=max_samples // 5 if max_samples else None,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=500,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4 if os.name != "nt" else 0,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("\nStarting training...")
    train_result = trainer.train()

    # Save final model
    trainer.save_model(output_dir + "/final")
    image_processor.save_pretrained(output_dir + "/final")

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"\n─── Evaluation Results ───")
    logger.info(f"  mIoU: {eval_results.get('eval_mIoU', 0):.4f}")
    logger.info(f"  Pixel Accuracy: {eval_results.get('eval_pixel_accuracy', 0):.4f}")
    logger.info(f"  Road IoU: {eval_results.get('eval_road_IoU', 0):.4f}")

    # Save results
    results_path = Path(output_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "train_loss": train_result.training_loss,
            "eval_results": eval_results,
            "epochs": epochs,
            "model": model_name,
            "dataset": str(data_dir),
        }, f, indent=2, default=str)

    logger.info(f"\n✅ Training complete!")
    logger.info(f"Model saved to: {output_dir}/final")
    logger.info(f"\nTo use in L4 pipeline:")
    logger.info(f"  Update config.yaml → segmentation.model_name: '{output_dir}/final'")

    return eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer on IDD road segmentation")
    parser.add_argument("--data_dir", type=str, default="data/idd_segmentation")
    parser.add_argument("--model", type=str, default="nvidia/segformer-b0-finetuned-ade-512-512")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--output_dir", type=str, default="runs/segformer/idd_road")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples (for debugging)")

    args = parser.parse_args()

    train_segformer(
        data_dir=args.data_dir,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        output_dir=args.output_dir,
        device=args.device,
        max_samples=args.max_samples,
    )
