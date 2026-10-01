"""Freeze the official IDD Lite validation image/mask pairs, without copying data."""
import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_pipeline import digest
from src.evaluation.ground_truth import binary_road_counts


def prepare(root, output, limit=None, seed=42):
    root = Path(root).resolve()
    masks = sorted((root / 'gtFine/val').rglob('*_label.png'))
    masks = [path for path in masks if not path.name.endswith('_inst_label.png')]
    if not masks or (limit is not None and limit < 1):
        raise ValueError('Expected official IDD Lite gtFine/val/*/*_label.png; limit must be positive')
    candidates = []
    for mask in masks:
        relative = mask.relative_to(root / 'gtFine/val')
        image = root / 'leftImg8bit/val' / relative.parent / (mask.name.removesuffix('_label.png') + '_image.jpg')
        if not image.is_file():
            raise ValueError(f'Missing paired validation image: {image}')
        candidates.append((image, mask, str(relative.with_suffix(''))))
    expected_images = set((root / 'leftImg8bit/val').rglob('*_image.jpg'))
    if {image for image, _, _ in candidates} != expected_images:
        raise ValueError('Unpaired validation images or duplicate masks')
    if limit is not None and limit < len(candidates):
        candidates = sorted(random.Random(seed).sample(candidates, limit), key=lambda item: item[2])
    schema = {'encoding': 'official IDD Lite level1Id', 'road_ids': [0],
              'valid_ids': list(range(7)), 'ignore_ids': [255],
              'target': 'drivable area: road, parking and drivable fallback'}
    samples = []
    for image, mask, sample_id in candidates:
        frame, labels = cv2.imread(str(image)), cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
        if frame is None or labels is None:
            raise ValueError(f'Unreadable image/mask: {sample_id}')
        binary_road_counts(np.zeros(frame.shape[:2], np.uint8), labels,
                           schema['road_ids'], schema['valid_ids'], schema['ignore_ids'])
        samples.append({'id': sample_id, 'image': str(image), 'mask': str(mask),
                        'image_sha256': digest(image), 'mask_sha256': digest(mask)})
    manifest = {'schema_version': 1, 'dataset': 'IDD Lite; https://idd.insaan.iiit.ac.in/',
                'split': 'val', 'label_schema': schema,
                'selection': {'available_images': len(masks), 'selected_images': len(samples),
                              'policy': 'all validation images' if len(samples) == len(masks) else 'seeded random subset before inference',
                              'seed': seed, 'drives': sorted({s['id'].split('/')[0] for s in samples})},
                'samples': samples}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Extracted idd20k_lite directory')
    parser.add_argument('--output', default=str(ROOT / 'runs/idd_lite_val_manifest.json'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    manifest = prepare(args.root, args.output, args.limit, args.seed)
    print(json.dumps(manifest['selection'], indent=2))
    print(f'Manifest SHA-256: {digest(args.output)}')


if __name__ == '__main__':
    main()
