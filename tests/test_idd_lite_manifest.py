"""Test pairing and frozen inputs using generated fixtures, not IDD data."""
import cv2
import numpy as np
import pytest

from scripts.evaluate_accuracy import load_manifest
from scripts.prepare_idd_lite_eval import prepare


def fixture_dataset(root, count=3):
    images = root / 'leftImg8bit/val/drive'
    masks = root / 'gtFine/val/drive'
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for index in range(count):
        assert cv2.imwrite(str(images / f'{index}_image.jpg'), np.zeros((8, 12, 3), np.uint8))
        assert cv2.imwrite(str(masks / f'{index}_label.png'), np.zeros((8, 12), np.uint8))
        assert cv2.imwrite(str(masks / f'{index}_inst_label.png'), np.zeros((8, 12), np.uint16))


def test_all_pairs_exclude_instance_masks_and_freeze_hashes(tmp_path):
    fixture_dataset(tmp_path)
    output = tmp_path / 'manifest.json'
    result = prepare(tmp_path, output)
    assert result['selection']['available_images'] == 3
    assert result['selection']['selected_images'] == 3
    assert result['label_schema']['road_ids'] == [0]
    assert len(load_manifest(output)['samples']) == 3
    image = tmp_path / 'leftImg8bit/val/drive/0_image.jpg'
    assert cv2.imwrite(str(image), np.ones((8, 12, 3), np.uint8) * 255)
    with pytest.raises(ValueError, match='Changed frozen image'):
        load_manifest(output)


def test_subset_selection_is_seeded_and_reproducible(tmp_path):
    fixture_dataset(tmp_path, count=6)
    a = prepare(tmp_path, tmp_path / 'a.json', limit=2, seed=42)
    b = prepare(tmp_path, tmp_path / 'b.json', limit=2, seed=42)
    assert a == b
    assert a['selection']['selected_images'] == 2
    assert 'subset before inference' in a['selection']['policy']


def test_unpaired_validation_image_fails(tmp_path):
    fixture_dataset(tmp_path)
    image = tmp_path / 'leftImg8bit/val/drive/extra_image.jpg'
    assert cv2.imwrite(str(image), np.zeros((8, 12, 3), np.uint8))
    with pytest.raises(ValueError, match='Unpaired'):
        prepare(tmp_path, tmp_path / 'manifest.json')


def test_missing_paired_validation_image_fails(tmp_path):
    fixture_dataset(tmp_path)
    (tmp_path / 'leftImg8bit/val/drive/0_image.jpg').unlink()
    with pytest.raises(ValueError, match='Missing paired'):
        prepare(tmp_path, tmp_path / 'manifest.json')


def test_unknown_label_encoding_fails(tmp_path):
    fixture_dataset(tmp_path)
    mask = tmp_path / 'gtFine/val/drive/0_label.png'
    assert cv2.imwrite(str(mask), np.full((8, 12), 7, np.uint8))
    with pytest.raises(ValueError, match='Unexpected'):
        prepare(tmp_path, tmp_path / 'manifest.json')
