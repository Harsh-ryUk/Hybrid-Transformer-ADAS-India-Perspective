"""Ground-truth metrics independent of models and runtime timing."""
import contextlib
import io

import numpy as np


def binary_road_counts(prediction, labels, road_ids, valid_ids, ignore_ids):
    """Count all valid pixels; never exclude failed predictions or crop misses."""
    prediction, labels = np.asarray(prediction), np.asarray(labels)
    if prediction.ndim != 2 or prediction.shape != labels.shape:
        raise ValueError('Prediction and label masks must have the same 2D shape')
    if not road_ids or not set(road_ids) <= set(valid_ids) or set(valid_ids) & set(ignore_ids):
        raise ValueError('Road IDs must be valid and ignore IDs must be disjoint')
    unexpected = set(np.unique(labels).tolist()) - set(valid_ids) - set(ignore_ids)
    if unexpected:
        raise ValueError(f'Unexpected ground-truth label IDs: {sorted(unexpected)}')
    valid = np.isin(labels, valid_ids)
    truth = np.isin(labels, road_ids) & valid
    predicted = (prediction > 0) & valid
    return {
        'tp': int(np.count_nonzero(predicted & truth)),
        'fp': int(np.count_nonzero(predicted & ~truth)),
        'fn': int(np.count_nonzero(~predicted & truth)),
        'tn': int(np.count_nonzero(~predicted & ~truth & valid)),
        'valid_pixels': int(np.count_nonzero(valid)),
        'ignored_pixels': int(np.count_nonzero(~valid)),
    }


def road_summary(counts):
    totals = {key: sum(row[key] for row in counts) for key in
              ('tp', 'fp', 'fn', 'tn', 'valid_pixels', 'ignored_pixels')}
    tp, fp, fn = (totals[key] for key in ('tp', 'fp', 'fn'))
    if totals['valid_pixels'] == 0:
        raise ValueError('No valid ground-truth pixels')
    return {**totals, 'IoU': tp / (tp + fp + fn) if tp + fp + fn else None,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'aggregation': 'global pixel counts; not mean of per-image IoUs'}


def coco_detection_metrics(images, annotations, predictions, category_names):
    """COCO bbox AP, including classes with GT but zero predictions."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if not images or not annotations or not category_names:
        raise ValueError('COCO AP requires images, categories and at least one GT box')
    categories = [{'id': index + 1, 'name': name} for index, name in enumerate(category_names)]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {'images': images, 'annotations': annotations, 'categories': categories, 'info': {}}
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': images, 'annotations': [], 'categories': categories}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'bbox')
        evaluator.params.imgIds = [image['id'] for image in images]
        evaluator.params.catIds = [category['id'] for category in categories]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    per_class = {}
    for index, name in enumerate(category_names):
        values = evaluator.eval['precision'][:, :, index, 0, -1]
        values = values[values >= 0]
        per_class[name] = float(values.mean()) if values.size else None
    return {'mAP50_95': float(evaluator.stats[0]), 'AP50': float(evaluator.stats[1]),
            'AP75': float(evaluator.stats[2]), 'per_class_AP50_95': per_class,
            'method': 'pycocotools COCOeval bbox; maxDets=100; area=all',
            'ground_truth_boxes': len(annotations), 'prediction_boxes': len(predictions)}
