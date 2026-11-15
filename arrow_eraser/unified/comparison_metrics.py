"""Pixel-level metrics for segmentation and the actual erasure mask."""
import numpy as np


def fuse_and_erase(rgb, labels):
    """Either independent model can protect a bond; both use the original RGB."""
    if not labels or any(x.shape != rgb.shape[:2] for x in labels):
        raise ValueError('Missing or misaligned predictions')
    if any(not set(np.unique(x)) <= {0, 1, 2} for x in labels):
        raise ValueError('Predictions must use three-class IDs 0/1/2')
    arrow = np.logical_or.reduce([x == 1 for x in labels])
    bond = np.logical_or.reduce([x == 2 for x in labels])
    erased = arrow & ~bond
    pred = np.zeros(rgb.shape[:2], np.uint8)
    pred[arrow] = 1
    pred[bond] = 2
    out = rgb.copy()
    out[erased] = 255
    return pred, erased, out


def counts(pred, erased, target):
    if pred.shape != target.shape or erased.shape != target.shape:
        raise ValueError('Prediction/GT dimensions differ')
    if not set(np.unique(target)) <= {0, 1, 2, 255}:
        raise ValueError('GT must use 0 background, 1 arrow, 2 bond, 255 ignore')
    valid = target != 255
    cm = np.bincount((target[valid].astype(int) * 3 + pred[valid]).ravel(), minlength=9).reshape(3, 3)
    return {'confusion': cm.tolist(), 'arrow_pixels': int((target == 1).sum()),
            'bond_pixels': int((target == 2).sum()), 'background_pixels': int((target == 0).sum()),
            'arrow_remaining': int(((target == 1) & ~erased).sum()),
            'bond_deleted': int(((target == 2) & erased).sum()),
            'background_deleted': int(((target == 0) & erased).sum()),
            'valid_erased': int((valid & erased).sum())}


def ratio(a, b):
    return float(a / b) if b else None


def summarize(items):
    cm = sum((np.asarray(x['confusion']) for x in items), np.zeros((3, 3), dtype=np.int64))
    result = {'images': len(items), 'confusion': cm.tolist()}
    for i, name in enumerate(['background', 'arrow', 'bond']):
        tp, actual, proposed = cm[i, i], cm[i].sum(), cm[:, i].sum()
        result[name + '_iou'] = ratio(tp, actual + proposed - tp)
        result[name + '_dice'] = ratio(2 * tp, actual + proposed)
    for name, num, den in [('arrow_residual_rate', 'arrow_remaining', 'arrow_pixels'),
                           ('bond_false_erasure_rate', 'bond_deleted', 'bond_pixels'),
                           ('background_erasure_rate', 'background_deleted', 'background_pixels')]:
        numerator, denominator = sum(x[num] for x in items), sum(x[den] for x in items)
        result[name] = ratio(numerator, denominator)
        result[name + '_counts'] = [numerator, denominator]
    times = [x['seconds'] for x in items if 'seconds' in x]
    if times:
        result.update(mean_seconds=float(np.mean(times)), median_seconds=float(np.median(times)),
                      p95_seconds=float(np.percentile(times, 95)))
    return result
