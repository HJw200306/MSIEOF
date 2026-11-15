"""表4统一评分器。

在页面坐标中计算 ``|P∩G|/|G|``，使用匈牙利算法完成一对一最大权匹配，
再按论文规定的阈值划分 Complete 和 Partial。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import RECALL_THRESHOLD


def _bbox_intersects(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _pixel_intersection(gt: Dict, pred: Dict) -> int:
    """计算两个实例在相交 bbox 内的前景交集像素数。"""
    gx1, gy1, gx2, gy2 = gt["bbox"]
    px1, py1, px2, py2 = pred["bbox"]
    ix1, iy1 = max(gx1, px1), max(gy1, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    gt_mask = gt["mask"][iy1 - gy1:iy2 - gy1, ix1 - gx1:ix2 - gx1]
    pred_mask = pred["mask"][iy1 - py1:iy2 - py1, ix1 - px1:ix2 - px1]
    return int(np.logical_and(gt_mask, pred_mask).sum())


def score_page(
    gt_instances: List[Dict],
    pred_instances: List[Dict],
    threshold: float = RECALL_THRESHOLD,
) -> Dict:
    """对一页进行一对一匹配并返回表4所需统计。"""
    n_gt = len(gt_instances)
    n_pred = len(pred_instances)

    if n_gt == 0 and n_pred == 0:
        return dict(
            n_gt=0, n_pred=0, n_complete=0, n_missed=0, n_partial=0,
            n_false_positive=0, coverage_rate=1.0,
            matches=[], gt_recall_per_inst=[],
        )
    if n_gt == 0:
        return dict(
            n_gt=0, n_pred=n_pred, n_complete=0, n_missed=0, n_partial=0,
            n_false_positive=n_pred, coverage_rate=1.0,
            matches=[], gt_recall_per_inst=[],
        )
    if n_pred == 0:
        return dict(
            n_gt=n_gt, n_pred=0, n_complete=0, n_missed=n_gt, n_partial=0,
            n_false_positive=0, coverage_rate=0.0,
            matches=[], gt_recall_per_inst=[None] * n_gt,
        )

    recalls = np.zeros((n_gt, n_pred), dtype=np.float32)
    for gt_index, gt in enumerate(gt_instances):
        for pred_index, pred in enumerate(pred_instances):
            if not _bbox_intersects(gt["bbox"], pred["bbox"]):
                continue
            intersection = _pixel_intersection(gt, pred)
            if intersection:
                recalls[gt_index, pred_index] = intersection / max(1, gt["area"])

    gt_indices, pred_indices = linear_sum_assignment(-recalls)
    matches = []
    gt_recall_per_inst = [None] * n_gt
    matched_predictions = set()
    for gt_index, pred_index in zip(gt_indices, pred_indices):
        recall = float(recalls[gt_index, pred_index])
        if recall <= 0.0:
            continue
        matches.append((int(gt_index), int(pred_index), recall))
        gt_recall_per_inst[gt_index] = recall
        matched_predictions.add(int(pred_index))

    n_complete = sum(recall >= threshold for _, _, recall in matches)
    n_partial = sum(recall < threshold for _, _, recall in matches)
    n_missed = n_gt - len(matches)
    n_false_positive = n_pred - len(matched_predictions)
    return dict(
        n_gt=n_gt,
        n_pred=n_pred,
        n_complete=n_complete,
        n_missed=n_missed,
        n_partial=n_partial,
        n_false_positive=n_false_positive,
        coverage_rate=n_complete / n_gt,
        matches=matches,
        gt_recall_per_inst=gt_recall_per_inst,
    )
