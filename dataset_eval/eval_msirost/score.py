"""Scorer：1 对 1 匹配 + Recall≥阈值 判定"完整检出"。

- 用**页面坐标 bbox 相交**先粗筛（大部分 GT-Pred 对没有交集，直接零），避免在 O(N×M)
  的完整矩阵上算像素级 IoU。
- 有交集的对才计算 `|P ∩ G| / |G|`。
- 用 `scipy.optimize.linear_sum_assignment` 做**最大权 1 对 1 匹配**（成本 = -recall），
  防"一个大 blob 通吃 12 个分子"作弊。
- Recall ≥ RECALL_THRESHOLD 才判为"完整检出"。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import RECALL_THRESHOLD


def _bbox_intersects(a: Tuple[int, int, int, int],
                     b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _pixel_intersection(gt: Dict, pred: Dict) -> int:
    """两个实例前景像素的交集大小（在 mask bbox 相交子区域内计算）。"""
    gx1, gy1, gx2, gy2 = gt["bbox"]
    px1, py1, px2, py2 = pred["bbox"]
    ix1 = max(gx1, px1)
    iy1 = max(gy1, py1)
    ix2 = min(gx2, px2)
    iy2 = min(gy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    gm = gt["mask"][iy1 - gy1:iy2 - gy1, ix1 - gx1:ix2 - gx1]
    pm = pred["mask"][iy1 - py1:iy2 - py1, ix1 - px1:ix2 - px1]
    return int(np.logical_and(gm, pm).sum())


def score_page(
    gt_instances: List[Dict],
    pred_instances: List[Dict],
    threshold: float = RECALL_THRESHOLD,
) -> Dict:
    """对一张页面打分。

    Returns:
        {
          'n_gt', 'n_pred',
          'n_complete'        Recall≥threshold 的匹配对数
          'n_missed'          未匹配到 pred 的 GT 数（漏检）
          'n_partial'         匹配到但 Recall<threshold 的 GT 数（不完整）
          'n_false_positive'  未匹配到 GT 的 pred 数（多画）
          'coverage_rate'     n_complete / n_gt（主指标）
          'matches'           [(gt_idx, pred_idx, recall), ...]  仅列匹配上的
          'gt_recall_per_inst' [recall or None, ...] 长度 = n_gt
        }
    """
    n_gt = len(gt_instances)
    n_pred = len(pred_instances)

    # 无 GT / 无 Pred 的边界
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

    # 覆盖率矩阵 R[j, i] = |P_i ∩ G_j| / |G_j|
    R = np.zeros((n_gt, n_pred), dtype=np.float32)
    for j, gt in enumerate(gt_instances):
        for i, pred in enumerate(pred_instances):
            if not _bbox_intersects(gt["bbox"], pred["bbox"]):
                continue
            inter = _pixel_intersection(gt, pred)
            if inter == 0:
                continue
            R[j, i] = inter / max(1, gt["area"])

    # 匈牙利：最大化总 recall（cost = -R）。矩阵可非方，linear_sum_assignment 支持。
    row_idx, col_idx = linear_sum_assignment(-R)

    # 阈值过滤：recall=0 的匹配视为"没匹配上"
    matches: List[Tuple[int, int, float]] = []
    gt_recall_per_inst: List = [None] * n_gt
    matched_pred = set()
    for gj, pi in zip(row_idx, col_idx):
        r = float(R[gj, pi])
        if r <= 0.0:
            continue
        matches.append((int(gj), int(pi), r))
        gt_recall_per_inst[gj] = r
        matched_pred.add(int(pi))

    n_complete = sum(1 for _, _, r in matches if r >= threshold)
    n_partial = sum(1 for _, _, r in matches if r < threshold)
    n_missed = n_gt - len(matches)
    n_false_positive = n_pred - len(matched_pred)

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
