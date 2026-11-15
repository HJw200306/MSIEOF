"""可视化：每页画一张 4-in-1 对比图，把 GT / 预测 / 匹配关系画在原页面上。

颜色约定
--------
GT 分子：
  - 完整检出（recall ≥ 阈值） → 绿色实线
  - 部分匹配（recall > 0 但 < 阈值） → 橙色虚线
  - 完全漏检                        → 红色实线（粗）
预测分子：
  - 匹配上 GT                       → 蓝色实线
  - 未匹配（误检 / 多画）           → 紫色虚线
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .config import RECALL_THRESHOLD


def _draw_bbox(ax, bbox, color, linestyle="-", lw=1.5, label=None):
    x1, y1, x2, y2 = bbox
    ax.add_patch(Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        fill=False, edgecolor=color, linestyle=linestyle, linewidth=lw,
    ))
    if label:
        ax.text(x1 + 2, y1 + 12, label, color=color, fontsize=7,
                bbox=dict(facecolor="white", alpha=0.6, pad=0.5, edgecolor="none"))


def save_page_overlay(
    page_rgb: np.ndarray,
    gt_instances: List[Dict],
    pred_instances: List[Dict],
    score: Dict,
    out_path: Path,
    page_stem: str,
) -> None:
    """在页面上画 GT / Pred / 匹配关系。"""
    matches = score["matches"]
    matched_pred_ids = {pi for _, pi, _ in matches}

    gt_status = ["missed"] * len(gt_instances)
    for gj, _, r in matches:
        gt_status[gj] = "complete" if r >= RECALL_THRESHOLD else "partial"

    fig, ax = plt.subplots(figsize=(10, 14))
    ax.imshow(page_rgb)
    ax.set_axis_off()

    for j, gt in enumerate(gt_instances):
        status = gt_status[j]
        if status == "complete":
            color, ls, lw = "#2ca02c", "-", 2.0
        elif status == "partial":
            color, ls, lw = "#ff7f0e", "--", 1.8
        else:
            color, ls, lw = "#d62728", "-", 2.5
        _draw_bbox(ax, gt["bbox"], color, ls, lw, label=f"G{j}")

    for i, pred in enumerate(pred_instances):
        if i in matched_pred_ids:
            color, ls = "#1f77b4", "-"
        else:
            color, ls = "#9467bd", "--"
        _draw_bbox(ax, pred["bbox"], color, ls, 1.2, label=f"P{i}")

    title = (
        f"[{page_stem}]  GT={score['n_gt']}  Pred={score['n_pred']}  "
        f"Complete={score['n_complete']}  Partial={score['n_partial']}  "
        f"Missed={score['n_missed']}  FP={score['n_false_positive']}  "
        f"Coverage={score['coverage_rate']*100:.1f}%  (thr={RECALL_THRESHOLD})"
    )
    ax.set_title(title, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
