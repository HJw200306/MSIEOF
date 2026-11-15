r"""GT vs 项目输出 的像素级双色叠加可视化。

对每张 PatCID 页面画一张 PNG：
  - GT 前景像素（build_gt 迁移自 MPOCSR 的精细 mask 并集）→ 半透明蓝 (0,0,255)
  - 项目流水线输出的前景像素（extract 洪水 + post_segment 过滤后的 kept 并集，
    再 ∩ 全图二值化前景）→ 半透明红 (255,0,0)
  - 两层半透明叠加处自然呈紫/品红 = 抓对的部分。

为什么项目侧要再 ∩ binary(220)？
--------------------------------
评估用的预测 mask = 洪水填充结果，它里面包含 YOLO 多边形内部的**白色背景像素**
（是区域掩码，不是墨水掩码）。对 recall = |P∩G|/|G| 无害（与 G 求交时白背景被
自动过滤），但如果直接拿来涂红，红色会盖满整个多边形区域、看不清笔画。所以
**仅在可视化层**把 P 也抽成前景：P_fg = P ∩ binary(220)。这一步不改评估口径，
per_page.csv / summary 的覆盖率数字不受影响。

除了这张双色叠加，本文件还会同步落两套“过程图”，与生产流水线
（utils/extract.py、utils/post_segment.py）完全同款渲染，方便逐步排查：
  gt_vs_pred/overlay/            本文件的 GT(蓝) vs 预测(红) 像素级叠加
  gt_vs_pred/extract/           extract 步：polygon + 扩边 bbox + 阅读顺序编号
                                + Tab10 洪水保留区
  gt_vs_pred/post_seg/keep,      post_seg 步：UNet++ 前景绿 + YOLO 多边形蓝 + 判定
  gt_vs_pred/post_seg/reject     （keep=判为分子，reject=被前景占比门控剔除）

读图
----
  紫/品红  GT 前景被流水线覆盖到了（= recall 的来源，越多越好）
  纯蓝      GT 有、流水线没有（漏检 / 漏原子标签 / GT 桥接伪影）
  纯红      流水线有、GT 没有（洪水拉进来的邻居 / 噪声）

注意纯蓝里混有 GT 构建的 MST 桥接线（2px 直线）和 3×3 膨胀环，这类是 GT 侧伪影，
不是流水线真的漏了什么，但确实在拉低 recall——这张图能把"真漏检"和"GT 伪影惩罚"
分开看。

命令行 / PyCharm
----------------
  python -m dataset_eval.eval_msirost.viz_gt_vs_pred --limit 10
  python -m dataset_eval.eval_msirost.viz_gt_vs_pred --subsets benchmark_D2C-RND --offices jp
  python -m dataset_eval.eval_msirost.viz_gt_vs_pred            # 全量

也可以在 PyCharm 里直接右键 Run 本文件（下面的 __package__ shim 处理了相对导入）。
"""

from __future__ import annotations

# 允许 PyCharm 右键 Run 直接运行：__package__ 未设置时把项目根塞进 sys.path。
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "dataset_eval.eval_msirost"

import argparse
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from . import build_gt, predict, score
from .build_gt import GLOBAL_BIN_THR
from .config import (
    EVAL_ROOT,
    MSIROST_ROOT,
    OFFICES,
    PATCID_RESTORED_ROOT,
    SUBSETS,
)

# 让 `from utils import viz_utils` 从 msirost 项目根解析（predict 里已经 insert 过，
# 这里再兜底一次，保证单独跑本文件时也能 import）。
import sys as _sys
if str(MSIROST_ROOT) not in _sys.path:
    _sys.path.insert(0, str(MSIROST_ROOT))
from utils import viz_utils as _viz  # type: ignore  # noqa: E402

# 输出目录：E:\其他文章及数据集\eval\vis\gt_vs_pred\
GT_VS_PRED_DIR = EVAL_ROOT / "vis" / "gt_vs_pred"
# 过程可视化子目录（与生产流水线 vis/extract、vis/post_seg 同款）：
#   overlay/      — 本文件原本的 GT(蓝) vs 预测(红) 像素级叠加图
#   extract/      — extract 步：polygon + 扩边 bbox + 阅读顺序编号 + Tab10 洪水区
#   post_seg/keep, post_seg/reject — post_seg 步：UNet++ 前景绿 + YOLO 多边形蓝 + 判定
GT_VS_PRED_OVERLAY_DIR = GT_VS_PRED_DIR / "overlay"
EXTRACT_VIS_DIR = GT_VS_PRED_DIR / "extract"
POST_SEG_VIS_DIR = GT_VS_PRED_DIR / "post_seg"

# 双色配色（RGB, 0-255）+ 叠加透明度
#   alpha = 叠加透明度，无量纲 [0,1]，作用于整片掩码区域；
#   调大 → 颜色更实但会盖住底图笔画；调小 → 颜色不明显；0.5 是两者都能看清的折中。
COLOR_GT: Tuple[int, int, int] = (0, 0, 255)      # 蓝 — GT 前景
COLOR_PRED: Tuple[int, int, int] = (255, 0, 0)    # 红 — 项目输出前景
ALPHA_GT = 0.5
ALPHA_PRED = 0.5


def _iter_pages(
    subsets: Iterable[str],
    offices: Iterable[str],
) -> Iterable[Tuple[str, str, Path, Path]]:
    """yield (subset, office, page_png, mask_png)（与 main._iter_pages 同口径）。"""
    for subset in subsets:
        for office in offices:
            d = PATCID_RESTORED_ROOT / subset / office
            if not d.exists():
                continue
            for png in sorted(d.glob("*.png")):
                if png.name.endswith(".mask.png") or png.name.endswith(".vis.png"):
                    continue
                mask = png.with_suffix(".mask.png")
                if not mask.exists():
                    continue
                yield subset, office, png, mask


def _paste_union(instances: List[Dict], page_hw: Tuple[int, int]) -> np.ndarray:
    """把若干实例的 bbox 内 bool mask 贴回页面坐标并取并集，返回页面尺寸 bool。"""
    H, W = page_hw
    canvas = np.zeros((H, W), dtype=bool)
    for inst in instances:
        x1, y1, x2, y2 = inst["bbox"]
        m = inst["mask"]
        # 边界裁剪，防越界（bbox 已按页面尺寸夹紧，这里再兜底）
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        sub = m[y1c - y1:y2c - y1, x1c - x1:x2c - x1]
        canvas[y1c:y2c, x1c:x2c] |= sub
    return canvas


def make_page_overlay(
    page_png: Path,
    mask_png: Path,
    page_stem_full: str,
    out_dir: Path = GT_VS_PRED_OVERLAY_DIR,
    viz_extract_dir: Path = EXTRACT_VIS_DIR,
    viz_post_dir: Path = POST_SEG_VIS_DIR,
) -> Dict:
    """构建并保存一页的 GT(蓝) vs 项目输出(红) 像素级叠加图。

    Returns:
        统计 dict（供打印 / 汇总）：n_gt, n_pred, coverage_rate,
        gt_fg_px, pred_fg_px, overlap_px, pixel_coverage, out_path。
    """
    page_bgr = cv2.imread(str(page_png), cv2.IMREAD_COLOR)
    if page_bgr is None:
        raise FileNotFoundError(page_png)
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    H, W = page_bgr.shape[:2]

    # 1) GT 前景并集（蓝）
    gt_insts, _gt_label = build_gt.build_gt_for_page(page_png, mask_png)
    g_all = _paste_union(gt_insts, (H, W))

    # 2) 项目输出前景并集（红）：kept 预测的洪水 mask 贴回页面 ∩ 全图二值化前景
    #    同时让流水线把 extract / post_seg 过程图落到评估输出树（同款渲染器）。
    pred_records = predict.predict_for_page(
        page_png, page_stem_full,
        viz_extract_dir=viz_extract_dir,
        viz_post_dir=viz_post_dir,
    )
    pred_insts = predict.build_page_pred_masks(pred_records, (H, W))
    p_region = _paste_union(pred_insts, (H, W))       # 含多边形内白背景的区域掩码
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_fg = cv2.threshold(gray, GLOBAL_BIN_THR, 255, cv2.THRESH_BINARY_INV)
    fg_bool = binary_fg > 0
    p_all = p_region & fg_bool                        # 抽前景后的项目输出（用于涂红）

    # 3) 统计：实例级 coverage（与 4-in-1 图一致）+ 页级像素 coverage
    s = score.score_page(gt_insts, pred_insts)
    gt_fg_px = int(g_all.sum())
    pred_fg_px = int(p_all.sum())
    overlap_px = int((g_all & p_all).sum())
    pixel_cov = overlap_px / gt_fg_px if gt_fg_px > 0 else float("nan")

    # 4) 画图：复用项目 viz_utils.save_overlay_figure（同款 legend / dpi=120 / 自动缩放）
    title = (
        f"[gt_vs_pred] {page_stem_full}  "
        f"GT={s['n_gt']} Pred={s['n_pred']}  "
        f"Coverage(inst)={s['coverage_rate'] * 100:.1f}%  "
        f"Coverage(px)={pixel_cov * 100:.1f}%"
    )
    out_path = out_dir / f"{page_stem_full}_overlay.png"
    _viz.save_overlay_figure(
        img_rgb=page_rgb,
        overlays=[
            (g_all, COLOR_GT, ALPHA_GT),      # 先铺蓝（GT）
            (p_all, COLOR_PRED, ALPHA_PRED),  # 再叠红（项目输出）→ 重叠处呈紫/品红
        ],
        title=title,
        legend_items=[
            ("GT foreground (blue)", COLOR_GT),
            ("pipeline output (red)", COLOR_PRED),
        ],
        out_path=out_path,
    )

    return {
        "n_gt": s["n_gt"],
        "n_pred": s["n_pred"],
        "coverage_rate": s["coverage_rate"],
        "gt_fg_px": gt_fg_px,
        "pred_fg_px": pred_fg_px,
        "overlap_px": overlap_px,
        "pixel_coverage": pixel_cov,
        "out_path": out_path,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--offices", nargs="+", default=list(OFFICES))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 页，0=不限")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 页")
    args = ap.parse_args()

    GT_VS_PRED_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_VIS_DIR.mkdir(parents=True, exist_ok=True)
    POST_SEG_VIS_DIR.mkdir(parents=True, exist_ok=True)

    todo = list(_iter_pages(args.subsets, args.offices))
    todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[gt_vs_pred] {len(todo)} pages → {GT_VS_PRED_DIR}")
    print(f"  overlay(GT vs pred) → {GT_VS_PRED_OVERLAY_DIR}")
    print(f"  extract 过程图      → {EXTRACT_VIS_DIR}")
    print(f"  post_seg 过程图     → {POST_SEG_VIS_DIR}\\{{keep,reject}}")

    n_ok = 0
    n_err = 0
    for idx, (subset, office, page_png, mask_png) in enumerate(todo, 1):
        stem = page_png.stem
        page_stem_full = f"{subset}__{office}__{stem}"
        t0 = time.time()
        try:
            r = make_page_overlay(page_png, mask_png, page_stem_full)
            n_ok += 1
            print(
                f"[{idx}/{len(todo)}] {page_stem_full}  "
                f"GT={r['n_gt']} Pred={r['n_pred']} "
                f"CovInst={r['coverage_rate'] * 100:.1f}% "
                f"CovPx={r['pixel_coverage'] * 100:.1f}%  "
                f"({time.time() - t0:.1f}s)"
            )
        except Exception as e:
            n_err += 1
            print(f"[{idx}/{len(todo)}] ERR {page_stem_full}: {e}")
            traceback.print_exc(limit=3)
        finally:
            predict.free_gpu_memory()

    print("\n" + "=" * 60)
    print(f"done: {n_ok} overlays, {n_err} errors")
    print(f"→ {GT_VS_PRED_DIR}")


if __name__ == "__main__":
    main()
