"""预测前景 vs GT 前景 的像素级"对/错"三色对比（分子级 + 页面级两种粒度）。

与 viz_gt_vs_pred.py 的区别
---------------------------
- viz_gt_vs_pred：**整页**半透明双色叠加（GT 蓝 + 预测红，重叠处靠混色呈紫），
  混色在密集页面上不够直观。
- 本脚本：不靠"混色"，而是把每个像素**硬分类**成三种互斥情形，各给一个纯色，
  一眼能分清哪里对、哪里错。且同时出两种粒度（--level 控制）：

    分子级 diff_per_mol\  — 每个 GT 分子裁一张小图（对齐 vis_fail/post_seg/keep 的
                            _mol### 粒度），裁窗 = GT bbox 外扩 CROP_PAD。
    页面级 diff_page\     — 每张页面一张整页图，GT 前景并集 vs 预测前景并集，同款三色。

  三色约定（两种粒度一致）：
    重合  (预测 ∩ GT)   → 绿色  = 抓对的前景像素（recall 的来源，越多越好）
    漏检  (GT \ 预测)    → 红色  = GT 有、预测没覆盖到（漏原子标签/短取代基/GT桥接伪影）
    误检  (预测 \ GT)    → 蓝色  = 预测有、GT 没有（洪水拉进来的邻居墨水/噪声）

前景口径（与 viz_gt_vs_pred 完全一致，不改任何评估数字）
------------------------------------------------------
- GT 前景：build_gt 迁移自 MPOCSR 的精细 mask（本分子的 inst["mask"]）。
- 预测前景：kept 预测洪水 mask 贴回页面 → ∩ 全图二值化前景 binary(220)。
  （洪水 mask 是"区域掩码"含多边形内白背景，∩binary 后只留真实墨水笔画才好上色。）

--matched-only
  默认裁窗内**所有**预测前景都参与对比（邻居分子的墨水会算成蓝色误检）；
  加 --matched-only 后只用"匈牙利匹配到本 GT 的那个预测实例"，得到最干净的
  单分子对比（推荐看单个分子对错时用）。

命令行 / PyCharm
----------------
  python -m dataset_eval.eval_msirost.viz_diff_per_mol --limit 5            # 两种都出
  python -m dataset_eval.eval_msirost.viz_diff_per_mol --level page         # 只出页面级
  python -m dataset_eval.eval_msirost.viz_diff_per_mol --level mol --only-fail --matched-only
  # 也可在 PyCharm 里右键 Run 本文件（下方 __package__ shim 处理相对导入）。
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
    RECALL_THRESHOLD,
    SUBSETS,
)

# 让 `from utils import viz_utils` 从 msirost 项目根解析（predict 里已 insert 过，兜底再来一次）。
import sys as _sys
if str(MSIROST_ROOT) not in _sys.path:
    _sys.path.insert(0, str(MSIROST_ROOT))
from utils import viz_utils as _viz  # type: ignore  # noqa: E402

# 输出目录（都在 E:\其他文章及数据集\eval\vis\ 下）：
#   diff_per_mol\      — 每个 GT 分子一张裁图（三色），所有分子
#   diff_per_mol_fail\ — 仅低覆盖率分子（recall < RECALL_THRESHOLD）
#   diff_page\         — 每张页面一张整页图（同款三色，GT 并集 vs 预测并集）
DIFF_PER_MOL_DIR = EVAL_ROOT / "vis" / "diff_per_mol"
DIFF_PER_MOL_FAIL_DIR = EVAL_ROOT / "vis" / "diff_per_mol_fail"
DIFF_PAGE_DIR = EVAL_ROOT / "vis" / "diff_page"

# 三色（RGB, 0-255）+ 叠加透明度。三类像素互斥，各上一个纯色（不混色）。
#   alpha = 叠加透明度，无量纲 [0,1]，作用于对应类别的像素上；
#   调大 → 颜色更实但盖住底图笔画；调小 → 颜色淡、能看清笔画但对比不明显。0.55 是折中。
COLOR_TP: Tuple[int, int, int] = (0, 200, 0)      # 绿 — 重合（预测∩GT）= 抓对
COLOR_FN: Tuple[int, int, int] = (230, 0, 0)      # 红 — 漏检（GT\预测）
COLOR_FP: Tuple[int, int, int] = (0, 90, 255)     # 蓝 — 误检（预测\GT）
ALPHA = 0.55

# 裁窗 pad：在 GT bbox 四周各外扩多少像素，无量纲=像素，几何尺度=单分子 bbox；
# 调大 → 看到更多上下文（也更容易把邻居墨水算成蓝色误检）；调小 → 只盯本分子。
CROP_PAD = 12


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


def _paste_inst(inst: Dict, page_hw: Tuple[int, int]) -> np.ndarray:
    """把单个实例 bbox 内的 bool mask 贴回页面坐标，返回页面尺寸 bool。"""
    H, W = page_hw
    canvas = np.zeros((H, W), dtype=bool)
    x1, y1, x2, y2 = inst["bbox"]
    m = inst["mask"]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(W, x2), min(H, y2)
    if x2c <= x1c or y2c <= y1c:
        return canvas
    sub = m[y1c - y1:y2c - y1, x1c - x1:x2c - x1]
    canvas[y1c:y2c, x1c:x2c] |= sub
    return canvas


def _paste_union(instances: List[Dict], page_hw: Tuple[int, int]) -> np.ndarray:
    """若干实例贴回页面并取并集。"""
    H, W = page_hw
    canvas = np.zeros((H, W), dtype=bool)
    for inst in instances:
        canvas |= _paste_inst(inst, (H, W))
    return canvas

def _build_page_ctx(
    page_png: Path,
    mask_png: Path,
    page_stem_full: str,
) -> Dict:
    """跑一遍 GT + 预测 + 打分，返回分子级/页面级共用的上下文。

    预测和 GT 都跑一次即可，分子级和页面级两张图共享，省一半 GPU/CPU 开销。

    Returns dict:
        page_rgb, H, W, gt_insts, pred_insts, fg_bool,
        p_all_page (页级预测前景并集 bool), g_all_page (页级 GT 前景并集 bool),
        score (score_page 结果), gt_to_pred, recalls。
    """
    page_bgr = cv2.imread(str(page_png), cv2.IMREAD_COLOR)
    if page_bgr is None:
        raise FileNotFoundError(page_png)
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    H, W = page_bgr.shape[:2]

    # 1) GT 实例 + 页级 GT 前景并集
    gt_insts, _gt_label = build_gt.build_gt_for_page(page_png, mask_png)
    g_all_page = _paste_union(gt_insts, (H, W))

    # 2) 预测实例 + 页级预测前景（洪水区域 ∩ 全图二值化前景）
    pred_records = predict.predict_for_page(page_png, page_stem_full)
    pred_insts = predict.build_page_pred_masks(pred_records, (H, W))
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    _, binary_fg = cv2.threshold(gray, GLOBAL_BIN_THR, 255, cv2.THRESH_BINARY_INV)
    fg_bool = binary_fg > 0
    p_all_page = _paste_union(pred_insts, (H, W)) & fg_bool

    # 3) 打分 → 每个 GT 的 recall 和它匹配到的预测实例 idx
    s = score.score_page(gt_insts, pred_insts)
    gt_to_pred: Dict[int, int] = {gj: pi for gj, pi, _ in s["matches"]}
    recalls = s["gt_recall_per_inst"]

    return {
        "page_rgb": page_rgb, "H": H, "W": W,
        "gt_insts": gt_insts, "pred_insts": pred_insts, "fg_bool": fg_bool,
        "p_all_page": p_all_page, "g_all_page": g_all_page,
        "score": s, "gt_to_pred": gt_to_pred, "recalls": recalls,
    }


def make_page_overlay_diff(
    ctx: Dict,
    page_stem_full: str,
    out_dir: Path = DIFF_PAGE_DIR,
) -> Dict:
    """页面级三色对比：整页 GT 前景并集 vs 预测前景并集，逐像素硬分三类。

    重合(绿)=预测∩GT，漏检(红)=GT\预测，误检(蓝)=预测\GT。与分子级同款配色。

    Returns: 页级统计 dict。
    """
    page_rgb = ctx["page_rgb"]
    g = ctx["g_all_page"]
    p = ctx["p_all_page"]
    s = ctx["score"]

    tp = g & p
    fn = g & ~p
    fp = p & ~g
    gt_px = int(g.sum())
    rec_px = (int(tp.sum()) / gt_px) if gt_px > 0 else float("nan")

    title = (
        f"[diff-page] {page_stem_full}  "
        f"GT={s['n_gt']} Pred={s['n_pred']}  "
        f"Coverage(inst)={s['coverage_rate'] * 100:.1f}%  "
        f"recall(px)={rec_px * 100:.1f}%"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{page_stem_full}_pagediff.png"
    _viz.save_overlay_figure(
        img_rgb=page_rgb,
        overlays=[
            (fp, COLOR_FP, ALPHA),   # 先蓝（误检）
            (fn, COLOR_FN, ALPHA),   # 再红（漏检）
            (tp, COLOR_TP, ALPHA),   # 最后绿（重合），保证抓对的最醒目
        ],
        title=title,
        legend_items=[
            ("overlap  pred∩GT (correct)", COLOR_TP),
            ("missed   GT\\pred", COLOR_FN),
            ("false-pos pred\\GT", COLOR_FP),
        ],
        out_path=out_path,
    )
    return {
        "gt_px": gt_px, "tp_px": int(tp.sum()),
        "fn_px": int(fn.sum()), "fp_px": int(fp.sum()),
        "pixel_recall": rec_px, "out_path": out_path,
    }


def make_mol_diffs(
    ctx: Dict,
    page_stem_full: str,
    out_dir: Path = DIFF_PER_MOL_DIR,
    fail_dir: Path = DIFF_PER_MOL_FAIL_DIR,
    matched_only: bool = False,
    only_fail: bool = False,
) -> List[Dict]:
    """对一页的每个 GT 分子各出一张三色对比裁图。

    Returns: 每个已出图分子的统计 dict 列表。
    """
    page_rgb = ctx["page_rgb"]
    H, W = ctx["H"], ctx["W"]
    gt_insts = ctx["gt_insts"]
    pred_insts = ctx["pred_insts"]
    fg_bool = ctx["fg_bool"]
    p_all_page = ctx["p_all_page"]
    gt_to_pred = ctx["gt_to_pred"]
    recalls = ctx["recalls"]

    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict] = []

    for gj, gt in enumerate(gt_insts):
        rec = recalls[gj]
        if rec is None:
            status = "missed"
        elif rec >= RECALL_THRESHOLD:
            status = "complete"
        else:
            status = "partial"
        if only_fail and status == "complete":
            continue

        # 本分子 GT 前景（页坐标）
        g_page = _paste_inst(gt, (H, W))

        # 参与对比的预测前景：默认整页预测前景；matched_only 时只用匹配到的那个预测实例
        if matched_only:
            pi = gt_to_pred.get(gj)
            if pi is not None:
                p_page = _paste_inst(pred_insts[pi], (H, W)) & fg_bool
            else:
                p_page = np.zeros((H, W), dtype=bool)
        else:
            p_page = p_all_page

        # 裁窗 = GT bbox 外扩 CROP_PAD
        gx1, gy1, gx2, gy2 = gt["bbox"]
        cx1 = max(0, gx1 - CROP_PAD)
        cy1 = max(0, gy1 - CROP_PAD)
        cx2 = min(W, gx2 + CROP_PAD)
        cy2 = min(H, gy2 + CROP_PAD)

        crop_rgb = page_rgb[cy1:cy2, cx1:cx2]
        g = g_page[cy1:cy2, cx1:cx2]
        p = p_page[cy1:cy2, cx1:cx2]

        tp = g & p          # 重合  → 绿
        fn = g & ~p         # 漏检  → 红
        fp = p & ~g         # 误检  → 蓝
        gt_px = int(g.sum())
        rec_px = (int(tp.sum()) / gt_px) if gt_px > 0 else float("nan")

        rec_tag = "None" if rec is None else f"{rec * 100:.1f}%"
        title = (
            f"[diff] {page_stem_full}_mol{gj:03d}  status={status}  "
            f"coverage(inst)={rec_tag}  coverage(px in crop)={rec_px * 100:.1f}%"
        )
        save_dir = fail_dir if status != "complete" else out_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{page_stem_full}_mol{gj:03d}_diff.png"
        _viz.save_overlay_figure(
            img_rgb=crop_rgb,
            overlays=[
                (fp, COLOR_FP, ALPHA),   # 先蓝（误检）
                (fn, COLOR_FN, ALPHA),   # 再红（漏检）
                (tp, COLOR_TP, ALPHA),   # 最后绿（重合），保证抓对的最醒目
            ],
            title=title,
            legend_items=[
                ("overlap  pred∩GT (correct)", COLOR_TP),
                ("missed   GT\\pred", COLOR_FN),
                ("false-pos pred\\GT", COLOR_FP),
            ],
            out_path=out_path,
        )
        results.append({
            "gt_idx": gj, "status": status, "recall": rec,
            "gt_px": gt_px, "tp_px": int(tp.sum()),
            "fn_px": int(fn.sum()), "fp_px": int(fp.sum()),
            "out_path": out_path,
        })

    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--offices", nargs="+", default=list(OFFICES))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 页，0=不限")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 页")
    ap.add_argument("--matched-only", action="store_true",
                    help="分子级：每个 GT 只与匹配到它的那个预测实例对比（最干净的单分子视图）")
    ap.add_argument("--only-fail", action="store_true",
                    help="分子级：只出 recall<阈值 的分子（partial/missed），跳过完整检出的")
    ap.add_argument("--level", choices=["mol", "page", "both"], default="both",
                    help="出哪种图：mol=只分子级，page=只页面级，both=两者都出（默认）")
    args = ap.parse_args()

    do_mol = args.level in ("mol", "both")
    do_page = args.level in ("page", "both")
    if do_mol:
        DIFF_PER_MOL_DIR.mkdir(parents=True, exist_ok=True)
        DIFF_PER_MOL_FAIL_DIR.mkdir(parents=True, exist_ok=True)
    if do_page:
        DIFF_PAGE_DIR.mkdir(parents=True, exist_ok=True)

    todo = list(_iter_pages(args.subsets, args.offices))
    todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[diff] {len(todo)} pages  level={args.level}  "
          f"(matched_only={args.matched_only}, only_fail={args.only_fail})")
    if do_mol:
        print(f"  分子级 → {DIFF_PER_MOL_DIR}")
    if do_page:
        print(f"  页面级 → {DIFF_PAGE_DIR}")

    n_mol = 0
    n_page = 0
    n_err = 0
    for idx, (subset, office, page_png, mask_png) in enumerate(todo, 1):
        page_stem_full = f"{subset}__{office}__{page_png.stem}"
        t0 = time.time()
        try:
            ctx = _build_page_ctx(page_png, mask_png, page_stem_full)
            n_this_mol = 0
            if do_mol:
                res = make_mol_diffs(
                    ctx, page_stem_full,
                    matched_only=args.matched_only, only_fail=args.only_fail,
                )
                n_this_mol = len(res)
                n_mol += n_this_mol
            if do_page:
                make_page_overlay_diff(ctx, page_stem_full)
                n_page += 1
            print(f"[{idx}/{len(todo)}] {page_stem_full}  "
                  f"{n_this_mol} mol"
                  f"{' +1 page' if do_page else ''}  ({time.time() - t0:.1f}s)")
        except Exception as e:
            n_err += 1
            print(f"[{idx}/{len(todo)}] ERR {page_stem_full}: {e}")
            traceback.print_exc(limit=3)
        finally:
            predict.free_gpu_memory()

    print("\n" + "=" * 60)
    print(f"done: {n_mol} molecule diffs, {n_page} page diffs, {n_err} page errors")
    if do_mol:
        print(f"→ 分子级 {DIFF_PER_MOL_DIR}")
    if do_page:
        print(f"→ 页面级 {DIFF_PAGE_DIR}")


if __name__ == "__main__":
    main()



