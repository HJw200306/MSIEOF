r"""跑 msirost 完整流水线（extract → fragment_cc → post_segment）得到每页的预测实例。

对一张页面：
    1) YOLOv11-seg 检测分子 polygon（utils/extract.py 的模型与口径，
       conf=EXTRACT_CONF，classes=[MOL_CLASS_ID]）；
    2) 每个实例独立洪水填充：扩边 bbox（±0.10·各自边长）内
       seed=raw polygon、host=raw ∪ 整页二值图，8 邻接形态学重建
       （复用 extract._page_binary / _morph_reconstruct，与生产流水线
       逐像素一致）；
    3) crop 落盘到 crops_root/<page_stem_full>/，命名
       <page_stem_full>_mol###.png + sidecar .meta.npz（阅读顺序编号，
       与 vis_all/post_seg 的 overlay 名对齐）；
    4) fragment_cc 邻近残片清理（utils/fragment_cleanup.py）：按同页其它
       实例的 polygon 支持度，擦掉洪水填充带进来的邻近分子残片 / 页边
       裁断残片；擦除像素同步从 record 的 mask 中扣除；
    5) 每个 crop 跑 post_segment.classify_image（UNet++ fg_ratio 门控，
       threshold=POST_SEG_THRESHOLD）得到 keep/reject。

注意：预测实例的 mask 是**洪水区域掩码**，polygon 内部的白背景也在里面。
对 recall = |P∩G|/|G| 无害（GT 前景不含白背景），但可视化涂色前要
∩ binary(220) 抽一次前景（见 viz_gt_vs_pred.py 顶部说明）。

接口（main.py / viz_*.py / collect_failures.py 依赖）
----------------------------------------------------
    predict_for_page(page_png, page_stem_full) -> List[record]
        record = {"bbox", "mask"(bbox 内 bool 洪水掩码，已扣除 fragment_cc
                  擦除的像素), "kept", "fg_ratio", "crop_path",
                  "polygon"(crop 局部), "raw_bbox"(页面坐标),
                  "fragment_cleanup_applied", "fragment_cleanup_info"}，
              含 keep 和 reject 的全部实例。
    build_page_pred_masks(records, (H, W)) -> List[inst]
        仅 kept 实例，inst = {"bbox", "mask", "area"}（score.py 口径）。
    free_gpu_memory()
        每页收尾释放显存缓存（模型常驻不卸载）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import (
    COLOR_AWARE_S_BG_THR,
    COLOR_AWARE_V_BG_THR,
    EXTRACT_CONF,
    POST_SEG_THRESHOLD,
)
from dataset_eval.eval_msirost.fg_utils import color_aware_ink_mask
from utils.pipeline import extract as _extract, post_segment as _post_segment

from .fragment_cleanup_integration import apply_fragment_cleanup_to_crops
import torch

# post_seg 的 UNet++ 常驻缓存（跨页面复用，免得每页重载 300MB 权重）。
_pseg_model = None
_pseg_device: Optional[torch.device] = None


def _get_pseg_model():
    global _pseg_model, _pseg_device
    if _pseg_model is None:
        _pseg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pseg_model = _post_segment._load_model(_pseg_device)
    return _pseg_model, _pseg_device


def prepare() -> None:
    """在正式遍历页面前加载模型。"""
    _extract._get_model()
    _get_pseg_model()


def predict_for_page(
    page_png: Path,
    page_stem_full: str,
    *,
    crops_root: Path,
    viz_extract_dir: Optional[Path] = None,
    viz_post_dir: Optional[Path] = None,
    color_aware: bool = False,
    fragment_cleanup: bool = True,
    use_page_lines: bool = False,
) -> List[Dict]:
    """对一张页面跑 extract → fragment_cc → post_segment，返回全部预测实例记录。

    crop 及 sidecar 写入 crops_root/<page_stem_full>/，
    post_seg 的 keep 副本写入 crops_root/_keep_sink/。

    过程可视化（与生产流水线 vis/extract、vis/post_seg 完全同款）：
      - viz_extract_dir 非 None → 落一张整页 extract 叠加图（polygon + 扩边
        bbox + 阅读顺序编号 + Tab10 洪水保留区）到该目录；
      - viz_post_dir 非 None → 每个 crop 落一张 post_seg 叠加图（UNet++ 前景
        绿 + YOLO 多边形蓝 + 判定）到 <viz_post_dir>/{keep,reject}/。
    两者均复用 utils/extract 与 utils/post_segment 的原渲染函数，保证观感一致。

    color_aware: True → 洪水填充的 host 二值用 fg_utils.color_aware_ink_mask(close_r=3)
        （HSV 背景判据）替代 extract._page_binary 的 Otsu 灰度二值化，让洪水能沿有色
        分子键蔓延（期刊页）；对无彩色专利页与 Otsu 近似。默认 False，PatCID 流程不变。
    crops_root: 本次表4评测的独立裁图目录。
    fragment_cleanup: 是否在 extract 与 post_seg 之间跑 fragment_cc。默认 True，
        与 ``utils/ablation_unified.py`` 的 ``full`` 配置、生产管线一致；设 False
        可复现“无残片清理”的对照结果。
    use_page_lines: 是否把整页直线证据（页边框 / 表格线）传给 fragment_cc。
        默认 False，与 backend/run_pipeline.py 的 denoise 阶段一致。

    Returns:
        [{"bbox": (x1, y1, x2, y2) 页面坐标（扩边后）,
          "mask": (bh, bw) bool 洪水掩码（含 polygon 内白背景；已扣除
                  fragment_cc 擦除的像素）,
          "kept": bool,     # post_seg 判为 2D 分子
          "fg_ratio": float,
          "crop_path": Path,
          "polygon": (N, 2) crop 局部坐标,
          "raw_bbox": (x1, y1, x2, y2) 页面坐标（YOLO 紧包围盒）,
          "fragment_cleanup_applied": bool,
          "fragment_cleanup_info": {"removed", "review", "erase_px"}}, ...]
        按阅读顺序排列（与 crop 编号一致）。
    """
    page = Image.open(page_png).convert("RGB")
    page_arr = np.array(page)
    h, w = page_arr.shape[:2]

    # --- 等比缩放：长边超过 _MAX_YOLO_SIDE 时先缩再送 YOLO，坐标事后还原 ---
    _MAX_YOLO_SIDE = 4000
    _scale = min(_MAX_YOLO_SIDE / max(h, w), 1.0)
    yolo_arr = (cv2.resize(page_arr, (int(w * _scale), int(h * _scale)),
                           interpolation=cv2.INTER_AREA)
                if _scale < 1.0 else page_arr)

    # --- 1) YOLO polygon 检测（与 extract.extract_from_page 同参）---
    model = _extract._get_model()
    results = model.predict(
        source=yolo_arr, conf=EXTRACT_CONF,
        classes=[_extract.MOL_CLASS_ID], verbose=False,
    )

    if color_aware:
        binary = color_aware_ink_mask(
            page_arr, v_bg_thr=COLOR_AWARE_V_BG_THR,
            s_bg_thr=COLOR_AWARE_S_BG_THR, close_r=3,
        )
    else:
        binary = _extract._page_binary(page_arr, close_r=3)

    raw_masks: List[np.ndarray] = []
    raw_bboxes: List[Tuple[int, int, int, int]] = []
    exp_bboxes: List[Tuple[int, int, int, int]] = []
    polygons: List[np.ndarray] = []
    for result in results:
        masks = result.masks
        if masks is None or len(masks) == 0:
            continue
        for contour_xy in masks.xy:
            if len(contour_xy) < 3:
                continue
            if _scale < 1.0:
                contour_xy = contour_xy / _scale
            mask_full = _extract._contour_to_mask(contour_xy, (h, w))
            ys, xs = np.where(mask_full > 0)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            raw_bboxes.append((x1, y1, x2, y2))
            exp = _extract._expand_bbox_per_axis((x1, y1, x2, y2), 0.10, h, w)
            raw_masks.append(mask_full)
            exp_bboxes.append(exp)
            polygons.append(np.asarray(contour_xy, dtype=np.float32))

    if not exp_bboxes:
        return []

    # --- 2) 每实例独立洪水填充 ---
    flood_local: List[np.ndarray] = []   # bbox 内 uint8 0/255
    crops: List[np.ndarray] = []
    for raw, bbox in zip(raw_masks, exp_bboxes):
        bx1, by1, bx2, by2 = bbox
        sub_raw = raw[by1:by2, bx1:bx2]
        sub_black = binary[by1:by2, bx1:bx2]
        sub_host = cv2.bitwise_or(sub_raw, sub_black)
        sub_recon = _extract._morph_reconstruct(sub_raw, sub_host)
        flood_local.append(sub_recon)

        sub_img = page_arr[by1:by2, bx1:bx2].copy()
        sub_img[sub_recon == 0] = 255
        crops.append(sub_img)

    # --- 3) crop + sidecar 落盘（阅读顺序编号）---
    crops_dir = crops_root / page_stem_full
    crops_dir.mkdir(parents=True, exist_ok=False)

    order = _extract._reading_order(exp_bboxes)
    records: List[Dict] = []
    for new_idx, old_idx in enumerate(order):
        crop_path = crops_dir / f"{page_stem_full}_mol{new_idx:03d}.png"
        Image.fromarray(crops[old_idx]).save(crop_path)

        bx1, by1, bx2, by2 = exp_bboxes[old_idx]
        rx1, ry1, rx2, ry2 = raw_bboxes[old_idx]
        poly_local = (
            polygons[old_idx] - np.array([bx1, by1], dtype=np.float32)
        ).astype(np.float32)
        # 与 extract.py 的 sidecar 同口径落盘页面级几何：目录级工具
        # （utils.fragment_cleanup.run / cleanup_image）不用再猜 crop 原点。
        np.savez_compressed(
            crops_dir / f"{page_stem_full}_mol{new_idx:03d}.meta.npz",
            mask=flood_local[old_idx].astype(np.uint8),
            polygon=poly_local,
            bbox_norm=np.array([bx1 / w, by1 / h, bx2 / w, by2 / h],
                               dtype=np.float32),
            bbox_raw_norm=np.array([rx1 / w, ry1 / h, rx2 / w, ry2 / h],
                                   dtype=np.float32),
            page_size=np.array([h, w], dtype=np.int32),
            page_id=page_stem_full,
        )
        records.append({
            "name": crop_path.stem,
            "bbox": exp_bboxes[old_idx],
            "exp_bbox": exp_bboxes[old_idx],
            "raw_bbox": raw_bboxes[old_idx],
            "polygon": poly_local,
            "mask": flood_local[old_idx] > 0,
            "kept": False,
            "fg_ratio": 0.0,
            "crop_path": crop_path,
            # 内存里已有 crop，交给 fragment_cc 直接用，避免再读一次盘；
            # 集成层用完会 pop 掉这个临时键。
            "_crop_rgb": crops[old_idx],
        })

    # --- 3.5) extract 过程叠加图（与 vis/extract 同款渲染器）---
    if viz_extract_dir is not None:
        inst_final_masks: List[np.ndarray] = []
        for local, bbox in zip(flood_local, exp_bboxes):
            bx1, by1, bx2, by2 = bbox
            full = np.zeros((h, w), dtype=np.uint8)
            full[by1:by2, bx1:bx2] = local
            inst_final_masks.append(full)
        _extract._save_extract_overlay(
            page_rgb=page_arr,
            inst_polygons=[[p] for p in polygons],
            inst_final_masks=inst_final_masks,
            inst_bboxes=exp_bboxes,
            order=order,
            page_stem=page_stem_full,
            shape_hw=(h, w),
            out_dir=viz_extract_dir,
        )

    # --- 3.6) fragment_cc 邻近残片清理（extract 之后、过滤之前）---
    # 统一走 dataset_eval.eval_coverage.fragment_cleanup_integration，算法本体是
    # utils/fragment_cleanup.py。清理会覆盖写回 crop，并从 rec["mask"] 扣掉擦除像素，
    # 后续过滤与 score 都基于清理后的结果。
    if fragment_cleanup:
        apply_fragment_cleanup_to_crops(
            records, page_rgb=page_arr, use_page_lines=use_page_lines)
    # 无论是否跑清理都丢掉临时图像引用，避免 records 把整页 crop 常驻内存。
    for rec in records:
        rec.pop("_crop_rgb", None)

    # --- 4) post_seg 二分类过滤 ---
    pseg_model, device = _get_pseg_model()
    keep_sink = crops_root / "_keep_sink"
    for rec in records:
        is_mol, ratio, _ = _post_segment.classify_image(
            model=pseg_model, img_path=rec["crop_path"], output_dir=keep_sink,
            device=device, threshold=POST_SEG_THRESHOLD,
            visualize=viz_post_dir is not None, viz_dir=viz_post_dir,
        )
        rec["kept"] = bool(is_mol)
        rec["fg_ratio"] = float(ratio)

    return records


def build_page_pred_masks(
    records: List[Dict], page_hw: Tuple[int, int]
) -> List[Dict]:
    """把 kept 预测记录转成 score.py 口径的实例列表（bbox 夹紧到页面内）。

    Returns:
        [{"bbox": (x1, y1, x2, y2), "mask": bbox 内 bool, "area": int}, ...]
    """
    H, W = page_hw
    insts: List[Dict] = []
    for rec in records:
        if not rec["kept"]:
            continue
        x1, y1, x2, y2 = rec["bbox"]
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(W, x2), min(H, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        m = rec["mask"][y1c - y1:y2c - y1, x1c - x1:x2c - x1]
        insts.append({
            "bbox": (x1c, y1c, x2c, y2c),
            "mask": m,
            "area": int(m.sum()),
        })
    return insts


def free_gpu_memory() -> None:
    """每页收尾：释放 CUDA 缓存（模型保持常驻，不重复加载权重）。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
