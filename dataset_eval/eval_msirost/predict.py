r"""跑 msirost 完整流水线（extract + post_segment）得到每页的预测实例。

对一张页面：
    1) YOLOv11-seg 检测分子 polygon（utils/extract.py 的模型与口径，
       conf=EXTRACT_CONF，classes=[MOL_CLASS_ID]）；
    2) 每个实例独立洪水填充：扩边 bbox（±0.10·各自边长）内
       seed=raw polygon、host=raw ∪ 整页二值图，8 邻接形态学重建
       （复用 extract._page_binary / _morph_reconstruct，与生产流水线
       逐像素一致）；
    3) crop 落盘到 CROPS_TMP_DIR/<page_stem_full>/，命名
       <page_stem_full>_mol###.png + sidecar .meta.npz（阅读顺序编号，
       与 vis_all/post_seg 的 overlay 名对齐）；
    4) 每个 crop 跑 post_segment.classify_image（UNet++ fg_ratio 门控，
       threshold=POST_SEG_THRESHOLD）得到 keep/reject。

注意：预测实例的 mask 是**洪水区域掩码**，polygon 内部的白背景也在里面。
对 recall = |P∩G|/|G| 无害（GT 前景不含白背景），但可视化涂色前要
∩ binary(220) 抽一次前景（见 viz_gt_vs_pred.py 顶部说明）。

接口（main.py / viz_*.py / collect_failures.py 依赖）
----------------------------------------------------
    predict_for_page(page_png, page_stem_full) -> List[record]
        record = {"bbox", "mask"(bbox 内 bool 洪水掩码), "kept", "fg_ratio",
                  "crop_path"}，含 keep 和 reject 的全部实例。
    build_page_pred_masks(records, (H, W)) -> List[inst]
        仅 kept 实例，inst = {"bbox", "mask", "area"}（score.py 口径）。
    free_gpu_memory()
        每页收尾释放显存缓存（模型常驻不卸载）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import (
    COLOR_AWARE_S_BG_THR,
    COLOR_AWARE_V_BG_THR,
    CROPS_TMP_DIR,
    EXTRACT_CONF,
    MSIROST_ROOT,
    POST_SEG_THRESHOLD,
)
from .fg_utils import color_aware_ink_mask

if str(MSIROST_ROOT) not in sys.path:
    sys.path.insert(0, str(MSIROST_ROOT))

from utils import extract as _extract  # noqa: E402
from utils import post_segment as _post_segment  # noqa: E402
import torch  # noqa: E402

# post_seg 的 UNet++ 常驻缓存（跨页面复用，免得每页重载 300MB 权重）。
_pseg_model = None
_pseg_device: Optional[torch.device] = None

# classify_image 要求一个 keep 输出目录（它会把通过的 crop 复制过去）。
# 评估只关心判定结果，副本统一丢进 sink 目录，可事后删除。
_KEEP_SINK = CROPS_TMP_DIR / "_keep_sink"


def _get_pseg_model():
    global _pseg_model, _pseg_device
    if _pseg_model is None:
        _pseg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pseg_model = _post_segment._load_model(_pseg_device)
    return _pseg_model, _pseg_device


def predict_for_page(
    page_png: Path,
    page_stem_full: str,
    viz_extract_dir: Optional[Path] = None,
    viz_post_dir: Optional[Path] = None,
    color_aware: bool = False,
    crops_root: Optional[Path] = None,
) -> List[Dict]:
    """对一张页面跑 extract + post_segment，返回全部预测实例记录。

    crop 及 sidecar 写入 CROPS_TMP_DIR/<page_stem_full>/（每次覆盖重建），
    post_seg 的 keep 副本写入 CROPS_TMP_DIR/_keep_sink/。

    过程可视化（与生产流水线 vis/extract、vis/post_seg 完全同款）：
      - viz_extract_dir 非 None → 落一张整页 extract 叠加图（polygon + 扩边
        bbox + 阅读顺序编号 + Tab10 洪水保留区）到该目录；
      - viz_post_dir 非 None → 每个 crop 落一张 post_seg 叠加图（UNet++ 前景
        绿 + YOLO 多边形蓝 + 判定）到 <viz_post_dir>/{keep,reject}/。
    两者均复用 utils/extract 与 utils/post_segment 的原渲染函数，保证观感一致。

    color_aware: True → 洪水填充的 host 二值用 fg_utils.color_aware_ink_mask(close_r=3)
        （HSV 背景判据）替代 extract._page_binary 的 Otsu 灰度二值化，让洪水能沿有色
        分子键蔓延（期刊页）；对无彩色专利页与 Otsu 近似。默认 False，PatCID 流程不变。
    crops_root: crop 及 sidecar 的落盘根目录（默认 CROPS_TMP_DIR）；自己验证集模式传
        MYVAL_CROPS_TMP_DIR，与 PatCID 的临时产物隔离。

    Returns:
        [{"bbox": (x1, y1, x2, y2) 页面坐标（扩边后）,
          "mask": (bh, bw) bool 洪水掩码（含 polygon 内白背景）,
          "kept": bool,     # post_seg 判为 2D 分子
          "fg_ratio": float,
          "crop_path": Path}, ...]
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
    crops_dir = (crops_root or CROPS_TMP_DIR) / page_stem_full
    crops_dir.mkdir(parents=True, exist_ok=True)
    for old in crops_dir.glob("*"):
        try:
            old.unlink()
        except OSError:
            pass

    order = _extract._reading_order(exp_bboxes)
    records: List[Dict] = []
    for new_idx, old_idx in enumerate(order):
        crop_path = crops_dir / f"{page_stem_full}_mol{new_idx:03d}.png"
        Image.fromarray(crops[old_idx]).save(crop_path)

        bx1, by1, _bx2, _by2 = exp_bboxes[old_idx]
        poly_local = (
            polygons[old_idx] - np.array([bx1, by1], dtype=np.float32)
        ).astype(np.float32)
        np.savez_compressed(
            crops_dir / f"{page_stem_full}_mol{new_idx:03d}.meta.npz",
            mask=flood_local[old_idx].astype(np.uint8),
            polygon=poly_local,
        )
        records.append({
            "bbox": exp_bboxes[old_idx],
            "mask": flood_local[old_idx] > 0,
            "kept": False,
            "fg_ratio": 0.0,
            "crop_path": crop_path,
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

    # --- 4) post_seg 二分类过滤 ---
    pseg_model, device = _get_pseg_model()
    _KEEP_SINK.mkdir(parents=True, exist_ok=True)
    for rec in records:
        is_mol, ratio, _ = _post_segment.classify_image(
            model=pseg_model, img_path=rec["crop_path"], output_dir=_KEEP_SINK,
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
