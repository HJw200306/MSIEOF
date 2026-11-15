"""eval_msirost 到 fragment_cc 的薄封装。

调用位置：extract（YOLO + 洪水填充）之后、post_segment（UNet++ 过滤）之前。
算法本体在 ``utils.pipeline.fragment_cleanup``：这里复用其纯几何算法
``clean_crop``，补齐新版移除的 record 级编排（同页邻居、页边直线、mask 同步）。

record 几何契约（与 ``predict.predict_for_page`` 一致）
--------------------------------------------------------
    polygon   (N, 2) float  当前 crop 局部坐标
    raw_bbox  (x1, y1, x2, y2) int  页面坐标（YOLO 紧包围盒）
    exp_bbox  (x1, y1, x2, y2) int  页面坐标（crop 区域；兼容 bbox）
    mask      (H, W) bool   crop 内洪水掩码（清理后原地扣掉擦除像素）
    crop_path Path          crop 图片路径（清理后原地覆盖写回）
    _crop_rgb 可选          内存中的 RGB crop，缺省从 crop_path 读盘
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from utils.pipeline.fragment_cleanup import clean_crop, page_lines  # noqa: E402


def _exp_bbox(rec: Dict) -> Tuple[int, int, int, int]:
    box = rec.get("exp_bbox", rec.get("bbox"))
    if box is None:
        raise KeyError("record 缺少 crop 区域：需要 exp_bbox 或 bbox")
    return tuple(int(v) for v in box)


def _polygon_page(rec: Dict) -> np.ndarray:
    poly = np.asarray(rec["polygon"], dtype=np.float32)
    bx1, by1 = _exp_bbox(rec)[:2]
    return poly + np.array([bx1, by1], dtype=np.float32)


def _neighbors_for(records: List[Dict], target_idx: int,
                   origin: Tuple[float, float]) -> List[Dict]:
    """同页其它实例的 polygon 平移到当前 crop 坐标系。"""
    neighbors: List[Dict] = []
    for j, other in enumerate(records):
        if j == target_idx or "polygon" not in other or "raw_bbox" not in other:
            continue
        poly_local = (_polygon_page(other)
                      - np.array(origin, dtype=np.float32)).astype(np.float32)
        neighbors.append({
            "name": str(other.get("name") or Path(other["crop_path"]).stem),
            "polygon": poly_local,
            "raw_bbox": tuple(int(v) for v in other["raw_bbox"]),
        })
    return neighbors


def _crop_page_line_mask(page_line_full: Optional[np.ndarray],
                         exp_bbox: Tuple[int, int, int, int],
                         shape_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    if page_line_full is None:
        return None
    x1, y1, x2, y2 = exp_bbox
    mask = page_line_full[y1:y2, x1:x2]
    h, w = shape_hw
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def apply_fragment_cleanup_to_crops(
    records: List[Dict],
    page_rgb: Optional[np.ndarray] = None,
    use_page_lines: bool = False,
) -> List[Dict]:
    """对同一页面的全部 crop 运行 fragment_cc（原地覆盖写回并同步 mask）。

    Args:
        records: ``predict.predict_for_page`` 产出的实例记录（几何契约见模块
            docstring）；若带 ``_crop_rgb`` 则复用内存中的 crop。
        page_rgb: 整页 RGB；仅 ``use_page_lines=True`` 时作为页边直线证据。
        use_page_lines: 默认 False，与 production 一致。

    Returns:
        同一个 records 列表，每项新增 ``fragment_cleanup_applied`` /
        ``fragment_cleanup_info``，并更新 ``crop_path``（覆盖写回）与 ``mask``
        （扣掉擦除像素）。
    """
    if not records:
        return records

    page_line_full = (
        page_lines(np.asarray(page_rgb))
        if (use_page_lines and page_rgb is not None) else None
    )

    for idx, rec in enumerate(records):
        crop_path = Path(rec["crop_path"])
        crop_rgb = rec.get("_crop_rgb")
        if crop_rgb is None:
            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                rec["fragment_cleanup_applied"] = False
                rec["fragment_cleanup_info"] = {"error": "imread_failed"}
                continue
            crop_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        crop_rgb = np.asarray(crop_rgb, dtype=np.uint8)
        h, w = crop_rgb.shape[:2]

        if "polygon" not in rec or "raw_bbox" not in rec:
            rec["fragment_cleanup_applied"] = False
            rec["fragment_cleanup_info"] = {"skipped": "incomplete_metadata"}
            continue

        exp_bbox = _exp_bbox(rec)
        neighbors = _neighbors_for(records, idx, origin=exp_bbox[:2])
        page_line_mask = _crop_page_line_mask(page_line_full, exp_bbox, (h, w))

        cleaned_rgb, erase_mask, info = clean_crop(
            crop_rgb,
            np.asarray(rec["polygon"], dtype=np.float32),
            tuple(int(v) for v in rec["raw_bbox"]),
            neighbors,
            page_line_mask,
        )

        ok, encoded = cv2.imencode(".png", cv2.cvtColor(cleaned_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError(f"无法写入清理后的裁图: {crop_path}")
        crop_path.write_bytes(encoded.tobytes())

        mask = rec.get("mask")
        if mask is not None and erase_mask.any():
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != erase_mask.shape:
                raise ValueError(
                    f"mask shape {mask.shape} != crop shape {erase_mask.shape} "
                    f"for {crop_path.name}")
            rec["mask"] = mask & ~erase_mask

        rec["fragment_cleanup_applied"] = True
        rec["fragment_cleanup_info"] = {
            "removed": len(info["removed_groups"]),
            "review": len(info["review_groups"]),
            "erase_px": int(erase_mask.sum()),
        }

    return records
