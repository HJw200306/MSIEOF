"""DECIMER-Segmentation 预测模块：整页图 → 每个分子的实例掩码。

基线模型
--------
DECIMER-Image-Segmentation（Kohulan Rajan et al.），Mask R-CNN，权重
`mask_rcnn_molecule.h5`（随 pip 包 `decimer_segmentation` 分发）。
检测置信度阈值由包内 `InferenceConfig.DETECTION_MIN_CONFIDENCE` 固定为
0.7（v1.3.0），本模块不覆盖。

关键点
------
`get_expanded_masks(image)` 返回 `(H, W, N)` 的**完整结构扩展掩码**（先 MRCNN
检测 seed，再在去表格线/横竖线的文档图上做洪水填充扩张，见包内
`complete_structure.complete_structure_mask`），这正是"完整分子结构提取"口径。
每个 mask 的 bbox 直接从 mask 自身非零像素外接框求出，与 `build_gt` / `score`
的 `(x1, y1, x2, y2)` + bbox 内 bool mask 约定完全一致，避免 mask 与 bbox 错位。

输入约定
--------
image 必须是 BGR ndarray（`cv2.imread` 默认），与包内 `get_expanded_masks`
的 `COLOR_BGR2GRAY` 分支一致。
"""

from __future__ import annotations

import os

# 必须在 import decimer_segmentation（进而 import tensorflow）之前设置，
# 否则 TF 会一次性占满全部显存，挤压同进程里 msirost 的 PyTorch 模型。
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")     # 压制 TF info/warning 日志
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")  # TF 按需增长显存

from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .image_io import read_image

# 惰性导入：import decimer_segmentation 会在模块级执行 `model = load_model()`，
# 加载权重，只在该方法真正被调用时才触发。
_expand_fn = None


def _get_expand_fn():
    global _expand_fn
    if _expand_fn is None:
        try:
            from decimer_segmentation import get_expanded_masks
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "DECIMER-Segmentation 不可用：当前 Python 环境缺少 "
                "`decimer_segmentation`（及其依赖 tensorflow）。\n"
                "请安装：pip install decimer-segmentation\n"
                f"原始错误: {e}"
            ) from e
        _expand_fn = get_expanded_masks
    return _expand_fn


def predict_decimer_for_page(page_png: Path) -> List[Dict]:
    """对一张页面跑 DECIMER-Segmentation，返回完整结构实例列表。

    Returns:
        [{"bbox": (x1, y1, x2, y2) 页面坐标,
          "mask": (bh, bw) bool 完整结构掩码,
          "area": int}, ...]
        与 score.score_page 的口径一致（bbox 是页面坐标、mask 是 bbox 内 bool）。
    """
    get_expanded_masks = _get_expand_fn()

    img = read_image(page_png, cv2.IMREAD_COLOR)

    masks = get_expanded_masks(img)  # (H, W, N) bool 完整结构掩码

    if masks is None or masks.ndim != 3 or masks.shape[2] == 0:
        return []

    instances: List[Dict] = []
    for i in range(masks.shape[2]):
        m = masks[:, :, i].astype(bool)
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        mask_local = m[y1:y2, x1:x2]
        instances.append({
            "bbox": (x1, y1, x2, y2),
            "mask": mask_local,
            "area": int(mask_local.sum()),
        })
    return instances


def free_gpu_memory() -> None:
    """清理 TensorFlow GPU 显存和 Python 对象缓存。

    TF 按需增长显存但不主动释放，长时间运行会累积到 OOM。
    强制触发 Keras backend 清理 + Python GC 防止内存泄漏。
    """
    try:
        import gc
        import tensorflow as tf
        from tensorflow.keras import backend as K

        # 清理 Keras session 缓存的张量
        K.clear_session()
        # 强制 Python 垃圾回收
        gc.collect()
    except Exception:
        pass  # 环境不支持时静默跳过
