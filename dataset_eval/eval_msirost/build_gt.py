r"""构建每页的 GT 分子实例（精细前景 mask + bbox）。

口径完全迁移自 MPOCSR (D:\workspace\MPOCSR-master\utils\process_patent.py) 的
get_refined_mask_and_bbox，与训练 GT 分布对齐（见 config.py 顶部说明）：

    1) 全图灰度 THRESH_BINARY_INV @ GLOBAL_BIN_THR 得到墨迹二值图；
    2) 实例来源（每个实例 = 一个 GT 分子）：
       - 优先"直接按 JSON 标签转换"：读 sibling <stem>.json，每个 shape
         逐一 fillPoly 成独立实例 label（相邻/接触的两个标注框**不会**被并成
         一个实例，避免把两个分子连成一条 MST 桥线）；
       - 若 .json 缺失/无有效 shape，回退到旧口径：对 mask.png 取 8 连通域，
         每个连通域 = 一个实例（接触的标注会被并成同一实例）。
    3) 对每个实例取 ROI = 实例 bbox 外扩 ROI_PAD，在 ROI 内做
       标注区域 ∩ 墨迹（= fillPoly ∩ binary，mask.png 已经是 fillPoly 的
       栅格化结果，直接用连通域像素等价）；
    4) 局部去噪：8 连通统计 ROI 内小连通域，仅当 area < TINY_CC_AREA 的
       小 CC 数量 ≥ TINY_CC_COUNT 时才把它们全部丢弃（少量小碎块视为
       原子标签/键的一部分，保留）；
    5) 3×3 矩形核膨胀 1 次（粘合断开的字符笔画）；
    6) MST 桥接：若仍有多个 8 连通域，按"组件间最近点对距离"建最小生成树，
       每条树边画一根 BRIDGE_THICKNESS px 直线，把实例连成单连通。

返回的实例 dict 与 score.py / visualize.py 约定一致：
    {"bbox": (x1, y1, x2, y2) 页面坐标, "mask": bbox 内 bool 数组, "area": int}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import COLOR_AWARE_S_BG_THR, COLOR_AWARE_V_BG_THR
from .fg_utils import color_aware_ink_mask

# 全图墨迹二值化阈值。
#   操作 = cv2.threshold(gray, GLOBAL_BIN_THR, 255, THRESH_BINARY_INV)，
#          灰度 ≤ 该值的像素判为墨迹（前景）；
#   单位 = 灰度值 [0, 255]；
#   几何尺度 = 单像素灰度，作用于整页；
#   调大（如 240）→ 把浅色底纹 / 扫描阴影也拉进前景，GT 面积虚胖、
#                   recall 分母变大、评估数字被压低；
#   调小（如 180）→ 浅印的骨架线 / 细笔画丢失，GT 缺笔画。
#   220 与 MPOCSR 训练时的取值一致，不要单独改这里（recover 侧的
#   find_annotated_pages_without_mol.py 也从这里导入以保持口径统一）。
GLOBAL_BIN_THR = 220

# ROI 外扩 pad。
#   操作 = 实例连通域 bbox 四周各外扩该值后裁 ROI，后续去噪/膨胀/桥接都在
#          ROI 内做；单位 = 像素；几何尺度 = 单实例 bbox；
#   调大 → 膨胀/桥接有更多活动空间，但更容易把 ROI 边缘的邻居墨迹带进来；
#   调小 → 膨胀可能在 ROI 边界被截断。MPOCSR 原值 = 5。
ROI_PAD = 5

# 局部去噪的两个参数（MPOCSR 原值 5 / 10）。
#   操作 = 统计 ROI 内 8 连通域，area < TINY_CC_AREA(单位=像素数) 的记为
#          "小碎块"；仅当小碎块个数 ≥ TINY_CC_COUNT(单位=个) 时全部丢弃；
#   几何尺度 = 单实例 ROI 内部；
#   调大 TINY_CC_AREA → 更多碎块被视为噪声候选，虚线键/小标点可能被误删；
#   调小 TINY_CC_COUNT → 去噪触发更频繁，扫描噪点少的干净页也会被删碎块。
TINY_CC_AREA = 5
TINY_CC_COUNT = 10

# MST 桥接线宽。
#   操作 = cv2.line(..., thickness=BRIDGE_THICKNESS) 连接组件最近点对；
#   单位 = 像素；几何尺度 = 组件间空隙；
#   调大 → 桥接伪影更粗、GT 面积虚增（这部分预测覆盖不到会拉低 recall）；
#   调小到 1 → 桥更细，但极端情况下细桥可能在后续处理中断开。MPOCSR 原值 = 2。
BRIDGE_THICKNESS = 2

# 桥接时每个组件参与最近点对搜索的轮廓点数上限。
#   操作 = 组件外轮廓点按等步长下采样到 ≤ 该值再算两两距离（纯性能参数）；
#   单位 = 点数；几何尺度 = 单组件外轮廓；
#   调大 → 最近点对更精确但 O(N²) 距离矩阵更慢；调小 → 桥接端点可能偏离
#   真实最近点几个像素（对 2px 桥线的评估影响可忽略）。
_BRIDGE_MAX_PTS = 400


def _shape_to_contour(shape: Dict) -> Optional[np.ndarray]:
    """LabelMe shape → cv2.fillPoly 用整数轮廓 (K,1,2)；不支持的形状返回 None。

    与 recover/mask_to_labelme.py 的 shape_to_contour 同口径：rectangle 两点/四点
    统一取外接矩形四角，polygon 至少 3 点。
    """
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "polygon")
    if shape_type == "rectangle":
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    elif shape_type != "polygon":
        return None
    elif len(points) < 3:
        return None
    return np.array(
        [[round(x), round(y)] for x, y in points], dtype=np.int32
    ).reshape(-1, 1, 2)


def _labels_from_json(
    json_path: Path, H: int, W: int
) -> Optional[Tuple[int, np.ndarray, np.ndarray]]:
    """按 JSON 标注逐 shape 栅格化成实例 label 图（每个 shape = 一个实例）。

    这是"直接按 JSON 标签转换"的核心：不再对二值 mask 做 connectedComponents，
    因此**相邻/接触的两个标注框不会被并成一个实例**（这正是把两个金刚烷酯类分子
    连成一条红色桥线的根因）。JSON 里 shape 的先后顺序即实例序号（1-based），
    后画的 shape 在重叠像素上覆盖先画的（与实例分离无关，仅影响重叠边界归属）。

    Returns:
        与 connectedComponentsWithStats 的前三项同形状的 (n, labels, stats)：
            n      : 实例数 + 1（含背景 0，与 cv2 约定一致）；
            labels : (H, W) int32，0=背景，i=第 i 个 shape；
            stats  : (n, 5) int32，列含 CC_STAT_LEFT/TOP/WIDTH/HEIGHT/AREA。
        若 JSON 读取失败或无有效 shape，返回 None（调用方回退到二值 mask 连通域）。
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    shapes = data.get("shapes") or []
    labels = np.zeros((H, W), dtype=np.int32)
    idx = 0
    for shape in shapes:
        contour = _shape_to_contour(shape)
        if contour is None:
            continue
        idx += 1
        cv2.fillPoly(labels, [contour], idx)

    if idx == 0:
        return None

    # 每个 label 的外接框 + 面积（对齐 cv2.CC_STAT_* 的列布局）。
    stats = np.zeros((idx + 1, 5), dtype=np.int32)
    for ci in range(1, idx + 1):
        ys, xs = np.nonzero(labels == ci)
        if xs.size == 0:
            continue
        x, y = int(xs.min()), int(ys.min())
        stats[ci, cv2.CC_STAT_LEFT] = x
        stats[ci, cv2.CC_STAT_TOP] = y
        stats[ci, cv2.CC_STAT_WIDTH] = int(xs.max()) - x + 1
        stats[ci, cv2.CC_STAT_HEIGHT] = int(ys.max()) - y + 1
        stats[ci, cv2.CC_STAT_AREA] = int(xs.size)
    return idx + 1, labels, stats


def _denoise_local(content: np.ndarray) -> np.ndarray:
    """局部去噪：仅当 area<TINY_CC_AREA 的小 CC 达到 TINY_CC_COUNT 个才丢弃它们。"""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(content, connectivity=8)
    tiny = [ci for ci in range(1, n) if stats[ci, cv2.CC_STAT_AREA] < TINY_CC_AREA]
    if len(tiny) < TINY_CC_COUNT:
        return content
    out = content.copy()
    out[np.isin(labels, tiny)] = 0
    return out


def _comp_contour_pts(comp: np.ndarray) -> np.ndarray:
    """组件外轮廓点 (N, 2)=(x, y)，等步长下采样到 ≤ _BRIDGE_MAX_PTS。"""
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
    if len(pts) > _BRIDGE_MAX_PTS:
        step = int(np.ceil(len(pts) / _BRIDGE_MAX_PTS))
        pts = pts[::step]
    return pts


def _bridge_mst(mask: np.ndarray) -> np.ndarray:
    """MST 桥接：多个 8 连通域间沿最小生成树画 BRIDGE_THICKNESS px 直线连成单连通。

    树边权 = 两组件外轮廓最近点对的欧氏距离；桥线端点即该最近点对。
    """
    n, labels = cv2.connectedComponents(mask, connectivity=8)
    n_comp = n - 1
    if n_comp <= 1:
        return mask

    pts_per_comp = [
        _comp_contour_pts((labels == ci).astype(np.uint8)) for ci in range(1, n)
    ]

    # 两两最近点对（距离平方即可，MST 只比大小）
    INF = float("inf")
    dist = np.full((n_comp, n_comp), INF, dtype=np.float64)
    endpoints: Dict[Tuple[int, int], Tuple[Tuple[int, int], Tuple[int, int]]] = {}
    for a in range(n_comp):
        pa = pts_per_comp[a].astype(np.float32)
        for b in range(a + 1, n_comp):
            pb = pts_per_comp[b].astype(np.float32)
            d2 = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=-1)
            ia, ib = np.unravel_index(int(np.argmin(d2)), d2.shape)
            dist[a, b] = dist[b, a] = float(d2[ia, ib])
            endpoints[(a, b)] = (
                tuple(int(v) for v in pa[ia]),
                tuple(int(v) for v in pb[ib]),
            )

    # Prim 最小生成树
    out = mask.copy()
    in_tree = [False] * n_comp
    in_tree[0] = True
    for _ in range(n_comp - 1):
        best = (INF, -1, -1)
        for a in range(n_comp):
            if not in_tree[a]:
                continue
            for b in range(n_comp):
                if in_tree[b]:
                    continue
                if dist[a, b] < best[0]:
                    best = (dist[a, b], a, b)
        _, a, b = best
        if b < 0:
            break
        in_tree[b] = True
        p1, p2 = endpoints[(min(a, b), max(a, b))]
        cv2.line(out, p1, p2, 255, BRIDGE_THICKNESS)
    return out


def build_gt_for_page(
    page_png: Path,
    mask_png: Path,
    color_aware: bool = False,
    json_path: Optional[Path] = None,
) -> Tuple[List[Dict], np.ndarray]:
    """对一张页面构建 GT 实例列表 + 页面尺寸的实例 label 图。

    实例来源优先 sibling <stem>.json（每个 shape = 一个实例，相邻框不合并）；
    .json 缺失/无有效 shape 时回退到 mask_png 的 8 连通域。

    Args:
        page_png: restored 页面原图路径（同目录 <stem>.json 若存在则优先使用）。
        mask_png: recover_pages.py 输出的标注 mask（前景=255），JSON 缺失时的回退源。
        color_aware: True → 用 fg_utils.color_aware_ink_mask（HSV 背景判据）替代
            固定 220 灰度阈值提取整页墨迹，捕捉期刊页里的有色分子键；对无彩色专利页
            与旧口径等价。默认 False，PatCID 流程保持旧灰度口径不变。
        json_path: 显式指定 LabelMe 标注路径（自己验证集的 json 与图像不同目录时用）；
            None 时回退到 page_png.with_suffix(".json")（sibling json）。

    Returns:
        (instances, gt_label)
        instances: [{"bbox": (x1, y1, x2, y2), "mask": bool (bh, bw), "area": int}, ...]
                   bbox 为页面坐标、mask 为 bbox 内精细前景；
                   ROI 内无墨迹的标注（纯空白框）被跳过。
        gt_label:  (H, W) int32，0=背景，i=第 i 个实例（1-based，与
                   instances 下标 +1 对应）。
    """
    if color_aware:
        page_color = cv2.imread(str(page_png), cv2.IMREAD_COLOR)
        if page_color is None:
            raise FileNotFoundError(page_png)
        H, W = page_color.shape[:2]
        binary_global = color_aware_ink_mask(
            page_color, v_bg_thr=COLOR_AWARE_V_BG_THR, s_bg_thr=COLOR_AWARE_S_BG_THR,
        )
    else:
        page_gray = cv2.imread(str(page_png), cv2.IMREAD_GRAYSCALE)
        if page_gray is None:
            raise FileNotFoundError(page_png)
        H, W = page_gray.shape[:2]
        _, binary_global = cv2.threshold(
            page_gray, GLOBAL_BIN_THR, 255, cv2.THRESH_BINARY_INV
        )

    # 优先"直接按 JSON 标签转换"：每个 shape = 一个实例，相邻框不会被并成一个。
    # 只有当 .json 缺失/无有效 shape 时，才回退到二值 mask 的 8 连通域
    # （旧口径：接触/相邻的标注会被并成同一实例并被 MST 桥线连起来）。
    if json_path is None:
        json_path = page_png.with_suffix(".json")
    labeled = _labels_from_json(json_path, H, W) if json_path.exists() else None
    if labeled is not None:
        n, labels, stats = labeled
    else:
        ann = cv2.imread(str(mask_png), cv2.IMREAD_GRAYSCALE)
        if ann is None:
            raise FileNotFoundError(mask_png)
        if ann.shape[:2] != (H, W):
            ann = cv2.resize(ann, (W, H), interpolation=cv2.INTER_NEAREST)
        ann_bin = (ann > 127).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            ann_bin, connectivity=8
        )

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    instances: List[Dict] = []
    gt_label = np.zeros((H, W), dtype=np.int32)

    for ci in range(1, n):
        x, y, w, h = (
            stats[ci, cv2.CC_STAT_LEFT], stats[ci, cv2.CC_STAT_TOP],
            stats[ci, cv2.CC_STAT_WIDTH], stats[ci, cv2.CC_STAT_HEIGHT],
        )
        rx1, ry1 = max(0, x - ROI_PAD), max(0, y - ROI_PAD)
        rx2, ry2 = min(W, x + w + ROI_PAD), min(H, y + h + ROI_PAD)

        # 标注区域 ∩ 墨迹（mask.png 已是 fillPoly 栅格化结果，连通域像素即多边形内部）
        region = (labels[ry1:ry2, rx1:rx2] == ci).astype(np.uint8) * 255
        content = cv2.bitwise_and(binary_global[ry1:ry2, rx1:rx2], region)
        content = _denoise_local(content)
        if not content.any():
            continue

        content = cv2.dilate(content, kernel3, iterations=1)
        content = _bridge_mst(content)

        ys, xs = np.nonzero(content)
        bx1, bx2 = int(xs.min()), int(xs.max()) + 1
        by1, by2 = int(ys.min()), int(ys.max()) + 1
        inst_mask = content[by1:by2, bx1:bx2] > 0
        bbox = (rx1 + bx1, ry1 + by1, rx1 + bx2, ry1 + by2)

        instances.append({
            "bbox": bbox,
            "mask": inst_mask,
            "area": int(inst_mask.sum()),
        })
        gt_label[bbox[1]:bbox[3], bbox[0]:bbox[2]][inst_mask] = len(instances)

    return instances, gt_label
