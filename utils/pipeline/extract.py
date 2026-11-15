"""化学分子结构提取模块。

输入：单张页面图片 或 PDF 文件。
流程：
  1. 实例分割模型 (model/best.pt) 检测精准分子结构轮廓 (polygon)。
     YOLO 给出几个 polygon 就视为几个分子实例，后处理不再做合并。
  2. 将轮廓栅格化为二值掩码，求 raw bbox（polygon 紧包围盒）。
  3. 整页 Otsu 二值化 + 闭运算（close_r=3，粘原子标签字符）得到黑像素图。
  4. raw bbox 各方向独立外扩 bbox_pad_ratio=0.10 倍各自边长（裁到图像
     边界），扩边后 bbox = 洪水填充的硬上界。
  5. 对每个实例独立做洪水填充：
     - 在扩边 bbox 子图内，seed = raw polygon，host = raw ∪ black，
       8 邻接形态学重建 → 从 polygon 沿黑像素吞并到稳定或撞 bbox；
     - 不排除其它实例的 raw polygon —— 若两个 YOLO 实例属于同一分子，
       它们洪水后的输出会重叠（这是预期行为，每个实例各出一张图）。
  6. 按扩边 bbox 裁原图，非洪水保留区域抹成纯白。
输出：保存到项目根目录下的 crop_dataset/ 文件夹中，按阅读顺序编号。
"""

from pathlib import Path
from typing import List, Optional, Union  # noqa: F401  (Optional used in viz signatures)

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO

from utils.base.paths import PROJECT_ROOT
MODEL_PATH = PROJECT_ROOT / "model" / "YOLOv11_best.pt"
INPUT_DIR = Path("E:/MSIREOT_DATA/DSeg/Semantic_Segmentation/patentAndjournal/images/test")
OUTPUT_DIR = PROJECT_ROOT / "crop_dataset"
DEBUG_OUTPUT_DIR = PROJECT_ROOT / "crop_dataset_debug"

# 与 YOLOv11/Seg_data.yaml 的 names 映射保持一致：0=mol, 1=non_mol。
# extract 只关心分子结构，predict 时用 classes=[MOL_CLASS_ID] 把 non_mol 过滤掉。
MOL_CLASS_ID = 0

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

_model: Optional[YOLO] = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def pdf_to_pages(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    """按指定 DPI（默认 300）逐页渲染 PDF 为 PIL 图像列表。"""
    pages: List[Image.Image] = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            pages.append(img)
    return pages


def _contour_to_mask(contour_xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """将多边形轮廓 (Nx2 float 坐标) 栅格化为 H×W 的 uint8 二值掩码。"""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.round(contour_xy).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], color=255)
    return mask


def _dilate_mask_in_bbox(mask_full: np.ndarray,
                         bbox: tuple[int, int, int, int],
                         radius_px: int = 10) -> np.ndarray:
    """在 bbox 子图内用圆形结构元对掩码做向外膨胀。

    radius_px 近似一个化学键长度（默认 10 px，对应直径 21 的椭圆核），
    使原子标签 / 离主体几像素的小标记被纳入；子图边界即扩张硬上界。

    返回与 mask_full 同形状的 uint8 掩码（仅 bbox 区域被修改），值为 0/255。
    """
    x1, y1, x2, y2 = bbox
    sub = mask_full[y1:y2, x1:x2]
    r = max(1, int(radius_px))
    k = 2 * r + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    sub_dilated = cv2.dilate(sub, kernel, iterations=1)

    out = np.zeros_like(mask_full)
    out[y1:y2, x1:x2] = sub_dilated
    return out


def _expand_bbox_per_axis(bbox: tuple[int, int, int, int],
                          ratio: float,
                          h: int, w: int) -> tuple[int, int, int, int]:
    """各方向独立按 ratio 倍各自边长外扩 bbox，并裁到图像边界。

    ratio=0.10 表示左右各 +0.10·bw、上下各 +0.10·bh，
    扩后宽 = (1+2·ratio)·bw、高 = (1+2·ratio)·bh，中心不变。
    """
    x1, y1, x2, y2 = bbox
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(round(ratio * bw))
    pad_y = int(round(ratio * bh))
    x1e = max(0, x1 - pad_x)
    y1e = max(0, y1 - pad_y)
    x2e = min(w, x2 + pad_x)
    y2e = min(h, y2 + pad_y)
    return (x1e, y1e, x2e, y2e)


def _page_binary(page_arr: np.ndarray, close_r: int = 3) -> np.ndarray:
    """整页灰度 → Otsu 二值化（黑前景）+ 闭运算（粘原子标签字符）。

    close_r 控制椭圆结构元半径（核大小 = 2·close_r+1）：把离主体几像素的
    原子标签 / 虚线键碎块跟主体粘成同一连通域，让后续洪水填充能从 polygon
    走到字符。close_r=0 即关闭闭运算（孤立的原子标签会丢）。
    """
    gray = cv2.cvtColor(page_arr, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if close_r > 0:
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*close_r+1, 2*close_r+1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    return binary


def _morph_reconstruct(seed: np.ndarray, mask: np.ndarray,
                       max_iter: int = 2000) -> np.ndarray:
    """形态学重建（按膨胀）：从 seed 出发，在 mask 约束内沿连通域反复膨胀至稳定。

    结果 = "从 seed 能沿着 mask 的黑像素到达的所有像素"。
    seed / mask 均为 uint8 0/255，8 邻接（3×3 矩形核）。
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cur = cv2.bitwise_and(seed, mask)
    for _ in range(max_iter):
        dil = cv2.dilate(cur, kernel)
        dil = cv2.bitwise_and(dil, mask)
        if np.array_equal(dil, cur):
            break
        cur = dil
    return cur


def _reading_order(boxes: List[tuple[int, int, int, int]]) -> List[int]:
    """按"从左到右、从上到下"返回 boxes 的排序索引。

    行聚类：先按 y 中心排序，用高度中位数的一半作为行容差；
    顺序扫描归并到当前行均值，超过容差则开新行；行内按 x 中心升序。
    """
    if not boxes:
        return []
    centers = [((y1 + y2) / 2, (x1 + x2) / 2, y2 - y1) for x1, y1, x2, y2 in boxes]
    order_by_y = sorted(range(len(boxes)), key=lambda i: centers[i][0])
    h_med = float(np.median([c[2] for c in centers]))
    tol = max(1.0, 0.5 * h_med)

    rows: List[List[int]] = []
    row_mean_y: List[float] = []
    for i in order_by_y:
        cy = centers[i][0]
        if rows and abs(cy - row_mean_y[-1]) <= tol:
            rows[-1].append(i)
            row_mean_y[-1] = float(np.mean([centers[j][0] for j in rows[-1]]))
        else:
            rows.append([i])
            row_mean_y.append(cy)

    ordered: List[int] = []
    for row in rows:
        row.sort(key=lambda i: centers[i][1])
        ordered.extend(row)
    return ordered


# matplotlib tab10 调色板（RGB 0-255），用于可视化中给相邻分子上不同颜色。
TAB10: List[tuple[int, int, int]] = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def _build_adjacency_reading(bboxes: List[tuple[int, int, int, int]],
                             order: List[int]) -> List[List[int]]:
    """阅读顺序意义下的相邻图：

      - 阅读顺序里 i 和 i+1 相邻；
      - 同列（x 中心聚类）上下相邻的两个分子相邻。

    返回邻接表（与 bboxes 同长，元素是邻居 old_idx 列表）。
    """
    n = len(bboxes)
    adj: List[set] = [set() for _ in range(n)]
    if n == 0:
        return [list(s) for s in adj]

    # 1) 阅读顺序里 i 和 i+1
    for k in range(len(order) - 1):
        a, b = order[k], order[k + 1]
        adj[a].add(b)
        adj[b].add(a)

    # 2) 同列上下相邻
    centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1)
               for x1, y1, x2, y2 in bboxes]
    w_med = float(np.median([c[2] for c in centers]))
    tol = max(1.0, 0.5 * w_med)

    order_by_x = sorted(range(n), key=lambda i: centers[i][0])
    cols: List[List[int]] = []
    col_mean_x: List[float] = []
    for i in order_by_x:
        cx = centers[i][0]
        if cols and abs(cx - col_mean_x[-1]) <= tol:
            cols[-1].append(i)
            col_mean_x[-1] = float(np.mean([centers[j][0] for j in cols[-1]]))
        else:
            cols.append([i])
            col_mean_x.append(cx)
    for col in cols:
        col.sort(key=lambda i: centers[i][1])
        for k in range(len(col) - 1):
            a, b = col[k], col[k + 1]
            adj[a].add(b)
            adj[b].add(a)

    return [sorted(s) for s in adj]


def _greedy_color(adj: List[List[int]], tiebreak_rank: List[int]) -> List[int]:
    """Welsh–Powell 贪心染色：按度数降序遍历，tiebreaker 用阅读顺序。

    返回每个节点的色号（从 0 开始）。
    """
    n = len(adj)
    nodes = sorted(range(n), key=lambda i: (-len(adj[i]), tiebreak_rank[i]))
    color = [-1] * n
    for i in nodes:
        used = {color[j] for j in adj[i] if color[j] >= 0}
        c = 0
        while c in used:
            c += 1
        color[i] = c
    return color


def _save_diff_triptych(page_arr: np.ndarray,
                        raw_mask: np.ndarray,
                        dilated_mask: np.ndarray,
                        contours_xy: List[np.ndarray],
                        bbox: tuple[int, int, int, int],
                        out_path: Path,
                        pad_ratio: float) -> None:
    """保存单 group 的三联诊断图（仅在 crop_dataset_debug/ 下用于诊断）。

    左：raw mask 裁剪（白底，仅 group 内所有 raw polygon 像素的并）
    中：flood mask 裁剪（白底，与 crop_dataset/ 输出一致）
    右：flood 裁剪 + 环带 (flood \\ raw) 淡黄半透明 +
        红色原 polygon 细线（每个成员一根） + 绿色 flood 轮廓粗线
    标题写出 pad_ratio 与环带像素数。
    """
    x1, y1, x2, y2 = bbox
    sub_img = page_arr[y1:y2, x1:x2]
    sub_raw = raw_mask[y1:y2, x1:x2]
    sub_dil = dilated_mask[y1:y2, x1:x2]

    raw_white = np.full_like(sub_img, 255)
    raw_white[sub_raw > 0] = sub_img[sub_raw > 0]

    dil_white = np.full_like(sub_img, 255)
    dil_white[sub_dil > 0] = sub_img[sub_dil > 0]

    # 环带：dilated AND NOT raw
    ring = cv2.bitwise_and(sub_dil, cv2.bitwise_not(sub_raw))
    ring_px = int((ring > 0).sum())

    diag = dil_white.copy()
    # 环带淡黄半透明（YELLOW=(255,235,120)，alpha=0.45）
    alpha = 0.45
    yellow = np.array([255, 235, 120], dtype=np.float32)
    ring_bool = ring > 0
    diag[ring_bool] = (alpha * yellow + (1 - alpha) * diag[ring_bool].astype(np.float32)
                       ).clip(0, 255).astype(np.uint8)

    # 红色原 polygon 细线（坐标减去 bbox 左上角；多成员 group 每个成员画一根）
    for poly in contours_xy:
        poly_local = np.round(poly - np.array([x1, y1])).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(diag, [poly_local], isClosed=True, color=(255, 0, 0), thickness=1)

    # 绿色 dilated 轮廓粗线
    cnts, _ = cv2.findContours(sub_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(diag, cnts, -1, (0, 200, 0), thickness=2)

    # 拼三联图
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    axes[0].imshow(raw_white); axes[0].set_title("raw mask"); axes[0].axis("off")
    axes[1].imshow(dil_white); axes[1].set_title("flood mask"); axes[1].axis("off")
    axes[2].imshow(diag)
    axes[2].set_title(f"diff (pad={pad_ratio:.2f}, ring={ring_px}px)\nred=raw, green=flood, yellow=ring")
    axes[2].axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def extract_from_page(
    page: Image.Image,
    page_stem: str,
    output_dir: Path,
    conf: float = 0.25,
    visualize: bool = False,
    bbox_pad_ratio: float = 0.10,
    debug_dir: Union[Path, None] = None,
    viz_dir: Union[Path, None] = None,
) -> List[Path]:
    """对单张页面图像运行实例分割模型，按"扩边 bbox + 沿黑像素洪水填充"的掩码裁剪并保存每个分子。

    流程：
      - 整页 Otsu 二值化 + 闭运算（close_r=3，粘原子标签字符）；
      - raw bbox = polygon 紧包围盒；扩边 bbox = 各方向独立 ±bbox_pad_ratio·各自边长
        （裁到图像边界），扩边后宽/高 = (1+2·ratio)·原值，中心不变；
      - 每个 YOLO 实例独立：在扩边 bbox 子图内 seed=raw polygon、host=raw ∪ black
        做 8 邻接形态学重建，不排除其它实例的 raw；
      - 按扩边 bbox 裁原图，非洪水保留像素抹成纯白。

    参数 bbox_pad_ratio：bbox 各方向外扩比例，单位 = 各自边长的倍数。
      默认 0.10 即左右各 +10%·宽、上下各 +10%·高，扩后 bbox 面积 ≈ 1.44× 原值。
      调大：洪水活动空间更大，能容纳更长的取代基链，也更容易把邻居拉进来。
      调小：可能装不下苯环 / 长烷基链伸出 polygon 的部分。

    YOLO 给出几个 polygon 就输出几张图；属于同一分子的多个实例不会被合并，
    它们的输出会包含同一分子的重叠内容（每张以各自 polygon 为种子）。

    若 debug_dir 不为 None，会额外为每个分子保存一张三联诊断图
    （raw / flood / 环带+双轮廓）到该目录。output_dir 内容不受影响。
    若 viz_dir 不为 None，页面级叠加图写到该目录而非默认的 vis/extract/。

    输出按阅读顺序（从左到右、从上到下，行聚类）依次编号。

    每个分子除 `<page_stem>_mol###.png` 外，还写一份 sidecar
    `<page_stem>_mol###.meta.npz`：
      mask          —— 洪水填充后的最终掩码（crop 局部，uint8，取值 {0,1}）
      polygon       —— YOLO 原始多边形（crop 局部坐标，float32，N×2）
      bbox_norm     —— 扩边 bbox（= crop 区域）归一化到整页 [0,1]，顺序
                       [x0, y0, x1, y1]
      bbox_raw_norm —— YOLO 多边形紧包围盒，同样归一化、同顺序（贴着分子本体）
      page_size     —— 整页尺寸 [h, w]（像素），把上述归一化值还原成像素坐标用
    """
    model = _get_model()
    page_arr = np.array(page)
    h, w = page_arr.shape[:2]
    results = model.predict(source=page_arr, conf=conf, classes=[MOL_CLASS_ID], verbose=False)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 整页黑像素二值图（Otsu + 闭运算粘原子标签字符），供洪水填充用。
    binary = _page_binary(page_arr, close_r=3)

    raw_masks: List[np.ndarray] = []
    bboxes: List[tuple[int, int, int, int]] = []  # 扩边后 bbox（裁剪硬上界）
    contours: List[np.ndarray] = []
    for result in results:
        masks = result.masks
        if masks is None or len(masks) == 0:
            continue
        for contour_xy in masks.xy:
            if len(contour_xy) < 3:
                continue
            mask = _contour_to_mask(contour_xy, (h, w))
            ys, xs = np.where(mask > 0)
            if xs.size == 0:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            raw_bbox = (x1, y1, x2, y2)
            exp_bbox = _expand_bbox_per_axis(raw_bbox, bbox_pad_ratio, h, w)
            raw_masks.append(mask)
            bboxes.append(exp_bbox)
            contours.append(np.asarray(contour_xy, dtype=np.float32))

    # 每个 YOLO 实例独立做一次洪水填充：seed = raw polygon，host = raw ∪ black，
    # 8 邻接 3×3 膨胀至稳定或撞扩边 bbox。不排除其它实例 raw。
    inst_bboxes: List[tuple[int, int, int, int]] = []
    inst_polygons: List[List[np.ndarray]] = []
    inst_raw_masks: List[np.ndarray] = []
    inst_final_masks: List[np.ndarray] = []
    crops: List[np.ndarray] = []

    for raw, bbox, contour in zip(raw_masks, bboxes, contours):
        x1, y1, x2, y2 = bbox
        sub_raw = raw[y1:y2, x1:x2]
        sub_black = binary[y1:y2, x1:x2]
        sub_host = cv2.bitwise_or(sub_raw, sub_black)
        sub_recon = _morph_reconstruct(sub_raw, sub_host)

        kept = np.zeros((h, w), dtype=np.uint8)
        kept[y1:y2, x1:x2] = sub_recon

        sub_img = page_arr[y1:y2, x1:x2].copy()
        sub_img[sub_recon == 0] = 255

        crops.append(sub_img)
        inst_bboxes.append(bbox)
        inst_polygons.append([contour])
        inst_raw_masks.append(raw)
        inst_final_masks.append(kept)

    order = _reading_order(inst_bboxes)
    saved: List[Path] = []
    for new_idx, old_idx in enumerate(order):
        out_path = output_dir / f"{page_stem}_mol{new_idx:03d}.png"
        Image.fromarray(crops[old_idx]).save(out_path)
        saved.append(out_path)

        # 写 sidecar：洪水掩码 + YOLO polygon（均换算为 crop 内局部坐标），
        # 供下游 post_segment.py 把"连通域范围"建立在 extract 的洪水结果上，
        # 并在 vis/post_seg/ overlay 里画出原 polygon 框。
        # 另存 bbox_norm / page_size：mask 与 polygon 都是 crop 局部坐标，单看
        # sidecar 无法知道这个分子在整页的哪一处；要还原页面级位置（前端在整页
        # 图上画检测框、后端按坐标排序）就必须额外落盘，否则信息在此处丢失。
        bx1, by1, bx2, by2 = inst_bboxes[old_idx]
        local_mask = inst_final_masks[old_idx][by1:by2, bx1:bx2].astype(np.uint8)
        poly_full = inst_polygons[old_idx][0]
        poly_local = (poly_full - np.array([bx1, by1], dtype=poly_full.dtype)).astype(np.float32)
        # 两套页面级框，都归一化到 [0,1]，顺序 [x0, y0, x1, y1]：
        #   bbox_norm     —— 扩边 bbox，即 crop 的实际区域（比分子本体大，
        #                    宽高各为紧包围盒的 1+2·bbox_pad_ratio 倍）
        #   bbox_raw_norm —— YOLO 多边形紧包围盒，贴合物本身的检测范围
        # 之所以都存：前端整页预览要画的是"检测到了什么"，贴合本体的紧框更准；
        # 而 crop 区域另有用途（回溯裁剪范围）。两者相差 20%（默认 ratio），
        # 只存一个则另一种口径只能靠猜。
        bbox_norm = np.array([bx1 / w, by1 / h, bx2 / w, by2 / h], dtype=np.float32)
        rys, rxs = np.where(inst_raw_masks[old_idx] > 0)
        bbox_raw_norm = (
            np.array([rxs.min(), rys.min(), rxs.max() + 1, rys.max() + 1], dtype=np.float32)
            / np.array([w, h, w, h], dtype=np.float32)
        )
        meta_path = output_dir / f"{page_stem}_mol{new_idx:03d}.meta.npz"
        np.savez_compressed(
            meta_path,
            mask=local_mask,
            polygon=poly_local,
            bbox_norm=bbox_norm,
            bbox_raw_norm=bbox_raw_norm,
            page_size=np.array([h, w], dtype=np.int32),
        )

        if debug_dir is not None:
            diff_path = debug_dir / f"{page_stem}_mol{new_idx:03d}_diff.png"
            _save_diff_triptych(
                page_arr=page_arr,
                raw_mask=inst_raw_masks[old_idx],
                dilated_mask=inst_final_masks[old_idx],
                contours_xy=inst_polygons[old_idx],
                bbox=inst_bboxes[old_idx],
                out_path=diff_path,
                pad_ratio=bbox_pad_ratio,
            )

    if visualize and inst_bboxes:
        _save_extract_overlay(page_arr, inst_polygons, inst_final_masks,
                              inst_bboxes, order, page_stem, (h, w),
                              out_dir=viz_dir)

    return saved


def _save_extract_overlay(
    page_rgb: np.ndarray,
    inst_polygons: List[List[np.ndarray]],
    inst_final_masks: List[np.ndarray],
    inst_bboxes: List[tuple[int, int, int, int]],
    order: List[int],
    page_stem: str,
    shape_hw: tuple[int, int],
    out_dir: Optional[Path] = None,
) -> None:
    """画 原始 polygon + 扩边 bbox + 阅读顺序编号 + 洪水保留区域的彩色叠加。

    每个 YOLO 实例一种 Tab10 颜色；相邻实例（阅读顺序里相邻 + 同列上下
    相邻）通过 Welsh–Powell 贪心染色保证邻居异色。注意：属于同一分子
    的多个实例不会被合并，它们的洪水保留区域可能在画面上重叠。

    out_dir 为 None 时写到默认 vis/extract/；传入目录则改写到该目录
    （供 eval 流水线把过程图落到评估输出树，见
    dataset_eval/eval_msirost/viz_gt_vs_pred.py）。
    """
    from matplotlib.patches import Polygon, Rectangle

    from utils.base.viz_utils import (
        COLOR_BBOX,
        VIZ_EXTRACT,
        save_overlay_figure,
    )

    n = len(inst_bboxes)
    # 阅读顺序排名：tiebreak 用
    rank = [0] * n
    for new_idx, old_idx in enumerate(order):
        rank[old_idx] = new_idx

    adj = _build_adjacency_reading(inst_bboxes, order)
    color_ids = _greedy_color(adj, tiebreak_rank=rank)
    colors_per_inst: List[tuple[int, int, int]] = [
        TAB10[c % len(TAB10)] for c in color_ids
    ]

    # 按色号分组叠加，每色号 OR 成一个 mask
    overlays = []
    color_groups: dict[int, np.ndarray] = {}
    for idx, m in enumerate(inst_final_masks):
        c = color_ids[idx]
        bin_m = (m > 0).astype(np.uint8)
        if c in color_groups:
            color_groups[c] = cv2.bitwise_or(color_groups[c], bin_m)
        else:
            color_groups[c] = bin_m.copy()
    for c, m in sorted(color_groups.items()):
        overlays.append((m.astype(bool), TAB10[c % len(TAB10)], 0.35))

    def _extras(ax):
        for idx, polys in enumerate(inst_polygons):
            col = colors_per_inst[idx]
            col_unit = (col[0] / 255, col[1] / 255, col[2] / 255)
            for c in polys:
                ax.add_patch(Polygon(c, closed=True, fill=False,
                                     edgecolor=col_unit, linewidth=1.0))
        for new_idx, old_idx in enumerate(order):
            x1, y1, x2, y2 = inst_bboxes[old_idx]
            col = colors_per_inst[old_idx]
            col_unit = (col[0] / 255, col[1] / 255, col[2] / 255)
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                   edgecolor=col_unit, linewidth=1.2))
            ax.text(x1, max(0, y1 - 4), f"mol{new_idx:03d}",
                    color="white", fontsize=8,
                    bbox=dict(facecolor=col_unit, edgecolor="none",
                              alpha=0.9, pad=1.5))

    n_colors_used = len(color_groups)
    target_dir = out_dir if out_dir is not None else VIZ_EXTRACT
    save_overlay_figure(
        img_rgb=page_rgb,
        overlays=overlays,
        title=f"[extract] {page_stem}  ({n} molecules, dilated, {n_colors_used} colors)",
        legend_items=[("kept (flood) mask", TAB10[0]),
                      ("bbox / polygon", COLOR_BBOX)],
        out_path=target_dir / f"{page_stem}_overlay.png",
        extras=_extras,
    )


def extract(input_path: Union[str, Path], conf: float = 0.25, visualize: bool = False,
            debug_crops: bool = True, output_dir: Union[str, Path, None] = None,
            debug_dir: Union[str, Path, None] = None,
            viz_dir: Union[str, Path, None] = None) -> List[Path]:
    """提取输入文件中的化学分子结构图像。

    Args:
        input_path: 单张页面图片或 PDF 文件路径。
        conf: 模型置信度阈值。
        visualize: 是否生成可视化叠加图到 vis/extract/。
        debug_crops: 是否同时为每个分子生成三联诊断图到 crop_dataset_debug/。
        output_dir: 裁剪图输出目录。None 用默认 crop_dataset/（服务化时传任务目录）。
        debug_dir: 三联诊断图目录。None 用默认 crop_dataset_debug/。
        viz_dir: 可视化叠加图目录。None 用 vis/extract/。

    Returns:
        保存到 output_dir 下的所有裁剪图片路径列表。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    out_dir = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if not debug_crops:
        dbg_dir = None
    else:
        dbg_dir = Path(debug_dir) if debug_dir is not None else DEBUG_OUTPUT_DIR
        dbg_dir.mkdir(parents=True, exist_ok=True)

    viz = Path(viz_dir) if viz_dir is not None else None
    suffix = input_path.suffix.lower()
    saved: List[Path] = []

    if suffix == ".pdf":
        for page_idx, page in enumerate(pdf_to_pages(input_path)):
            stem = f"{input_path.stem}_p{page_idx:03d}"
            saved.extend(extract_from_page(page, stem, out_dir, conf=conf,
                                           visualize=visualize, debug_dir=dbg_dir,
                                           viz_dir=viz))
    elif suffix in IMAGE_SUFFIXES:
        page = Image.open(input_path).convert("RGB")
        saved.extend(extract_from_page(page, input_path.stem, out_dir, conf=conf,
                                       visualize=visualize, debug_dir=dbg_dir,
                                       viz_dir=viz))
    else:
        raise ValueError(f"Unsupported input type: {suffix}. Expected PDF or image.")

    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract chemical structures from a page image or PDF.")
    parser.add_argument(
        "input",
        nargs="?",
        help="Path to a page image, PDF file, or directory. Defaults to project's input/ folder.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--visualize", action="store_true", help="生成可视化叠加图到 vis/extract/")
    parser.add_argument("--debug-crops", dest="debug_crops", action="store_true",
                        help="同时为每个分子生成三联诊断图到 crop_dataset_debug/（默认开启）")
    parser.add_argument("--no-debug-crops", dest="debug_crops", action="store_false",
                        help="关闭三联诊断图")
    parser.set_defaults(visualize=True, debug_crops=True)
    args = parser.parse_args()

    target = Path(args.input) if args.input else INPUT_DIR
    if target.is_dir():
        files = [p for p in sorted(target.iterdir())
                 if p.suffix.lower() in IMAGE_SUFFIXES or p.suffix.lower() == ".pdf"]
    else:
        files = [target]

    total = 0
    for f in files:
        results = extract(f, conf=args.conf, visualize=args.visualize,
                          debug_crops=args.debug_crops)
        print(f"[{f.name}] -> {len(results)} crops")
        total += len(results)
    print(f"Done. Saved {total} crops in total to {OUTPUT_DIR}")
