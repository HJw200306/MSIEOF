"""分子键语义分割掩膜生成模块。

策略（v7 —— cairosvg bond-only 重渲染）:
    1. 从 RDKit SVG 中移除所有非键元素 (atom text / 无 class 元素)
    2. 用 cairosvg 以同样的 scale 重新渲染 → 与原始图像使用同一渲染引擎
    3. bond-only 渲染图中的任意非白色像素 = 键像素
       — 完美对齐 (同引擎同 scale)
       — 天然排除文字 (已被移除)
       — 不受键颜色影响 (任何颜色 ≠ 白色)
       — 双键/楔形键/虚线键与可视结果完全一致
    4. pos_var 键补充 (不在 SVG 中)
    5. AP marker / 括号 overlay 条件移除
"""

import io
import re
import xml.etree.ElementTree as ET

import cairosvg
import cv2
import numpy as np
from PIL import Image

from utils.base.image_loader import ImageLoader

_img_loader = ImageLoader()

SVG_NS = '{http://www.w3.org/2000/svg}'

# AP marker style IDs —— 条件移除
_AP_STYLE_IDS = {'wave_ap', 'dash_ap', 'sine_ap', 'zigzag_ap', 'star_ap', 'wave_bond', 'dash_bond'}


# ============================================================
# 工具
# ============================================================
def _make_poly_mask(shape, polygonverts, expand=0):
    H, W = shape
    m = np.zeros((H, W), dtype=np.uint8)
    pts = np.array(polygonverts, dtype=np.float32).reshape(-1, 2).copy()
    if expand > 0:
        c = pts.mean(axis=0)
        d = pts - c
        n = np.linalg.norm(d, axis=1, keepdims=True)
        n = np.where(n < 1e-6, 1.0, n)
        pts = pts + d / n * expand
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)
    pts_int = (pts + 0.5).astype(np.int32)
    cv2.fillPoly(m, [pts_int], 255)
    return m


def _make_bond_only_svg(svg_text):
    """移除 SVG 中所有非键元素, 仅保留 <path class='bond-N'> 节点。"""
    root = ET.fromstring(svg_text)

    # 递归移除: 遍历所有元素, 收集要删的, 从父节点删除
    def _remove_non_bond(parent):
        to_remove = []
        for child in parent:
            tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag
            if tag == 'path':
                cls = child.get('class', '')
                if not re.match(r'bond-\d+', cls):
                    to_remove.append(child)
            elif tag == 'text':
                to_remove.append(child)
            # 递归处理子节点 (g, svg 等)
            _remove_non_bond(child)
        for child in to_remove:
            parent.remove(child)

    _remove_non_bond(root)
    return ET.tostring(root, encoding='unicode')


def _svg_to_grayscale(svg_text, scale):
    """将 SVG 文本用 cairosvg 渲染为灰度 numpy 数组。"""
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), scale=scale)
    cv_img = _img_loader.load_cv_img(png_bytes)
    return cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)


# ============================================================
# 1. Bond-only SVG 重渲染 → 完美键掩膜
# ============================================================
def extract_bond_mask_from_bond_svg(svg_text, svg_scale, ref_size):
    """用 cairosvg 重渲染 bond-only SVG, 生成与原始图像完全对齐的键掩膜。

    原理: 移除原子文本 / 非键 path 后的 SVG 用同一渲染引擎 +
    同一 scale 重新渲染, 渲染结果中任意非白像素即为键。

    Args:
        svg_text: RDKit 生成的完整 SVG 字符串
        svg_scale: 原始渲染使用的 scale
        ref_size: (H, W) 参考尺寸 —— 掩膜将 resize/crop 到此尺寸
                  以确保与原图尺寸完全一致

    Returns:
        mask: uint8, 255=键, 0=背景, shape=(H, W)
    """
    bond_svg = _make_bond_only_svg(svg_text)
    gray = _svg_to_grayscale(bond_svg, svg_scale)

    # 确保尺寸与参考一致 (cairosvg 两次渲染可能因浮点差 1px)
    H_ref, W_ref = ref_size
    H, W = gray.shape
    if H != H_ref or W != W_ref:
        fixed = np.full((H_ref, W_ref), 255, dtype=np.uint8)
        copy_h, copy_w = min(H, H_ref), min(W, W_ref)
        fixed[:copy_h, :copy_w] = gray[:copy_h, :copy_w]
        gray = fixed

    # 非白色像素 = 键 (threshold=250 宽容反锯齿灰边)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    return mask


# ============================================================
# 2. 暗色像素检测 (仅用于 pos_var 键补充)
# ============================================================
def detect_dark_pixels(cv_img, threshold=180):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return dark


# ============================================================
# 3. Overlay 处理
# ============================================================
def handle_overlay_priority(mask, cv_img, placed_overlays,
                             bond_obbs_indexed, removed_bonds, ap_meta_list=None,
                             pos_meta_list=None):
    if ap_meta_list is None:
        ap_meta_list = []
    if pos_meta_list is None:
        pos_meta_list = []
    H, W = mask.shape[:2]
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    ap_idx = 0
    pos_idx = 0

    for poly, style_id in placed_overlays:
        if poly is None or len(poly) < 3:
            continue

        poly_mask = _make_poly_mask((H, W), poly)

        if style_id in _AP_STYLE_IDS:
            if ap_idx < len(ap_meta_list):
                meta = ap_meta_list[ap_idx]
                is_endpoint = meta['is_endpoint']
                opaque_band = meta['opaque_band']
            else:
                is_endpoint = True
                opaque_band = False
            ap_idx += 1

            if is_endpoint == 'endpoint' or opaque_band:
                mask[poly_mask > 0] = 0
        elif style_id == 'pos_var':
            if pos_idx < len(pos_meta_list) and pos_meta_list[pos_idx].get('opaque_band', False):
                mask[poly_mask > 0] = 0
            else:
                roi_bool = poly_mask > 0
                total = np.sum(roi_bool)
                if total > 0:
                    white_ratio = np.sum((gray > 245) & roi_bool) / total
                    if white_ratio > 0.15:
                        mask[(gray > 245) & roi_bool] = 0
            pos_idx += 1
        else:
            roi_bool = poly_mask > 0
            total = np.sum(roi_bool)
            if total == 0:
                continue
            white_ratio = np.sum((gray > 245) & roi_bool) / total
            if white_ratio > 0.15:
                mask[(gray > 245) & roi_bool] = 0

    return mask


# ============================================================
# 4. pos_var 键补充 (不在 SVG 中, 仍用暗色像素提取)
# ============================================================
def extract_pos_var_bonds(cv_img, placed_overlays):
    H, W = cv_img.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    dark = detect_dark_pixels(cv_img, threshold=180)

    for poly, style_id in placed_overlays:
        if style_id != 'pos_var':
            continue
        poly_mask = _make_poly_mask((H, W), poly, expand=3)
        single = cv2.bitwise_and(dark, poly_mask)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        single = cv2.morphologyEx(single, cv2.MORPH_CLOSE, kernel_close)
        mask = cv2.bitwise_or(mask, single)

    return mask


# ============================================================
# 5. 掩膜平移辅助
# ============================================================
def shift_mask_to_match(mask, new_h, new_w, shift_dy, shift_dx):
    """画布扩张后将掩膜平移到新位置。"""
    shifted = np.zeros((new_h, new_w), dtype=np.uint8)
    H_old, W_old = mask.shape
    copy_h = min(H_old, new_h - shift_dy)
    copy_w = min(W_old, new_w - shift_dx)
    if copy_h > 0 and copy_w > 0 and shift_dy >= 0 and shift_dx >= 0:
        shifted[shift_dy:shift_dy + copy_h, shift_dx:shift_dx + copy_w] = \
            mask[:copy_h, :copy_w]
    return shifted


# ============================================================
# 6. 主入口
# ============================================================
def generate_bond_segmentation_mask(cv_img, bond_obbs_indexed, removed_bonds,
                                     placed_overlays, atom_boxes, root, svg_scale,
                                     ap_meta_list=None, pos_meta_list=None):
    """生成分子键语义分割掩膜。

    流程:
        1. Bond-only SVG cairosvg 重渲染 → 键掩膜
        2. pos_var 键补充
        3. Overlay 条件移除
        4. 移除孤立噪点
    """
    # 此函数已废弃, 保留兼容旧调用方。
    # 新流程在 mol_markush_gen 中直接调用分段函数。
    if ap_meta_list is None:
        ap_meta_list = []
    if pos_meta_list is None:
        pos_meta_list = []

    mask = extract_bond_mask_from_bond_svg(
        svg_text=None, svg_scale=svg_scale, ref_size=cv_img.shape[:2])
    # (svg_text=None 走不了此路径, 这里仅接口保留)

    pos_var_mask = extract_pos_var_bonds(cv_img, placed_overlays)
    mask = cv2.bitwise_or(mask, pos_var_mask)

    mask = handle_overlay_priority(mask, cv_img, placed_overlays,
                                    bond_obbs_indexed, removed_bonds,
                                    ap_meta_list, pos_meta_list)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < 4:
            mask[labels == i] = 0

    return mask


# ============================================================
# 6. 逐键掩膜提取: 解析 SVG bond path → OpenCV 逐键绘制 → 精确 per-bond mask
# ============================================================
def extract_per_bond_masks_from_svg(svg_root, svg_scale, ref_size):
    """从 RDKit SVG root 中提取每个 bond 的独立像素掩膜。

    对每个 bond-N path, 解析其 d 属性中的线段端点, 用 OpenCV 以与 cairosvg
    相同的 scale 绘制到独立掩膜上, 得到每个键的精确前景像素区域。

    Args:
        svg_root: ElementTree root of the full SVG
        svg_scale: same scale used for cairosvg rendering
        ref_size: (H, W) of the reference image

    Returns:
        {bond_idx: mask (H,W) uint8}, 仅包含有前景像素的键。
    """
    H_ref, W_ref = ref_size
    per_bond = {}

    for path in svg_root.iter(f'{SVG_NS}path'):
        class_attr = path.get('class', '')
        m = re.match(r'bond-(\d+)\b', class_attr)
        if not m:
            continue
        bond_idx = int(m.group(1))
        d_attr = path.get('d', '')

        # 解析 stroke-width
        style = path.get('style', '')
        sw = 2.0  # default
        sm = re.search(r'stroke-width\s*:\s*([\d.]+)', style)
        if sm:
            sw = float(sm.group(1))

        # 解析路径中的线段端点: M x,y L x,y ...
        pts_svg = []
        for cm in re.finditer(r'[ML]\s*([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)', d_attr):
            pts_svg.append((float(cm.group(1)), float(cm.group(2))))

        if len(pts_svg) < 2:
            continue

        mask = np.zeros((H_ref, W_ref), dtype=np.uint8)
        thickness_px = max(1, int(round(sw * svg_scale * 0.95)))

        for i in range(0, len(pts_svg) - 1, 2):
            p1 = np.array(pts_svg[i]) * svg_scale
            p2 = np.array(pts_svg[i + 1]) * svg_scale
            # 确保在画布范围内
            p1_int = (int(round(np.clip(p1[0], 0, W_ref - 1))),
                      int(round(np.clip(p1[1], 0, H_ref - 1))))
            p2_int = (int(round(np.clip(p2[0], 0, W_ref - 1))),
                      int(round(np.clip(p2[1], 0, H_ref - 1))))
            cv2.line(mask, p1_int, p2_int, 255, thickness=thickness_px,
                     lineType=cv2.LINE_AA)

        if np.any(mask):
            per_bond[bond_idx] = mask

    return per_bond


def compute_per_bond_lengths(per_bond_masks):
    """从逐键掩膜计算每个键的像素长度 (minAreaRect 长轴)。

    Returns:
        {bond_idx: length_px}
    """
    lengths = {}
    for bond_idx, mask in per_bond_masks.items():
        ys, xs = np.where(mask)
        if len(ys) < 3:
            continue
        pts = np.column_stack([xs, ys]).astype(np.float32)
        try:
            rect = cv2.minAreaRect(pts)
            w, h = rect[1]
            lengths[bond_idx] = max(w, h)
        except Exception:
            continue
    return lengths


def filter_candidates_by_bond_length(candidates, per_bond_lengths,
                                      min_ratio=0.55, hard_min_px=8.0):
    """按键长过滤候选, 剔除过短键上的候选。

    每个候选的关联键长必须 ≥ max(min_ratio × 平均键长, hard_min_px)。

    Args:
        candidates: list of dict, 每项至少含 'bond_idx' 或 'left_bond_idx'/'right_bond_idx'
        per_bond_lengths: {bond_idx: length_px}
        min_ratio: 相对平均键长的最小比例
        hard_min_px: 绝对最小像素长度 (兜底)

    Returns:
        (filtered_candidates, avg_len, threshold)
    """
    if not candidates or not per_bond_lengths:
        return candidates, 0.0, 0.0

    all_lengths = list(per_bond_lengths.values())
    avg_len = float(np.mean(all_lengths)) if all_lengths else 20.0
    threshold = max(avg_len * min_ratio, hard_min_px)

    filtered = []
    for cand in candidates:
        # 收集候选涉及的 bond_idx
        bond_ids = set()
        if 'bond_idx' in cand:
            bond_ids.add(cand['bond_idx'])
        if 'left_bond_idx' in cand:
            bond_ids.add(cand['left_bond_idx'])
        if 'right_bond_idx' in cand:
            bond_ids.add(cand['right_bond_idx'])
        if 'on_bond_idx' in cand:
            bond_ids.add(cand['on_bond_idx'])

        if not bond_ids:
            filtered.append(cand)
            continue

        # 所有关联键都必须达标
        ok = True
        for bi in bond_ids:
            bl = per_bond_lengths.get(bi, avg_len)  # 无数据则假设达标
            if bl < threshold:
                ok = False
                break
        if ok:
            filtered.append(cand)

    return filtered, avg_len, threshold


# ============================================================
# 7. 可视化
# ============================================================
def visualize_bond_mask(cv_img, mask, alpha=0.5):
    vis = cv_img.copy()
    overlay = np.zeros_like(vis)
    overlay[mask > 0] = [0, 0, 255]
    return cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
