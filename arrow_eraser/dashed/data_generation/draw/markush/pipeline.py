"""Markush 变异共享原语。

设计目标: 让每个 Markush 变异 (连接点、括号、片段嫁接...) 只需要回答三件事:
    1. 我的候选原子/键是什么?         -> find_*_candidates(mol)
    2. 我的 overlay 长什么样?         -> 各 variation 自己渲染 PIL RGBA
    3. overlay 的哪个像素该贴在哪里?  -> paste_with_anchor(...)

抠下来的几何/光栅化细节集中在这里, 避免在每个 variation 里重复实现旋转矩阵 +
alpha 复合 + 画布扩边逻辑。
"""

import math
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
from rdkit import Chem

# 复用 drawer.py 中已经过验证的图像粘贴 / 透明处理工具,
# 避免重复实现 alpha 复合 + 画布扩边逻辑。
from draw.drawer import (
    paste_rotated_image as _paste_centered,
    remove_transparent_areas,
)

NS = '{http://www.w3.org/2000/svg}'


# ============================================================
# 1. 原子像素坐标 (覆盖所有原子, 包括未渲染文本的碳)
# ============================================================
def get_atom_pixel_coords(drawer, n_atoms, svg_scale):
    """drawer.GetDrawCoords 在 FinishDrawing 之后依然有效, 返回 SVG 单位坐标,
    乘上 svg_scale 即为 PNG 像素坐标。这里覆盖每一个原子 (包含纯碳, 不依赖
    collect_atom_boxes 那种仅对'画出文本'的原子才能拿到坐标的方式)。"""
    out = {}
    for i in range(n_atoms):
        p = drawer.GetDrawCoords(i)
        out[i] = (p.x * svg_scale, p.y * svg_scale)
    return out


# ============================================================
# 2. 键 OBB —— 按 bond_idx 索引
# ============================================================
def collect_bond_obbs_indexed(root, scale, default_width=2.0):
    """与 mol_obb_gen.collect_bond_obbs 同义, 但返回 {bond_idx: 4x2 polygon},
    方便变异时按 bond_idx 精确定位 / 抹除。"""
    bond_paths = {}
    for path in root.iter(f'{NS}path'):
        class_attr = path.get('class', '')
        m = re.match(r'bond-(\d+)\b', class_attr)
        if not m:
            continue
        bond_idx = int(m.group(1))
        d = path.get('d', '')
        pts = []
        for cm in re.finditer(r'[ML]\s*([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)', d):
            pts.append((float(cm.group(1)), float(cm.group(2))))
        style = path.get('style', '')
        sw = default_width
        sm = re.search(r'stroke-width\s*:\s*([\d.]+)', style)
        if sm:
            sw = float(sm.group(1))
        info = bond_paths.setdefault(bond_idx, {'pts': [], 'width': sw})
        info['pts'].extend(pts)
        info['width'] = max(info['width'], sw)

    out = {}
    for bond_idx, info in bond_paths.items():
        pts = info['pts']
        if len(pts) < 2:
            continue
        half_w = info['width'] * 0.5 * 0.9
        expanded = []
        for i in range(0, len(pts) - 1, 2):
            p1 = np.array(pts[i])
            p2 = np.array(pts[i + 1])
            v = p2 - p1
            n = np.linalg.norm(v)
            if n < 1e-6:
                expanded.append(p1)
                continue
            perp = np.array([-v[1], v[0]]) / n * half_w
            expanded.extend([p1 + perp, p1 - perp, p2 + perp, p2 - perp])
        expanded.extend([np.array(p) for p in pts])
        arr = np.array(expanded, dtype=np.float32) * scale
        rect = cv2.minAreaRect(arr)
        out[bond_idx] = cv2.boxPoints(rect)
    return out


# ============================================================
# 3. 候选筛选: 端点碳 + 单键 (AP 用)
#    其他变异以后在这里追加 find_*_candidates 即可。
# ============================================================
def find_terminal_c_single_bond_candidates(mol):
    """返回 [{atom_idx, neighbor_idx, bond_idx}, ...]:
        - 元素 = C
        - degree == 1
        - 唯一的那根键是单键
        - 不在芳香环里 (避免破坏苯环结构)
        - 键当前没有立体方向 (BondDir == NONE)
        - 端点 C 与其邻居都不是 stereo center (CHI_UNSPECIFIED) ——
          这是为了挡住 "find 时 BondDir 还是 NONE, 但 set_liti / DrawMolecule 之后会
          被自动渲染成楔形键" 的情况. 楔形键即使是单键也不应被替换为 AP marker
          (会丢失立体语义)。

    用于 ap_variation.draw_attachment_points。
    """
    candidates = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or atom.GetAtomicNum() == 0:
            continue
        if atom.GetDegree() != 1:
            continue
        if atom.GetIsAromatic():
            continue
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        bonds = atom.GetBonds()
        if len(bonds) != 1:
            continue
        bond = bonds[0]
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.GetBondDir() != Chem.BondDir.NONE:
            continue
        neighbor = bond.GetOtherAtom(atom)
        if neighbor.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        candidates.append({
            'atom_idx': atom.GetIdx(),
            'neighbor_idx': neighbor.GetIdx(),
            'bond_idx': bond.GetIdx(),
        })
    return candidates


# ============================================================
# 4. 去背景: 把要被 overlay 替换掉的原始像素涂白
# ============================================================
def erase_polygons(cv_img, polygons, color=(255, 255, 255), inflate=0):
    """把 polygons 列表里的每个多边形区域用 `color` 填充到 cv_img (in-place)。
    inflate > 0 时会先沿外法线方向稍微扩大多边形, 用于覆盖键末端的描边毛刺。
    polygons 可以是 list[Nx2] 或 list[None] (None 跳过)。
    """
    if not polygons:
        return
    H, W = cv_img.shape[:2]
    for poly in polygons:
        if poly is None:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 3:
            continue
        if inflate > 0:
            center = pts.mean(axis=0)
            dirs = pts - center
            norms = np.linalg.norm(dirs, axis=1, keepdims=True)
            norms = np.where(norms < 1e-6, 1.0, norms)
            pts = pts + dirs / norms * inflate
        ipts = np.clip(np.round(pts).astype(np.int32), 0, None)
        ipts[:, 0] = np.minimum(ipts[:, 0], W - 1)
        ipts[:, 1] = np.minimum(ipts[:, 1], H - 1)
        cv2.fillPoly(cv_img, [ipts], color)


# ============================================================
# 5. 几何小工具
# ============================================================
def screen_angle_deg(p_from, p_to):
    """返回从 p_from 指向 p_to 的角度 (度数, 屏幕坐标系: y 向下).
    水平向右 = 0°, 向下 = 90°, 向左 = 180°/-180°, 向上 = -90°。"""
    dx = p_to[0] - p_from[0]
    dy = p_to[1] - p_from[1]
    return math.degrees(math.atan2(dy, dx))


def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# ============================================================
# 6. 旋转 + 锚点对齐 + 粘贴 —— Markush overlay 的核心原语
# ============================================================
def rotate_pil_around_anchor(pil_rgba, angle_deg, anchor_xy):
    """围绕 overlay 内的 (anchor_x, anchor_y) 像素点旋转 angle_deg 度,
    返回 (旋转后图像, 锚点在新图中的位置).

    angle_deg 的语义按屏幕坐标 (Y 向下) 解读: 0 = 不旋转, 90 = 把原本指向 +x
    的方向旋转到 +y (屏幕向下)。等价于 "希望 overlay 的原始水平指向最终在屏幕上
    指向 angle_deg 方向"。
    """
    if not isinstance(pil_rgba, Image.Image):
        pil_rgba = Image.fromarray(pil_rgba)
    pil_rgba = pil_rgba.convert('RGBA')
    w, h = pil_rgba.size
    ax, ay = anchor_xy

    # 先把 overlay 贴到一个足够大 (够容纳任意旋转) 的画布中央, 让 anchor 落在画布中心,
    # 再围绕画布中心旋转 —— 这样 anchor 在旋转后依然位于画布中心。
    diag = int(math.ceil(math.hypot(w, h))) + 2
    canvas_size = (w + 2 * diag, h + 2 * diag)
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    cx, cy = canvas_size[0] // 2, canvas_size[1] // 2
    paste_xy = (int(round(cx - ax)), int(round(cy - ay)))
    canvas.paste(pil_rgba, paste_xy)

    # PIL.Image.rotate 的 angle 在 PIL 文档里写作 "CCW 度数"。在屏幕 (Y 向下) 上,
    # PIL CCW = 数学 CW; 即 PIL rotate(α) 会把屏幕方向 0° 旋到屏幕方向 -α。
    # 我们要让屏幕方向 0° 旋到屏幕方向 angle_deg, 所以传入 -angle_deg。
    rotated = canvas.rotate(-angle_deg, center=(cx, cy), expand=True,
                            resample=Image.Resampling.BICUBIC,
                            fillcolor=(0, 0, 0, 0))
    pre_w, pre_h = rotated.size  # expand=True 后的画布尺寸 (含透明边)

    bbox = rotated.getbbox()
    if bbox is None:
        return rotated, (rotated.width / 2, rotated.height / 2)
    cropped = rotated.crop(bbox)

    # PIL rotate(expand=True) 把"旋转前 canvas 中心"映射到"旋转后 canvas 中心",
    # 因此 anchor 在 pre-crop 坐标里就是 (pre_w/2, pre_h/2)。再扣掉 bbox 左上偏移
    # 即可得到 post-crop 坐标。
    new_anchor = (pre_w / 2 - bbox[0], pre_h / 2 - bbox[1])
    return cropped, new_anchor


def paste_with_anchor(background_cv, pil_overlay, world_anchor, overlay_anchor):
    """把 pil_overlay 粘到 background_cv 上, 使 overlay_anchor 像素落到 world_anchor。
    如果 overlay 越过画布边缘, 画布会向对应方向扩张白边, 返回 (new_bg, (off_x, off_y)) ——
    off_x/off_y 是 background_cv 原坐标系到新画布坐标系的平移量, 调用方必须把它应用到
    所有此前已记录的坐标上 (atom_boxes, bond_obbs, 之前粘贴过的 overlay polygon)。
    """
    if not isinstance(pil_overlay, Image.Image):
        pil_overlay = Image.fromarray(pil_overlay)
    pil_overlay = pil_overlay.convert('RGBA')
    w, h = pil_overlay.size
    ax, ay = overlay_anchor

    # paste_centered 会把 image 中心对齐到 target_point。我们想让 (ax, ay) 对齐到
    # world_anchor, 所以图像中心应当对齐到 world_anchor + (w/2 - ax, h/2 - ay)。
    center_x = int(round(world_anchor[0] + w / 2 - ax))
    center_y = int(round(world_anchor[1] + h / 2 - ay))
    new_bg, off_x, off_y = _paste_centered(background_cv, pil_overlay, (center_x, center_y))
    return new_bg, (off_x, off_y)


def compute_pasted_obb(pil_overlay, world_anchor, overlay_anchor, canvas_off):
    """根据 paste_with_anchor 的输入参数 + 返回的偏移, 计算 overlay 在新画布坐标系下的
    紧致 OBB 多边形 (用 overlay 的 alpha mask 求 minAreaRect)。
    返回 (4, 2) np.float32 顶点。若 overlay 无非透明像素返回 None。
    """
    arr = np.array(pil_overlay.convert('RGBA'))
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if xs.size < 3:
        return None

    w, h = pil_overlay.size
    ax, ay = overlay_anchor
    # paste_with_anchor 内部用 int(round(world_anchor + w/2 - ax)) 作为图像中心,
    # 再用 w // 2 / h // 2 算左上角。我们必须严格复刻这套整数算术, 不能简化为
    # "world_anchor - overlay_anchor", 否则会产生 ±1 像素的亚像素位差。
    center_x = int(round(world_anchor[0] + w / 2 - ax)) + canvas_off[0]
    center_y = int(round(world_anchor[1] + h / 2 - ay)) + canvas_off[1]
    top_left_x = center_x - w // 2
    top_left_y = center_y - h // 2

    pts = np.column_stack([xs + top_left_x, ys + top_left_y]).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    return cv2.boxPoints(rect).astype(np.float32)


# ============================================================
# 7. 平移坐标 (便于 AP 粘贴后或 padding 后批量更新各项几何记录)
# ============================================================
def shift_axis_boxes(boxes_dict, dx, dy):
    """axis-aligned box: [x1, y1, x2, y2]"""
    if not dx and not dy:
        return
    for k, b in boxes_dict.items():
        boxes_dict[k] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy]


def shift_polygons_dict(d, dx, dy):
    """dict of (N, 2) numpy polygons"""
    if not dx and not dy:
        return
    for k in d:
        d[k] = d[k].copy()
        d[k][:, 0] += dx
        d[k][:, 1] += dy


def shift_polygon_list(lst, dx, dy):
    """list of (polygon (N,2) np.float32, ...other fields)"""
    if not dx and not dy:
        return
    for i, item in enumerate(lst):
        poly = item[0].copy()
        poly[:, 0] += dx
        poly[:, 1] += dy
        lst[i] = (poly,) + tuple(item[1:])


def shift_atom_pixel_coords(coords, dx, dy):
    if not dx and not dy:
        return
    for k in coords:
        x, y = coords[k]
        coords[k] = (x + dx, y + dy)


# ============================================================
# 8. RDKit 2D 坐标旋转 —— 让括号更多以水平方式绘制
# ============================================================
def rotate_mol_2d_coords(mol, angle_deg):
    """绕分子质心旋转全部原子的 2D 坐标 angle_deg 度。

    在 AllChem.Compute2DCoords 之后、DrawMolecule 之前调用,
    后续的 drawer.GetDrawCoords / SVG 渲染全部基于旋转后的坐标。
    """
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    if n == 0:
        return
    cx = sum(conf.GetAtomPosition(i).x for i in range(n)) / n
    cy = sum(conf.GetAtomPosition(i).y for i in range(n)) / n
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    for i in range(n):
        pos = conf.GetAtomPosition(i)
        dx, dy = pos.x - cx, pos.y - cy
        conf.SetAtomPosition(i, (cx + dx * cos_a - dy * sin_a,
                                  cy + dx * sin_a + dy * cos_a,
                                  0.0))


def compute_bracket_axis_angle_deg(mol, chain_candidates, ring1_candidates=None):
    """从 chain / ring1 候选里收集 bracket 轴方向, 用圆均值估算主方向 (度)。

    返回 None 表示没有足够候选来判定方向。
    """
    conf = mol.GetConformer()
    angles = []

    def _bond_mid_xy(bond):
        a = conf.GetAtomPosition(bond.GetBeginAtomIdx())
        b = conf.GetAtomPosition(bond.GetEndAtomIdx())
        return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)

    for c in chain_candidates:
        b_left = mol.GetBondWithIdx(c['left_bond_idx'])
        b_right = mol.GetBondWithIdx(c['right_bond_idx'])
        lx, ly = _bond_mid_xy(b_left)
        rx, ry = _bond_mid_xy(b_right)
        dx, dy = rx - lx, ry - ly
        if math.hypot(dx, dy) > 1e-6:
            # 括号轴方向; 括号高度方向 ⟂ axis, 我们希望 axis 接近水平
            angles.append(math.atan2(dy, dx))

    if ring1_candidates:
        for c in ring1_candidates:
            a_idx = c['atom_idx']
            other_idx = c['on_bond_other']
            ax = conf.GetAtomPosition(a_idx)
            ox = conf.GetAtomPosition(other_idx)
            lx = (ax.x + ox.x) / 2.0
            ly = (ax.y + ox.y) / 2.0
            rx = 2 * ax.x - lx
            ry = 2 * ax.y - ly
            dx, dy = rx - lx, ry - ly
            if math.hypot(dx, dy) > 1e-6:
                angles.append(math.atan2(dy, dx))

    if not angles:
        return None

    # 圆均值: 括号轴方向在 [0, π) 范围内 (括号不分正反方向)
    # 所以先 ×2 把 π 周期映射到 2π, 求均值后再 ÷2
    sin_sum = sum(math.sin(2 * a) for a in angles)
    cos_sum = sum(math.cos(2 * a) for a in angles)
    mean_2a = math.atan2(sin_sum, cos_sum)
    mean_rad = mean_2a / 2.0
    # 归一化到 (-90°, 90°] (即水平 ±90°)
    mean_deg = math.degrees(mean_rad)
    if mean_deg > 90:
        mean_deg -= 180
    elif mean_deg <= -90:
        mean_deg += 180
    return mean_deg


# ============================================================
# 9. 共线单键对查找 —— 前置括号判定
# ============================================================
def _bond_direction_xy(mol, bond_idx):
    """返回键的方向向量 (dx, dy) 在 RDKit 坐标空间, 未归一化。"""
    bond = mol.GetBondWithIdx(bond_idx)
    conf = mol.GetConformer()
    a = conf.GetAtomPosition(bond.GetBeginAtomIdx())
    b = conf.GetAtomPosition(bond.GetEndAtomIdx())
    return (b.x - a.x, b.y - a.y)


def _bond_midpoint_xy(mol, bond_idx):
    bond = mol.GetBondWithIdx(bond_idx)
    conf = mol.GetConformer()
    a = conf.GetAtomPosition(bond.GetBeginAtomIdx())
    b = conf.GetAtomPosition(bond.GetEndAtomIdx())
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


# ============================================================
# 10. 重叠检查工具
# ============================================================
# 重叠检查用的类别标签 (简化三分组)
# 0 = text (atom_text / other_text)
# 1 = bond
# 2 = overlay (bracket / AP / pos_var)
_CAT_TEXT = 0
_CAT_BOND = 1
_CAT_OVERLAY = 2


def check_candidate_overlap(candidate_poly, existing_polys, img_shape,
                            strict_threshold=0.20, lenient_threshold=0.50):
    """检查 candidate_poly 是否与已有多边形过度重叠 (分类别容忍度)。

    策略:
        - 候选 vs 文本/overlay → strict (低容忍, 默认 0.20)
        - 候选 vs 仅键 → lenient (高容忍, 默认 0.50)

    Args:
        candidate_poly: (N,2) np.ndarray — 候选多边形顶点
        existing_polys: list of (poly, category) — category: 0=text, 1=bond, 2=overlay
        img_shape: (H, W) — 图像尺寸
        strict_threshold: 文本/overlay 重叠阈值
        lenient_threshold: 键重叠阈值

    Returns:
        True 若候选应被拒绝 (重叠过度), 否则 False.
    """
    if not existing_polys:
        return False
    H, W = img_shape[:2]

    # 分离 strict (text + overlay) 和全部已有多边形
    strict_polys = [p for p, cat in existing_polys if cat in (_CAT_TEXT, _CAT_OVERLAY)]
    all_polys = [p for p, _ in existing_polys]

    def _build_map(polys):
        cm = np.zeros((H, W), dtype=np.uint8)
        for poly in polys:
            ipoly = np.array(poly, dtype=np.int32).reshape(-1, 2)
            x1 = max(0, int(ipoly[:, 0].min()))
            y1 = max(0, int(ipoly[:, 1].min()))
            x2 = min(W, int(ipoly[:, 0].max()) + 1)
            y2 = min(H, int(ipoly[:, 1].max()) + 1)
            if x2 <= x1 or y2 <= y1:
                continue
            h, w = y2 - y1, x2 - x1
            m = np.zeros((h, w), dtype=np.uint8)
            shifted = ipoly.copy()
            shifted[:, 0] -= x1
            shifted[:, 1] -= y1
            cv2.fillPoly(m, [shifted], 1)
            cm[y1:y2, x1:x2] += m
        return cm

    def _check(candidate, count_map, threshold):
        ipoly = np.array(candidate, dtype=np.int32).reshape(-1, 2)
        x1 = max(0, int(ipoly[:, 0].min()))
        y1 = max(0, int(ipoly[:, 1].min()))
        x2 = min(W, int(ipoly[:, 0].max()) + 1)
        y2 = min(H, int(ipoly[:, 1].max()) + 1)
        if x2 <= x1 or y2 <= y1:
            return False
        h, w = y2 - y1, x2 - x1
        m = np.zeros((h, w), dtype=np.uint8)
        shifted = ipoly.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1
        cv2.fillPoly(m, [shifted], 1)
        area = int(m.sum())
        if area < 4:
            return False
        cm_slice = count_map[y1:y2, x1:x2]
        ov = int(((cm_slice > 0) & (m > 0)).sum())
        return ov / area > threshold

    # 先检查 strict (文本+overlay) 重叠
    if strict_polys:
        strict_map = _build_map(strict_polys)
        if _check(candidate_poly, strict_map, strict_threshold):
            return True

    # 再检查全部重叠 (含键)
    if all_polys:
        total_map = _build_map(all_polys)
        if _check(candidate_poly, total_map, lenient_threshold):
            return True

    return False


def estimate_overlay_obb(pil_size, world_anchor, overlay_anchor, angle_deg=0.0):
    """在不实际粘贴的情况下估算 overlay 在世界坐标系中的 OBB。

    模拟 rotate_pil_around_anchor + paste_with_anchor 的几何变换,
    用于放置前预检查重叠。

    Args:
        pil_size: (w, h) PIL 图像尺寸
        world_anchor: (x, y) 世界坐标锚点
        overlay_anchor: (x, y) PIL 图像内锚点
        angle_deg: 屏幕坐标系旋转角度 (度)

    Returns:
        (4, 2) np.float32 估算 OBB 顶点
    """
    w, h = pil_size
    ax, ay = overlay_anchor
    wx, wy = world_anchor

    # 四角相对锚点的偏移
    corners = np.array([
        [-ax, -ay],
        [w - ax, -ay],
        [w - ax, h - ay],
        [-ax, h - ay],
    ], dtype=np.float64)

    if abs(angle_deg) > 0.1:
        # PIL rotate(-angle_deg) 等价于屏幕 CW 旋转 angle_deg
        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rx = corners[:, 0] * cos_a + corners[:, 1] * sin_a
        ry = -corners[:, 0] * sin_a + corners[:, 1] * cos_a
        corners[:, 0] = rx
        corners[:, 1] = ry

    corners[:, 0] += wx
    corners[:, 1] += wy
    return corners.astype(np.float32)


def _build_existing_polys(atom_boxes, bond_obbs_indexed, placed_overlays,
                          exclude_bond_ids=None):
    """构建已有多边形列表 (含类别标签), 供 check_candidate_overlap 使用。

    Args:
        exclude_bond_ids: 可选, set of bond_idx 需要从已有多边形中排除。
            用于 AP/位置变异放置在键上时排除自身目标键。

    Returns:
        list of (poly (4,2) np.float32, category int).
        category: 0=_CAT_TEXT (atom/other_text), 1=_CAT_BOND, 2=_CAT_OVERLAY
    """
    exclude = exclude_bond_ids or set()
    polys = []
    for box in atom_boxes.values():
        x1, y1, x2, y2 = box
        polys.append((np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32),
                      _CAT_TEXT))
    for bidx, obb in bond_obbs_indexed.items():
        if bidx in exclude:
            continue
        polys.append((np.array(obb, dtype=np.float32), _CAT_BOND))
    for ov_poly, _ in placed_overlays:
        polys.append((np.array(ov_poly, dtype=np.float32), _CAT_OVERLAY))
    return polys


def find_collinear_bond_pair_candidates(mol, *, angle_tol_deg=15.0, min_separation=0.20):
    """查找共线的两个单键对, 优先用于括号绘制。

    在 RDKit 2D 坐标中, zigzag 链上交替的键方向分别为 ±30° (相对水平),
    键中点连线则几乎水平。判定条件:
        - 两个键都是无环单键
        - 方向几乎平行 (夹角 < angle_tol_deg, 两个都在 ~+30° 或都在 ~-30°)
        - 两键不共用原子
        - 两键中点距离 > min_separation × 分子最大跨度
        - 两键之间路径的原子都是非环非芳香 (保证括号不跨环)

    不要求中点连线与键方向平行 —— zigzag 链的中点连线是水平的,
    而键方向是 ±30°, 它们本身就不平行。

    Returns:
        list[{'left_bond_idx', 'right_bond_idx', 'path_atoms', 'inner_atoms', 'dist'}]
        按两键中点间距降序排列 (优先用跨度最大的对).
    """
    tol = math.radians(angle_tol_deg)
    cos_tol = math.cos(tol)

    # 候选键: acyclic 单键, 非楔形/虚线, 非手性中心
    elig = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        if bond.IsInRing():
            continue
        if bond.GetBondDir() != Chem.BondDir.NONE:
            continue
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if b.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        elig.append(bond.GetIdx())

    if len(elig) < 2:
        return []

    # 预计算键方向 & 中点
    dirs = {}
    mids = {}
    for bi in elig:
        dirs[bi] = _bond_direction_xy(mol, bi)
        mids[bi] = _bond_midpoint_xy(mol, bi)

    # 计算分子最大跨度作为 min_separation 基准
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    ys = [conf.GetAtomPosition(i).y for i in range(mol.GetNumAtoms())]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    min_dist = span * min_separation

    cands = []
    for i in range(len(elig)):
        for j in range(i + 1, len(elig)):
            bi, bj = elig[i], elig[j]
            bond_i = mol.GetBondWithIdx(bi)
            bond_j = mol.GetBondWithIdx(bj)

            # 共享原子检查
            shared = {bond_i.GetBeginAtomIdx(), bond_i.GetEndAtomIdx()} & \
                     {bond_j.GetBeginAtomIdx(), bond_j.GetEndAtomIdx()}
            if shared:
                continue

            # 方向检查: 平行 或 反平行 (即两个键"同向")
            dxi, dyi = dirs[bi]
            dxj, dyj = dirs[bj]
            len_i = math.hypot(dxi, dyi)
            len_j = math.hypot(dxj, dyj)
            if len_i < 1e-6 or len_j < 1e-6:
                continue
            dot = (dxi * dxj + dyi * dyj) / (len_i * len_j)
            if abs(dot) < cos_tol:  # |cos(θ)| ≈ 1 且 > cos_tol
                continue

            # 中点间距
            mx_i, my_i = mids[bi]
            mx_j, my_j = mids[bj]
            len_mid = math.hypot(mx_j - mx_i, my_j - my_i)
            if len_mid < min_dist:
                continue

            # 路径检查: 两键之间可通过非环非芳香路径连接
            b_i_atoms = {bond_i.GetBeginAtomIdx(), bond_i.GetEndAtomIdx()}
            b_j_atoms = {bond_j.GetBeginAtomIdx(), bond_j.GetEndAtomIdx()}
            path_ok = False
            for start_a in b_i_atoms:
                if path_ok:
                    break
                for end_a in b_j_atoms:
                    if start_a == end_a:
                        path_ok = True
                        break
                    visited = {start_a}
                    queue = [(start_a, [start_a])]
                    while queue:
                        node, path = queue.pop(0)
                        if node == end_a:
                            path_ok = True
                            break
                        atom = mol.GetAtomWithIdx(node)
                        for nb in atom.GetNeighbors():
                            nbi = nb.GetIdx()
                            if nbi in visited:
                                continue
                            if nb.GetIsAromatic() or nb.IsInRing():
                                continue
                            visited.add(nbi)
                            queue.append((nbi, path + [nbi]))
                    if path_ok:
                        break
                if not path_ok:
                    break

            if not path_ok:
                continue

            cands.append({
                'left_bond_idx': bi,
                'right_bond_idx': bj,
                'path_atoms': [],
                'inner_atoms': [],
                'dist': len_mid,
            })

    cands.sort(key=lambda c: -c['dist'])
    return cands
