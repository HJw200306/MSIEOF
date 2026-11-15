"""取代基连接点变异 (Attachment Point variation)。

行为约定:
- 候选 = 端点碳 + 与之相连的单键 (find_terminal_c_single_bond_candidates).
- 对每个候选独立按 `prob` 概率决定是否替换。
- 替换时:
    1. 渲染一个 marker overlay (波浪 / 折线 / 虚线 / 圆波)。
       星号 (*) 已改用 RDKit 原生 atomLabel 绘制, 不再走 PIL overlay。
    2. 让 overlay 的"连接点" 对齐端点 C 的像素位置,
       并把 overlay 整体旋转到与键方向同向 (从 neighbor 指向 C 的方向)。
    3. 用 paste_with_anchor 完成 alpha 复合; 若导致画布扩边, 把所有外部
       几何记录 (atom_coords / atom_boxes / bond_obbs / 之前粘的 placed_overlays)
       同步平移, 以保证后续步骤 (粘第二个 AP / 加 padding / overlap 检查) 坐标一致。

注意: 不主动抹除端点键或端点原子文本的像素。原始 drawer.py 中"被设为透明"的
只是高亮圈, 键和原子文本的墨水仍在 SVG 中; marker 通过 alpha 覆盖到它们上方
即可。是否要把这根键 / 这个原子文本从 YOLO 标注里裁掉, 由 has_excessive_overlap
按像素覆盖比例自然处理。

返回的 placed_overlays 中, 每个元素的多边形已用 minAreaRect 紧贴 marker 像素,
直接进入 has_excessive_overlap 即可被纳入像素覆盖检查。

类别名:
    wave_ap   —— 圆波 (circle_wave) 端点
    dash_ap   —— 虚线 (dashed) 端点
    wave_bond —— 波浪 (sine) 整键
    dash_bond —— 锯齿 (zigzag) 整键
"""

import math
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from draw.line_draw import draw_connect_line
from draw.markush.pipeline import (
    paste_with_anchor,
    rotate_pil_around_anchor,
    compute_pasted_obb,
    remove_transparent_areas,
    screen_angle_deg,
    dist,
    shift_atom_pixel_coords,
    shift_axis_boxes,
    shift_polygons_dict,
    shift_polygon_list,
    check_candidate_overlap,
    estimate_overlay_obb,
    _build_existing_polys,
)


# line_draw 内部用 matplotlib lw=thickness (单位 points) 配合 1-inch 高的画布
# (dpi=72, ylim=(-2,2) → ~18 px/单位); thickness > ~8 时线宽超过画布可视区域,
# 渲染结果会出现黑色矩形伪影。所以我们先把 thickness 钳在安全范围, 再用 PIL
# 把 overlay 等比放大到目标 bond_pixel_width 量级。
_RENDER_THICKNESS_SAFE_MAX = 8.0
_RENDER_THICKNESS_SAFE_MIN = 1.5


# AP 风格 -> (line_draw 的 line_type, connect_point_type, 是否"端点 marker")
# is_endpoint = True 表示锚点直接放到 terminal C; False 表示替换整根键。
_AP_STYLES = [
    # 1. 端点截断 (endpoint) 组
    ('wave_ap',   'circle_wave', 'center', 'endpoint'), # 圆波
    ('dash_ap',   'dashed',      'center', 'endpoint'), # 虚线
    ('sine_ap',   'wave',        'center', 'endpoint'), # 正弦波浪
    ('zigzag_ap', 'zigzag',      'center', 'endpoint'), # 锯齿折线

    # 2. 中间截断 (middle) 组
    ('wave_ap',   'circle_wave', 'center', 'middle'),
    ('dash_ap',   'dashed',      'center', 'middle'),
    ('sine_ap',   'wave',        'center', 'middle'),
    ('zigzag_ap', 'zigzag',      'center', 'middle'),
]

# 同步修改权重分配 (长度需与 _AP_STYLES 保持一致，共 8 项，总和为 1.0)
# 这里假设你希望端点截断出现的概率 (0.7) 大于中间截断 (0.3)
_AP_WEIGHTS = [
    0.20, 0.20, 0.15, 0.15,  # 端点截断的权重
    0.08, 0.08, 0.07, 0.07   # 中间截断的权重
]

# 原 star_ap 权重 (0.20) —— star 不再通过 PIL overlay 渲染,
# 而是由 mol_markush_gen 在 DrawMolecule 之前直接把 atomLabel 设为 '*',
# 利用 RDKit 原生原子标签绘制。保留此常量供上层预选概率参考。
STAR_AP_PROB = 0.20


def _render_line_overlay(style_id, line_type, cp_type, bond_pixel_width,
                         opaque_band=False):
    """绕过 line_draw 内部的 connect_point_type='endpoint' 限制 (只对 dashed 有效)。

    bond_pixel_width: marker 期望的实际像素粗细 (与键描边同量级)。
    opaque_band: True 时在 marker 上下补一条白色不透明带, 然后整图变成不透明 ——
                 这是 drawer.py 原本的"视觉截断键"特性 (paste 时白带覆盖底下的键,
                 让键看起来在 marker 处被切断 / 不完全相连)。False 时 overlay 仍
                 保留透明背景, 底下的键墨水会从空隙透出来。

    实现策略: 先在 matplotlib 安全 thickness 区间内渲染一份基准图, 然后用 PIL
    resize 到目标尺寸 —— 直接用大 thickness 喂给 matplotlib 会触发画布截断 (黑框)。
    返回 (PIL RGBA, anchor_xy, base_direction_deg)。
        base_direction_deg = 从 anchor 指向 overlay 几何中心的角度 (屏幕坐标系)。
    """
    target = max(1.0, bond_pixel_width * random.uniform(0.85, 1.15))
    # 在安全区间渲染, 之后再缩放
    render_t = min(_RENDER_THICKNESS_SAFE_MAX,
                   max(_RENDER_THICKNESS_SAFE_MIN, target))
    pil, cp = draw_connect_line(thickness=render_t, line_type=line_type,
                                connect_point_type=cp_type)
    if not isinstance(pil, Image.Image):
        pil = Image.fromarray(pil)
    pil = pil.convert('RGBA')

    # 等比缩放使 overlay 的"线粗"接近 target
    if abs(target - render_t) > 1e-3:
        scale = target / render_t
        new_w = max(2, int(round(pil.size[0] * scale)))
        new_h = max(2, int(round(pil.size[1] * scale)))
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        cp = [cp[0] * scale, cp[1] * scale]

    # 上下加白带 + 整图去透明 → marker 带不透明白底, 粘贴时会盖住键。
    # 参考 drawer.py 的原始做法 (perpendicular padding + remove_transparent_areas)。
    if opaque_band:
        pad_h = max(1, int(round(random.uniform(0.4, 0.9) * target)))
        pil = ImageOps.expand(pil, border=(0, pad_h, 0, pad_h),
                              fill=(255, 255, 255, 255))
        cp = [cp[0], cp[1] + pad_h]
        pil = remove_transparent_areas(pil)

    w, h = pil.size
    cx, cy = w / 2, h / 2
    ax, ay = cp[0], cp[1]
    dx, dy = cx - ax, cy - ay
    if dx * dx + dy * dy < 1e-6:
        # connect_point 落在几何中心 (典型: circle_wave/wave/zigzag center 模式)
        # 此时 overlay 没有天然朝向, 当作水平向右 (+x) 即可, 旋转时无所谓。
        base_dir = 0.0
    else:
        base_dir = math.degrees(math.atan2(dy, dx))
    return pil, (ax, ay), base_dir


def draw_attachment_points(cv_img, mol, *,
                           candidates,
                           atom_coords, atom_boxes, bond_obbs_indexed,
                           placed_overlays,
                           bond_line_width, svg_scale, font_size_px,
                           prob=0.5, max_count=None):
    """主入口。

    Args:
        cv_img: BGR np.ndarray, 会被原地修改。
        mol: RDKit Mol (用于读 bond_idx 等元信息, 不修改)。
        candidates: pipeline.find_terminal_c_single_bond_candidates(mol) 的结果。
        atom_coords: {atom_idx: (px, py)}, 会随画布扩张被 in-place shift。
        atom_boxes: {atom_idx: [x1,y1,x2,y2]} (mol_obb_gen 收集的), in-place shift。
        bond_obbs_indexed: {bond_idx: (4,2) polygon}, in-place shift。
        placed_overlays: list[(poly (4,2), class_name str)] 之前所有粘贴 marker 的
                        多边形列表, 会随扩张 in-place shift; AP 自身的 polygon 也
                        append 进来 (统一管理)。
        bond_line_width: opts.bondLineWidth (SVG 单位).
        svg_scale:       SVG -> PNG 的缩放倍率;
                         marker 实际像素粗细 = bond_line_width * svg_scale。
        font_size_px:    opts.fixedFontSize * svg_scale, 用于 star marker 大小。
        prob: 单个候选独立选中概率。
        max_count: 全图最多放几个 AP (None 不限).

    Returns:
        (new_cv_img, reserved_atoms set[int], removed_bonds set[int],
         ap_meta list[dict]). ap_meta 与 placed_overlays 中 AP 条目一一对应，
         每项 = {'is_endpoint': bool, 'opaque_band': bool}。
    """
    if not candidates:
        return cv_img, set(), set(), [], 0

    # SVG 的 bondLineWidth 是 SVG 单位 (typical 0.4~0.9), 实际 PNG 中键的像素粗细
    # 是 bond_line_width * svg_scale。让 marker 与键同量级才不会显得过细。
    bond_pixel_width = max(0.8, bond_line_width * svg_scale)

    random.shuffle(candidates)
    reserved_atoms = set()
    removed_bonds = set()
    ap_meta_list = []  # 每个 AP 的元数据，与 placed_overlays 中 AP 一一对应

    placed_count = 0
    for cand in candidates:
        if max_count is not None and placed_count >= max_count:
            break
        if random.random() > prob:
            continue
        atom_idx = cand['atom_idx']
        neigh_idx = cand['neighbor_idx']
        bond_idx = cand['bond_idx']

        # 像素坐标 (可能在多次粘贴过程中被 shift 过, 所以每次循环都重新读取)
        if atom_idx not in atom_coords or neigh_idx not in atom_coords:
            continue
        c_xy = atom_coords[atom_idx]
        n_xy = atom_coords[neigh_idx]
        bond_len = dist(c_xy, n_xy)
        if bond_len < 4:
            continue

        # 决定风格
        idx = random.choices(range(len(_AP_STYLES)), weights=_AP_WEIGHTS, k=1)[0]
        style = _AP_STYLES[idx]
        style_id = style[0]
        _, line_type, cp_type, is_endpoint = style

        # 渲染 overlay
        try:
            # 50% 概率给 marker 加白带 (原 drawer.py 行为) —— 这样 marker 会
            # 视觉上"截断"底下的键, 让键与连接点不完全相连; 另 50% 保持透明
            # 背景, 键墨水从 marker 空隙透出来。
            opaque_band = random.random() < 0.5
            overlay, anchor, _base_dir = _render_line_overlay(
                style_id, line_type, cp_type, bond_pixel_width,
                opaque_band=opaque_band)
        except Exception:
            continue

        # overlay 尺寸不应远超键长 (尤其 wave_bond/dash_bond 整键替换)
        ow, oh = overlay.size
        max_overlay_dim = bond_len * 3.5 if is_endpoint == 'endpoint' else bond_len * 1.8
        # 若过大, 整体等比缩小
        long_side = max(ow, oh)
        if long_side > max_overlay_dim:
            scale = max_overlay_dim / long_side
            new_w = max(2, int(round(ow * scale)))
            new_h = max(2, int(round(oh * scale)))
            overlay = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)
            anchor = (anchor[0] * scale, anchor[1] * scale)

        # 计算粘贴方向 (屏幕坐标系)

        # 2. 获取该键的精准 OBB (4x2 多边形)
        obb = bond_obbs_indexed[bond_idx]

        # 3. 通过 OBB 的长边计算键的绝对像素方向
        v1 = obb[1] - obb[0]
        v2 = obb[2] - obb[1]
        bond_vec = v1 if np.linalg.norm(v1) > np.linalg.norm(v2) else v2

        # 4. 利用原子坐标定一个粗略的“向外”基准（neighbor -> terminal C）
        approx_out_vec = np.array(c_xy) - np.array(n_xy)

        # 5. 如果 OBB 向量与“向外”基准方向相反，则翻转 180 度
        if np.dot(bond_vec, approx_out_vec) < 0:
            bond_vec = -bond_vec

        # 6. 计算绝对视觉角度
        bond_out_dir = math.degrees(math.atan2(bond_vec[1], bond_vec[0]))

        # 7. 使 overlay 垂直于键: 原版 drawer.py rotate_around_point 的
        #    PIL 旋转角为 (-angle + 90°), 等价于屏幕空间旋转 angle - 90°.
        #    rotate_pil_around_anchor 的语义是 "屏幕空间旋转 target_dir",
        #    因此 target_dir = bond_out_dir - 90.0 即可复现原版行为.
        target_dir = bond_out_dir - 90.0

        # 旋转 overlay
        rotated, new_anchor = rotate_pil_around_anchor(overlay, target_dir, anchor)

        # 8. 决定粘贴的绝对位置
        if is_endpoint == 'endpoint':
            # 端点截断：依然贴在端点 C 附近
            world_anchor = c_xy
        else:
            # 中间截断：完美对齐到 OBB 的几何中心
            world_anchor = (obb[:, 0].mean(), obb[:, 1].mean())

        # ---- 放置前重叠预检查 (排除自身目标键, AP 天然覆盖在键上) ----
        existing_polys = _build_existing_polys(atom_boxes, bond_obbs_indexed,
                                                placed_overlays, exclude_bond_ids={bond_idx})
        est_obb = estimate_overlay_obb(rotated.size, world_anchor, new_anchor, 0.0)
        if check_candidate_overlap(est_obb, existing_polys, cv_img.shape[:2]):
            continue  # 跳过此候选, 不拒绝整张图

        # ================== 替换结束 ==================

        new_bg, off = paste_with_anchor(cv_img, rotated, world_anchor, new_anchor)
        # 画布扩张时同步所有已有坐标
        if off != (0, 0):
            shift_atom_pixel_coords(atom_coords, *off)
            shift_axis_boxes(atom_boxes, *off)
            shift_polygons_dict(bond_obbs_indexed, *off)
            shift_polygon_list(placed_overlays, *off)
            world_anchor = (world_anchor[0] + off[0], world_anchor[1] + off[1])
        cv_img = new_bg

        # 计算 overlay 在新画布坐标下的 OBB —— canvas_off 已应用到 world_anchor,
        # 这里传 (0, 0) 避免二次累加。
        obb = compute_pasted_obb(rotated, world_anchor, new_anchor, (0, 0))
        if obb is None:
            continue

        placed_overlays.append((obb.astype(np.float32), style_id))
        ap_meta_list.append({'is_endpoint': is_endpoint, 'opaque_band': opaque_band})
        reserved_atoms.add(atom_idx)

        placed_count += 1

    return cv_img, reserved_atoms, removed_bonds, ap_meta_list, placed_count
