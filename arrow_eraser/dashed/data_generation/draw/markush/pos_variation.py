"""取代基位置变异 (Substituent Position Variation)。

在环上的任意位置画一个"开放取代基": 取代基上的连接键中点与环键的中点垂直相交。
视觉上表达"该取代基可在该环的任意原子上出现".

改进绘制思路:
    1. 使用 RDKit 渲染真实的取代基片段 (如 *C, *C(F)(F)F 等)，背景全透明。
    2. 提取取代基上的 D-R (Dummy 到 真实原子) 的键，以其为中心对齐到环键中点。
    3. 利用正则表达式直接解析 RDKit SVG 的 <path>，精准获取被缩短过的视觉化学键坐标，
       生成完美贴合且绝对不包含文本的 OBB 掩膜多边形。
    4. 若触发 opaque_band，直接在底层主画布上沿 D-R 键的路径绘制白色粗线，自然截断环键。
"""

import io
import math
import random
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import cairosvg

from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Geometry import Point3D

from draw.markush.pipeline import (
    paste_with_anchor,
    rotate_pil_around_anchor,
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

_OPAQUE_BAND_PROB = 0.6

_POS_VAR_SMILES = [
    '*C',          # 甲基
    '*CC',         # 乙基
    '*C(C)C'
]


# ============================================================
# 候选搜索
# ============================================================
def find_ring_pos_variation_candidates(mol, *, min_ring_size=4, max_ring_size=8):
    """返回 [{'ring_atoms': [...], 'ring_bond_idxs': [...], 'bond_idx', 'bond_a', 'bond_b'}]."""
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return []
    atom_rings = ri.AtomRings()
    bond_rings = ri.BondRings()

    out = []
    for ridx in range(len(atom_rings)):
        ring_atoms = list(atom_rings[ridx])
        if not (min_ring_size <= len(ring_atoms) <= max_ring_size):
            continue
        ring_atom_set = set(ring_atoms)
        ring_bonds = list(bond_rings[ridx])

        for bidx in ring_bonds:
            bond = mol.GetBondWithIdx(bidx)
            # 仅允许无立体结构的单键或芳香键, 避免楔形/虚线/双键等被 markush overlay 覆盖
            if bond.GetBondType() not in (Chem.BondType.SINGLE, Chem.BondType.AROMATIC):
                continue
            if bond.GetBondDir() != Chem.BondDir.NONE:
                continue
            a = bond.GetBeginAtom()
            b = bond.GetEndAtom()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            if b.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            if _has_outer_substituent(a, ring_atom_set) or \
               _has_outer_substituent(b, ring_atom_set):
                continue
            if a.GetSymbol() != 'C' or b.GetSymbol() != 'C':
                continue
            if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 0 for i in ring_atoms):
                continue
            out.append({
                'ring_atoms': ring_atoms,
                'ring_bond_idxs': ring_bonds,
                'bond_idx': bidx,
                'bond_a': a.GetIdx(),
                'bond_b': b.GetIdx(),
            })
    return out


def _has_outer_substituent(atom, ring_atom_set):
    """该原子是否有任何"非环内"的显式邻居."""
    for nb in atom.GetNeighbors():
        if nb.GetIdx() not in ring_atom_set:
            return True
    return False


# ============================================================
# RDKit 取代基与 SVG 精准掩膜提取
# ============================================================
def _get_visual_bond_lines(svg_text):
    """通过解析 SVG 的 path d 属性，获取绝对视觉的键段线段，完美避开文本框."""
    lines = []
    try:
        root = ET.fromstring(svg_text)
        for path in root.iter('{http://www.w3.org/2000/svg}path'):
            cls = path.get('class', '')
            # RDKit的键通常带有 bond-0, bond-1 等class
            if re.match(r'bond-\d+', cls):
                d = path.get('d', '')
                # 匹配 M x,y L x,y 的格式
                matches = re.findall(r'M\s*([0-9.-]+)[,\s]+([0-9.-]+)\s*L\s*([0-9.-]+)[,\s]+([0-9.-]+)', d)
                for m in matches:
                    x1, y1, x2, y2 = map(float, m)
                    lines.append(((x1, y1), (x2, y2)))
    except Exception:
        pass
    return lines


def _get_line_obb(p1, p2, w):
    """根据线段两点和线宽，生成 OBB 多边形."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    l = math.hypot(dx, dy)
    if l < 1e-3:
        return None
    nx = -dy / l * w / 2
    ny = dx / l * w / 2
    return [
        (p1[0] + nx, p1[1] + ny),
        (p2[0] + nx, p2[1] + ny),
        (p2[0] - nx, p2[1] - ny),
        (p1[0] - nx, p1[1] - ny)
    ]


def _render_rdkit_pos_variation_overlay(bond_pixel_width, font_size_px, avg_bond_px):
    """用 RDKit 渲染真实取代基。使用自身的键，不画任何延长线。"""

    smi = random.choice(_POS_VAR_SMILES)
    mol = Chem.MolFromSmiles(smi)

    dummy_idx = -1
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 or atom.GetSymbol() == '*':
            dummy_idx = atom.GetIdx()
            atom.SetProp('atomLabel', ' ')  # 隐藏 Dummy 原子的符号
            break

    nbr_idx = mol.GetAtomWithIdx(dummy_idx).GetNeighbors()[0].GetIdx()

    # 强制将 dummy -> C 的键旋转至正 X 轴 (0度方向)
    AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pd_3d = conf.GetAtomPosition(dummy_idx)
    pn_3d = conf.GetAtomPosition(nbr_idx)

    dx = pn_3d.x - pd_3d.x
    dy = pn_3d.y - pd_3d.y
    angle = math.atan2(dy, dx)

    for i in range(mol.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        px = p.x - pd_3d.x
        py = p.y - pd_3d.y
        rx = px * math.cos(-angle) - py * math.sin(-angle)
        ry = px * math.sin(-angle) + py * math.cos(-angle)
        conf.SetAtomPosition(i, Point3D(rx, ry, 0.0))

    # 配置渲染器，强制透明背景
    drawer = Draw.MolDraw2DSVG(-1, -1)
    opts = drawer.drawOptions()
    opts.clearBackground = False
    opts.scalingFactor = max(10.0, avg_bond_px / 1.5)  # 使得取代基键长匹配主图平均键长
    opts.bondLineWidth = bond_pixel_width
    opts.fixedFontSize = int(max(6, font_size_px))
    opts.scaleBondWidth = False
    opts.useBWAtomPalette()
    opts.explicitMethyl = False
    opts.highlightRadius = 0
    opts.highlightBondWidth = 0
    drawer.SetDrawOptions(opts)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    svg_text = drawer.GetDrawingText()
    png_bytes = cairosvg.svg2png(bytestring=svg_text.encode('utf-8'), scale=1.0)
    full_img = Image.open(io.BytesIO(png_bytes)).convert('RGBA')

    # 获取 Dummy 和 Neighbor 在绘制结果中的真实像素坐标
    pt_d = drawer.GetDrawCoords(dummy_idx)
    pt_r = drawer.GetDrawCoords(nbr_idx)

    # 锚点设定为 Dummy-Neighbor 连接键的【绝对中点】
    anchor_x = (pt_d.x + pt_r.x) / 2.0
    anchor_y = (pt_d.y + pt_r.y) / 2.0
    real_anchor = (anchor_x, anchor_y)

    d_r_length = dist((pt_d.x, pt_d.y), (pt_r.x, pt_r.y))

    # 解析出 RDKit 取代基内部真正渲染出来的黑线坐标 (跳过文字空间)
    visual_lines = _get_visual_bond_lines(svg_text)
    local_polys = []
    for line in visual_lines:
        b_poly = _get_line_obb(line[0], line[1], max(2, bond_pixel_width * 2.5))
        if b_poly:
            local_polys.append(b_poly)

    # Bbox 裁剪与锚点、多边形的同步校正
    bbox = full_img.getbbox()
    if bbox is None:
        return full_img, real_anchor, 0.0, local_polys, d_r_length

    bx0 = max(0, bbox[0] - 1)
    by0 = max(0, bbox[1] - 1)
    bx1 = min(full_img.size[0], bbox[2] + 1)
    by1 = min(full_img.size[1], bbox[3] + 1)

    cropped = full_img.crop((bx0, by0, bx1, by1))
    final_anchor = (real_anchor[0] - bx0, real_anchor[1] - by0)

    final_polys = []
    for poly in local_polys:
        adjusted = [(pt[0] - bx0, pt[1] - by0) for pt in poly]
        final_polys.append(adjusted)

    return cropped, final_anchor, 0.0, final_polys, d_r_length


# ============================================================
# 主入口
# ============================================================
def draw_position_variations(cv_img, mol, *,
                              candidates,
                              atom_coords, atom_boxes, bond_obbs_indexed,
                              placed_overlays,
                              bond_line_width, svg_scale, font_size_px,
                              prob=0.50, max_count=3,
                              guaranteed_first=False):
    """主入口.

    Args:
        guaranteed_first: 若为 True, 首个候选以 100% 概率尝试, 成功后后续
            候选概率衰减为 prob * 0.55.

    Returns:
        (new_cv_img, reserved_atoms, pos_meta_list, placed_count).
    """
    if not candidates:
        return cv_img, set(), [], 0

    bond_pixel_width = max(0.8, bond_line_width * svg_scale)
    used_rings = set()
    reserved_atoms = set()
    pos_meta_list = []

    candidates_shuffled = list(candidates)
    random.shuffle(candidates_shuffled)

    def _effective_prob():
        if not guaranteed_first:
            return prob
        if placed_count == 0:
            return 1.0
        return prob * 0.55

    placed_count = 0
    for cand in candidates_shuffled:
        if placed_count >= max_count:
            break
        if random.random() >= _effective_prob():
            continue

        ring_key = tuple(sorted(cand['ring_atoms']))
        if ring_key in used_rings:
            continue

        ring_atoms = cand['ring_atoms']
        bond_a = cand['bond_a']
        bond_b = cand['bond_b']

        if any(i not in atom_coords for i in ring_atoms):
            continue
        if bond_a not in atom_coords or bond_b not in atom_coords:
            continue

        rx = sum(atom_coords[i][0] for i in ring_atoms) / len(ring_atoms)
        ry = sum(atom_coords[i][1] for i in ring_atoms) / len(ring_atoms)
        ring_center = (rx, ry)

        # 环上被截断键的中点
        bond_mid = ((atom_coords[bond_a][0] + atom_coords[bond_b][0]) / 2.0,
                    (atom_coords[bond_a][1] + atom_coords[bond_b][1]) / 2.0)

        n = len(ring_atoms)
        perim = 0.0
        cnt = 0
        for k in range(n):
            a_idx = ring_atoms[k]
            b_idx = ring_atoms[(k + 1) % n]
            perim += dist(atom_coords[a_idx], atom_coords[b_idx])
            cnt += 1
        if cnt == 0:
            continue

        avg_bond = perim / cnt
        opaque_band = random.random() < _OPAQUE_BAND_PROB

        try:
            overlay, anchor, base_dir, local_polys, d_r_length = _render_rdkit_pos_variation_overlay(
                bond_pixel_width=bond_pixel_width,
                font_size_px=font_size_px,
                avg_bond_px=avg_bond,
            )
        except Exception:
            continue

        # 目标方向 = 从环中心 → 环键中点 (外法线方向)
        target_dir = screen_angle_deg(ring_center, bond_mid)

        # 1. 旋转 overlay
        rotated, new_anchor = rotate_pil_around_anchor(
            overlay, target_dir - base_dir, anchor)

        world_anchor = bond_mid

        # ---- 放置前重叠预检查 (排除自身环键, 位置变异天然覆盖在环键上) ----
        existing_polys = _build_existing_polys(atom_boxes, bond_obbs_indexed,
                                                placed_overlays, exclude_bond_ids={cand['bond_idx']})
        est_obb = estimate_overlay_obb(rotated.size, world_anchor, new_anchor, 0.0)
        if check_candidate_overlap(est_obb, existing_polys, cv_img.shape[:2]):
            continue  # 跳过此候选

        # 2. 物理涂白截断：在主画布上直接沿要贴上去的单键路径画一条较粗的纯白线
        if opaque_band:
            dx = math.cos(math.radians(target_dir)) * (d_r_length / 2.0)
            dy = math.sin(math.radians(target_dir)) * (d_r_length / 2.0)
            # 以 bond_mid 为中心，分别向内外延伸半个键长
            pt_in = (int(round(bond_mid[0] - dx)), int(round(bond_mid[1] - dy)))
            pt_out = (int(round(bond_mid[0] + dx)), int(round(bond_mid[1] + dy)))

            halo_w = max(int(bond_pixel_width * 3.5), 6)
            cv2.line(cv_img, pt_in, pt_out, (255, 255, 255), thickness=halo_w)

        # 3. 粘贴 overlay
        new_bg, off = paste_with_anchor(cv_img, rotated, world_anchor, new_anchor)

        if off != (0, 0):
            shift_atom_pixel_coords(atom_coords, *off)
            shift_axis_boxes(atom_boxes, *off)
            shift_polygons_dict(bond_obbs_indexed, *off)
            shift_polygon_list(placed_overlays, *off)
            world_anchor = (world_anchor[0] + off[0], world_anchor[1] + off[1])

        cv_img = new_bg

        # 4. 映射 SVG 精准掩膜多边形到全局坐标
        angle_rad = math.radians(target_dir - base_dir)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        for poly in local_polys:
            world_poly = []
            for pt in poly:
                dx_p = pt[0] - anchor[0]
                dy_p = pt[1] - anchor[1]
                rx_p = dx_p * cos_a - dy_p * sin_a
                ry_p = dx_p * sin_a + dy_p * cos_a
                wx = world_anchor[0] + rx_p
                wy = world_anchor[1] + ry_p
                world_poly.append((wx, wy))

            placed_overlays.append((np.array(world_poly, dtype=np.float32), 'pos_var'))
            # 阻止下游 bond_segmentation 删除这些纯键像素
            pos_meta_list.append({'opaque_band': False})

        used_rings.add(ring_key)
        reserved_atoms.update(ring_atoms)
        placed_count += 1

    return cv_img, reserved_atoms, pos_meta_list, placed_count