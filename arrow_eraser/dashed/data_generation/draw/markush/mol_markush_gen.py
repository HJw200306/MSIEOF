"""Markush 版分子图生成 + YOLO-OBB 数据集 CLI。

在 mol_obb_gen 的渲染管线基础上, 集成 markush 子模块的多类变异:
    1. 取代基异变 (sub_variation): 任意原子按概率换成 R/X/Y 等符号文本。
    2. 连接点异变 (ap_variation):  端点 C + 单键 → 波浪/折线/虚线/星号 overlay。
    3. 重复次数变异 (repeat_variation): 链/整分子/环单原子 → 括号 + 下标。
    4. 取代基位置变异 (pos_variation): 环上"任意位置可取代"标记。

关键差异 (相对 mol_obb_gen):
    - 不再依赖 SMILES 里必须有 `*` 才能出 R 基团; 任意端点原子都可以被改写。
    - 渲染后会再做一次基于像素的 overlay 操作 (AP), 因此 overlap 检查必须把
      AP polygon 也算进来。
    - 默认输出 5 类标签: 0=left_bracket, 1=right_bracket,
      2=bracket_subscript, 3=wave_ap, 4=dash_ap。
      atom_text / other_text / bond / star_ap / pos_var 在 fold_classes_markush 阶段过滤掉。

数据集 CLI 与 mol_obb_gen 几乎相同, 输出到独立目录避免覆盖。
"""

import argparse
import math  # noqa
import multiprocessing as mp
import os
import random
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

cv2.setNumThreads(1)

from utils.base.image_loader import ImageLoader  # noqa: E402
from draw.mol_draw import (  # noqa: E402
    rotate_molecule_90,
    flip_molecule,
    set_liti,
    format_html_subscripts_superscripts,
    get_best_font,
)
from draw.mol_obb_gen import (  # noqa: E402
    parse_svg_dims,
    collect_atom_boxes,
    add_random_other_texts,
    axis_box_to_polygon,
    compute_mol_bbox,
    labels_to_yolo_obb,
    has_excessive_overlap,
    iter_smiles,
    visualize_labels,
    html_to_ocr_plain,
    make_r_symbol,
    make_conservative_dummy_symbol,
    OCRBalancer,
    _crop_obb_or_axis,
)
from draw.markush.pipeline import (  # noqa: E402
    get_atom_pixel_coords,
    collect_bond_obbs_indexed,
    find_terminal_c_single_bond_candidates,
    shift_atom_pixel_coords,
    shift_axis_boxes,
    shift_polygons_dict,
    shift_polygon_list,
    rotate_mol_2d_coords,
    compute_bracket_axis_angle_deg,
    find_collinear_bond_pair_candidates,
    check_candidate_overlap,
    estimate_overlay_obb,
    _build_existing_polys,
)
from draw.markush.sub_variation import (  # noqa: E402
    pick_substitution_atoms,
    apply_substitutions,
)
from draw.markush.ap_variation import draw_attachment_points, STAR_AP_PROB  # noqa: E402
from draw.markush.repeat_variation import (  # noqa: E402
    find_chain_repeat_candidates,
    find_whole_molecule_repeat_candidate,
    find_ring_single_atom_repeat_candidates,
    draw_repeat_brackets,
)
from draw.markush.pos_variation import (  # noqa: E402
    find_ring_pos_variation_candidates,
    draw_position_variations,
)


img_loader = ImageLoader()


def _get_adjacent_bond_ids(mol, bond_ids):
    """返回与 bond_ids 中任一键相邻的所有键 (含自身) 的 idx 集合。

    两键相邻 = 它们共享一个原子。用于确保 AP/括号/位置变异不会落在相邻键上,
    避免视觉重叠。
    """
    result = set(bond_ids)
    for bond_idx in bond_ids:
        bond = mol.GetBondWithIdx(bond_idx)
        for atom_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()):
            atom = mol.GetAtomWithIdx(atom_idx)
            for nb_bond in atom.GetBonds():
                result.add(nb_bond.GetIdx())
    return result


# placed_overlays 第二项 (style_id 字符串) → 内部类别 id。
# 内部 id 不与 atom_text(0)/bond(1)/other_text(2) 冲突,
# fold_classes_markush 负责映射到最终输出类别并过滤不需要的类。
OVERLAY_STYLE_TO_CLASS = {
    # AP markers (ap_variation) — 内部 id 3/4, 最终输出 4/5
    'wave_ap': 3, 'sine_ap': 3, 'zigzag_ap': 3,     # → 连接点1 (折线/波浪线)
    'dash_ap': 4, 'dash_bond': 4,      # → 连接点2 (虚线)
    'star_ap': -1,                      # 过滤 (不输出)
    # repeat brackets — 内部 id 5/6/7, 最终输出 1/2/3
    'left_bracket': 5,                  # → 左括号
    'right_bracket': 6,                 # → 右括号
    'bracket_subscript': 7,             # → 括号下标
    # 环位置变异 — 过滤
    'pos_var': -1,
}


# ============================================================
# 主绘制函数
# ============================================================
def draw_markush_obb(smiles, *, svg_scale=None, with_padding=True,
                    sub_prob=0.40,
                    ap_prob=0.90, ap_max=5,
                    repeat_chain_prob=0.75, repeat_whole_prob=0.35,
                    repeat_ring1_prob=0.35, repeat_max=4,
                    pos_prob=0.65, pos_max=4,
                    other_text_max=6,
                    generate_bond_mask=False):
    """返回 (cv_img, labels, ocr_samples, overlay_debug, bond_mask) 或 None。

    labels: list[(class_id, polygon (4,2))]
        内部 id: 0=atom_text, 1=bond, 2=other_text,
                 3=wave_ap/bond, 4=dash_ap/bond,
                 5=left_bracket, 6=right_bracket, 7=bracket_subscript,
                 -1=star_ap/pos_var (待过滤)
        (上层 fold_classes_markush 映射到最终输出类别.)
    overlay_debug: list[(poly, style_id_str)], 用于 debug 可视化.
    bond_mask: 分子键语义分割掩膜 (仅当 generate_bond_mask=True 时生成)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol, canonOrient=True)

    if random.random() < 0.4:
        rotate_molecule_90(mol, random.choice([True, False]),
                           random.choice([0, 1, 2, 3]))
        flip_molecule(mol, random.choice([True, False]),
                      random.choice([True, False]))

    # set_liti 必须在所有候选选取之前调用，这样 BondDir 检查
    # 才能正确过滤掉被设为楔形/虚线的键，避免 AP/括号画在立体键上。
    if random.random() < 0.5:
        mol = set_liti(mol)

    # ================================================================
    # 候选选取顺序: 括号优先 → AP → 位置变异.
    # 后续变异必须避开前者已占用的键及其相邻键, 避免视觉重叠.
    # ================================================================

    # ---- 1. 括号候选 (最高优先级) ----
    # 前置: 共线单键对检查 —— 若找到直接用于括号绘制
    collinear_pairs = find_collinear_bond_pair_candidates(mol)
    use_collinear_brackets = False

    if collinear_pairs:
        best = collinear_pairs[0]
        conf = mol.GetConformer()
        def _bmid_xy(bi):
            b = mol.GetBondWithIdx(bi)
            a1 = conf.GetAtomPosition(b.GetBeginAtomIdx())
            a2 = conf.GetAtomPosition(b.GetEndAtomIdx())
            return ((a1.x + a2.x) / 2.0, (a1.y + a2.y) / 2.0)
        lmx, lmy = _bmid_xy(best['left_bond_idx'])
        rmx, rmy = _bmid_xy(best['right_bond_idx'])
        bracket_angle = math.degrees(math.atan2(rmy - lmy, rmx - lmx))
        if abs(bracket_angle) > 30:
            rotate_mol_2d_coords(mol, -bracket_angle)
        from draw.markush.repeat_variation import _find_simple_path_between_bonds
        b_left = mol.GetBondWithIdx(best['left_bond_idx'])
        b_right = mol.GetBondWithIdx(best['right_bond_idx'])
        path = _find_simple_path_between_bonds(mol, b_left, b_right, 6)
        if path is not None:
            repeat_chain_candidates = [{
                'left_bond_idx': best['left_bond_idx'],
                'right_bond_idx': best['right_bond_idx'],
                'path_atoms': path,
                'inner_atoms': path[1:-1],
            }]
            repeat_whole_candidate = None
            repeat_ring1_candidates = []
            use_collinear_brackets = True

    if not use_collinear_brackets:
        repeat_chain_candidates = find_chain_repeat_candidates(mol)
        repeat_whole_candidate = find_whole_molecule_repeat_candidate(mol)
        repeat_ring1_candidates = find_ring_single_atom_repeat_candidates(mol)

        bracket_axis_deg = compute_bracket_axis_angle_deg(
            mol, repeat_chain_candidates, repeat_ring1_candidates)
        if bracket_axis_deg is not None and abs(bracket_axis_deg) > 30:
            rotate_mol_2d_coords(mol, -bracket_axis_deg)

    repeat_reserve_hint = set()
    repeat_reserved_bonds = set()
    for c in repeat_chain_candidates:
        repeat_reserve_hint.update(c['inner_atoms'])
        repeat_reserved_bonds.add(c['left_bond_idx'])
        repeat_reserved_bonds.add(c['right_bond_idx'])
    for c in repeat_ring1_candidates:
        repeat_reserve_hint.add(c['atom_idx'])
        repeat_reserved_bonds.add(c['on_bond_idx'])

    # 括号预留键 + 相邻键 → AP/pos_var 必须避开
    bracket_forbidden_bonds = _get_adjacent_bond_ids(mol, repeat_reserved_bonds)

    # ---- 2. AP 候选 (中优先级, 避开括号键及其相邻键) ----
    ap_candidates_all = find_terminal_c_single_bond_candidates(mol)
    # 过滤掉 bond_idx 落在 bracket_forbidden_bonds 里的候选
    ap_candidates_all = [c for c in ap_candidates_all
                         if c['bond_idx'] not in bracket_forbidden_bonds]
    ap_selected = [c for c in ap_candidates_all if random.random() < ap_prob]

    ap_star_atoms = set()
    ap_line_candidates = []
    for c in ap_selected:
        if random.random() < STAR_AP_PROB:
            ap_star_atoms.add(c['atom_idx'])
        else:
            ap_line_candidates.append(c)

    for atom_idx in ap_star_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        atom.SetProp('atomLabel', '*')

    ap_reserve_hint = {c['atom_idx'] for c in ap_selected}
    ap_reserved_bonds = {c['bond_idx'] for c in ap_selected}

    # AP 预留键 + 相邻键 → pos_var 必须避开
    ap_forbidden_bonds = _get_adjacent_bond_ids(mol, ap_reserved_bonds)

    # ---- 3. 环位置变异候选 ----
    # 括号主要落在链键上, 环变异落在环键上, 二者天然不冲突;
    # 只排除 AP 相邻键 + 括号/AP 自身键 (不扩展相邻), 避免过度过滤。
    pos_candidates = find_ring_pos_variation_candidates(mol)
    pos_forbidden = repeat_reserved_bonds | ap_forbidden_bonds
    pos_candidates = [c for c in pos_candidates
                      if c['bond_idx'] not in pos_forbidden]
    pos_reserve_hint = set()
    for c in pos_candidates:
        pos_reserve_hint.update(c['ring_atoms'])

    sub_reserved = ap_reserve_hint | repeat_reserve_hint | pos_reserve_hint

    # ---- 取代基异变 ----
    sub_atoms = pick_substitution_atoms(
        mol, prob=sub_prob, reserved=sub_reserved,
        max_total=max(2, int(mol.GetNumAtoms() * 0.4)))
    # 处理 SMILES 里原本就有的 `*` (与 mol_obb_gen 行为一致, 但分离逻辑):
    star_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == '*']
    for sa in star_atoms:
        if sa not in sub_atoms:
            sub_atoms.append(sa)
    sub_ocr_truth = apply_substitutions(
        mol, sub_atoms, show_r_sym_prob=0.7)

    # ---- 收集杂原子兜底 OCR 真值 ----
    atom_display_ocr = dict(sub_ocr_truth)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in atom_display_ocr:
            continue
        sym = atom.GetSymbol()
        if sym not in ('C', '*'):
            atom_display_ocr[idx] = sym
    # star_ap 原子 (RDKit 原生 '*' 标签) 补充 OCR 真值
    for atom_idx in ap_star_atoms:
        atom_display_ocr[atom_idx] = '*'

    # ---- DrawMolecule ----
    drawer = Draw.MolDraw2DSVG(-1, -1)
    opts = drawer.drawOptions()

    use_bw = random.random() < 0.8
    if use_bw:
        opts.useBWAtomPalette()

    bondLineWidth = random.choices([0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                   weights=[0.05, 0.2, 0.3, 0.25, 0.15, 0.05],
                                   k=1)[0]
    opts.bondLineWidth = bondLineWidth
    opts.scaleBondWidth = False

    # 键长与字体大小反向约束: 短键长 + 大字体 → 文本挤占键的位置 → 渲染异常。
    # 用一个 size_tier (0=紧凑, 1=适中, 2=宽松) 同时约束两者。
    size_tier = random.choices([0, 1, 2], weights=[0.3, 0.4, 0.3], k=1)[0]
    if size_tier == 0:
        bond_scale_mul = random.uniform(0.7, 0.9)
        fixed_font = random.choices([5, 6], weights=[0.4, 0.6], k=1)[0]
    elif size_tier == 1:
        bond_scale_mul = random.uniform(0.80, 1.0)
        fixed_font = random.choices([6, 7, 8], weights=[0.3, 0.5, 0.2], k=1)[0]
    else:
        bond_scale_mul = random.uniform(0.90, 1.20)
        fixed_font = random.choices([8, 9, 10], weights=[0.3, 0.4, 0.3], k=1)[0]

    opts.scalingFactor = 14.4 / max(Draw.MeanBondLength(mol), 1e-3) * bond_scale_mul
    opts.multipleBondOffset = random.uniform(0.13, 0.24)
    opts.fixedFontSize = fixed_font
    opts.additionalAtomLabelPadding = random.uniform(0.04, 0.10)
    opts.highlightRadius = 0
    opts.highlightBondWidth = 0
    opts.fontFile = get_best_font()
    opts.explicitMethyl = False
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.annotationFontScale = 0.5
    opts.atomLabelFontSize = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    root, svg_w_raw, svg_h_raw = parse_svg_dims(svg)

    if svg_scale is None:
        svg_max_raw = max(svg_w_raw, svg_h_raw, 1.0)
        target_buckets = [(300, 450), (450, 600), (600, 800), (800, 1000)]
        lo, hi = random.choice(target_buckets)
        target = random.randint(lo, hi)
        svg_scale = max(2.0, min(14.0, target / svg_max_raw))

    atom_boxes = collect_atom_boxes(root, svg_scale)
    bond_obbs_indexed = collect_bond_obbs_indexed(
        root, svg_scale, default_width=bondLineWidth * 2.0)
    # 全原子 (含未渲染文字的碳) 像素坐标 —— AP 需要它来定位端点 C
    atom_coords = get_atom_pixel_coords(drawer, mol.GetNumAtoms(), svg_scale)

    png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'), scale=svg_scale)
    cv_img = img_loader.load_cv_img(png_bytes)

    # ---- 基础分子重叠预检查: 若原子文本/键自身过度重叠, 直接重绘 ----
    base_labels = []
    for box in atom_boxes.values():
        base_labels.append((0, axis_box_to_polygon(box)))
    for obb in bond_obbs_indexed.values():
        base_labels.append((1, obb))
    if has_excessive_overlap(base_labels, cv_img.shape, threshold=0.50,
                             class_thresholds={0: 0.20, 1: 0.55}):
        return None

    # ---- 生成键掩膜基图 (bond-only SVG cairosvg 重渲染, overlay 前) ----
    bond_mask = None
    per_bond_masks = None
    per_bond_lengths = {}
    if generate_bond_mask:
        from draw.markush.bond_segmentation import (
            extract_bond_mask_from_bond_svg,
            extract_pos_var_bonds,
            handle_overlay_priority,
            shift_mask_to_match,
            extract_per_bond_masks_from_svg,
            compute_per_bond_lengths,
            filter_candidates_by_bond_length,
        )
        bond_mask = extract_bond_mask_from_bond_svg(svg, svg_scale, cv_img.shape[:2])

        # 逐键掩膜提取 + 键长过滤: 防止 markush overlay 绘制在过短的键上
        per_bond_masks = extract_per_bond_masks_from_svg(
            root, svg_scale, cv_img.shape[:2])
        per_bond_lengths = compute_per_bond_lengths(per_bond_masks)

        # 对候选键做长度与资格检查, 剔除短键候选
        repeat_chain_candidates, _avg_chain, _thr_chain = \
            filter_candidates_by_bond_length(repeat_chain_candidates, per_bond_lengths)
        repeat_ring1_candidates, _avg_r1, _thr_r1 = \
            filter_candidates_by_bond_length(repeat_ring1_candidates, per_bond_lengths)
        ap_line_candidates, _avg_ap, _thr_ap = \
            filter_candidates_by_bond_length(ap_line_candidates, per_bond_lengths)
        pos_candidates, _avg_pos, _thr_pos = \
            filter_candidates_by_bond_length(pos_candidates, per_bond_lengths)

        # 用于计算每步增量偏移的参考原子
        _ref_atom = 0 if atom_coords else None
        _prev_h, _prev_w = cv_img.shape[:2]

    # ---- 1. 重复次数变异 (括号, 最高优先级) ----
    placed_overlays = []  # list[(poly (4,2), class_name)]
    if bond_mask is not None:
        _prev_coord = atom_coords[_ref_atom]
    cv_img, repeat_reserved, bracket_placed = draw_repeat_brackets(
        cv_img, mol,
        chain_candidates=repeat_chain_candidates,
        whole_candidate=repeat_whole_candidate,
        ring1_candidates=repeat_ring1_candidates,
        atom_coords=atom_coords,
        atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_line_width=bondLineWidth,
        svg_scale=svg_scale,
        font_size_px=opts.fixedFontSize * svg_scale,
        chain_prob=repeat_chain_prob,
        whole_prob=repeat_whole_prob,
        ring1_prob=repeat_ring1_prob,
        max_count=repeat_max,
        guaranteed_first=True,
    )
    if bond_mask is not None:
        new_h, new_w = cv_img.shape[:2]
        curr_coord = atom_coords[_ref_atom]
        step_dx = int(round(curr_coord[0] - _prev_coord[0]))
        step_dy = int(round(curr_coord[1] - _prev_coord[1]))
        if step_dx != 0 or step_dy != 0 or new_h != _prev_h or new_w != _prev_w:
            bond_mask = shift_mask_to_match(bond_mask, new_h, new_w, step_dy, step_dx)
            if per_bond_masks is not None:
                for bi in list(per_bond_masks.keys()):
                    per_bond_masks[bi] = shift_mask_to_match(
                        per_bond_masks[bi], new_h, new_w, step_dy, step_dx)
        _prev_coord = curr_coord
        _prev_h, _prev_w = new_h, new_w

    # ---- 2. 取代基连接点 (AP, 中优先级) ----
    if bond_mask is not None:
        _prev_coord = atom_coords[_ref_atom]
    cv_img, ap_reserved, removed_bonds, ap_meta_list, ap_placed = draw_attachment_points(
        cv_img, mol,
        candidates=ap_line_candidates,
        atom_coords=atom_coords,
        atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_line_width=bondLineWidth,
        svg_scale=svg_scale,
        font_size_px=opts.fixedFontSize * svg_scale,
        prob=1.0,
        max_count=ap_max,
    )
    if bond_mask is not None:
        new_h, new_w = cv_img.shape[:2]
        curr_coord = atom_coords[_ref_atom]
        step_dx = int(round(curr_coord[0] - _prev_coord[0]))
        step_dy = int(round(curr_coord[1] - _prev_coord[1]))
        if step_dx != 0 or step_dy != 0 or new_h != _prev_h or new_w != _prev_w:
            bond_mask = shift_mask_to_match(bond_mask, new_h, new_w, step_dy, step_dx)
            if per_bond_masks is not None:
                for bi in list(per_bond_masks.keys()):
                    per_bond_masks[bi] = shift_mask_to_match(
                        per_bond_masks[bi], new_h, new_w, step_dy, step_dx)
        _prev_coord = curr_coord
        _prev_h, _prev_w = new_h, new_w

    # ---- 3. 取代基位置变异 (环上"任意位置可取代", 最低优先级) ----
    if bond_mask is not None:
        _prev_coord = atom_coords[_ref_atom]
    cv_img, pos_reserved, pos_meta_list, pos_placed = draw_position_variations(
        cv_img, mol,
        candidates=pos_candidates,
        atom_coords=atom_coords,
        atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_line_width=bondLineWidth,
        svg_scale=svg_scale,
        font_size_px=opts.fixedFontSize * svg_scale,
        prob=pos_prob,
        max_count=pos_max,
        guaranteed_first=(bracket_placed == 0),
    )
    if bond_mask is not None:
        new_h, new_w = cv_img.shape[:2]
        curr_coord = atom_coords[_ref_atom]
        step_dx = int(round(curr_coord[0] - _prev_coord[0]))
        step_dy = int(round(curr_coord[1] - _prev_coord[1]))
        if step_dx != 0 or step_dy != 0 or new_h != _prev_h or new_w != _prev_w:
            bond_mask = shift_mask_to_match(bond_mask, new_h, new_w, step_dy, step_dx)
            if per_bond_masks is not None:
                for bi in list(per_bond_masks.keys()):
                    per_bond_masks[bi] = shift_mask_to_match(
                        per_bond_masks[bi], new_h, new_w, step_dy, step_dx)
        _prev_coord = curr_coord
        _prev_h, _prev_w = new_h, new_w

    # ---- 保底检查: 括号/AP/位置变异 三者都没画出来 → 跳过这张分子 ----
    if bracket_placed == 0 and ap_placed == 0 and pos_placed == 0:
        return None

    # ---- 应用 overlay 修正 (pos_var 补充 + AP/括号移除) ----
    if bond_mask is not None:
        pos_var_mask = extract_pos_var_bonds(cv_img, placed_overlays)
        bond_mask = cv2.bitwise_or(bond_mask, pos_var_mask)
        bond_mask = handle_overlay_priority(
            bond_mask, cv_img, placed_overlays,
            bond_obbs_indexed, removed_bonds, ap_meta_list, pos_meta_list)
        # 移除孤立噪点
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            bond_mask, connectivity=8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 4:
                bond_mask[labels == i] = 0

    # ---- 随机加白边以放置外部文本 ----
    H, W = cv_img.shape[:2]
    if with_padding:
        pad_l = random.randint(int(0.05 * W), int(0.30 * W))
        pad_r = random.randint(int(0.05 * W), int(0.30 * W))
        pad_t = random.randint(int(0.05 * H), int(0.30 * H))
        pad_b = random.randint(int(0.05 * H), int(0.30 * H))
        new_W, new_H = W + pad_l + pad_r, H + pad_t + pad_b
        canvas = np.full((new_H, new_W, 3), 255, dtype=np.uint8)
        canvas[pad_t:pad_t + H, pad_l:pad_l + W] = cv_img
        cv_img = canvas
        shift_axis_boxes(atom_boxes, pad_l, pad_t)
        shift_polygons_dict(bond_obbs_indexed, pad_l, pad_t)
        shift_atom_pixel_coords(atom_coords, pad_l, pad_t)
        shift_polygon_list(placed_overlays, pad_l, pad_t)

        # 同步平移掩膜
        if bond_mask is not None:
            mask_canvas = np.zeros((new_H, new_W), dtype=np.uint8)
            mask_canvas[pad_t:pad_t + H, pad_l:pad_l + W] = bond_mask
            bond_mask = mask_canvas
        if per_bond_masks is not None:
            for bi in list(per_bond_masks.keys()):
                pm = per_bond_masks[bi]
                pm_canvas = np.zeros((new_H, new_W), dtype=np.uint8)
                pm_canvas[pad_t:pad_t + H, pad_l:pad_l + W] = pm
                per_bond_masks[bi] = pm_canvas

    # ---- 分子 bbox (含 AP overlay) ----
    mol_bbox = compute_mol_bbox(atom_boxes, list(bond_obbs_indexed.values()))
    # 加入 AP polygon
    for poly, _ in placed_overlays:
        xs = poly[:, 0]
        ys = poly[:, 1]
        if mol_bbox is None:
            mol_bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
        else:
            mol_bbox[0] = min(mol_bbox[0], xs.min())
            mol_bbox[1] = min(mol_bbox[1], ys.min())
            mol_bbox[2] = max(mol_bbox[2], xs.max())
            mol_bbox[3] = max(mol_bbox[3], ys.max())
    if mol_bbox is None:
        H2, W2 = cv_img.shape[:2]
        mol_bbox = [W2 * 0.2, H2 * 0.2, W2 * 0.8, H2 * 0.8]
    pb = 6
    mol_bbox_safe = [mol_bbox[0] - pb, mol_bbox[1] - pb,
                     mol_bbox[2] + pb, mol_bbox[3] + pb]

    # ---- 加入随机外部文本 ----
    atom_font_px = opts.fixedFontSize * svg_scale
    other_text_items = add_random_other_texts(
        cv_img, mol_bbox_safe,
        target_font_px=atom_font_px,
        max_count=random.randint(0, other_text_max),
    )

    # ---- 汇总标注 ----
    labels = []
    H, W = cv_img.shape[:2]

    def _clip_poly(poly):
        poly = np.array(poly, dtype=np.float32).copy()
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
        return poly

    ocr_samples = []

    for atom_idx, box in atom_boxes.items():
        poly = _clip_poly(axis_box_to_polygon(box))
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((0, poly))
        plain = atom_display_ocr.get(atom_idx)
        if plain:
            ocr_samples.append((box, plain))

    for bond_idx, obb in bond_obbs_indexed.items():
        if bond_idx in removed_bonds:
            continue
        poly = _clip_poly(obb)
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((1, poly))

    for corners, plain in other_text_items:
        poly = _clip_poly(corners)
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((2, poly))
        ocr_samples.append((corners, plain))

    for ov_poly, style_id in placed_overlays:
        poly = _clip_poly(ov_poly)
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        # style_id 可能携带调试标签 (如 'left_bracket|font_times'),
        # 提取基础名称用于类别映射
        base_style = style_id.split('|')[0] if '|' in style_id else style_id
        cls_id = OVERLAY_STYLE_TO_CLASS.get(base_style, -1)
        labels.append((cls_id, poly))

    return cv_img, labels, ocr_samples, placed_overlays, bond_mask, \
        atom_boxes, bond_obbs_indexed, removed_bonds, per_bond_masks, \
        mol, atom_display_ocr, atom_coords


# ============================================================
# YOLO 标签 + 类别折叠
# ============================================================
# 内部 id → 最终输出 id (num_classes=5 模式, 不含 text)
# -1 表示过滤 (不输出该标签)
_INTERNAL_TO_OUTPUT_5 = {
    0: -1,  # atom_text → 过滤 (不检测文本)
    1: -1,  # bond → 过滤
    2: -1,  # other_text → 过滤 (不检测文本)
    3: 3,   # wave_ap/bond → 连接点1 (折线/波浪线)
    4: 4,   # dash_ap/bond → 连接点2 (虚线)
    5: 0,   # left_bracket → 左括号
    6: 1,   # right_bracket → 右括号
    7: 2,   # bracket_subscript → 括号下标
}

# 内部 id → 最终输出 id (num_classes=8 模式, 包含 text)
_INTERNAL_TO_OUTPUT_8 = {
    0: 0,   # atom_text → 原子文本
    1: -1,  # bond → 过滤（改用语义分割）
    2: 1,   # other_text → 其他文本
    3: 4,   # wave_ap/bond → 连接点1 (折线/波浪线)
    4: 5,   # dash_ap/bond → 连接点2 (虚线)
    5: 2,   # left_bracket → 左括号
    6: 3,   # right_bracket → 右括号
    7: 6,   # bracket_subscript → 括号下标
}


def fold_classes_markush(labels, num_classes):
    """num_classes:
        5 (默认) -> 0=left_bracket, 1=right_bracket,
                   2=bracket_subscript, 3=wave_ap, 4=dash_ap
                   过滤: atom_text, other_text, bond, star_ap, pos_var
        8 (新增) -> 0=atom_text, 1=other_text, 2=left_bracket, 3=right_bracket,
                   4=wave_ap, 5=dash_ap, 6=bracket_subscript
                   过滤: bond (改用语义分割), star_ap, pos_var
        11 -> 保留内部 id (0=atom_text, 1=bond, 2=other_text,
                            3=wave_ap, 4=dash_ap, 5=left_bracket,
                            6=right_bracket, 7=bracket_subscript)
              过滤: star_ap(-1), pos_var(-1)
    """
    if num_classes >= 11:
        out = []
        for cls, poly in labels:
            if cls < 0:  # 过滤 star_ap/pos_var
                continue
            out.append((cls, poly))
        return out

    if num_classes == 8:
        out = []
        for cls, poly in labels:
            new_cls = _INTERNAL_TO_OUTPUT_8.get(cls, -1)
            if new_cls < 0:
                continue
            out.append((new_cls, poly))
        return out

    # num_classes == 5 (default)
    out = []
    for cls, poly in labels:
        new_cls = _INTERNAL_TO_OUTPUT_5.get(cls, -1)
        if new_cls < 0:
            continue
        out.append((new_cls, poly))
    return out


# ============================================================
# 多进程 worker
# ============================================================
def _worker_init():
    cv2.setNumThreads(1)
    seed = (os.getpid() * 2654435761) ^ int.from_bytes(os.urandom(4), 'little')
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


# 重试时概率放大系数: 第 N 次重试 (N>=1) 的倍率
_RETRY_BOOST = [1.0, 2.0, 3.0, 4.5]

# 检查 labels 里是否有至少一个不被 fold_classes_markush 过滤的标签
def _has_valid_label(labels, num_classes=5):
    if num_classes == 8:
        mapping = _INTERNAL_TO_OUTPUT_8
    else:
        mapping = _INTERNAL_TO_OUTPUT_5
    for cls, _ in labels:
        if mapping.get(cls, -1) >= 0:
            return True
    return False


def _gen_worker(smi, num_classes=5, generate_bond_mask=False):
    """带概率放大的重试生成: 若首次无有效 overlay, 提升 AP/括号概率重试。"""
    for attempt, boost in enumerate(_RETRY_BOOST):
        try:
            result = draw_markush_obb(
                smi,
                ap_prob=min(1.0, 0.90 * boost),
                repeat_chain_prob=min(1.0, 0.75 * boost),
                repeat_whole_prob=min(1.0, 0.35 * boost),
                repeat_ring1_prob=min(1.0, 0.35 * boost),
                pos_prob=min(1.0, 0.65 * boost),
                generate_bond_mask=generate_bond_mask,
            )
        except Exception:
            print(f"Worker Error on {smi}: {traceback.format_exc()}")
            return None

        if result is None:
            continue
        _, labels, _, _, _, _, _, _, _, _, _, _ = result
        if not labels:
            continue
        if _has_valid_label(labels, num_classes):
            return result

    # 最后一次尝试返回 (即使无有效标签, 上层会跳过)
    return result


# ============================================================
# 数据集生成
# ============================================================

# num_classes=5 模式下的可视化色板 + 名称
_MARKUSH_CLASS_NAMES_5 = {
    0: ('left_bracket', (0, 255, 128)),
    1: ('right_bracket', (0, 255, 255)),
    2: ('bracket_sub', (255, 128, 0)),
    3: ('wave_ap', (0, 128, 255)),
    4: ('dash_ap', (128, 0, 255)),
}

# num_classes=8 模式下的可视化色板 + 名称
_MARKUSH_CLASS_NAMES_8 = {
    0: ('atom_text', (255, 0, 0)),
    1: ('other_text', (255, 128, 0)),
    2: ('left_bracket', (0, 255, 128)),
    3: ('right_bracket', (0, 255, 255)),
    4: ('wave_ap', (0, 128, 255)),
    5: ('dash_ap', (128, 0, 255)),
    6: ('bracket_sub', (255, 0, 255)),
}


def visualize_labels_markush(img, labels, placed_overlays=None, num_classes=5):
    """与 visualize_labels 类似, 但在 overlay 多边形旁标注 style 名称.

    placed_overlays: 可选, list[(poly, style_id_str)], 用于在 vis 图上标注
                     "wave_ap"/"dash_bond" 等文本, 方便排查渲染异常.
    num_classes: 类别数量，决定使用哪个色板
    """
    class_names = _MARKUSH_CLASS_NAMES_8 if num_classes == 8 else _MARKUSH_CLASS_NAMES_5
    vis = img.copy()
    for cls, poly in labels:
        info = class_names.get(cls)
        color = info[1] if info else (0, 0, 255)
        pts = poly.astype(np.int32)
        cv2.polylines(vis, [pts], True, color, 2)
        # 在多边形中心标注类别名
        cx = int(poly[:, 0].mean())
        cy = int(poly[:, 1].mean())
        name = info[0] if info else f'cls{cls}'
        cv2.putText(vis, name, (cx - 14, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 若提供了原始 placed_overlays, 额外用细虚线标出每个 overlay 的 style_id,
    # 以便区分同一大类下的不同绘制风格 (如 wave_ap vs wave_bond).
    if placed_overlays:
        style_color = {}
        style_palette = [
            (255, 0, 0), (0, 200, 0), (0, 0, 255),
            (200, 200, 0), (200, 0, 200), (0, 200, 200),
        ]
        for i, (poly, style_id) in enumerate(placed_overlays):
            if style_id not in style_color:
                style_color[style_id] = style_palette[len(style_color) % len(style_palette)]
            color = style_color[style_id]
            pts = poly.astype(np.int32)
            # 细虚线框
            for j in range(4):
                pt1 = tuple(pts[j])
                pt2 = tuple(pts[(j + 1) % 4])
                # 手动虚线: 每 6px 画一段
                import math as _math
                dx, dy = pt2[0] - pt1[0], pt2[1] - pt1[1]
                seg_len = _math.hypot(dx, dy)
                if seg_len < 1:
                    continue
                dash = 6
                steps = max(1, int(seg_len / dash))
                for s in range(steps):
                    if s % 2 == 1:
                        continue
                    t0 = s / steps
                    t1 = min((s + 0.5) / steps, 1.0)
                    p0 = (int(pt1[0] + dx * t0), int(pt1[1] + dy * t0))
                    p1 = (int(pt1[0] + dx * t1), int(pt1[1] + dy * t1))
                    cv2.line(vis, p0, p1, color, 1)
            # 标注 style_id
            cx = int(poly[:, 0].mean())
            cy = int(poly[:, 1].min()) - 6
            cv2.putText(vis, style_id, (cx - 20, max(cy, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1)

    return vis


def generate_dataset(out_dir, n_train, n_val, csv_path, num_classes=5,
                    vis_n=0, seed=42, ocr_export=True, overlap_threshold=0.5,
                    workers=1, ocr_balance_cap=50, ocr_balance_alpha=1.5,
                    debug=False, start_idx=0, generate_bond_mask=False):
    random.seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    splits = {'train': n_train, 'val': n_val}
    smiles_pool = iter_smiles(csv_path, limit=max(n_train, n_val) * 8)
    total_needed = n_train + n_val
    if len(smiles_pool) < total_needed:
        smiles_pool = (smiles_pool * (total_needed // len(smiles_pool) + 1))[:total_needed]

    if ocr_export:
        ocr_root = out_dir / 'ocr'
        ocr_root.mkdir(parents=True, exist_ok=True)

    pool = None
    if workers and workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init)
        print(f"[init] multiprocessing pool: workers={workers}")

    try:
        idx_global = start_idx
        for split, n in splits.items():
            img_dir = out_dir / split / 'images'
            lbl_dir = out_dir / split / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = out_dir / split / 'vis'
            if vis_n > 0:
                vis_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = out_dir / split / 'debug_overlays'
            if debug:
                debug_dir.mkdir(parents=True, exist_ok=True)

            # 分子键掩膜目录
            mask_dir = None
            if generate_bond_mask:
                mask_dir = out_dir / split / 'masks'
                mask_dir.mkdir(parents=True, exist_ok=True)

            ocr_img_dir = None
            ocr_label_lines = []
            ocr_counter = start_idx
            balancer = None
            if ocr_export:
                ocr_img_dir = ocr_root / 'images' / split
                ocr_img_dir.mkdir(parents=True, exist_ok=True)
                balancer = OCRBalancer(soft_cap=ocr_balance_cap, alpha=ocr_balance_alpha)

            max_attempts = n * 20
            state = {'produced': 0, 'idx': idx_global, 'fed': 0}

            def _feed_gen():
                while state['fed'] < max_attempts:
                    if state['produced'] >= n:
                        return
                    smi = smiles_pool[state['idx'] % len(smiles_pool)]
                    state['idx'] += 1
                    state['fed'] += 1
                    yield smi

            if pool is not None:
                # 使用 functools.partial 传递额外参数
                from functools import partial
                worker_fn = partial(_gen_worker, num_classes=num_classes, generate_bond_mask=generate_bond_mask)
                results_iter = pool.imap_unordered(worker_fn, _feed_gen(), chunksize=2)
            else:
                def _seq_iter():
                    for smi in _feed_gen():
                        try:
                            yield draw_markush_obb(smi, generate_bond_mask=generate_bond_mask)
                        except Exception:
                            yield None
                results_iter = _seq_iter()

            rejected_overlap = 0
            rejected_empty = 0
            attempts = 0
            ocr_kept = 0
            ocr_dropped = 0
            pbar = tqdm(total=n, desc=f'[{split}]', unit='img',
                        dynamic_ncols=True, smoothing=0.1)

            for result in results_iter:
                attempts += 1
                if state['produced'] >= n:
                    break
                if result is None:
                    continue
                img, labels, ocr_samples, placed_overlays, bond_mask, \
                    atom_boxes, bond_obbs_indexed, removed_bonds, per_bond_masks, \
                    _mol, _atom_ocr, _atom_coords = result
                if not labels:
                    continue
                if has_excessive_overlap(labels, img.shape, threshold=overlap_threshold,
                                         class_thresholds={
                                             0: 0.20, 1: 0.55, 2: 0.20,
                                             3: 0.25, 4: 0.25, 5: 0.25,
                                             6: 0.25, 7: 0.25,
                                         }):
                    rejected_overlap += 1
                    pbar.set_postfix_str(
                        f'rej={rejected_overlap} emp={rejected_empty} ocr={ocr_kept}/{ocr_kept + ocr_dropped}',
                        refresh=False)
                    continue
                labels_fold = fold_classes_markush(labels, num_classes)
                # 折叠后无有效标签 (全是 text/bond/star_ap/pos_var): 跳过
                if not labels_fold:
                    rejected_empty += 1
                    pbar.set_postfix_str(
                        f'rej={rejected_overlap} emp={rejected_empty} ocr={ocr_kept}/{ocr_kept + ocr_dropped}',
                        refresh=False)
                    continue
                h, w = img.shape[:2]
                produced = state['produced']
                name = f"{split}_{produced + start_idx:06d}"
                cv2.imwrite(str(img_dir / f"{name}.png"), img)
                with open(lbl_dir / f"{name}.txt", 'w', encoding='utf-8') as f:
                    f.write(labels_to_yolo_obb(labels_fold, w, h))

                # 保存分子键掩膜
                if generate_bond_mask and bond_mask is not None:
                    cv2.imwrite(str(mask_dir / f"{name}.png"), bond_mask)

                if produced < vis_n:
                    vis = visualize_labels_markush(img, labels_fold, placed_overlays, num_classes)
                    cv2.imwrite(str(vis_dir / f"{name}.png"), vis)

                    # 可视化掩膜（前 vis_n 张）
                    if generate_bond_mask and bond_mask is not None:
                        from draw.markush.bond_segmentation import visualize_bond_mask
                        vis_mask = visualize_bond_mask(img, bond_mask, alpha=0.3)
                        cv2.imwrite(str(vis_dir / f"{name}_mask.png"), vis_mask)

                    # 调试可视化：逐键前景像素 + 原子框 + overlay
                    if generate_bond_mask:
                        debug_bond = img.copy()
                        # 绿色 = 键前景轮廓 (从逐键掩膜提取)
                        if per_bond_masks:
                            # 先绘制所有键掩膜半透明叠加
                            bond_overlay = np.zeros_like(debug_bond)
                            for bond_idx, pmask in per_bond_masks.items():
                                if bond_idx in removed_bonds:
                                    continue
                                # 为每个键分配不同色相避免混淆
                                hue = (bond_idx * 43) % 180
                                color_hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
                                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0]
                                color_bgr = tuple(int(c) for c in color_bgr)
                                bond_overlay[pmask > 0] = color_bgr
                            debug_bond = cv2.addWeighted(debug_bond, 0.65, bond_overlay, 0.35, 0)
                            # 标注 bond_idx
                            for bond_idx, pmask in per_bond_masks.items():
                                if bond_idx in removed_bonds:
                                    continue
                                ys, xs = np.where(pmask)
                                if len(ys) > 0:
                                    cx, cy = int(xs.mean()), int(ys.mean())
                                    cv2.putText(debug_bond, str(bond_idx), (cx - 8, cy + 4),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                            # 已移除的键用红色虚线 OBB 标出
                            for bond_idx, obb in bond_obbs_indexed.items():
                                if bond_idx not in removed_bonds:
                                    continue
                                pts = obb.astype(np.int32)
                                for j in range(4):
                                    pt1, pt2 = tuple(pts[j]), tuple(pts[(j + 1) % 4])
                                    cv2.line(debug_bond, pt1, pt2, (0, 0, 255), 1,
                                             lineType=cv2.LINE_AA)
                        else:
                            # 回退：无逐键掩膜时用 OBB 线框
                            for bond_idx, obb in bond_obbs_indexed.items():
                                if bond_idx in removed_bonds:
                                    continue
                                pts = obb.astype(np.int32)
                                cv2.polylines(debug_bond, [pts], True, (0, 255, 0), 1)
                        # 蓝色 = 原子框
                        for atom_idx, box in atom_boxes.items():
                            x1, y1, x2, y2 = [int(v) for v in box]
                            cv2.rectangle(debug_bond, (x1, y1), (x2, y2), (255, 0, 0), 1)
                        # 黄色 = overlay 多边形
                        for ov_poly, style_id in placed_overlays:
                            pts = ov_poly.astype(np.int32)
                            cv2.polylines(debug_bond, [pts], True, (0, 255, 255), 1)
                            cx, cy = int(ov_poly[:, 0].mean()), int(ov_poly[:, 1].mean())
                            cv2.putText(debug_bond, style_id[:6], (cx - 16, cy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1)
                        cv2.imwrite(str(vis_dir / f"{name}_bond_obb.png"), debug_bond)

                if debug and placed_overlays:
                    h_debug, w_debug = img.shape[:2]
                    debug_img = img.copy()
                    for ov_poly, style_id in placed_overlays:
                        pts = ov_poly.astype(np.int32)
                        cv2.polylines(debug_img, [pts], True, (0, 0, 255), 1)
                        cx = int(ov_poly[:, 0].mean())
                        cy = int(ov_poly[:, 1].mean())
                        cv2.putText(debug_img, style_id, (cx - 20, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                        # 裁剪 overlay 区域并单独保存
                        x1 = max(0, int(ov_poly[:, 0].min()) - 4)
                        y1 = max(0, int(ov_poly[:, 1].min()) - 4)
                        x2 = min(w_debug, int(ov_poly[:, 0].max()) + 4)
                        y2 = min(h_debug, int(ov_poly[:, 1].max()) + 4)
                        if x2 > x1 and y2 > y1:
                            crop = img[y1:y2, x1:x2]
                            crop_name = f"{name}_{style_id}_{x1}_{y1}.png"
                            cv2.imwrite(str(debug_dir / crop_name), crop)
                            # 检测疑似黑色方框: crop 内 >80% 像素接近纯黑 (<30)
                            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                            black_ratio = (gray < 30).mean()
                            if black_ratio > 0.8:
                                print(f"[DEBUG] 疑似黑色方框: {crop_name} "
                                      f"black_ratio={black_ratio:.2f}")
                    cv2.imwrite(str(debug_dir / f"{name}_all.png"), debug_img)

                if ocr_export:
                    for region, text in ocr_samples:
                        if not balancer.maybe_keep(text):
                            ocr_dropped += 1
                            continue
                        crop, ok = _crop_obb_or_axis(img, region)
                        if not ok:
                            continue
                        ocr_counter += 1
                        ocr_name = f"{split}_word_{ocr_counter:07d}.png"
                        cv2.imwrite(str(ocr_img_dir / ocr_name), crop)
                        rel = f"images/{split}/{ocr_name}"
                        ocr_label_lines.append(f"{rel}\t{text}")
                        ocr_kept += 1

                state['produced'] += 1
                pbar.update(1)
                pbar.set_postfix_str(
                    f'rej={rejected_overlap} emp={rejected_empty} ocr={ocr_kept}/{ocr_kept + ocr_dropped}',
                    refresh=False)

            pbar.close()
            idx_global = state['idx']

            if ocr_export:
                label_file = ocr_root / f"{split}_label.txt"
                with open(label_file, 'w', encoding='utf-8') as f:
                    f.write("\n".join(ocr_label_lines))
                uniq = len(balancer.counts) if balancer else 0
                top = sorted(balancer.counts.items(), key=lambda kv: -kv[1])[:5] if balancer else []
                print(f"[ocr/{split}] kept={ocr_kept} dropped={ocr_dropped} "
                      f"unique={uniq} top5={top} -> {label_file}")

            print(f"[{split}] done. produced={state['produced']} attempts={attempts} "
                  f"rejected_overlap={rejected_overlap} rejected_empty={rejected_empty}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT_DIR / 'dataset' / 'yolo_markush_obb'))
    ap.add_argument('--csv', default=str(ROOT_DIR / 'dataset' / 'train_200k.csv'))
    ap.add_argument('--train', type=int, default=10000)
    ap.add_argument('--val', type=int, default=500)
    ap.add_argument('--num-classes', type=int, default=5, choices=[5, 8, 11],
                    help='5: l_bracket/r_bracket/bracket_sub/wave_ap/dash_ap (无text); '
                         '8: atom_text/other_text/l_bracket/r_bracket/wave_ap/dash_ap/bracket_sub (无bond); '
                         '11: 全粒度内部 id')
    ap.add_argument('--vis', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-ocr', default=True, action='store_true')
    ap.add_argument('--overlap', type=float, default=0.3)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--ocr-cap', type=int, default=50)
    ap.add_argument('--ocr-alpha', type=float, default=1.5)
    ap.add_argument('--debug', action='store_true',
                    help='保存 overlay 裁剪图 + 标注图到 debug_overlays/ 目录, '
                         '用于排查渲染异常 (如黑色方框)')
    ap.add_argument('--start-idx', type=int, default=0,
                    help='起始编号偏移 (续生成时使用, 如已生成 5000 张则设 --start-idx 5000)')
    ap.add_argument('--generate-bond-mask', action='store_true',
                    help='生成分子键语义分割掩膜 (保存到 masks/ 目录)')
    args = ap.parse_args()

    generate_dataset(args.out, args.train, args.val, args.csv,
                     num_classes=args.num_classes, vis_n=args.vis, seed=args.seed,
                     ocr_export=not args.no_ocr,
                     overlap_threshold=args.overlap,
                     workers=args.workers,
                     ocr_balance_cap=args.ocr_cap,
                     ocr_balance_alpha=args.ocr_alpha,
                     debug=args.debug,
                     start_idx=args.start_idx,
                     generate_bond_mask=args.generate_bond_mask)
    print('done.')


if __name__ == '__main__':
    main()
