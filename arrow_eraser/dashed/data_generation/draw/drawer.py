import math
import random
from typing import Union, List

import cv2
import numpy as np
from PIL import Image, ImageOps
from rdkit import Chem

from utils.base.image_loader import ImageLoader
from draw.line_draw import draw_connect_line
from draw.mol_draw import draw_molecule_like_chemdraw, break_smile, RGROUP_SYMBOLS
from draw.bracket_draw import draw_bracket_pair, draw_bracket_subscript, generate_subscript_content

img_loader = ImageLoader()

cls_mapping = [
    'left_bracket',       # 0
    'right_bracket',      # 1
    'bracket_subscript',  # 2
    'wave_AP',            # 3
    'star_AP',            # 4
    'dash_AP',            # 5
    'wave_bond',          # 6
    'dash_bond',          # 7
    'AP',                 # 8
] + RGROUP_SYMBOLS       # 9+


def draw_rgroup_struct(smiles, svg_scale=3):
    """入口：始终绘制R基团，独立叠加括号(50%)和连接线(50%)。"""
    result = draw_molecule_like_chemdraw(
        smiles, random_rotate=True, svg_scale=svg_scale,
        show_r_sym=True, return_all_boxes=True)
    if result is None:
        return None
    background, all_boxes, r_sym_ids, r_sym_list, font_size = result

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 收集 R 基团标注框和类别
    boxes = []
    classes = []
    for i, idx in enumerate(r_sym_ids):
        if idx in all_boxes:
            boxes.append(convert_bbox_to_polygon(all_boxes[idx]))
            cls_name = r_sym_list[i]
            if cls_name in cls_mapping:
                classes.append(cls_mapping.index(cls_name))

    # 独立叠加括号 (50%)
    if random.random() < 0.90:
        try:
            b_result = _draw_bracket_overlay(mol, background, all_boxes, font_size, svg_scale)
            if b_result:
                b_boxes, b_classes = b_result
                boxes.extend(b_boxes)
                classes.extend(b_classes)
        except Exception:
            pass

    # 独立叠加连接线 (50%)
    if random.random() < 0.50:
        try:
            c_result = _draw_connect_line_overlay(background, all_boxes, svg_scale)
            if c_result:
                c_boxes, c_classes = c_result
                boxes.extend(c_boxes)
                classes.extend(c_classes)
        except Exception:
            pass

    return background, boxes, classes


def _draw_connect_line_overlay(background, all_boxes, svg_scale):
    """在已有分子图像上叠加连接线。通过原子包围盒计算边缘信息。"""
    edge_info = _compute_edge_info_from_boxes(all_boxes, svg_scale)
    if edge_info is None:
        return None
    edge_center, center, angle, thickness = edge_info

    pp = random.choice(["c", "e"])
    paste_point = center if "c" == pp else edge_center

    line_type = random.choices(('circle_wave', 'wave', 'zigzag', 'dashed'),
                               weights=[0.25, 0.25, 0.25, 0.25], k=1)[0]
    transparent = random.choice([True, False])
    if line_type == 'dashed':
        connect_point_type = random.choice(['center', 'endpoint'])
    else:
        connect_point_type = 'center'

    line_img_rgba, connect_point = draw_connect_line(thickness, line_type=line_type,
                                                     connect_point_type=connect_point_type)

    width, height = line_img_rgba.size
    if not transparent:
        pad_height = round(random.uniform(0.3, 0.8) * thickness)
        padding = (0, pad_height, 0, pad_height)
        fill_color = (255, 255, 255)
        line_img_rgba = ImageOps.expand(line_img_rgba, border=padding, fill=fill_color)
        connect_point[0] += pad_height
        line_img_rgba = remove_transparent_areas(line_img_rgba)

    rotated_src = rotate_around_point(line_img_rgba, angle, connect_point)
    h, w = rotated_src.size
    line_center_x, line_center_y = w // 2, h // 2
    polygon = get_rotated_rect_points(line_center_x, line_center_y, width, height, angle)
    background, bg_off_x, bg_off_y = paste_rotated_image(background, rotated_src, paste_point)
    offset_x = round(paste_point[0] + bg_off_x - line_center_x)
    offset_y = round(paste_point[1] + bg_off_y - line_center_y)
    polygon = [[point[0] + offset_x, point[1] + offset_y] for point in polygon]
    polygon = fix_polygon_bounds(polygon, background.shape)

    line_type_to_cls = {
        'circle_wave': 'wave_AP',
        'dashed': 'dash_AP',
        'wave': 'wave_bond',
        'zigzag': 'dash_bond',
    }
    cls = line_type_to_cls.get(line_type, 'AP')

    return [polygon], [cls_mapping.index(cls)]


def _draw_bracket_overlay(mol, background, all_boxes, font_size, svg_scale):
    """在已有分子图像上叠加括号和可选下标，支持三种定位模式。"""
    atom_centers = {}
    for idx, box in all_boxes.items():
        atom_centers[idx] = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

    # 按概率选择括号定位模式
    rand = random.random()
    if rand < 0.6:
        bracket_info = _get_single_carbon_bracket_info(mol, atom_centers, all_boxes, font_size, svg_scale)
        bracket_type = random.choice(['paren', 'square'])
    elif rand < 0.9:
        bracket_info = _get_whole_molecule_bracket_info(mol, atom_centers, all_boxes, svg_scale)
        bracket_type = 'square'
    else:
        bracket_info = _get_fragment_bracket_info(mol, atom_centers, all_boxes, svg_scale)
        bracket_type = random.choice(['paren', 'square'])

    if bracket_info is None:
        return None

    lx, ly, rx, ry, angle_rad, height = bracket_info

    thickness = svg_scale * 0.6
    bracket_height = max(int(height), svg_scale * 6)

    left_br, right_br = draw_bracket_pair(bracket_type, bracket_height, thickness)

    rotation_deg = -math.degrees(angle_rad)
    if abs(rotation_deg) > 0.5:
        left_br = left_br.rotate(rotation_deg, expand=True,
                                  resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))
        right_br = right_br.rotate(rotation_deg, expand=True,
                                    resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))

    polygons = []
    classes = []
    cum_x = 0
    cum_y = 0

    # —— 粘贴左括号 ——
    background, off_x, off_y = paste_rotated_image(background, left_br, (round(lx), round(ly)))
    cum_x += off_x
    cum_y += off_y

    lw, lh = left_br.size
    left_rect = [[lx + cum_x - lw / 2, ly + cum_y - lh / 2],
                 [lx + cum_x + lw / 2, ly + cum_y - lh / 2],
                 [lx + cum_x + lw / 2, ly + cum_y + lh / 2],
                 [lx + cum_x - lw / 2, ly + cum_y + lh / 2]]
    polygons.append(fix_polygon_bounds(left_rect, background.shape))
    classes.append(0)  # left_bracket

    # —— 粘贴右括号 ——
    prev_off_x, prev_off_y = off_x, off_y
    background, off_x, off_y = paste_rotated_image(
        background, right_br, (round(rx + prev_off_x), round(ry + prev_off_y)))
    cum_x += off_x
    cum_y += off_y

    rw, rh = right_br.size
    right_rect = [[rx + cum_x - rw / 2, ry + cum_y - rh / 2],
                  [rx + cum_x + rw / 2, ry + cum_y - rh / 2],
                  [rx + cum_x + rw / 2, ry + cum_y + rh / 2],
                  [rx + cum_x - rw / 2, ry + cum_y + rh / 2]]
    polygons.append(fix_polygon_bounds(right_rect, background.shape))
    classes.append(1)  # right_bracket

    # —— 粘贴下标 ——
    sub_content = generate_subscript_content()
    sub_pos = random.choices(['right', 'left', 'none'], weights=[0.7, 0.2, 0.1], k=1)[0]
    if sub_content and sub_pos != 'none':
        sub_img = draw_bracket_subscript(sub_content, font_size=svg_scale * 4)
        if sub_img:
            sw, sh = sub_img.size
            if sub_pos == 'right':
                sx = round(rx + cum_x + rw / 2 + sw / 2 + 2)
                sy = round(ry + cum_y + rh / 2)
            else:
                sx = round(lx + cum_x - lw / 2 - sw / 2 - 2)
                sy = round(ly + cum_y + lh / 2)

            background, off_x, off_y = paste_rotated_image(background, sub_img, (sx, sy))
            sub_rect = [[sx + off_x - sw / 2, sy + off_y - sh / 2],
                        [sx + off_x + sw / 2, sy + off_y - sh / 2],
                        [sx + off_x + sw / 2, sy + off_y + sh / 2],
                        [sx + off_x - sw / 2, sy + off_y + sh / 2]]
            polygons.append(fix_polygon_bounds(sub_rect, background.shape))
            classes.append(2)  # bracket_subscript

    return polygons, classes


def _get_single_carbon_bracket_info(mol, atom_centers, all_boxes, font_size, svg_scale):
    """模式A (60%)：找到仅有两个碳邻居的碳原子，括号放在其两根键上。
    括号高度基于SVG渲染的原子文本大小。"""
    candidates = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != 'C' or atom.GetAtomicNum() == 0:
            continue
        neighbors = [n for n in atom.GetNeighbors()
                     if n.GetSymbol() == 'C' and n.GetAtomicNum() != 0]
        if len(neighbors) == 2:
            candidates.append((atom.GetIdx(), neighbors[0].GetIdx(), neighbors[1].GetIdx()))

    if not candidates:
        return None

    center_idx, n1_idx, n2_idx = random.choice(candidates)

    cx, cy = atom_centers[center_idx]
    n1x, n1y = atom_centers[n1_idx]
    n2x, n2y = atom_centers[n2_idx]

    # 按主方向确定左右邻居
    x_spread = max(cx, n1x, n2x) - min(cx, n1x, n2x)
    y_spread = max(cy, n1y, n2y) - min(cy, n1y, n2y)
    if x_spread >= y_spread:
        if n1x <= n2x:
            left_idx, right_idx = n1_idx, n2_idx
            left_x, left_y = n1x, n1y
            right_x, right_y = n2x, n2y
        else:
            left_idx, right_idx = n2_idx, n1_idx
            left_x, left_y = n2x, n2y
            right_x, right_y = n1x, n1y
    else:
        if n1y <= n2y:
            left_idx, right_idx = n1_idx, n2_idx
            left_x, left_y = n1x, n1y
            right_x, right_y = n2x, n2y
        else:
            left_idx, right_idx = n2_idx, n1_idx
            left_x, left_y = n2x, n2y
            right_x, right_y = n1x, n1y

    # 左括号：放在左邻-中心碳的键中点上
    lx = (left_x + cx) / 2
    ly = (left_y + cy) / 2

    # 右括号：根据三原子共线性决定是否放在键上
    v1x, v1y = left_x - cx, left_y - cy
    v2x, v2y = right_x - cx, right_y - cy
    l1 = math.hypot(v1x, v1y)
    l2 = math.hypot(v2x, v2y)

    if l1 > 0 and l2 > 0:
        dot = (v1x * v2x + v1y * v2y) / (l1 * l2)
        if dot < -0.866:  # cos(150°) 判定共线
            rx = (cx + right_x) / 2
            ry = (cy + right_y) / 2
        else:
            scale = l1 / l2 if l2 > 0 else 1.0
            rx = cx + v2x * scale
            ry = cy + v2y * scale
    else:
        rx = (cx + right_x) / 2
        ry = (cy + right_y) / 2

    angle_rad = math.atan2(ry - ly, rx - lx)

    # 单原子括号高度基于SVG渲染的实际文本大小
    atom_h = all_boxes[center_idx][3] - all_boxes[center_idx][1]
    height = max(atom_h * 2.0, font_size * 1.2)

    return lx, ly, rx, ry, angle_rad, height


def _get_whole_molecule_bracket_info(mol, atom_centers, all_boxes, svg_scale):
    """模式B (30%)：方括号包裹整个分子主体。"""
    valid_boxes = {}
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 0:
            continue
        if idx in all_boxes:
            valid_boxes[idx] = all_boxes[idx]

    if not valid_boxes:
        return None

    x1 = min(b[0] for b in valid_boxes.values())
    y1 = min(b[1] for b in valid_boxes.values())
    x2 = max(b[2] for b in valid_boxes.values())
    y2 = max(b[3] for b in valid_boxes.values())

    mol_w = x2 - x1
    mol_h = y2 - y1
    margin = max(mol_w, mol_h) * 0.12

    if mol_w >= mol_h:
        lx = x1 - margin
        ly = (y1 + y2) / 2
        rx = x2 + margin
        ry = (y1 + y2) / 2
        height = mol_h * 1.3
    else:
        lx = (x1 + x2) / 2
        ly = y2 + margin
        rx = (x1 + x2) / 2
        ry = y1 - margin
        height = mol_w * 1.3

    angle_rad = math.atan2(ry - ly, rx - lx)
    height = max(height, svg_scale * 10)

    return lx, ly, rx, ry, angle_rad, height


def _get_fragment_bracket_info(mol, atom_centers, all_boxes, svg_scale):
    """模式C (~10%)：括号包裹分子片段，左括号在某个具体键上。"""
    repeat_atoms = select_repeat_unit_atoms(mol)
    if repeat_atoms is None or len(repeat_atoms) < 2:
        return None

    selected_boxes = {i: all_boxes[i] for i in repeat_atoms if i in all_boxes}
    if len(selected_boxes) < 2:
        return None

    left_bond_mid = _find_leftmost_bond_in_selection(mol, repeat_atoms, atom_centers)
    if left_bond_mid is None:
        xs = [(atom_centers[i][0], atom_centers[i][1]) for i in repeat_atoms if i in atom_centers]
        if not xs:
            return None
        xs.sort(key=lambda p: p[0])
        lx, ly = xs[0]
    else:
        lx, ly = left_bond_mid

    x1 = min(b[0] for b in selected_boxes.values())
    y1 = min(b[1] for b in selected_boxes.values())
    x2 = max(b[2] for b in selected_boxes.values())
    y2 = max(b[3] for b in selected_boxes.values())

    margin = (y2 - y1) * 0.18
    rx = x2 + margin
    ry = (y1 + y2) / 2

    angle_rad = math.atan2(ry - ly, rx - lx)
    height = max((y2 - y1) * 1.2, svg_scale * 8)

    return lx, ly, rx, ry, angle_rad, height


def _find_leftmost_bond_in_selection(mol, atom_indices, atom_centers):
    """在选中的原子集合中找到x坐标最小的键，返回其中点坐标。"""
    atom_set = set(atom_indices)
    best_mid = None
    best_x = float('inf')

    for bond in mol.GetBonds():
        i1 = bond.GetBeginAtomIdx()
        i2 = bond.GetEndAtomIdx()
        if i1 not in atom_set or i2 not in atom_set:
            continue
        if i1 not in atom_centers or i2 not in atom_centers:
            continue

        x1, y1 = atom_centers[i1]
        x2, y2 = atom_centers[i2]
        mid_x = (x1 + x2) / 2

        if mid_x < best_x:
            best_x = mid_x
            best_mid = (mid_x, (y1 + y2) / 2)

    return best_mid


def select_repeat_unit_atoms(mol, min_atoms=2, max_atoms=8):
    """
    从分子中随机选择一个连通原子子集作为重复单元。
    BFS 从随机非 * 原子出发，收集 2~min(8, 非*原子总数) 个连通原子。
    排除 dummy 原子 (atomic number 0)。保证连通性。
    """
    non_star = [atom.GetIdx() for atom in mol.GetAtoms()
                if atom.GetSymbol() != '*' and atom.GetAtomicNum() != 0]
    if len(non_star) < min_atoms:
        return None

    start = random.choice(non_star)
    target_count = random.randint(min_atoms, min(max_atoms, len(non_star)))
    visited = {start}
    frontier = [start]

    while len(visited) < target_count and frontier:
        current = random.choice(frontier)
        frontier.remove(current)
        atom = mol.GetAtomWithIdx(current)
        neighbors = [n.GetIdx() for n in atom.GetNeighbors()
                     if n.GetSymbol() != '*' and n.GetAtomicNum() != 0
                     and n.GetIdx() not in visited]
        if neighbors:
            random.shuffle(neighbors)
            for n in neighbors:
                if len(visited) >= target_count:
                    break
                visited.add(n)
                frontier.append(n)

    if len(visited) < min_atoms:
        return None
    return list(visited)


def _compute_edge_info_from_boxes(all_boxes, svg_scale):
    """从原子包围盒计算分子边缘信息（用于连接线叠加）。

    Returns:
        (edge_center, center, angle, thickness) 或 None
    """
    if not all_boxes:
        return None

    x1 = min(b[0] for b in all_boxes.values())
    y1 = min(b[1] for b in all_boxes.values())
    x2 = max(b[2] for b in all_boxes.values())
    y2 = max(b[3] for b in all_boxes.values())

    cx = round((x1 + x2) / 2)
    cy = round((y1 + y2) / 2)
    mol_w = x2 - x1
    mol_h = y2 - y1

    if mol_w >= mol_h:
        angle = random.choice([0, 180])
        if random.random() < 0.5:
            edge_center = (round(x1), cy)
        else:
            edge_center = (round(x2), cy)
    else:
        angle = random.choice([90, 270])
        if random.random() < 0.5:
            edge_center = (cx, round(y1))
        else:
            edge_center = (cx, round(y2))

    thickness = svg_scale * random.uniform(0.5, 0.7)
    return edge_center, (cx, cy), angle, thickness


def fix_polygon_bounds(polygon, image_shape):
    height, width = image_shape[:2]
    fixed_polygon = []
    for (x, y) in polygon:
        new_x = max(0, min(width - 1, x))
        new_y = max(0, min(height - 1, y))
        fixed_polygon.append((new_x, new_y))
    return fixed_polygon


def get_rotated_rect_points(cx, cy, width, height, angle_degrees):
    angle = math.radians(angle_degrees + 90)
    half_w = width / 2
    half_h = height / 2
    corners = np.array([
        [-half_w, -half_h],
        [half_w, -half_h],
        [half_w, half_h],
        [-half_w, half_h]
    ])
    cos_theta, sin_theta = math.cos(angle), math.sin(angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    rotated_corners = (corners @ rotation_matrix.T) + [cx, cy]
    return np.round(rotated_corners).astype(np.int32)


def remove_transparent_areas(img_rgba):
    data = np.array(img_rgba)
    rgb = data[..., :3]
    alpha = data[..., 3]
    alpha_scale = alpha.astype(float) / 255.0
    inverse_alpha_scale = 1.0 - alpha_scale
    alpha_scale = alpha_scale[..., np.newaxis]
    inverse_alpha_scale = inverse_alpha_scale[..., np.newaxis]
    new_rgb = np.around(
        rgb.astype(float) * alpha_scale + 255.0 * inverse_alpha_scale
    ).clip(0, 255).astype(np.uint8)
    data[..., :3] = new_rgb
    data[..., 3] = 255
    return Image.fromarray(data)


def paste_rotated_image(background: np.ndarray, rotated_src: Image.Image, target_point: (int, int)) -> tuple:
    rotated_np = np.array(rotated_src)
    rotated_src = cv2.cvtColor(rotated_np, cv2.COLOR_RGBA2BGRA)

    h, w = rotated_src.shape[:2]
    x_center, y_center = target_point

    desired_x_start = x_center - w // 2
    desired_y_start = y_center - h // 2
    desired_x_end = desired_x_start + w
    desired_y_end = desired_y_start + h

    left_extend = max(0, -desired_x_start)
    right_extend = max(0, desired_x_end - background.shape[1])
    top_extend = max(0, -desired_y_start)
    bottom_extend = max(0, desired_y_end - background.shape[0])

    if any([left_extend, right_extend, top_extend, bottom_extend]):
        border_color = (255, 255, 255)
        new_bg = cv2.copyMakeBorder(
            background,
            top=top_extend,
            bottom=bottom_extend,
            left=left_extend,
            right=right_extend,
            borderType=cv2.BORDER_CONSTANT,
            value=border_color
        )
        new_x_center = x_center + left_extend
        new_y_center = y_center + top_extend
    else:
        new_bg = background.copy()
        new_x_center, new_y_center = x_center, y_center

    x_start = new_x_center - w // 2
    y_start = new_y_center - h // 2
    x_end = x_start + w
    y_end = y_start + h

    if x_end <= x_start or y_end <= y_start:
        return new_bg, left_extend, top_extend

    alpha = rotated_src[:, :, 3].astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    foreground = rotated_src[:, :, :3].astype(float)

    roi = new_bg[y_start:y_end, x_start:x_end].astype(float)
    blended = (foreground * alpha + roi * (1 - alpha)).clip(0, 255).astype(np.uint8)
    new_bg[y_start:y_end, x_start:x_end] = blended

    return new_bg, left_extend, top_extend


def rotate_around_point(img, angle, center):
    img_rgba = img.convert('RGBA')
    w, h = img.size
    cx, cy = center

    diagonal = int(math.sqrt(w ** 2 + h ** 2))
    new_size = (w + 2 * diagonal, h + 2 * diagonal)

    expanded_img = Image.new('RGBA', new_size, (0, 0, 0, 0))
    paste_x = new_size[0] // 2 - cx
    paste_y = new_size[1] // 2 - cy
    expanded_img.paste(img_rgba, (paste_x, paste_y))

    rotated_img = expanded_img.rotate(
        angle * -1 + 90,
        center=(new_size[0] // 2, new_size[1] // 2),
        expand=True,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0)
    )

    bbox = rotated_img.getbbox()
    rotated_img = rotated_img.crop(bbox)

    return rotated_img


def convert_bbox_to_polygon(bbox: List[Union[int, float]]) -> List[List[Union[int, float]]]:
    x_min, y_min, x_max, y_max = bbox
    return [
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max],
    ]


if __name__ == '__main__':
    test_smi = '*c1cn(CC)nn1'
    for i in range(3):
        background, polygons, classes = draw_rgroup_struct(test_smi, 3)
        print(f'Test {i}: mode classes={classes}, num_polygons={len(polygons)}')
        cv2.imshow(f'test_{i}', background)
    cv2.waitKey(0)
