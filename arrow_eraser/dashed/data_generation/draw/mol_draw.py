import math
import os
import random
import re
import string
import xml.etree.ElementTree as ET
from typing import List

import cairosvg
import cv2
import numpy as np
from matplotlib import font_manager
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

from utils.base.image_loader import ImageLoader

# from IPython.display import display

img_loader = ImageLoader()

RGROUP_SYMBOLS = ['*', 'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10', 'R11', 'R12',
                  'Ra', 'Rb', 'Rc', 'Rd', "R'", "R''", "R'''", "R1'", "R2'", "R3'", 'X', 'Y', 'Z', 'Q', 'Ar']

WINDOWS_CANDIDATE_FONTS = [
    # --- 无衬线字体 (Sans-serif) : 现代、屏幕显示与化学绘图软件默认 ---
    'Arial', 'Tahoma', 'Microsoft Sans Serif',
    'Calibri', 'Verdana', 'Segoe UI', 'Corbel',
    # --- 衬线字体 (Serif) : 经典文献、学术期刊与教科书常用 ---
    'Times New Roman', 'Georgia', 'Cambria', 'Palatino Linotype',
    'Century Schoolbook', 'Garamond', 'Book Antiqua', 'Constantia',
    # --- 等宽字体 (Monospace) : 代码、SMILES 序列或对齐要求高的场景 ---
    'Courier New', 'Consolas', 'Lucida Console'
]
LINUX_CANDIDATE_FONTS = [
    'Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans',  # 核心推荐 (最常用，兼容性最好)
    'Calibri', 'Verdana', 'Segoe UI', 'Noto Sans',  # Google 多语言支持
    'Times New Roman', 'Georgia', 'Palatino',  # 学术期刊常用
    'Courier New', 'Monaco', 'DejaVu Sans Mono',  # 等宽，符号支持好
    'Telex Regular', 'Roboto Regular'
]




def generate_random_dummy_symbol():
    """
    随机拼接无意义的基团字符串，涵盖所有常见元素、缩写，
    并强制覆盖所有 a-z, A-Z 字母，用于增强超分辨率模型的字母识别泛化能力。
    """
    # 1. 核心有机元素
    organic_atoms = ['C', 'H', 'O', 'N', 'S', 'P']
    # 2. 卤素
    halogens = ['F', 'Cl', 'Br', 'I']
    # 3. 常见金属与微量元素
    metals = ['Li', 'Na', 'K', 'Mg', 'Ca', 'B', 'Al', 'Si', 'Sn', 'Se', 'Te',
              'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Pd', 'Pt', 'Au', 'Ag', 'Ru', 'Rh']
    # 4. 常见基团缩写
    abbreviations = ['Me', 'Et', 'Pr', 'iPr', 'Bu', 'iBu', 'tBu', 'Ph', 'Bn', 'Ac',
                     'Ts', 'Tf', 'Ms', 'Boc', 'Cbz', 'Fmoc', 'Cy', 'TMS', 'TBS', 'PMB', 'Cp']

    # 5. [新增] 补充带有缺失大写字母的稀有元素和常见化学占位符 (补足 W, U, Y, X, Q, D, G 等)
    rare_elements = ['W', 'U', 'Y', 'Zr', 'Xe', 'Kr', 'Pb', 'Hg', 'Gd', 'Dy', 'Mo']
    dummy_atoms = ['X', 'Y', 'Z', 'Q', 'D', 'Ar', 'R']

    # 合并为一个大池子
    text_blocks = organic_atoms * 3 + halogens * 2 + metals + abbreviations * 2 + rare_elements + dummy_atoms

    # 6. [新增] 纯随机字母生成器，确保所有小写/大写字母都能无死角覆盖
    def get_random_chars():
        # 随机生成 1 到 2 个字母 (大小写混合)
        length = random.randint(1, 2)
        return "".join(random.choices(string.ascii_letters, k=length))

    num_fragments = random.randint(1, 4)
    symbol_parts = []

    for _ in range(num_fragments):
        # [优化] 15% 的概率直接硬塞纯随机字母（解决生僻小写字母问题），85% 的概率使用化学库
        if random.random() < 0.15:
            part = get_random_chars()
        else:
            part = random.choice(text_blocks)

        # 40% 的概率在后面跟一个数字 (2 到 9)
        if random.random() < 0.4:
            part += str(random.randint(2, 9))

        symbol_parts.append(part)

    symbol = "".join(symbol_parts)

    # 有极小概率（如 5%）加上电荷符号，模拟离子状态 (如 +, -, 2+, 3-)
    if random.random() < 0.05:
        charge = random.choice(['+', '-', '2+', '2-', '3+', '3-'])
        symbol += charge

    return symbol


def draw_molecule_like_chemdraw(smiles, random_rotate=False, svg_scale=16, show_r_sym=False, return_all_boxes=False):
    # 分子预处理
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    # mol = Chem.AddHs(mol, explicitOnly=True)  # 仅显式添加必要氢

    highlight_atoms = dict()
    highlight_bonds = dict()
    r_sym_atom_ids = []
    r_sym_list = []
    AllChem.Compute2DCoords(mol, canonOrient=True)

    # template = Chem.MolFromSmiles("C(=O)")  # 水平排列的模板
    # AllChem.Compute2DCoords(template)
    # AllChem.GenerateDepictionMatching2DStructure(mol, template)

    if random_rotate:
        rotate_molecule_90(mol, random.choice([True, False]), random.choice([0, 1, 2, 3]))
        flip_molecule(mol, random.choice([True, False]), random.choice([True, False]))

    # 配置绘图参数
    drawer = Draw.MolDraw2DSVG(-1, -1)
    draw_options = drawer.drawOptions()

    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == '*':
            if show_r_sym:
                r_sym = random.choice(RGROUP_SYMBOLS)
                r_sym_list.append(r_sym)
                # 直接调用优化后的格式化函数
                su_r_sym = format_html_subscripts_superscripts(r_sym)
                atom.SetProp("_displayLabel", su_r_sym)
                r_sym_atom_ids.append(i)
            else:
                r_sym_atom_ids.append(i)
                highlight_atoms[i] = (1, 0, 0)
                bond = atom.GetBonds()[0]
                assert bond.GetBondType() == Chem.BondType.SINGLE
                highlight_bonds[bond.GetIdx()] = (0, 0, 1)

                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                connected_atom = atom1 if atom2.GetIdx() == i else atom2
                highlight_atoms[connected_atom.GetIdx()] = (0, 1, 0)

        elif atom.GetDegree() < 2 and atom.GetSymbol() != 'H':
            # 【修改点 1】: 将出现基团替换的概率从 0.3 提高到 0.8 (可根据需求自行微调)
            replace_function_groups = random.choices((True, False), weights=[0.8, 0.2], k=1)[0]

            if replace_function_groups:
                if show_r_sym:
                    # 【修改点 2】: 调整 R 基团和普通 dummy 基团的竞争概率，让 dummy 出现可能性更大 (例如 0.3 vs 0.7)
                    replace_r_sym = random.choices((True, False), weights=[0.3, 0.7], k=1)[0]
                    if replace_r_sym:
                        r_sym = random.choice(RGROUP_SYMBOLS)
                        r_sym_list.append(r_sym)
                        su_r_sym = format_html_subscripts_superscripts(r_sym)
                        atom.SetProp("_displayLabel", su_r_sym)
                        r_sym_atom_ids.append(i)
                        continue

                # 【修改点 3】: 给 dummy_sym 也套上格式化函数，解决 CF3 等无法显示下标的问题
                dummy_sym = generate_random_dummy_symbol()
                su_dummy_sym = format_html_subscripts_superscripts(dummy_sym, random_sup_prob=0.15)
                atom.SetProp("_displayLabel", su_dummy_sym)

    liti = random.choice([True, False])
    if liti:
        mol = set_liti(mol)

    ACS_mode = random.choice([True, False])
    if ACS_mode:
        # Draw.SetACS1996Mode(draw_options, Draw.MeanBondLength(mol))
        # 基础设置
        draw_options.useBWAtomPalette()  # 设为黑白
    bondLineWidth = random.choices([0.5, 0.6, 0.7], weights=[0.25, 0.5, 0.25], k=1)[0]
    draw_options.bondLineWidth = bondLineWidth
    draw_options.scaleBondWidth = False
    draw_options.scalingFactor = 14.4 / Draw.MeanBondLength(mol) * random.uniform(0.8, 0.9)
    draw_options.multipleBondOffset = random.uniform(0.16, 0.2)
    draw_options.fixedFontSize = random.choices([6, 7, 8], weights=[0.25, 0.5, 0.25], k=1)[0]
    draw_options.additionalAtomLabelPadding = random.uniform(0.06, 0.07)

    # 高亮效果设置，辅助计算键的位置
    draw_options.highlightRadius = 0  # 原子高亮圆半径为0
    draw_options.highlightBondWidth = 0  # 键高亮宽度为0
    # draw_options.highlightBondWidthMultiplier = 2

    # 随机内容，如字体、标签
    # if random.choice([True, False]):
    draw_options.fontFile = get_best_font()  # 随机字体

    draw_options.explicitMethyl = random.choices([False, True], weights=[0.8, 0.2], k=1)[0]
    draw_options.addAtomIndices = random.choices([False, True], weights=[0.4, 0.6], k=1)[0]  # 提高数字概率提高鲁棒性
    draw_options.addStereoAnnotation = random.choices([False, True], weights=[0.9, 0.1], k=1)[0]
    draw_options.annotationFontScale = random.choices([0.5, 0.6, 0.7], weights=[0.6, 0.3, 0.2], k=1)[0]
    draw_options.atomLabelFontSize = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]

    # 绘制并保存
    drawer.DrawMolecule(mol, highlightAtoms=list(highlight_atoms.keys()), highlightAtomColors=highlight_atoms,
                        highlightBonds=list(highlight_bonds.keys()), highlightBondColors=highlight_bonds)
    drawer.FinishDrawing()
    svg_content = drawer.GetDrawingText()

    if return_all_boxes:
        font_size = draw_options.fixedFontSize * svg_scale
        half_box = font_size * 0.45
        all_boxes = {}
        for i in range(mol.GetNumAtoms()):
            pos = drawer.GetDrawCoords(i)
            x, y = pos.x * svg_scale, pos.y * svg_scale
            all_boxes[i] = [x - half_box, y - half_box, x + half_box, y + half_box]
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), scale=svg_scale)
        cv_img = img_loader.load_cv_img(png_bytes)
        return cv_img, all_boxes, r_sym_atom_ids, r_sym_list, font_size

    atom_boxes = get_atoms_box(svg_content, r_sym_atom_ids, scale=svg_scale)
    if show_r_sym:
        png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), scale=svg_scale)
        cv_img = img_loader.load_cv_img(png_bytes)
        return cv_img, atom_boxes, r_sym_list

    svg_content, boxes = svg_string_parse(svg_content, svg_scale)
    png_bytes = cairosvg.svg2png(bytestring=svg_content.encode('utf-8'), scale=svg_scale)
    cv_img = img_loader.load_cv_img(png_bytes)
    max_coutour = find_max_contour_in_polygon(cv_img, boxes['polygon'], boxes.get('exclude', None))
    # cv2.drawContours(cv_img, [max_coutour], -1, (0, 255, 0), 2)

    angle = calculate_angle(boxes)
    # print(angle)

    M = cv2.moments(max_coutour)
    assert M["m00"] != 0
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    center = (cX, cY)

    x1, y1, x2, y2 = map(int, boxes['box'])
    cv_img[y1:y2, x1:x2] = [255, 255, 255]
    point = ((x1 + x2) // 2, (y1 + y2) // 2)
    edge_center = get_edge_center(max_coutour, center, angle, point)
    # cv2.circle(cv_img, center, 2, (255, 255, 255))
    # draw_angled_line(cv_img, center, angle)
    thickness = svg_scale * bondLineWidth  # get_thickness(max_coutour, center, angle)

    return cv_img, edge_center, center, angle, thickness


def rotate_molecule_90(mol, clockwise=True, times=1):
    """旋转分子90度（默认顺时针）"""
    conf = mol.GetConformer()
    # 计算几何中心
    center_x, center_y = 0.0, 0.0
    for i in range(conf.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        center_x += pos.x
        center_y += pos.y
    center_x /= conf.GetNumAtoms()
    center_y /= conf.GetNumAtoms()
    # 旋转每个原子
    for t in range(times):
        for i in range(conf.GetNumAtoms()):
            x, y, z = conf.GetAtomPosition(i)
            # 平移至原点
            x -= center_x
            y -= center_y
            # 旋转90度
            if clockwise:  # 顺时针
                new_x = y
                new_y = -x
            else:  # 逆时针
                new_x = -y
                new_y = x
            # 平移回原位置
            new_x += center_x
            new_y += center_y
            # 更新原子位置
            conf.SetAtomPosition(i, (new_x, new_y, z))


def flip_molecule(mol, vertical: bool = True, horizontal: bool = True):
    x_plus = -1 if horizontal else 1
    y_plus = -1 if vertical else 1
    # 获取分子中的共轭物（Conformer），并调整每个原子的位置以模拟翻转
    conf = mol.GetConformer()
    for i in range(conf.GetNumAtoms()):
        # 获取当前原子的位置
        x, y, z = conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z
        conf.SetAtomPosition(i, (x_plus * x, y_plus * y, z))


def set_liti(mol):
    # 转为可编辑的分子对象
    mol = Chem.RWMol(mol)
    # 获取所有单键
    single_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            single_bonds.append(bond.GetIdx())
    # 随机选择一个单键
    if single_bonds:
        selected_bond_idx = random.choice(single_bonds)
        # 设置该键为楔形键（从某个原子出发）
        bond = mol.GetBondWithIdx(selected_bond_idx)
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()

        # 设置立体化学方向（楔形键）
        # 通常设置为从较小编号原子指向较大编号原子
        if begin_atom.GetIdx() < end_atom.GetIdx():
            mol.GetBondWithIdx(selected_bond_idx).SetBondDir(
                random.choice([Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH]))
        else:
            mol.GetBondWithIdx(selected_bond_idx).SetBondDir(
                random.choice([Chem.BondDir.BEGINDASH, Chem.BondDir.BEGINWEDGE]))
    # 转换为标准 Mol 对象
    fixed_mol = mol.GetMol()
    return fixed_mol


def get_thickness(contour, center, angle):
    """计算轮廓在垂直方向上的厚度（保证恰好两个交点）

    参数：
    contour : 轮廓点集（形状为[N,1,2]的numpy数组）
    center  : 中心点坐标 (cx, cy)
    angle   : 主方向角度（度）

    返回：
    厚度值（浮点型）
    """
    # 将角度转换为法向量
    theta = np.deg2rad(angle)
    A = np.cos(theta)  # 法向量的x分量
    B = np.sin(theta)  # 法向量的y分量
    cx, cy = center

    # 准备存储交点
    intersections = []
    contour_pts = contour.squeeze(axis=1)

    # 遍历所有轮廓边
    for i in range(len(contour_pts)):
        p1 = contour_pts[i]
        p2 = contour_pts[(i + 1) % len(contour_pts)]
        x1, y1 = p1
        x2, y2 = p2

        # 计算线段参数方程与直线方程的交点
        denominator = A * (x2 - x1) + B * (y2 - y1)
        if np.isclose(denominator, 0):
            continue  # 忽略平行边

        t = -(A * (x1 - cx) + B * (y1 - cy)) / denominator
        if 0 <= t <= 1:
            # 计算精确交点坐标
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            intersections.append((x, y))

    assert len(intersections) > 1

    pt1, pt2 = np.array(intersections[0]), np.array(intersections[-1])
    return np.linalg.norm(pt1 - pt2)


def get_edge_center(contour, center, angle, point_x):
    """获取连接点"""
    # 步骤1：找到轮廓上距离point_x最近的点y
    contour_pts = np.array([pt[0] for pt in contour])  # 提取轮廓点坐标
    distances = np.linalg.norm(contour_pts - point_x, axis=1)
    min_idx = np.argmin(distances)
    y_point = tuple(contour_pts[min_idx])

    # 角度处理
    angle_rad = np.deg2rad(angle)
    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # 步骤2：构建两条直线
    # --------------------------------------------------
    # 直线1：过y_point且与angle方向垂直的直线
    # 直线方程：cosθ*(x - yx) + sinθ*(y - yy) = 0
    # 即：cosθ*x + sinθ*y - (cosθ*yx + sinθ*yy) = 0
    # 系数：A1=cosθ, B1=sinθ, C1=-(cosθ*yx + sinθ*yy)
    # --------------------------------------------------
    # 直线2：从center沿angle方向延伸的直线
    # 参数方程：x = cx + t*cosθ, y = cy + t*sinθ
    # --------------------------------------------------

    # 联立方程求交点
    denominator = cos_theta ** 2 + sin_theta ** 2  # 恒等于1，保留用于公式展示
    numerator = cos_theta * (y_point[0] - center[0]) + sin_theta * (y_point[1] - center[1])

    t = numerator / denominator
    foot_x = int(center[0] + t * cos_theta)
    foot_y = int(center[1] + t * sin_theta)

    return (foot_x, foot_y)


def draw_angled_line(image, center, angle_deg, length=20, color=(33, 163, 102), thickness=1):
    """
    从中心点按指定角度绘制直线

    参数:
        image: 要绘制的图像
        center: 中心点坐标 (x, y)
        angle_deg: 角度（0-360度，0度指向右侧）
        length: 直线总长度（默认200像素）
        color: 线条颜色（默认绿色）
        thickness: 线宽（默认2像素）
    """
    angle_rad = np.deg2rad(angle_deg)

    # 计算端点坐标
    end_x = int(center[0] + length / 2 * np.cos(angle_rad))
    end_y = int(center[1] + length / 2 * np.sin(angle_rad))
    start_x = int(center[0] - length / 2 * np.cos(angle_rad))
    start_y = int(center[1] - length / 2 * np.sin(angle_rad))

    # 绘制直线
    cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)
    return image


def calculate_angle(data, angle_threshold=1):
    polygon = data['polygon']
    if len(polygon) != 4:
        raise ValueError("Polygon must be a quadrilateral.")

    # 提取所有边（起点、终点、向量）
    edges = []
    for i in range(4):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % 4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        edges.append((p1, p2, (dx, dy)))

    # 按边长降序排序
    edges.sort(key=lambda e: -math.hypot(e[2][0], e[2][1]))
    e1_p1, e1_p2, e1_vec = edges[0]  # 最长边
    e2_p1, e2_p2, e2_vec = edges[1]  # 次长边

    # 计算叉积和向量模长
    cross_product = e1_vec[0] * e2_vec[1] - e1_vec[1] * e2_vec[0]
    v1_norm = math.hypot(e1_vec[0], e1_vec[1])  # |v1|
    v2_norm = math.hypot(e2_vec[0], e2_vec[1])  # |v2|
    # 计算3叉积阈值
    max_allowed_cross = v1_norm * v2_norm * math.sin(math.radians(angle_threshold))
    if abs(cross_product) > max_allowed_cross:
        raise ValueError(f"Edges are not parallel. 多边形 {polygon} ")

    # 计算中线方向向量（与边平行）
    direction_vec = e1_vec if e1_vec != (0, 0) else e2_vec

    # 计算平均角度（以第一条边为准）
    angle_rad = math.atan2(direction_vec[1], direction_vec[0])
    avg_angle_deg = math.degrees(angle_rad) % 360

    return avg_angle_deg


def find_max_contour_in_polygon(cv_img, polygon, exclude=None):
    # 创建与图像大小相同的空白掩码
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)

    # 将多边形坐标转换为NumPy数组格式
    polygon_np = np.array(polygon, dtype=np.int32)

    # 在掩码上绘制填充多边形
    cv2.fillPoly(mask, [polygon_np], 255)

    # 处理排除区域
    if exclude is not None:
        exclude_mask = np.zeros_like(mask)

        if len(exclude) == 4 and all(isinstance(i, (int, float)) for i in exclude):
            # Box格式 (x1,y1,x2,y2)
            x1, y1, x2, y2 = map(int, exclude)
            exclude_mask[y1:y2, x1:x2] = 255
        else:
            # Polygon格式 [(x1,y1), (x2,y2),...]
            exclude_np = np.array(exclude, dtype=np.int32)
            cv2.fillPoly(exclude_mask, [exclude_np], 255)
        # 从主掩码中减去排除区域
        mask = cv2.subtract(mask, exclude_mask)

    # 转换为灰度图并进行二值化
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # 将二值图像与多边形掩码结合
    masked_binary = cv2.bitwise_and(binary, binary, mask=mask)

    # 寻找连通区域
    contours, _ = cv2.findContours(masked_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 过滤掉点数太少的轮廓（噪点）
    contours = [c for c in contours if c.shape[0] >= 5]

    # 返回面积最大的轮廓（如果没有则返回None）
    return max(contours, key=cv2.contourArea) if contours else None


def svg_string_parse(svg_string, scale=8):
    # 解析SVG文件
    root = ET.fromstring(svg_string)

    # 命名空间处理（SVG通常带有命名空间）
    ns = {'svg': 'http://www.w3.org/2000/svg'}

    # 记录形状信息
    boxes = {}

    # 查找红色和绿色圆形（ellipse）并记录
    for ellipse in root.findall(".//svg:ellipse", ns):
        if 'fill:#FF0000' in ellipse.get('style', ''):
            cx, cy = float(ellipse.get('cx')), float(ellipse.get('cy'))
            rx, ry = float(ellipse.get('rx')), float(ellipse.get('ry'))
            boxes['box'] = [cx - rx, cy - ry, cx + rx, cy + ry]
            boxes['box'] = [point * scale for point in boxes['box']]
            # 移除或隐藏圆形（设为透明）
            ellipse.set('style', ellipse.get('style').replace('#FF0000', 'none'))
        if 'fill:#00FF00' in ellipse.get('style', ''):
            cx, cy = float(ellipse.get('cx')), float(ellipse.get('cy'))
            rx, ry = float(ellipse.get('rx')), float(ellipse.get('ry'))
            boxes['exclude'] = [cx - rx, cy - ry, cx + rx, cy + ry]
            boxes['exclude'] = [point * scale for point in boxes['exclude']]
            # 移除或隐藏圆形（设为透明）
            ellipse.set('style', ellipse.get('style').replace('#00FF00', 'none'))

    # 查找蓝色矩形（实际是path）并记录
    for path in root.findall(".//svg:path", ns):
        if 'fill:#0000FF' in path.get('style', ''):
            path_data = path.get('d')
            points = re.findall(r'[ML]\s*([\d.]+),\s*([\d.]+)', path_data)
            polygon = [(float(p[0]) * scale, float(p[1]) * scale) for p in points]
            boxes['polygon'] = polygon
            # 移除或隐藏路径
            path.set('style', path.get('style').replace('#0000FF', 'none'))
        if 'fill:#00FF00' in path.get('style', ''):
            path_data = path.get('d')
            points = re.findall(r'[ML]\s*([\d.]+),\s*([\d.]+)', path_data)
            polygon = [(float(p[0]) * scale, float(p[1]) * scale) for p in points]
            boxes['exclude'] = polygon
            path.set('style', path.get('style').replace('#00FF00', 'none'))

    # 保存修改后的SVG
    svg_string = ET.tostring(root, encoding='utf-8', xml_declaration=True).decode('utf-8')

    # 打印记录的形状信息
    # print("记录的形状信息：", boxes)
    return svg_string, boxes


from utils.base.chem_utils import break_smile


def get_atoms_box(svg_content, atom_idxes, scale):
    # 解析SVG
    root = ET.fromstring(svg_content)
    boxes = []

    for atom_idx in atom_idxes:
        # 查找指定原子的所有path元素
        target_class = f'atom-{atom_idx}'

        # 收集该原子的所有坐标
        all_x_coords = []
        all_y_coords = []

        for path in root.iter('{http://www.w3.org/2000/svg}path'):
            class_attr = path.get('class', '')
            if class_attr == target_class:
                # 解析路径数据
                path_data = path.get('d', '')
                # 提取所有数字坐标
                coords = re.findall(r'[\d.]+', path_data)
                if not coords:
                    continue

                # 转换为浮点数
                float_coords = [float(c) for c in coords]
                # 分离x和y坐标并添加到总坐标列表
                x_coords = float_coords[::2]  # 偶数索引为x坐标
                y_coords = float_coords[1::2]  # 奇数索引为y坐标

                all_x_coords.extend(x_coords)
                all_y_coords.extend(y_coords)

        # 如果找到了坐标，计算整体边界框
        if all_x_coords and all_y_coords:
            min_x = min(all_x_coords)
            max_x = max(all_x_coords)
            min_y = min(all_y_coords)
            max_y = max(all_y_coords)
            scaled_bbox = [x * scale for x in [min_x, min_y, max_x, max_y]]
            boxes.append(scaled_bbox)

    return boxes


def get_all_atom_boxes(svg_content, scale):
    """解析 SVG 中所有原子的包围盒，返回 {atom_idx: [x1, y1, x2, y2], ...}"""
    root = ET.fromstring(svg_content)
    atom_coords = {}

    for path in root.iter('{http://www.w3.org/2000/svg}path'):
        class_attr = path.get('class', '')
        match = re.match(r'atom-(\d+)', class_attr)
        if not match:
            continue
        atom_idx = int(match.group(1))
        path_data = path.get('d', '')
        coords = re.findall(r'[\d.]+', path_data)
        if not coords:
            continue
        float_coords = [float(c) for c in coords]
        x_coords = float_coords[::2]
        y_coords = float_coords[1::2]
        if atom_idx not in atom_coords:
            atom_coords[atom_idx] = [[], []]
        atom_coords[atom_idx][0].extend(x_coords)
        atom_coords[atom_idx][1].extend(y_coords)

    result = {}
    for idx, (xs, ys) in atom_coords.items():
        result[idx] = [min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale]
    return result


# --- 1. 将字体分类定义为全局常量 ---
SANS_SERIF_FONTS = [
    'Arial', 'Tahoma', 'Microsoft Sans Serif',
    'Calibri', 'Verdana', 'Segoe UI', 'Corbel'
]

SERIF_FONTS = [
    'Times New Roman', 'Georgia', 'Cambria', 'Palatino Linotype',
    'Century Schoolbook', 'Garamond', 'Book Antiqua', 'Constantia'
]

MONOSPACE_FONTS = [
    'Courier New', 'Consolas', 'Lucida Console'
]

# --- 2. 核心提速：在模块加载时建立全局可用字体映射 (O(1) 查找) ---
# fontManager.ttflist 包含 Matplotlib 扫描到的所有可用字体
AVAILABLE_FONTS_MAP = {
    f.name: f.fname
    for f in font_manager.fontManager.ttflist
    if os.path.isfile(f.fname)
}


def get_best_font(serif_prob=0.7):
    """
    极速获取最佳可用字体的路径。
    忖线字体概率为0.7提高模型效果
    :param serif_prob: 抽取衬线字体 (Serif) 的概率，范围 0.0 到 1.0。
                       无衬线字体的概率自动为 1.0 - serif_prob。
    :return: 字体文件的绝对路径。
    """
    # 1. 根据概率决定首选与次选字体池
    if random.random() < serif_prob:
        primary_pool = SERIF_FONTS
        secondary_pool = SANS_SERIF_FONTS
    else:
        primary_pool = SANS_SERIF_FONTS
        secondary_pool = SERIF_FONTS

    # 2. 设定降级策略：首选 -> 次选 -> 等宽
    fallback_strategies = [primary_pool, secondary_pool, MONOSPACE_FONTS]

    # 3. 极速查找与抽取
    for pool in fallback_strategies:
        # 利用集合交集思想，快速过滤出当前系统中实际存在的字体
        available_candidates = [font for font in pool if font in AVAILABLE_FONTS_MAP]

        if available_candidates:
            # 随机挑选一个可用字体并直接返回其路径
            chosen_font = random.choice(available_candidates)
            return AVAILABLE_FONTS_MAP[chosen_font]

    # 4. 终极备选方案：如果自定义列表里的字体全军覆没，返回 Matplotlib 默认字体路径
    # 新版 matplotlib 会把字符串 'sans-serif' 当 fontconfig pattern 解析（- 触发 ParseException），
    # 传 list 才走 family-name 路径，行为稳定。
    return font_manager.findfont(font_manager.FontProperties(family=['sans-serif']))


def format_html_subscripts_superscripts(sym: str, random_sup_prob=0.15):
    """
    处理包含数字和电荷的字符串，将其转换为HTML格式的上标或下标。
    random_sup_prob: 数字被随机渲染为上标的概率，用于增加数据多样性。
    """
    # 1. 处理 R 基团 (例如: R1, R2)
    if re.match(r'^R\d+$', sym):
        format_choices = ['subscript', 'superscript', 'none']
        chosen_format = random.choice(format_choices)
        num = sym[1:]
        if chosen_format == 'subscript':
            return f"R<sub>{num}</sub>"
        elif chosen_format == 'superscript':
            return f"R<sup>{num}</sup>"
        return sym

    # 2. 排除不需要加下标的特例缩写（如果有的话，虽然现在是随机拼接，但以防万一）
    ignore_list = ['nBu', 'n-Bu', 'OiBu', 'OtBu', 'tBu', 't-Bu']
    if sym in ignore_list:
        return sym

    # 3. 处理电荷 (例如结尾的 +, -, 2+, 3-) -> 强制转为上标
    # 正则匹配末尾的数字加正负号，或单纯的正负号
    sym = re.sub(r'(\d*[+-])$', r'<sup>\1</sup>', sym)

    # 4. 处理剩余的普通数字
    # 定义一个替换函数，让数字有 random_sup_prob 的概率变成上标，否则变下标
    def replace_num(match):
        num = match.group(1)
        if random.random() < random_sup_prob:
            return f"<sup>{num}</sup>"
        else:
            return f"<sub>{num}</sub>"

    # 注意：为了防止把刚才电荷里已经处理好的 <sup>2</sup> 里面的数字再替换一遍，
    # 我们用一个简单的技巧：只匹配不在 HTML 标签内的数字。
    # 简便起见，我们可以先把 HTML 标签保护起来，或者用不匹配尖括号内内容的正则：
    # (?![^<]*>) 确保数字不在 < > 之间
    formatted_sym = re.sub(r'(\d+)(?![^<]*>)', replace_num, sym)

    return formatted_sym
