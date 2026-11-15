"""
分子图像 + 文本/键 OBB 数据集生成器
用于训练 YOLO-OBB 模型检测分子结构中的：
    class 0: atom_text  -- 原子(超原子/R基团/杂原子)文本框
    class 1: bond       -- 化学键 OBB
    class 2: other_text -- 围绕在分子周围的非分子标签文本 (降级类: 模拟图注、序号、标签等)

主要特性 (相对 mol_draw.py)：
    1. 更宽的参数随机区间：键长(scalingFactor)、字号、键宽、字体、padding 等。
    2. 在分子四周白边区域随机贴入额外文本（"other_text"）。
    3. 更保守的随机超原子: 多片段时只取 1-2 段、字符更短，避免文本重叠。
    4. 更丰富的随机 R 基团符号 (R_1, A^23, X^k, etc.)。
    5. 每个键、每个原子文本、每个其它文本都被标注为 OBB。
    6. 通过 --num-classes 可降级到二分类 (text, bond) — 见 fold_classes。
"""

import argparse
import math
import multiprocessing as mp
import os
import random
import re
import string
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cairosvg
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from tqdm import tqdm

# 关闭 OpenCV 内部线程: 工作负载里都是小图的 fillPoly/minAreaRect, 多线程只带来
# 调度开销 (典型症状: CPU 全核 100% 但吞吐反而比单线程更慢)。多进程并行时这也是
# 必须的, 否则会和 mp.Pool 形成线程超额。
cv2.setNumThreads(1)

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.base.image_loader import ImageLoader  # noqa: E402
from draw.mol_draw import (  # noqa: E402
    rotate_molecule_90,
    flip_molecule,
    set_liti,
    format_html_subscripts_superscripts,
    get_best_font,
    AVAILABLE_FONTS_MAP,
    SANS_SERIF_FONTS,
    SERIF_FONTS,
)


img_loader = ImageLoader()


# ============================================================
# 0. OCR 文本工具
# ============================================================
def html_to_ocr_plain(s):
    """把带 <sub>/<sup> 的 displayLabel 转换成 OCR 训练用的纯文本。
    直接提取标签内容并拼接, 不引入任何格式标记 (_、^、{})。
    OCR 任务只需识别字形本身: R<sub>1</sub> -> R1, A<sup>23</sup> -> A23。
    """
    if s is None:
        return ''
    s = re.sub(r'<sub>(.*?)</sub>', r'\1', s)
    s = re.sub(r'<sup>(.*?)</sup>', r'\1', s)
    s = re.sub(r'<[^>]+>', '', s)
    return s


# ============================================================
# 1. 符号/文本生成
# ============================================================
RGROUP_SYMBOLS = [
    '*', 'R', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9',
    'R10', 'R11', 'R12', 'Ra', 'Rb', 'Rc', 'Rd', "R'", "R''", "R'''",
    "R1'", "R2'", "R3'", 'X', 'Y', 'Z', 'Q', 'Ar',
    # 增补
    'A', 'A1', 'A2', 'A3', 'B1', 'B2', 'D', 'D1', 'E', 'E1', 'G', 'G1',
    'L', 'L1', 'L2', 'M', 'M1', 'T', 'T1', 'U', 'V', 'W',
    'Het', 'Cy', 'Alk', 'Aryl',
]


def _rand_letters(min_len=1, max_len=2, allow_lower=True):
    pool = string.ascii_letters if allow_lower else string.ascii_uppercase
    return "".join(random.choices(pool, k=random.randint(min_len, max_len)))


def make_r_symbol():
    """构造形式多样的 R 基团符号：R1, A23, R^a, (R)+, [A]- 等。"""
    base_pool = ['R', 'A', 'B', 'D', 'E', 'G', 'L', 'M', 'X', 'Y', 'Z', 'Q', 'T', 'U', 'V', 'W', 'Ar', 'Het', 'Cy']
    base = random.choice(base_pool)

    p = random.random()
    if p < 0.25:
        # 数字下标
        return f"{base}<sub>{random.randint(1, 30)}</sub>"
    elif p < 0.38:
        # 数字上标
        return f"{base}<sup>{random.randint(1, 30)}</sup>"
    elif p < 0.46:
        # 字母下标 (Ra, An)
        return f"{base}<sub>{_rand_letters(1, 2, allow_lower=True)}</sub>"
    elif p < 0.54:
        # 字母上标 (R^a)
        return f"{base}<sup>{_rand_letters(1, 1, allow_lower=True)}</sup>"
    elif p < 0.61:
        # 特殊字符上标: 撇号、星号、电荷、变量
        sc = random.choice(["'", "''", "'''", "*", "+", "-", "n", "m"])
        return f"{base}<sup>{sc}</sup>"
    elif p < 0.66:
        # 特殊字符下标
        sc = random.choice(["n", "m", "*"])
        return f"{base}<sub>{sc}</sub>"
    elif p < 0.74:
        # (原括号包裹分支已移除, 避免与 bracket 检测标签冲突)
        return random.choice(RGROUP_SYMBOLS)
    elif p < 0.84:
        # 与已有列表一致
        return random.choice(RGROUP_SYMBOLS)
    else:
        return base


def make_conservative_dummy_symbol():
    """保守版超原子: 仅 1-2 段、整体字符少，降低重叠风险。
    包含括号取代基变体如 (CH3)2、[NH4]+，以及上下标特殊字符。
    """
    organic = ['C', 'H', 'O', 'N', 'S', 'P', 'F', 'Cl', 'Br', 'I']
    metals = ['Li', 'Na', 'K', 'Mg', 'Ca', 'B', 'Al', 'Si', 'Sn', 'Se', 'Zn', 'Cu', 'Fe', 'Mn']
    abbr = ['Me', 'Et', 'Pr', 'iPr', 'Bu', 'tBu', 'Ph', 'Bn', 'Ac', 'Ts',
            'Tf', 'Ms', 'Boc', 'Cbz', 'Cy', 'TMS', 'TBS', 'PMB', 'Cp', 'OMe', 'OEt']
    pool = organic * 3 + metals + abbr * 2

    if random.random() < 0.65:
        # 单段
        part = random.choice(pool) if random.random() > 0.10 else _rand_letters(1, 2)
        if random.random() < 0.35:
            part += str(random.randint(2, 6))
        return part

    # 双段 (短)
    a = random.choice(pool)
    b = random.choice(pool)
    sym = a + b
    if random.random() < 0.30:
        sym += str(random.randint(2, 5))
    if len(sym) > 6:  # 保守: 超过 6 字符就回退
        sym = a
    return sym


# ============================================================
# 2. SVG 解析: 原子文本框 / 键 OBB
# ============================================================
NS = '{http://www.w3.org/2000/svg}'


def parse_svg_dims(svg_string):
    root = ET.fromstring(svg_string)
    w = float(root.get('width', '0').replace('px', ''))
    h = float(root.get('height', '0').replace('px', ''))
    return root, w, h


def collect_atom_boxes(root, scale):
    """返回 {atom_idx: [x1,y1,x2,y2]} (scaled, 像素坐标)。仅返回 SVG 中实际绘制了文字的原子。"""
    atom_coords = {}
    for path in root.iter(f'{NS}path'):
        class_attr = path.get('class', '')
        # 只匹配纯 atom-X (非 bond-X)
        m = re.fullmatch(r'atom-(\d+)', class_attr)
        if not m:
            continue
        atom_idx = int(m.group(1))
        d = path.get('d', '')
        nums = re.findall(r'[-+]?\d*\.\d+|[-+]?\d+', d)
        if not nums:
            continue
        coords = [float(x) for x in nums]
        xs = coords[::2]
        ys = coords[1::2]
        if not xs or not ys:
            continue
        atom_coords.setdefault(atom_idx, [[], []])
        atom_coords[atom_idx][0].extend(xs)
        atom_coords[atom_idx][1].extend(ys)
    return {
        idx: [min(xs) * scale, min(ys) * scale, max(xs) * scale, max(ys) * scale]
        for idx, (xs, ys) in atom_coords.items()
    }


def collect_bond_obbs(root, scale, default_width=2.0):
    """
    返回每个键的 OBB (4 个顶点) 列表。
    SVG 中每个键由 1~多 段 path 组成 (双键有两条平行线)，class 形如 'bond-X atom-Y atom-Z'。
    我们把同一个 bond-X 的所有端点 + 描边宽度一起送入 minAreaRect 计算旋转框。
    """
    bond_paths = {}  # bond_idx -> {'pts': [(x,y), ...], 'width': float}

    for path in root.iter(f'{NS}path'):
        class_attr = path.get('class', '')
        m = re.match(r'bond-(\d+)\b', class_attr)
        if not m:
            continue
        bond_idx = int(m.group(1))

        # 解析 d: "M x,y L x,y L x,y ..."
        d = path.get('d', '')
        pts = []
        for cm in re.finditer(r'[ML]\s*([-+]?\d*\.?\d+)[,\s]+([-+]?\d*\.?\d+)', d):
            pts.append((float(cm.group(1)), float(cm.group(2))))

        # 提取 stroke-width
        style = path.get('style', '')
        sw = default_width
        sm = re.search(r'stroke-width\s*:\s*([\d.]+)', style)
        if sm:
            sw = float(sm.group(1))

        info = bond_paths.setdefault(bond_idx, {'pts': [], 'width': sw})
        info['pts'].extend(pts)
        info['width'] = max(info['width'], sw)

    obbs = []
    for bond_idx, info in bond_paths.items():
        pts = info['pts']
        if len(pts) < 2:
            continue
        half_w = info['width'] * 0.5 * 0.9  # 略收缩, 避免过宽

        # 给端点按线段加宽 -> 多点扩展后 minAreaRect
        expanded = []
        # 对每条相邻线段, 向两侧法线方向扩 half_w
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
        # 也带上原始端点 (兜底)
        expanded.extend([np.array(p) for p in pts])

        arr = np.array(expanded, dtype=np.float32) * scale
        rect = cv2.minAreaRect(arr)
        box = cv2.boxPoints(rect)  # 4x2
        obbs.append(box)

    return obbs


# ============================================================
# 3. 围绕分子的随机文本
# ============================================================
OTHER_TEXT_SAMPLES = [
    # 编号 / 化合物标识
    'Compound {n}', 'Cpd {n}', 'cmpd {n}', 'Example {n}', 'Ex. {n}',
    'Fig. {n}', 'Figure {n}', 'Scheme {n}', 'Table {n}',
    '{n}{a}', '{a}-{n}', '{n}.', '{a}{n}',
    # 物性 / 浓度 / 单位
    'IC50 = {n} nM', 'EC50 = {n}.{n2} μM', 'pKa = {n}.{n2}',
    'mp {n}-{m} °C', '{n}% yield', 'ee = {n}%', 'dr = {n}:{m}',
    '{n} mg', '{n}.{n2} mmol', '{n} equiv', 'rt, {n} h',
    # 条件
    'DMF, 80 °C', 'THF, 0 °C', 'DCM, rt', 'EtOH, reflux',
    'NaH, DMF', 'K2CO3, MeCN', 'Pd(OAc)2', 'CuI, 110 °C',
    # 短描述
    'this work', 'ref. {n}', 'see SI', 'literature',
]


def _format_text_template(t):
    return (t
            .replace('{n}', str(random.randint(1, 99)))
            .replace('{n2}', str(random.randint(0, 9)))
            .replace('{m}', str(random.randint(1, 99)))
            .replace('{a}', random.choice(string.ascii_lowercase)))


# 仅衬线 + 无衬线字体: 排除等宽字体 (Courier New 等), 贴近化学论文打印体风格。
# 在模块加载时一次性过滤, 运行时直接从列表随机选取, 无额外开销。
_PRINT_FONT_POOL = [f for f in (SERIF_FONTS + SANS_SERIF_FONTS) if f in AVAILABLE_FONTS_MAP]


def _pick_font_path():
    if _PRINT_FONT_POOL:
        return AVAILABLE_FONTS_MAP[random.choice(_PRINT_FONT_POOL)]
    # 系统字体极少时降级到 matplotlib 默认 sans-serif
    return get_best_font(serif_prob=0.5)

def _measure_text(text, font):
    """返回 textbbox (left, top, right, bottom) — 文本绘制于 (0,0) 时的精确像素范围。"""
    img = Image.new('RGBA', (10, 10))
    draw = ImageDraw.Draw(img)
    return draw.textbbox((0, 0), text, font=font)


def _render_text_tile(text, font_path, font_size, angle_deg, color=(0, 0, 0)):
    """渲染一段文本到带 alpha 的 RGBA 图块, 并旋转 angle_deg 度。

    返回 (RGBA Image, 4 顶点 (本图块坐标系下))。顶点紧贴文本字形 (不含 padding)，
    并对负 bearing / 下降部 做了偏移补偿, 因此 OBB 与文本贴合且不会截断字尾。
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    bbox = _measure_text(text, font)  # (left, top, right, bottom)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw <= 0 or th <= 0:
        raise ValueError(f"empty text bbox for '{text}'")

    pad = max(2, font_size // 8)
    w = tw + 2 * pad
    h = th + 2 * pad
    tile = Image.new('RGBA', (w, h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(tile)
    # 关键: 用 -bbox[0]/-bbox[1] 抵消字体的负 bearing, 确保字形落在 (pad, pad)~(pad+tw, pad+th)
    draw.text((pad - bbox[0], pad - bbox[1]), text, font=font, fill=color + (255,))

    # 紧贴字形的 4 角 (不含 padding)
    corners = np.array([
        [pad, pad],
        [pad + tw, pad],
        [pad + tw, pad + th],
        [pad, pad + th],
    ], dtype=np.float32)

    if abs(angle_deg) > 0.1:
        rotated = tile.rotate(angle_deg, expand=True, resample=Image.BICUBIC)
        rw, rh = rotated.size
        theta = math.radians(angle_deg)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        cx_in, cy_in = w / 2, h / 2
        cx_out, cy_out = rw / 2, rh / 2
        new_corners = []
        for x, y in corners:
            dx, dy = x - cx_in, y - cy_in
            nx = dx * cos_t + dy * sin_t
            ny = -dx * sin_t + dy * cos_t
            new_corners.append([nx + cx_out, ny + cy_out])
        return rotated, np.array(new_corners, dtype=np.float32)

    return tile, corners


def add_random_other_texts(cv_img, mol_bbox, target_font_px, max_count=6):
    """
    在 cv_img 的白边区域贴入随机文本。
    mol_bbox: [x1,y1,x2,y2] 分子内容的实际边界。
    target_font_px: 期望的字号 (像素), 与原子文本字号同量级。
    返回新增的 [(corners_4x2, plain_text), ...] 列表。
    """
    H, W = cv_img.shape[:2]
    mx1, my1, mx2, my2 = mol_bbox
    placed = []
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)).convert('RGBA')

    count = random.randint(0, max_count)
    attempts = 0
    while len(placed) < count and attempts < count * 8:
        attempts += 1
        text = _format_text_template(random.choice(OTHER_TEXT_SAMPLES))
        # 以原子文本字号为基准随机抖动 (0.7x ~ 1.6x)
        fs_mul = random.uniform(0.7, 1.6)
        font_size = max(8, int(round(target_font_px * fs_mul)))
        font_path = _pick_font_path()
        # 标签文本保持正向, 仅允许轻微倾斜
        if random.random() < 0.15:
            angle = random.uniform(-5, 5)
        else:
            angle = 0
        try:
            tile, local_corners = _render_text_tile(text, font_path, font_size, angle)
        except Exception:
            continue
        tw, th = tile.size
        if tw >= W or th >= H:
            continue

        # 随机选择粘贴位置 (尝试避开分子)
        ok = False
        for _ in range(30):
            x = random.randint(0, W - tw)
            y = random.randint(0, H - th)
            tx1, ty1, tx2, ty2 = x, y, x + tw, y + th
            inter = not (tx2 < mx1 or tx1 > mx2 or ty2 < my1 or ty1 > my2)
            if inter:
                continue
            overlap_other = False
            for prev_corners, _ in placed:
                px1, py1 = prev_corners.min(axis=0)
                px2, py2 = prev_corners.max(axis=0)
                if not (tx2 < px1 or tx1 > px2 or ty2 < py1 or ty1 > py2):
                    overlap_other = True
                    break
            if overlap_other:
                continue
            ok = True
            break
        if not ok:
            continue

        pil_img.alpha_composite(tile, (x, y))
        global_corners = local_corners.copy()
        global_corners[:, 0] += x
        global_corners[:, 1] += y
        placed.append((global_corners, text))

    rgb = np.array(pil_img.convert('RGB'))
    cv_img[:] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return placed


# ============================================================
# 4. 主绘制函数
# ============================================================
def axis_box_to_polygon(box):
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def compute_mol_bbox(atom_boxes, bond_obbs):
    pts = []
    for b in atom_boxes.values():
        x1, y1, x2, y2 = b
        pts.extend([[x1, y1], [x2, y2]])
    for o in bond_obbs:
        pts.extend(o.tolist())
    if not pts:
        return None
    arr = np.array(pts)
    return [arr[:, 0].min(), arr[:, 1].min(), arr[:, 0].max(), arr[:, 1].max()]


def draw_molecule_obb(smiles, svg_scale=None, with_padding=True):
    """
    返回 (cv_img, labels) 其中 labels 为列表，每项: (class_id, polygon_4x2)
    class: 0=atom_text 1=bond 2=other_text
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

    drawer = Draw.MolDraw2DSVG(-1, -1)
    opts = drawer.drawOptions()

    # ---- 随机参数 ----
    show_r_sym = random.random() < 0.6
    use_bw = random.random() < 0.7
    if use_bw:
        opts.useBWAtomPalette()

    bondLineWidth = random.choices([0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                                   weights=[0.05, 0.2, 0.3, 0.25, 0.15, 0.05], k=1)[0]
    opts.bondLineWidth = bondLineWidth
    opts.scaleBondWidth = False
    # 大范围键长缩放
    bond_scale_mul = random.uniform(0.6, 1.05)
    opts.scalingFactor = 14.4 / max(Draw.MeanBondLength(mol), 1e-3) * bond_scale_mul
    opts.multipleBondOffset = random.uniform(0.13, 0.24)
    # 字号区间扩大
    opts.fixedFontSize = random.choices([5, 6, 7, 8, 9, 10, 12],
                                        weights=[0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05], k=1)[0]
    opts.additionalAtomLabelPadding = random.uniform(0.04, 0.10)
    opts.highlightRadius = 0
    opts.highlightBondWidth = 0
    opts.fontFile = get_best_font()
    opts.explicitMethyl = random.choices([False, True], weights=[0.85, 0.15], k=1)[0]
    # 关掉无 bbox 的额外注解, 避免出现不在标签框中的文字
    opts.addAtomIndices = False
    opts.addStereoAnnotation = False
    opts.annotationFontScale = 0.6
    opts.atomLabelFontSize = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2], k=1)[0]

    # ---- R 基团 / dummy 替换 ----
    r_sym_atom_ids = []
    highlight_atoms = {}
    highlight_bonds = {}
    atom_display = {}  # atom_idx -> raw display label (with HTML)
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() == '*':
            sym = make_r_symbol() if show_r_sym else random.choice(RGROUP_SYMBOLS)
            label = sym if '<' in sym else format_html_subscripts_superscripts(sym)
            atom.SetProp("_displayLabel", label)
            atom_display[i] = label
            r_sym_atom_ids.append(i)
        elif atom.GetDegree() < 2 and atom.GetSymbol() != 'H':
            if random.random() < 0.7:
                if show_r_sym and random.random() < 0.35:
                    sym = make_r_symbol()
                    label = sym if '<' in sym else format_html_subscripts_superscripts(sym)
                    atom.SetProp("_displayLabel", label)
                    atom_display[i] = label
                else:
                    dummy = make_conservative_dummy_symbol()
                    # 若 dummy 已含 HTML 标签 (括号变体), 直接使用; 否则交给 format 处理
                    label = dummy if '<' in dummy else format_html_subscripts_superscripts(dummy, random_sup_prob=0.12)
                    atom.SetProp("_displayLabel", label)
                    atom_display[i] = label
        else:
            # 杂原子等内置标签 - 用原子符号作为 OCR 真值兜底
            sym = atom.GetSymbol()
            if sym not in ('C',):
                atom_display[i] = sym

    if random.random() < 0.5:
        mol = set_liti(mol)

    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    root, svg_w_raw, svg_h_raw = parse_svg_dims(svg)

    if svg_scale is None:
        # 按"目标最终像素尺寸"等概率分桶选择 scale, 使图像大小分布均匀。
        # 原版固定倍率 random.choice([10..18]) 不考虑分子大小, 小分子产几百像素的图,
        # 复杂大分子能产 3000+ 像素的图 (并导致 has_excessive_overlap OOM)。
        # 现在按 max(svg_w, svg_h) 反推 scale, 把"最终图像最长边"落在 4 个等概率桶里。
        svg_max_raw = max(svg_w_raw, svg_h_raw, 1.0)
        target_buckets = [
            (300, 450),    # small
            (450, 600),    # medium
            (600, 800),   # large
            (800, 1000),  # xl
        ]
        lo, hi = random.choice(target_buckets)
        target = random.randint(lo, hi)
        # 限制极端值: scale 不低于 2 (避免文字糊成一团), 不高于 14 (兜底防御)
        svg_scale = max(2.0, min(14.0, target / svg_max_raw))

    atom_boxes = collect_atom_boxes(root, svg_scale)
    bond_obbs = collect_bond_obbs(root, svg_scale, default_width=bondLineWidth * 2.0)

    png_bytes = cairosvg.svg2png(bytestring=svg.encode('utf-8'), scale=svg_scale)
    cv_img = img_loader.load_cv_img(png_bytes)
    H, W = cv_img.shape[:2]

    # ---- 随机加白边以放置外部文本 ----
    if with_padding:
        pad_l = random.randint(int(0.05 * W), int(0.30 * W))
        pad_r = random.randint(int(0.05 * W), int(0.30 * W))
        pad_t = random.randint(int(0.05 * H), int(0.30 * H))
        pad_b = random.randint(int(0.05 * H), int(0.30 * H))
        new_W, new_H = W + pad_l + pad_r, H + pad_t + pad_b
        canvas = np.full((new_H, new_W, 3), 255, dtype=np.uint8)
        canvas[pad_t:pad_t + H, pad_l:pad_l + W] = cv_img
        cv_img = canvas

        # 平移所有坐标
        for k in atom_boxes:
            x1, y1, x2, y2 = atom_boxes[k]
            atom_boxes[k] = [x1 + pad_l, y1 + pad_t, x2 + pad_l, y2 + pad_t]
        for i in range(len(bond_obbs)):
            bond_obbs[i][:, 0] += pad_l
            bond_obbs[i][:, 1] += pad_t

    # ---- 计算分子整体 bbox 用于避让 ----
    mol_bbox = compute_mol_bbox(atom_boxes, bond_obbs)
    if mol_bbox is None:
        # 兜底: 用图片中心区域
        mol_bbox = [cv_img.shape[1] * 0.2, cv_img.shape[0] * 0.2,
                    cv_img.shape[1] * 0.8, cv_img.shape[0] * 0.8]
    # 给 mol_bbox 加一些保护边界
    pb = 6
    mol_bbox_safe = [mol_bbox[0] - pb, mol_bbox[1] - pb,
                     mol_bbox[2] + pb, mol_bbox[3] + pb]

    # ---- 加入随机外部文本 (字号以原子文本为基准) ----
    atom_font_px = opts.fixedFontSize * svg_scale
    other_text_items = add_random_other_texts(
        cv_img, mol_bbox_safe,
        target_font_px=atom_font_px,
        max_count=random.randint(0, 6),
    )

    # ---- 汇总标注 ----
    labels = []
    H, W = cv_img.shape[:2]

    def _clip_poly(poly):
        poly = np.array(poly, dtype=np.float32).copy()
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
        return poly

    # OCR 样本: [(axis_box (x1,y1,x2,y2), plain_text)]
    ocr_samples = []

    for atom_idx, box in atom_boxes.items():
        poly = _clip_poly(axis_box_to_polygon(box))
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((0, poly))
        # 若有 display label, 加入 OCR
        raw = atom_display.get(atom_idx)
        if raw:
            plain = html_to_ocr_plain(raw)
            if plain:
                ocr_samples.append((box, plain))

    for obb in bond_obbs:
        poly = _clip_poly(obb)
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((1, poly))

    for corners, plain in other_text_items:
        poly = _clip_poly(corners)
        if (poly.max(axis=0) - poly.min(axis=0)).min() < 1:
            continue
        labels.append((2, poly))
        # other_text 直接用平面文本 (已无 HTML)
        ocr_samples.append((corners, plain))

    return cv_img, labels, ocr_samples


# ============================================================
# 5. YOLO OBB 标签格式化
# ============================================================
def labels_to_yolo_obb(labels, img_w, img_h):
    lines = []
    for cls, poly in labels:
        pts = poly.reshape(-1, 2)
        if pts.shape[0] != 4:
            # 兜底转换为 minAreaRect
            rect = cv2.minAreaRect(pts.astype(np.float32))
            pts = cv2.boxPoints(rect)
        norm = pts.copy().astype(np.float64)
        norm[:, 0] /= img_w
        norm[:, 1] /= img_h
        norm = np.clip(norm, 0.0, 1.0)
        line = f"{cls} " + " ".join(f"{v:.6f}" for v in norm.flatten())
        lines.append(line)
    return "\n".join(lines)


def fold_classes(labels, num_classes):
    """3 -> 2 降级: 把 atom_text(0) 和 other_text(2) 合并成 text(0), bond(1) 不变。"""
    if num_classes >= 3:
        return labels
    out = []
    for cls, poly in labels:
        if cls == 2:
            cls = 0  # all text => 0
        out.append((cls, poly))
    return out


def has_excessive_overlap(labels, img_shape, threshold=0.4, skip_classes=None,
                          class_thresholds=None):
    """若任一框被其它框覆盖面积超阈值, 返回 True 表示应废弃。

    实现: 全图只保留一张 uint16 count_map; 每个标签仅在自身 bbox 局部范围内
    光栅化小掩码, 两遍扫描 (Pass1 累加, Pass2 比对 ratio)。

    Args:
        threshold: 默认阈值 (所有类别统一).
        skip_classes: 可选, set of class_id 在 overlap 计算中完全跳过.
        class_thresholds: 可选, dict of class_id → threshold 覆盖默认阈值.
            例: {0: 0.20, 1: 0.55, 3: 0.25} 为文本/键/overlay 分别设阈值.
    """
    if not labels:
        return False
    skip = skip_classes or set()
    ct = class_thresholds or {}
    H, W = img_shape[:2]

    # 预处理: 标签 -> (cls, int32 poly, 裁剪后的 bbox)。越界/零面积直接丢弃。
    items = []
    for cls, poly in labels:
        if cls in skip:
            continue
        ipoly = poly.astype(np.int32)
        x1 = max(0, int(ipoly[:, 0].min()))
        y1 = max(0, int(ipoly[:, 1].min()))
        x2 = min(W, int(ipoly[:, 0].max()) + 1)
        y2 = min(H, int(ipoly[:, 1].max()) + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        items.append((cls, ipoly, x1, y1, x2, y2))

    if not items:
        return False

    count_map = np.zeros((H, W), dtype=np.uint16)

    # Pass 1: 累加 count_map (每个标签只光栅化到自己的小 bbox)
    for _, ipoly, x1, y1, x2, y2 in items:
        h, w = y2 - y1, x2 - x1
        m = np.zeros((h, w), dtype=np.uint8)
        shifted = ipoly.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1
        cv2.fillPoly(m, [shifted], 1)
        count_map[y1:y2, x1:x2] += m

    # Pass 2: 逐标签计算被其它框覆盖的比例 (分类别阈值)
    for cls, ipoly, x1, y1, x2, y2 in items:
        h, w = y2 - y1, x2 - x1
        m = np.zeros((h, w), dtype=np.uint8)
        shifted = ipoly.copy()
        shifted[:, 0] -= x1
        shifted[:, 1] -= y1
        cv2.fillPoly(m, [shifted], 1)
        area = int(m.sum())
        if area < 4:  # 极小框跳过, 防除零
            continue
        cm_slice = count_map[y1:y2, x1:x2]
        ov = int(((cm_slice >= 2) & (m > 0)).sum())
        t = ct.get(cls, threshold)
        if ov / area > t:
            return True
    return False


# ============================================================
# 6. 数据集生成 CLI
# ============================================================
def visualize_labels(img, labels, palette=None):
    palette = palette or {0: (0, 165, 255), 1: (0, 255, 0), 2: (255, 0, 255)}
    vis = img.copy()
    for cls, poly in labels:
        pts = poly.astype(np.int32)
        cv2.polylines(vis, [pts], True, palette.get(cls, (0, 0, 255)), 2)
    return vis


def iter_smiles(csv_path, limit=None, shuffle=True):
    df = pd.read_csv(csv_path)
    if 'SMILES' not in df.columns:
        raise ValueError("csv 文件必须包含 'SMILES' 列")
    smis = df['SMILES'].dropna().tolist()
    if shuffle:
        random.shuffle(smis)
    if limit:
        smis = smis[:limit]
    return smis


class OCRBalancer:
    """按文本频次衰减采样的 OCR 样本均衡器。

    - 单一文本的样本数 < soft_cap 时全保留;
    - 超过后按 (soft_cap / count)^alpha 概率保留, alpha 越大衰减越激进。

    这样高频文本 (单字母原子 'O' 'N' 'H' 等) 会被压平, 而稀有的 R 基团符号 (R_1, A^23)
    几乎都被保留, 解决了原版直接累加导致的极度不平衡问题。
    """

    def __init__(self, soft_cap=50, alpha=1.5):
        self.counts = {}
        self.soft_cap = max(1, int(soft_cap))
        self.alpha = float(alpha)

    def maybe_keep(self, text):
        c = self.counts.get(text, 0)
        if c < self.soft_cap:
            keep = True
        else:
            keep_prob = (self.soft_cap / (c + 1)) ** self.alpha
            keep = random.random() < keep_prob
        if keep:
            self.counts[text] = c + 1
        return keep


def _worker_init():
    """子进程初始化: 限制 OpenCV 线程, 并基于 pid+随机熵 重置 RNG 避免多 worker 同构。"""
    cv2.setNumThreads(1)
    seed = (os.getpid() * 2654435761) ^ int.from_bytes(os.urandom(4), 'little')
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


def _gen_worker(smi):
    """子进程入口: 生成一张图; 任何异常返回 None, 避免单条样本拖垮整个 pool。"""
    try:
        return draw_molecule_obb(smi)
    except Exception:
        return None


def _crop_obb_or_axis(img, region, pad_ratio=0.10, min_pad=2):
    """region 可以是 4 个点的 OBB (np.ndarray Nx2) 或 axis-aligned [x1,y1,x2,y2]。
    返回 (crop_img, ok)。OBB 时做透视变换拉正；axis-aligned 时直接切片+padding。"""
    H, W = img.shape[:2]
    if isinstance(region, (list, tuple)) and len(region) == 4 and not isinstance(region[0], (list, tuple, np.ndarray)):
        x1, y1, x2, y2 = map(float, region)
        bw, bh = x2 - x1, y2 - y1
        pad = max(min_pad, int(round(max(bw, bh) * pad_ratio)))
        x1 = int(max(0, x1 - pad))
        y1 = int(max(0, y1 - pad))
        x2 = int(min(W, x2 + pad))
        y2 = int(min(H, y2 + pad))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None, False
        return img[y1:y2, x1:x2].copy(), True

    # OBB 路径: 4 点
    pts = np.array(region, dtype=np.float32)
    if pts.shape != (4, 2):
        return None, False
    # 估算 OBB 宽高
    e01 = np.linalg.norm(pts[1] - pts[0])
    e12 = np.linalg.norm(pts[2] - pts[1])
    if min(e01, e12) < 4:
        return None, False
    if e01 >= e12:
        w_out, h_out = int(round(e01)), int(round(e12))
        src = pts.astype(np.float32)
    else:
        w_out, h_out = int(round(e12)), int(round(e01))
        src = np.array([pts[1], pts[2], pts[3], pts[0]], dtype=np.float32)
    # padding
    pad = max(min_pad, int(round(max(w_out, h_out) * pad_ratio)))
    dst = np.array([[pad, pad],
                    [pad + w_out, pad],
                    [pad + w_out, pad + h_out],
                    [pad, pad + h_out]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(img, M, (w_out + 2 * pad, h_out + 2 * pad),
                               borderValue=(255, 255, 255))
    return crop, True


def generate_dataset(out_dir, n_train, n_val, csv_path, num_classes=3,
                    vis_n=0, seed=42, ocr_export=True, overlap_threshold=0.4,
                    workers=1, ocr_balance_cap=50, ocr_balance_alpha=1.5):
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
        idx_global = 0
        for split, n in splits.items():
            img_dir = out_dir / split / 'images'
            lbl_dir = out_dir / split / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            vis_dir = out_dir / split / 'vis'
            if vis_n > 0:
                vis_dir.mkdir(parents=True, exist_ok=True)

            ocr_img_dir = None
            ocr_label_lines = []
            ocr_counter = 0
            balancer = None
            if ocr_export:
                ocr_img_dir = ocr_root / 'images' / split
                ocr_img_dir.mkdir(parents=True, exist_ok=True)
                balancer = OCRBalancer(soft_cap=ocr_balance_cap, alpha=ocr_balance_alpha)

            max_attempts = n * 20
            # 用闭包状态控制 feed 的生成器, 一旦 produced 达到 n 立即停止吐 smi,
            # 避免多进程模式下白白处理一大批多余样本。
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
                results_iter = pool.imap_unordered(_gen_worker, _feed_gen(), chunksize=2)
            else:
                def _seq_iter():
                    for smi in _feed_gen():
                        try:
                            yield draw_molecule_obb(smi)
                        except Exception:
                            yield None
                results_iter = _seq_iter()

            rejected_overlap = 0
            attempts = 0
            ocr_kept = 0
            ocr_dropped = 0
            pbar = tqdm(total=n, desc=f'[{split}]', unit='img', dynamic_ncols=True,
                        smoothing=0.1)

            for result in results_iter:
                attempts += 1
                if state['produced'] >= n:
                    break
                if result is None:
                    continue
                img, labels, ocr_samples = result
                if not labels:
                    continue
                if has_excessive_overlap(labels, img.shape, threshold=overlap_threshold):
                    rejected_overlap += 1
                    pbar.set_postfix_str(
                        f'rej={rejected_overlap} ocr={ocr_kept}/{ocr_kept + ocr_dropped}',
                        refresh=False)
                    continue
                labels_fold = fold_classes(labels, num_classes)
                h, w = img.shape[:2]
                produced = state['produced']
                name = f"{split}_{produced:06d}"
                cv2.imwrite(str(img_dir / f"{name}.png"), img)
                with open(lbl_dir / f"{name}.txt", 'w', encoding='utf-8') as f:
                    f.write(labels_to_yolo_obb(labels_fold, w, h))
                if produced < vis_n:
                    vis = visualize_labels(img, labels_fold)
                    cv2.imwrite(str(vis_dir / f"{name}.png"), vis)

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
                    f'rej={rejected_overlap} ocr={ocr_kept}/{ocr_kept + ocr_dropped}',
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
                  f"rejected_overlap={rejected_overlap}")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(ROOT_DIR / 'dataset' / 'yolo_mol_obb'),
                    help='输出根目录')
    ap.add_argument('--csv', default=str(ROOT_DIR / 'dataset' / 'train_200k.csv'))
    ap.add_argument('--train', type=int, default=5000)
    ap.add_argument('--val', type=int, default=500)
    ap.add_argument('--num-classes', type=int, default=3, choices=[2, 3])
    ap.add_argument('--vis', type=int, default=10, help='每个 split 保存可视化样本数量')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no-ocr', action='store_true', help='不导出 OCR 训练数据')
    ap.add_argument('--overlap', type=float, default=0.4,
                    help='单框被其它框覆盖比例阈值; 超过则废弃整张图 (默认 0.4)')
    ap.add_argument('--workers', type=int, default=6,
                    help='多进程并行 worker 数 (默认 1 单进程; 设为 CPU 物理核数附近通常最快)')
    ap.add_argument('--ocr-cap', type=int, default=50,
                    help='OCR 同一文本的软上限, 超过后按概率衰减保留 (默认 50)')
    ap.add_argument('--ocr-alpha', type=float, default=1.5,
                    help='OCR 衰减强度, 越大对高频文本压制越强 (默认 1.5)')
    args = ap.parse_args()

    generate_dataset(args.out, args.train, args.val, args.csv,
                     num_classes=args.num_classes, vis_n=args.vis, seed=args.seed,
                     ocr_export=not args.no_ocr,
                     overlap_threshold=args.overlap,
                     workers=args.workers,
                     ocr_balance_cap=args.ocr_cap,
                     ocr_balance_alpha=args.ocr_alpha)
    print('done.')


if __name__ == '__main__':
    main()
