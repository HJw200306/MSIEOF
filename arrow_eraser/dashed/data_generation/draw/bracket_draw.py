"""Markush 括号渲染: matplotlib LaTeX 延展括号 + PIL 系统字体括号。

字体策略 (仅使用文档印刷常用字体, 排除艺术体/手写体/装饰体):
    1. matplotlib 数学排版 (3 款正文字体集: cm/stix/dejavuserif × 可变 DPI).
    2. PIL 系统字体渲染: 仅从系统字体中选取文档印刷标准字体 (Times/Arial/
       Cambria/Georgia/Palatino/Courier/STIX 等) 渲染 '[' / '(' 字符.
    3. PIL 线艺绘制: 用 ImageDraw 画方括号/圆括号, 纯几何风格, 不依赖字体.
"""

import io
import math
import os
import random
import string
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.ndimage import label, find_objects

# ---- matplotlib 数学字体集 (仅保留印刷正文字体) ----
_MATH_FONTSETS = ['cm', 'stix', 'dejavuserif']

# DPI 变化 (去掉 72——低分辨率渲染放大后模糊)
_DPIS = [96, 120, 150, 200]

# PIL 线艺括号样式
_PIL_LINE_STYLES = ['square', 'paren']


# ============================================================
# 系统 TrueType 字体扫描 (PIL 字体渲染用)
# ============================================================
# 文档印刷常用字体 (不含 Symbol——其编码将 ASCII 映射到希腊/数学符号, 会导致豆腐块)
_STANDARD_FONT_NAMES = [
    'times', 'timesnewroman', 'times new roman',
    'arial', 'helvetica',
    'cambria', 'calibri',
    'georgia', 'palatino',
    'courier', 'couriernew', 'courier new',
    'stix', 'computer modern', 'dejavu serif',
]
_CACHED_SYSTEM_FONTS = None

def _get_system_fonts():
    """返回系统上可用的标准印刷 TrueType 字体路径列表 (缓存)."""
    global _CACHED_SYSTEM_FONTS
    if _CACHED_SYSTEM_FONTS is not None:
        return _CACHED_SYSTEM_FONTS

    fonts = []
    # Windows
    win_font_dir = os.environ.get('WINDIR', r'C:\Windows') + r'\Fonts'
    # Linux / Mac
    for search_dir in [
        win_font_dir,
        '/usr/share/fonts',
        '/usr/local/share/fonts',
        '/System/Library/Fonts',
        os.path.expanduser('~/.fonts'),
        os.path.expanduser('~/Library/Fonts'),
    ]:
        if os.path.isdir(search_dir):
            for root, _dirs, files in os.walk(search_dir):
                for fn in files:
                    if fn.lower().endswith(('.ttf', '.otf', '.ttc')):
                        name_lower = fn.lower()
                        # 仅保留文档印刷常用字体
                        if any(std in name_lower for std in _STANDARD_FONT_NAMES):
                            fonts.append(os.path.join(root, fn))
    if not fonts:
        fonts.append('arial.ttf')  # PIL 兜底
    _CACHED_SYSTEM_FONTS = fonts
    return fonts


# ============================================================
# 通用工具
# ============================================================
def image_strip_rgba(img_rgba):
    """裁剪RGBA图像的透明边缘"""
    if isinstance(img_rgba, Image.Image):
        img_arr = np.array(img_rgba)
    else:
        img_arr = img_rgba
    alpha = img_arr[:, :, 3]
    mask = alpha > 0
    if not mask.any():
        return img_arr
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img_arr[y0:y1, x0:x1]


def extract_single_bracket_via_cc(img, is_left):
    """通过连通域分析提取图像最左侧/右侧的元素。"""
    img_arr = np.array(img)
    alpha = img_arr[:, :, 3]
    mask = alpha > 0
    labeled_mask, num_features = label(mask)
    if num_features < 2:
        return img_arr
    slices = find_objects(labeled_mask)
    components = []
    for i, slc in enumerate(slices):
        if slc is not None:
            components.append((i + 1, slc[1].start))
    components.sort(key=lambda item: item[1])
    target_label = components[0][0] if is_left else components[-1][0]
    result_arr = np.zeros_like(img_arr)
    is_bracket = (labeled_mask == target_label)
    result_arr[is_bracket] = img_arr[is_bracket]
    return result_arr


# ============================================================
# 1. matplotlib LaTeX 括号 — 双模式: \left/\right 或 \big 系列
# ============================================================
def draw_bracket_pair_mpl(bracket_type, height, thickness=2.0, dpi=None, fontset=None):
    """matplotlib LaTeX 渲染一对左右括号。

    随机选择两种模式:
      A. \left/\right 延展括号 — 高度自适应内容, 仅靠 fontset+DPI 变化字体.
      B. \bigl/\bigr 固定尺寸 — 可用 \mathbf{\bigl[} 施加数学样式变化.

    Args:
        bracket_type: 'square' 或 'paren'
        height: 括号高度 (像素)
        thickness: 最终图像宽度 ≈ thickness*4
        dpi: None 则随机选
        fontset: None 则随机选

    Returns:
        (left_img, right_img, tag)
    """
    if dpi is None:
        dpi = random.choice(_DPIS)
    if fontset is None:
        fontset = random.choice(_MATH_FONTSETS)

    plt.rcParams['mathtext.fontset'] = fontset

    left_b = '(' if bracket_type == 'paren' else '['
    right_b = ')' if bracket_type == 'paren' else ']'

    fontsize = 50
    step_height = fontsize * 1.2 * (dpi / 100)
    N = max(1, int(height / step_height))

    inner = r"\quad"
    for _ in range(N):
        inner = rf"\frac{{\quad}}{{{inner}}}"

    inner_formula = rf"\left{left_b} {inner} \right{right_b}"

    # 不再用 \mathrm/\mathbf/\mathit 等包裹 —— 这些样式会改变括号粗细/衬线形状，
    # 使字形偏离默认 \mathnormal 的外观，与文档印刷常用字体不一致。
    formula = '$' + inner_formula + '$'
    tag = f'mpl_{fontset}_dpi{dpi}'

    fig_h = max(2.0, (height * 1.5) / dpi)
    fig = plt.figure(figsize=(4, fig_h), dpi=dpi)
    fig.text(0.5, 0.5, formula, fontsize=fontsize, color='black',
             ha='center', va='center', usetex=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGBA")

    # 分别提取左右括号, 不使用镜像翻转
    left_arr = extract_single_bracket_via_cc(img, is_left=True)
    left_img = Image.fromarray(image_strip_rgba(left_arr))
    right_arr = extract_single_bracket_via_cc(img, is_left=False)
    right_img = Image.fromarray(image_strip_rgba(right_arr))

    lw, lh = left_img.size
    rw, rh = right_img.size
    # 等比缩放 —— 按高度缩放，宽度保持原始比例，避免强制宽度导致的拉伸形变
    if lh > 0 and rh > 0:
        l_new_w = max(1, int(round(lw * height / lh)))
        r_new_w = max(1, int(round(rw * height / rh)))
        left_img = left_img.resize((l_new_w, height), Image.Resampling.LANCZOS)
        right_img = right_img.resize((r_new_w, height), Image.Resampling.LANCZOS)

    return left_img, right_img, tag


# ============================================================
# 2. PIL 系统字体渲染 — 最大字体多样性来源
# ============================================================
def _try_load_font(size):
    """随机挑一个系统字体加载, 失败返回 PIL 默认字体。"""
    fonts = _get_system_fonts()
    for _ in range(10):
        path = random.choice(fonts)
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _render_single_bracket_char(char, height, font):
    """渲染单个括号字符到透明画布, 缩放到目标高度。返回 PIL RGBA 或 None。"""
    h = max(int(height), 8)
    render_size = max(h * 3, 40)
    tmp = Image.new('RGBA', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), char, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw < 1 or th < 1:
        return None
    pad = max(4, render_size // 10)
    canvas_w = tw + pad * 2
    canvas_h = th + pad * 2
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), char,
                              font=font, fill=(0, 0, 0, 255))
    cropped = Image.fromarray(image_strip_rgba(img))
    cw, ch = cropped.size
    if ch < 2:
        return None
    scale_factor = h / ch
    new_w = max(1, int(cw * scale_factor))
    return cropped.resize((new_w, h), Image.Resampling.LANCZOS)


def draw_bracket_pair_font(bracket_type, height, thickness=2.0):
    """用 PIL 系统字体分别渲染左右括号, 不使用镜像翻转。

    左右括号各自用真实字符 '[', ']' (或 '(', ')') 渲染,
    避免 FLIP_LEFT_RIGHT 在对称字形上产生两个相同括号的 bug。

    Args:
        bracket_type: 'square' 或 'paren'
        height: 目标高度 (像素)
        thickness: 最终图像宽度 ≈ thickness*4

    Returns:
        (left_img, right_img, tag) 或 None
    """
    h = max(int(height), 8)
    render_size = max(h * 3, 40)
    font = _try_load_font(render_size)

    if bracket_type == 'paren':
        left_char, right_char = '(', ')'
    else:
        left_char, right_char = '[', ']'

    left_img = _render_single_bracket_char(left_char, h, font)
    right_img = _render_single_bracket_char(right_char, h, font)
    if left_img is None or right_img is None:
        return None

    # 从字体路径提取短名称作为 tag
    try:
        font_name = os.path.splitext(os.path.basename(font.path or 'unknown'))[0][:16]
    except Exception:
        font_name = 'unknown'
    tag = f'font_{font_name}'
    return left_img, right_img, tag


# ============================================================
# 3. PIL 线艺绘制 — 非字体, 纯几何风格
# ============================================================
def _draw_left_square_line(h, t, serif_fraction=0.16):
    """PIL ImageDraw 画左方括号 [ 。竖线在左, 衬线向右伸出。"""
    h = max(int(h), 8)
    serif_w = max(int(h * serif_fraction), t * 2, 6)
    t = max(int(t), 2)
    pad = max(t * 2, 4)
    canvas_w = t + serif_w + pad * 2
    canvas_h = h + pad * 2
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ink = (0, 0, 0, 255)
    ht = t / 2.0
    vl = pad                     # 竖线左边缘
    vr = pad + t                 # 竖线右边缘
    ty = pad                     # 顶部 y
    by = pad + h                 # 底部 y

    pts = [
        (vl,          ty - ht),       # 左上
        (vr + serif_w, ty - ht),      # 上衬线右端上沿
        (vr + serif_w, ty + ht),      # 上衬线右端下沿
        (vr,          ty + ht),       # 竖线右侧, 上衬线下方
        (vr,          by - ht),       # 竖线右侧, 下衬线上方
        (vr + serif_w, by - ht),      # 下衬线右端上沿
        (vr + serif_w, by + ht),      # 下衬线右端下沿
        (vl,          by + ht),       # 左下
    ]
    draw.polygon(pts, fill=ink)
    return Image.fromarray(image_strip_rgba(img))


def _draw_right_square_line(h, t, serif_fraction=0.16):
    """PIL ImageDraw 画右方括号 ] 。竖线在右, 衬线向左伸出。"""
    h = max(int(h), 8)
    serif_w = max(int(h * serif_fraction), t * 2, 6)
    t = max(int(t), 2)
    pad = max(t * 2, 4)
    canvas_w = t + serif_w + pad * 2
    canvas_h = h + pad * 2
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ink = (0, 0, 0, 255)
    ht = t / 2.0
    vl = pad + serif_w            # 竖线左边缘 (衬线在左侧)
    vr = pad + serif_w + t        # 竖线右边缘
    ty = pad                      # 顶部 y
    by = pad + h                  # 底部 y

    pts = [
        (pad,        ty - ht),       # 上衬线左端上沿
        (vr,         ty - ht),       # 右上
        (vr,         ty + ht),       # 竖线右侧, 上衬线下方
        (vl,         ty + ht),       # 竖线左侧, 上衬线下方
        (vl,         by - ht),       # 竖线左侧, 下衬线上方
        (vr,         by - ht),       # 竖线右侧, 下衬线上方
        (vr,         by + ht),       # 右下
        (pad,        by + ht),       # 下衬线左端下沿
    ]
    draw.polygon(pts, fill=ink)
    return Image.fromarray(image_strip_rgba(img))


def _draw_left_paren_line(h, t):
    """PIL ImageDraw 画左圆括号 ( 。"""
    h = max(int(h), 10)
    t = max(int(t), 2)
    bw = max(int(h * 0.32), t * 3, 8)
    pad = t * 2 + 2
    canvas_w = bw + pad * 2
    canvas_h = h + pad * 2
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = [pad, pad, pad + bw * 2, pad + h]
    draw.arc(bbox, start=90, end=270, fill=(0, 0, 0, 255), width=t)
    return Image.fromarray(image_strip_rgba(img))


def _draw_right_paren_line(h, t):
    """PIL ImageDraw 画右圆括号 ) 。"""
    h = max(int(h), 10)
    t = max(int(t), 2)
    bw = max(int(h * 0.32), t * 3, 8)
    pad = t * 2 + 2
    canvas_w = bw + pad * 2
    canvas_h = h + pad * 2
    img = Image.new('RGBA', (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    bbox = [pad - bw, pad, pad + bw, pad + h]
    draw.arc(bbox, start=270, end=90, fill=(0, 0, 0, 255), width=t)
    return Image.fromarray(image_strip_rgba(img))


def draw_bracket_pair_line(bracket_type, height, thickness=2.0):
    """PIL 线艺绘制一对括号。左括号用多边形绘制, 右括号由左括号镜像得到,
    保证左右括号严格对称。

    bracket_type: 'square' | 'paren'
    """
    h = max(int(height), 8)
    t = max(thickness, 0.8)

    if bracket_type == 'paren':
        left = _draw_left_paren_line(h, t)
        right = ImageOps.mirror(left)
        tag = 'line_paren'
    else:  # square
        serif_frac = random.uniform(0.14, 0.20)  # 增大衬线比例, 避免与键混淆
        left = _draw_left_square_line(h, t, serif_fraction=serif_frac)
        right = ImageOps.mirror(left)
        tag = f'line_sq_s{serif_frac:.2f}'
    return left, right, tag


# ============================================================
# 统一入口
#   方括号 → 线艺几何绘制 (无形变, 形态统一)
#   圆括号 → matplotlib LaTeX 渲染 (弧线自然)
# ============================================================


def draw_bracket_pair(bracket_type, height, thickness=2.0):
    """绘制一对左右括号。

    方括号: 线艺路径 (纯几何, 粗细受 thickness 精确控制).
    圆括号: matplotlib LaTeX 渲染 (弧线自然).

    Args:
        bracket_type: 'square' 或 'paren'
        height: 括号高度 (像素)
        thickness: 线宽 (像素), 方括号直接用, 圆括号作参考

    Returns:
        (left_img, right_img, tag)
    """
    if bracket_type == 'square':
        return draw_bracket_pair_line('square', height, thickness)

    # 圆括号 — matplotlib LaTeX, 弧线最自然
    try:
        return draw_bracket_pair_mpl('paren', height, thickness)
    except Exception:
        return draw_bracket_pair_line('paren', height, thickness)


# ---- 保留旧版 draw_bracket 接口 (单侧, 供调试用) ----
def draw_bracket(bracket_type, height, thickness=2.0, serif_rel=0.15, curve_rel=0.22,
                  dpi=100, fontset=None):
    """绘制单个原生排版括号 (供调试用)。"""
    if fontset is None:
        fontset = random.choice(_MATH_FONTSETS)
    left_b = '(' if 'paren' in bracket_type else '['
    right_b = ')' if 'paren' in bracket_type else ']'
    is_left = 'left' in bracket_type

    plt.rcParams['mathtext.fontset'] = fontset
    fontsize = 50
    step_height = fontsize * 1.2 * (dpi / 100)
    N = max(1, int(height / step_height))
    inner = r"\quad"
    for _ in range(N):
        inner = rf"\frac{{\quad}}{{{inner}}}"
    formula = rf"$\left{left_b} {inner} \right{right_b}$"

    fig_h = max(2.0, (height * 1.5) / dpi)
    fig = plt.figure(figsize=(4, fig_h), dpi=dpi)
    fig.text(0.5, 0.5, formula, fontsize=fontsize, color='black',
             ha='center', va='center', usetex=False)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', transparent=True, bbox_inches='tight', pad_inches=0.0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    bracket_arr = extract_single_bracket_via_cc(img, is_left)
    cropped_img = Image.fromarray(image_strip_rgba(bracket_arr))
    orig_h = cropped_img.size[1]
    if orig_h > 0:
        target_w = max(1, int(thickness * 4))
        cropped_img = cropped_img.resize((target_w, height), Image.Resampling.LANCZOS)
    return cropped_img


# ============================================================
# 下标渲染
# ============================================================
def draw_bracket_subscript(content, font_size=14):
    """绘制括号下标文本 (PIL 系统字体, 带多重字体 fallback 防豆腐块)。"""
    if not content:
        return None
    font_size = max(int(font_size), 8)
    for _attempt in range(8):
        font = _try_load_font(font_size)
        result = _render_subscript_internal(content, font_size, font)
        if result is not None:
            rw, rh = result.size
            min_expected_w = max(3, font_size * 0.25 * len(content))
            if rw >= min_expected_w:
                return result
    return None


def _render_subscript_internal(content, font_size, font):
    tmp = Image.new('RGBA', (1, 1))
    bbox = ImageDraw.Draw(tmp).textbbox((0, 0), content, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    if tw < 1 or th < 1:
        return None
    pad = max(2, int(font_size * 0.2))
    w, h = tw + pad * 2, th + pad * 2
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((pad - bbox[0], pad - bbox[1]), content, font=font,
                               fill=(0, 0, 0, 255))
    return Image.fromarray(image_strip_rgba(img))


def generate_subscript_content():
    """随机生成下标内容。"""
    r = random.random()
    if r < 0.10:
        return ''
    elif r < 0.30:
        if random.random() < 0.4:
            lo, hi = sorted([random.randint(1, 6) for _ in range(2)])
            return f'{lo}-{hi}' if lo != hi else str(lo)
        else:
            return str(random.randint(1, 20))
    else:
        return random.choice(string.ascii_lowercase)


if __name__ == '__main__':
    print("Bracket generator test — rendering samples...")
    for bt in ['square', 'paren']:
        for _ in range(4):
            l, r = draw_bracket_pair(bt, height=80, thickness=2.0)
            print(f'  {bt}: left={l.size}, right={r.size}')
    for _ in range(3):
        c = generate_subscript_content()
        s = draw_bracket_subscript(c, font_size=14)
        if s:
            print(f'  subscript "{c}": {s.size}')
    print('Done.')
