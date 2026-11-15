import math
import random
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.base.image_loader import ImageLoader
from utils.base.image_utils import plt_fig_to_img, image_strip_rgba

img_loader = ImageLoader()
log = logging.getLogger(__name__)


def draw_connect_line(thickness: float, line_type: str = 'circle_wave', connect_point_type: str = 'center'):
    """
    绘制连接线，可选四种类型。 当连接点为虚线时，连接点可选中心点和端点，其余点只能选择中心点
    Args:
        thickness (float): 壁厚
        line_type (str): 线段类型
        connect_point_type (str): 连接点类型

    Returns:

    """
    line_types = ('circle_wave', 'wave', 'zigzag', 'dashed')
    if line_type not in line_types:
        line_type = 'circle_wave'
        connect_point_type = 'center'

    thickness = thickness * random.uniform(0.75, 1.1)

    # 限制最大最小绘制循环
    if line_type == 'dashed':
        img = draw_dashed_line(thickness=thickness)
        connect_point = [img.width // 2, 0] if connect_point_type == 'center' else [0, 0]
    else:
        if connect_point_type != 'center':
            log.warning('connect_point_type error, fixed!')

        if line_type == 'circle_wave':
            img = draw_circle_wave(thickness=thickness)
        elif line_type == 'zigzag':
            img = draw_cycle_line(thickness=thickness, line_type='zigzag')
        else:
            img = draw_cycle_line(thickness=thickness, line_type='wave')

        connect_point = find_vertical_black_center(img_loader.load_cv_img(img))
        if not connect_point:
            connect_point = [img.width // 2, img.height // 2]

    return img, connect_point


def find_vertical_black_center(image):
    # 确保图像是numpy数组
    cv_img = img_loader.load_cv_img(image)

    # 转换为灰度图
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    center_x = width // 2

    start_y = None  # 黑色区域起始y坐标
    end_y = None  # 黑色区域结束y坐标

    # 遍历垂直中心线的所有像素
    for y in range(height):
        pixel_value = gray[y, center_x]

        # 遇到黑色像素时记录起始位置
        if pixel_value == 0:
            if start_y is None:
                start_y = y
            # 继续跟踪当前黑色区域的结束位置
            end_y = y  # 动态更新结束位置
        # 非黑色像素且已找到起始位置时终止搜索
        elif start_y is not None:
            break  # 遇到非黑色像素且区域已开始则终止

    # 处理全黑情况（整个中心线都是黑色）
    if start_y is not None and end_y is None:
        end_y = height - 1

    # 计算中心点
    if start_y is not None and end_y is not None:
        center_y = (start_y + end_y) // 2

        return [center_x, center_y]
    else:
        # print("未找到竖直方向的黑色区域")
        return None


def draw_dashed_line(num_cycles: int = -1, thickness: float = 3, aspect_ratio=3):
    """
    绘制一条虚线并返回其图像。

    参数:
        num_cycles (int): 虚线的周期数，要求为3-6。
        thickness (int): 虚线的粗细，默认为 10。
        aspect_ratio (int): 长度比例，默认为3

    返回:
        PIL.Image: 包含虚线的图像对象。
    """
    # 动态调整 x 的范围以适应 num_cycles
    if num_cycles < 3 or num_cycles > 6:
        num_cycles = round(random.uniform(3, 6))

    x = np.linspace(0, 1, 1000)  # 增加点数以提高分辨率
    y = np.zeros_like(x)  # y 坐标固定为 0
    dash_length = aspect_ratio
    dashes = (dash_length, dash_length)  # 虚线样式

    # 根据实验计算出来的公式
    line_width = 250 / (2 * num_cycles - 1) / dash_length

    # 创建绘图
    plt.figure(figsize=(5, 1), dpi=72)  # 设置图像大小
    plt.plot(x, y, linestyle='--', dashes=dashes, linewidth=line_width, color='black')
    plt.axis("off")  # 关闭坐标轴

    img_rgba = plt_fig_to_img()
    img_rgba = image_strip_rgba(img_rgba)

    # 根据需要的壁厚压缩图像
    compression_factor = line_width / thickness
    height, width = img_rgba.shape[:2]
    new_height = int(height / compression_factor)
    new_width = int(width / compression_factor)
    # 分离通道
    rgb_channels = img_rgba[:, :, :3].astype(np.float32)
    alpha_channel = img_rgba[:, :, 3].astype(np.float32)

    # 预乘Alpha
    rgb_premultiplied = rgb_channels * (alpha_channel[:, :, np.newaxis] / 255.0)

    # 边缘扩展
    rgb_padded = cv2.copyMakeBorder(rgb_premultiplied, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    alpha_padded = cv2.copyMakeBorder(alpha_channel, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # 缩放
    resized_rgb = cv2.resize(rgb_padded, (new_width + 2, new_height + 2), interpolation=cv2.INTER_AREA)[1:-1, 1:-1]
    resized_alpha = cv2.resize(alpha_padded, (new_width + 2, new_height + 2), interpolation=cv2.INTER_LINEAR)[1:-1,
                    1:-1]

    # 解除预乘并处理除零
    resized_rgb = (resized_rgb / (resized_alpha[:, :, np.newaxis] / 255.0 + 1e-6)).clip(0, 255).astype(np.uint8)
    resized_alpha = resized_alpha.clip(0, 255).astype(np.uint8)
    # 重新组合 RGBA 图像
    resized_img = np.dstack((resized_rgb, resized_alpha))
    return Image.fromarray(resized_img)


def draw_circle_wave(num_cycles: float = -1, thickness: float = 10):
    # 恢复你原本的设计下限和上限
    if num_cycles < 4 or num_cycles > 11:
        num_cycles = random.uniform(4, 11)

    # 核心公式：让圆波的半径恒定为粗细的 1.5 倍，从根本上防止糊成一坨
    radius_px = 1.5 * thickness
    width_px = num_cycles * 2 * radius_px
    height_px = (radius_px * 2) + thickness * 4

    # 转换为 matplotlib 的 figsize (72 dpi)
    fig_width = max(0.5, width_px / 72.0)
    fig_height = max(0.2, height_px / 72.0)

    plt.figure(figsize=(fig_width, fig_height), dpi=72)

    x_all, y_all = [], []
    for i in range(int(num_cycles * 2)):
        # 调整 theta 走向保证连贯性，消除线段拼接处的断层
        if i % 2 == 0:
            theta = np.linspace(np.pi, 0, 100)
        else:
            theta = np.linspace(np.pi, 2 * np.pi, 100)

        x_center = radius_px + i * (2 * radius_px)
        x_circle = x_center + radius_px * np.cos(theta)
        y_circle = radius_px * np.sin(theta)

        x_all.extend(x_circle)
        y_all.extend(y_circle)

    # 加入圆角拼接参数 (solid_joinstyle/solid_capstyle)，让粗线条转折处更自然
    plt.plot(x_all, y_all, lw=thickness, color='black', solid_joinstyle='round', solid_capstyle='round')

    # 严格锁定坐标轴，防止画布自动缩放导致变形
    plt.xlim(0, width_px)
    plt.ylim(-height_px / 2, height_px / 2)
    plt.axis('off')

    img_rgba = plt_fig_to_img()
    return Image.fromarray(image_strip_rgba(img_rgba))


def draw_cycle_line(num_cycles: float = -1, thickness: float = 10, line_type='random'):
    if num_cycles < 4 or num_cycles > 10:
        num_cycles = random.uniform(4, 10)

    wavelength = thickness / 4
    amplitude = thickness / 20 * random.uniform(1, 2)

    line_types = ('wave', 'zigzag')
    if line_type not in line_types:
        line_type = random.choice(line_types)

    """通过周期数量控制线段长度"""
    y = 0
    total_length = num_cycles * wavelength  # 自动计算总长度
    t = np.linspace(0, total_length, 1000)
    x = t
    if line_type == 'wave':
        y = np.sin(2 * np.pi * t / wavelength) * amplitude
    elif line_type == 'zigzag':
        y = (4 * np.abs((t / wavelength) - np.floor(t / wavelength + 0.5)) - 0.5) * amplitude

    plt.figure(figsize=(total_length, 1), dpi=72)
    plt.gca().set_aspect('equal')

    plt.plot(x, y, lw=thickness, color='black')
    # 设置绘制区间（示例范围，你可以根据需要调整）
    # plt.xlim(0, 10)  # x轴范围
    plt.ylim(-2, 2)  # y轴范围
    plt.axis("off")
    img_rgba = plt_fig_to_img()
    return Image.fromarray(image_strip_rgba(img_rgba))


