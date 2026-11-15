"""图像辅助工具 (从 R_Groups_Segment/draw/line_draw.py 提取)。"""
import numpy as np
import matplotlib.pyplot as plt


def plt_fig_to_img() -> np.ndarray:
    """把当前 matplotlib figure 渲染成 RGBA np.ndarray, 同时关掉 figure。"""
    fig = plt.gcf()
    fig.patch.set_alpha(0.0)
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
    fig.canvas.draw()
    img_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_rgba = img_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close()
    return img_rgba


def image_strip_rgba(img_rgba: np.ndarray, bg_color=(255, 255, 255, 0)) -> np.ndarray:
    """裁掉 RGBA 图四周的纯背景像素 (默认: 白色 + alpha=0)。"""
    if img_rgba.ndim != 3 or img_rgba.shape[2] != 4:
        raise ValueError("输入数组必须是形状为 (height, width, 4) 的 RGBA 图像。")

    bg_arr = np.array(bg_color, dtype=img_rgba.dtype)
    mask = np.any(img_rgba != bg_arr, axis=-1)
    if not mask.any():
        raise ValueError("图像完全是背景颜色。")

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(cols[0]), int(cols[-1])
    return img_rgba[top:bottom + 1, left:right + 1]
