"""颜色感知的前景（墨迹）提取。

动机
----
旧口径（build_gt 的固定 220 灰度阈值、extract._page_binary 的 Otsu 灰度二值化）
都先把整页 BGR/RGB 转成灰度再阈值。对纯黑分子（专利页）没问题，但对**期刊页面
里带颜色的分子结构**（蓝/红/绿键、彩色取代基），灰度转换会把高饱和的亮色（黄、
青等）压到接近白纸的灰度值上，导致这些有色墨迹被判成背景而漏掉。实测在 EJMC 期刊
页上：固定 220 漏 ~12%，Otsu 漏 ~42% 的有色墨迹。

思路
----
不在灰度上判前景，而在 HSV 上判**背景（纸张）**：纸张 = 明亮(V 高) 且 低饱和(S 低)。
前景墨迹 = 背景取反。这样：
  - 黑 / 深灰：V 低 → 前景；
  - 任意饱和度的有色墨迹（含亮而高饱和的黄/青）：S 高 → 前景；
  - 仅明亮的无彩色像素（白纸、灰阶抗锯齿边缘）→ 背景。
对无彩色的专利页，S≈0，判据退化为 V<V_BG_THR，与旧的 gray≤(V_BG_THR-1) 一致
（已验证在 CN 专利页上与固定 220 的差异为 0 像素）。
"""

from __future__ import annotations

import cv2
import numpy as np


def color_aware_ink_mask(
    img: np.ndarray,
    v_bg_thr: int = 221,
    s_bg_thr: int = 40,
    close_r: int = 0,
) -> np.ndarray:
    """整页 → 墨迹二值图（前景=255, uint8）。

    参数
    ----
    img : (H, W, 3) 彩色图，BGR 或 RGB 均可（V/S 与通道顺序无关）。
    v_bg_thr : HSV V 明度阈值 [0,255]，单像素·整页。
        背景需满足 V >= v_bg_thr。调大 → 连接近白的淡像素也算前景，GT 虚胖、
        recall 分母膨胀、数字被压低；调小 → 淡骨架线丢失。与旧灰度阈值 220 对齐
        （V>=221 ⇔ gray>220 判背景）。
    s_bg_thr : HSV S 饱和度阈值 [0,255]，单像素·整页。
        背景需满足 S <= s_bg_thr。调大 → 连淡淡带色的像素也当纸张，淡色墨迹丢失；
        调小 → 一点点色边也被划为前景。
    close_r : 闭运算椭圆核半径（核大小 = 2·close_r+1），单位=像素，几何尺度=墨迹间隙。
        >0 时把离主体几像素的原子标签/虚线键碎块粘成同一连通域（对齐 extract.
        _page_binary 的 close_r=3，供洪水填充 host 用）；GT 侧用 0（不额外粘连）。

    返回
    ----
    (H, W) uint8，墨迹=255、背景=0。
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"expected (H,W,3) color image, got shape {img.shape}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 通道顺序不影响 V/S
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    bg = (v >= v_bg_thr) & (s <= s_bg_thr)
    binary = np.where(bg, 0, 255).astype(np.uint8)

    if close_r > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_r + 1, 2 * close_r + 1)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    return binary
