"""统一可视化工具:配色、目录、叠加 helper。

供 pipeline 各步复用，调用顺序：
  extract.py → fragment_cleanup.py → post_segment.py → arrow_sr.py → img2smiles.py

各阶段写到独立的 vis/ 子目录：
  vis/extract/  — extract.py: polygon + bbox + 洪水保留区域叠加
  vis/post_seg/ — post_segment.py: UNet++ 前景 + 连通域分组叠加
  vis/fragment_cleanup/ — fragment_cleanup.py: 邻近/截断片段清理叠加
  vis/sr/       — arrow_sr.py: HAT 超分前后对比
  vis/arrow/    — arrow_sr.py: 去箭头叠加（arrow 红 / bond 蓝）
  vis/molscribe/ — img2smiles.py: MolScribe 预测的原子点 + 键叠加

所有颜色使用 RGB 元组(0-255)。
"""

from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from utils.base.paths import PROJECT_ROOT

# 统一配色 (RGB, 0-255)
COLOR_ARROW: Tuple[int, int, int] = (255, 0, 0)      # 红 — 实线箭头 (split_tools/Unet 输出)
COLOR_ARROW_DASHED: Tuple[int, int, int] = (255, 140, 0)  # 橙 — 虚线箭头 (arrow_eraser UNet++ 输出)
COLOR_BOND: Tuple[int, int, int] = (0, 0, 255)       # 蓝 — 化学键 (Unet 原绿色重映射)
COLOR_FG: Tuple[int, int, int] = (0, 255, 0)         # 绿 — 前景/分子结构
COLOR_BBOX: Tuple[int, int, int] = (255, 255, 0)     # 黄 — 原始 bbox / polygon bbox
COLOR_MERGED: Tuple[int, int, int] = (255, 165, 0)   # 橙 — 合并后的分组框

# MolScribe 预测可视化（utils.pipeline.img2smiles）
COLOR_MS_ATOM: Tuple[int, int, int] = (220, 20, 60)         # 深红 — 原子位置（碳以外都画文字标签）
COLOR_MS_BOND_SINGLE: Tuple[int, int, int] = (30, 144, 255) # 道奇蓝 — 单键
COLOR_MS_BOND_DOUBLE: Tuple[int, int, int] = (0, 200, 0)    # 绿 — 双键
COLOR_MS_BOND_TRIPLE: Tuple[int, int, int] = (148, 0, 211)  # 紫 — 三键
COLOR_MS_BOND_AROMATIC: Tuple[int, int, int] = (255, 140, 0)# 橙 — 芳香键
COLOR_MS_BOND_WEDGE_SOLID: Tuple[int, int, int] = (0, 0, 0) # 黑 — 实楔键
COLOR_MS_BOND_WEDGE_DASH: Tuple[int, int, int] = (128, 128, 128)  # 灰 — 虚楔键

VIZ_ROOT = PROJECT_ROOT / "vis"
VIZ_EXTRACT = VIZ_ROOT / "extract"
VIZ_SR = VIZ_ROOT / "sr"
VIZ_ARROW = VIZ_ROOT / "arrow"
VIZ_POST = VIZ_ROOT / "post_seg"
VIZ_DENOISE = VIZ_ROOT / "fragment_cleanup"
VIZ_MOLSCRIBE = VIZ_ROOT / "molscribe"

# 各阶段的可视化子目录名。redirect_viz_root 按这张表重建全部指针。
_VIZ_STAGES = ("extract", "fragment_cleanup", "post_seg", "sr", "arrow", "molscribe")


def redirect_viz_root(root: Union[str, Path], create: bool = True) -> Path:
    """把所有 VIZ_* 指针指到 root 下的各阶段子目录，返回 root。

    用途：服务化时每个任务需要独立的可视化输出目录，否则并发任务会往同一个
    PROJECT_ROOT/vis/<stage>/ 里写，同名分子互相覆盖。

    生效原理：extract / fragment_cleanup / post_segment / arrow_sr / img2smiles 都在函数
    内部才 `from utils.base.viz_utils import VIZ_*`，每次调用重读模块属性，所以这里改
    完立刻生效——这与 ceshi/eval_full_pipeline.py 的既有做法一致，本函数只是把
    它扩展到全部六个可视化阶段。

    root 传 None 之外的路径即启用重定向；想恢复默认用 redirect_viz_root(PROJECT_ROOT / "vis")。
    """
    global VIZ_ROOT, VIZ_EXTRACT, VIZ_SR, VIZ_ARROW, VIZ_POST, VIZ_DENOISE, VIZ_MOLSCRIBE
    root = Path(root)
    VIZ_ROOT = root
    VIZ_EXTRACT = root / "extract"
    VIZ_SR = root / "sr"
    VIZ_ARROW = root / "arrow"
    VIZ_POST = root / "post_seg"
    VIZ_DENOISE = root / "fragment_cleanup"
    VIZ_MOLSCRIBE = root / "molscribe"
    if create:
        for name in _VIZ_STAGES:
            (root / name).mkdir(parents=True, exist_ok=True)
    return root


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _to_unit(color_rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    return (color_rgb[0] / 255.0, color_rgb[1] / 255.0, color_rgb[2] / 255.0)


def overlay_mask_on_image(
    img_rgb: np.ndarray,
    mask_bool: np.ndarray,
    color_rgb: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """把 color 以 alpha 半透明压到 img 的 mask 区域上,返回 uint8 ndarray。"""
    out = img_rgb.astype(np.float32).copy()
    if mask_bool.any():
        color = np.array(color_rgb, dtype=np.float32)
        out[mask_bool] = (1 - alpha) * out[mask_bool] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def save_overlay_figure(
    img_rgb: np.ndarray,
    overlays: Sequence[Tuple[np.ndarray, Tuple[int, int, int], float]],
    title: str,
    legend_items: Sequence[Tuple[str, Tuple[int, int, int]]],
    out_path: Path,
    extras: Optional[Callable] = None,
) -> None:
    """通用 overlay 保存入口。viz 失败不抛,只打印警告。

    Args:
        img_rgb: 底图 (H, W, 3) uint8 RGB。
        overlays: 依次叠加的 [(mask_bool, color_rgb, alpha), ...]。
        title: 图标题。
        legend_items: [(label, color_rgb), ...] 生成右上角 legend。
        out_path: 保存路径,父目录自动创建。
        extras: 可选回调 fn(ax) -> None,在叠加后画 bbox/编号/polygon 等。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch

        base = img_rgb
        for mask_bool, color, alpha in overlays:
            base = overlay_mask_on_image(base, mask_bool, color, alpha)

        h, w = base.shape[:2]
        fig_w = max(6.0, min(16.0, w / 150.0))
        fig_h = fig_w * h / max(w, 1)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(base)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()

        if extras is not None:
            extras(ax)

        if legend_items:
            handles = [Patch(facecolor=_to_unit(c), edgecolor="black", label=lbl)
                       for lbl, c in legend_items]
            ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.85)

        ensure_dir(out_path.parent)
        fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
        plt.close(fig)
    except Exception as e:
        try:
            from tqdm import tqdm
            tqdm.write(f"  ⚠️ 可视化保存失败 [{out_path.name}]: {e}")
        except Exception:
            print(f"  ⚠️ 可视化保存失败 [{out_path.name}]: {e}")


def save_side_by_side(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    left_title: str,
    right_title: str,
    suptitle: str,
    out_path: Path,
) -> None:
    """两图并排保存,用于 SR 前后对比。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        h = max(left_rgb.shape[0], right_rgb.shape[0])
        w = left_rgb.shape[1] + right_rgb.shape[1]
        fig_w = max(8.0, min(20.0, w / 200.0))
        fig_h = fig_w * h / max(w, 1)
        fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
        axes[0].imshow(left_rgb)
        axes[0].set_title(left_title, fontsize=10)
        axes[0].set_axis_off()
        axes[1].imshow(right_rgb)
        axes[1].set_title(right_title, fontsize=10)
        axes[1].set_axis_off()
        fig.suptitle(suptitle, fontsize=11)

        ensure_dir(out_path.parent)
        fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
        plt.close(fig)
    except Exception as e:
        try:
            from tqdm import tqdm
            tqdm.write(f"  ⚠️ 可视化保存失败 [{out_path.name}]: {e}")
        except Exception:
            print(f"  ⚠️ 可视化保存失败 [{out_path.name}]: {e}")
