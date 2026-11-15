"""把 find_lowres_arrow 命中（低清/箭头）的分子裁剪图生成 2×2 可视化网格，供人工复核判定规则。

对每个分子裁剪图：
  - 左上：原图
  - 右上：箭头类 mask（红 = 箭头，白 = 其余）
  - 左下：化学键类 mask（绿 = 键，白 = 其余）
  - 右下：叠加图（红 = 箭头，绿 = 键，其余为原图）
输出到 lr_arrow/visualize/arrow_hits/ 与 lowres_hits/，文件名编码指标：
  <name>__arrow<箭头像素>__bond<键像素>__prob<高清概率>.png
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.base.paths import PROJECT_ROOT

from utils.pipeline.arrow_sr import ArrowSRProcessor  # noqa: E402
from utils.pipeline.resolution_classifier import ResolutionClassifier  # noqa: E402

CROPS_DIR = PROJECT_ROOT / "lr_arrow" / "_crops_tmp"
OUT_DIR = PROJECT_ROOT / "lr_arrow" / "visualize"
CLASSIFIER = PROJECT_ROOT / "model" / "classifier_fixed.pkl"
ARROW_THRESHOLD = 100
LOW_THRESHOLD = 0.10


def build_grid(orig_rgb, red_mask, green_mask):
    white = np.full_like(orig_rgb, 255)
    red_only = white.copy()
    red_only[red_mask] = (255, 0, 0)
    green_only = white.copy()
    green_only[green_mask] = (0, 200, 0)
    overlay = orig_rgb.copy()
    overlay[red_mask] = (255, 0, 0)
    overlay[green_mask] = (0, 200, 0)
    top = np.hstack([orig_rgb, red_only])
    bottom = np.hstack([green_only, overlay])
    return np.vstack([top, bottom])


def main():
    crops = sorted(p for p in CROPS_DIR.glob("*.png") if not p.name.startswith("_"))
    if not crops:
        print(f"⚠️ 未找到裁剪图: {CROPS_DIR}")
        return

    arrow_model = ArrowSRProcessor(device="cuda", load_sr_model=False)
    classifier = ResolutionClassifier(str(CLASSIFIER), low_threshold=LOW_THRESHOLD)

    arrow_dir = OUT_DIR / "arrow_hits"
    lowres_dir = OUT_DIR / "lowres_hits"
    arrow_dir.mkdir(parents=True, exist_ok=True)
    lowres_dir.mkdir(parents=True, exist_ok=True)

    n_arrow = n_low = 0
    for p in crops:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # 箭头/键 mask
        masks = arrow_model.predict_masks(Image.fromarray(rgb))
        red = masks["arrow"]
        green = masks["bond"]
        arrow_px, bond_px = int(red.sum()), int(green.sum())

        # 低清概率
        res = classifier.predict(str(p))
        prob = res["prob"] if res.get("features") is not None else -1.0

        grid = build_grid(rgb, red, green)
        tag = f"{p.stem}__arrow{arrow_px}__bond{bond_px}__prob{prob:.3f}.png"

        if arrow_px > ARROW_THRESHOLD:
            Image.fromarray(grid).save(arrow_dir / tag)
            n_arrow += 1
        if prob >= 0 and prob < LOW_THRESHOLD:
            Image.fromarray(grid).save(lowres_dir / tag)
            n_low += 1

    print(f"✅ 可视化完成，共 {len(crops)} 个分子")
    print(f"   箭头命中: {n_arrow} → {arrow_dir}")
    print(f"   低清命中: {n_low} → {lowres_dir}")
    print("   文件名含义: __arrow<箭头像素>__bond<键像素>__prob<高清概率>")


if __name__ == "__main__":
    main()
