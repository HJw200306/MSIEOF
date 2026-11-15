"""实线箭头（solid arrow）分割模型的可视化脚本。

针对实线箭头三分类模型
``model/legacy_arrow_models/misp_weights_no_freeze.pth``（Unet / vgg，3 类）：
    0 = background, 1 = arrow（实线箭头）, 2 = bond（化学键）

与虚线箭头不同，实线箭头没有 3 类 GT 掩码，因此本脚本只做「预测可视化」，
不做 TP/FP/FN 与 IoU/Dice 统计。每张图输出 4 面板：
    original | class map(红=arrow 蓝=bond) | overlay | erased(箭头涂白)

erase 逻辑对应历史实线模型；当前生产实现位于 utils.pipeline.arrow_sr。
涂白 [255,255,255]；--refine 额外做 HSV 精细擦除。

Usage:
    python arrow_solid_val.py                         # 跑 val/images 全部
    python arrow_solid_val.py --limit 8               # 只跑前 8 张
    python arrow_solid_val.py --images a.png b.png    # 指定图片
    python arrow_solid_val.py --refine                # erase 额外做 HSV 精细擦除
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from split_tools.unet import Unet  # noqa: E402
from utils.base.viz_utils import COLOR_ARROW, COLOR_BOND  # noqa: E402

# ── Config ────────────────────────────────────────────────────
MODEL_PATH = _PROJECT_ROOT / "model" / "legacy_arrow_models" / "misp_weights_no_freeze.pth"
IMG_DIR = _PROJECT_ROOT / "arrow_eraser" / "dataset" / "val" / "images"
OUTPUT_DIR = _PROJECT_ROOT / "arrow_eraser" / "val_results" / "solid"
PRED_DIR = OUTPUT_DIR / "preds"
VIS_DIR = OUTPUT_DIR / "vis"

CLASS_NAMES = {0: "background", 1: "arrow", 2: "bond"}
# RGB 配色：arrow 红 / bond 蓝，与 utils.base.viz_utils 统一。
CLASS_RGB = {0: (0, 0, 0), 1: COLOR_ARROW, 2: COLOR_BOND}


# ── IO（规避 Windows 非 ASCII 路径） ──────────────────────────
def imread_rgb(path: Path) -> np.ndarray:
    """cv2.imdecode(np.fromfile) 读图，规避非 ASCII 路径，返回 RGB。"""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imwrite_bgr(path: Path, img_bgr: np.ndarray) -> None:
    """cv2.imencode + tofile 写图，规避非 ASCII 路径。"""
    ok, buf = cv2.imencode(Path(path).suffix, img_bgr)
    if not ok:
        raise IOError(f"encode failed: {path}")
    buf.tofile(str(path))


# ── 推理 ─────────────────────────────────────────────────────
def classify(model: Unet, img_rgb: np.ndarray) -> np.ndarray:
    """跑 3 类模型，返回 class map (H, W) uint8 ∈ {0,1,2}。"""
    return np.array(model.get_miou_png(Image.fromarray(img_rgb)))


# ── 可视化 ───────────────────────────────────────────────────
def class_map_rgb(cls: np.ndarray) -> np.ndarray:
    out = np.zeros((*cls.shape, 3), np.uint8)
    for c, rgb in CLASS_RGB.items():
        out[cls == c] = rgb
    return out


def overlay_rgb(img_rgb: np.ndarray, cls: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """在原图上半透明叠加 arrow(红) + bond(蓝)。"""
    out = img_rgb.astype(np.float32).copy()
    for c in (1, 2):
        m = cls == c
        if m.any():
            color = np.array(CLASS_RGB[c], dtype=np.float32)
            out[m] = (1 - alpha) * out[m] + alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def erase_solid_arrows(img_rgb: np.ndarray, cls: np.ndarray, refine: bool = False) -> np.ndarray:
    """把类别 1(arrow) 像素涂白；refine=True 时额外做 HSV 精细擦除。"""
    out = img_rgb.copy()
    mask = cls == 1
    if refine:
        mask = _refine_arrow_mask(img_rgb, mask)
    out[mask] = [255, 255, 255]
    return out


def _refine_arrow_mask(img_rgb: np.ndarray, mask: np.ndarray,
                       saturation_threshold: int = 30) -> np.ndarray:
    """HSV 精细擦除（参考 arrow_earser_val.erase_arrows）：膨胀 halo 内彩色像素一并擦掉。"""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    halo = cv2.dilate(mask_u8, kernel, iterations=3) > 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    is_colorful = hsv[..., 1] > saturation_threshold
    return mask | (halo & is_colorful)


def build_composite(img_rgb: np.ndarray, cls: np.ndarray) -> np.ndarray:
    """4 面板并排 + 顶部标题条 + 底部统计条，返回 BGR。"""
    p_orig = img_rgb
    p_map = class_map_rgb(cls)
    p_over = overlay_rgb(img_rgb, cls)
    p_eras = erase_solid_arrows(img_rgb, cls)

    h = p_orig.shape[0]
    div = np.full((h, 3, 3), 80, np.uint8)
    row = np.hstack([p_orig, div, p_map, div, p_over, div, p_eras])

    labels = ["original", "class map  (red=arrow, blue=bond)", "overlay", "erased (arrow -> white)"]
    seg = row.shape[1] // 4
    title = np.full((24, row.shape[1], 3), 0, np.uint8)
    for i, lab in enumerate(labels):
        cv2.putText(title, lab, (i * seg + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    stats = (f"arrow={int((cls == 1).sum())}px   "
             f"bond={int((cls == 2).sum())}px   "
             f"background={int((cls == 0).sum())}px")
    bar = np.full((30, row.shape[1], 3), 0, np.uint8)
    cv2.putText(bar, stats, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA)

    composite_rgb = np.vstack([title, row, bar])
    return cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR)


# ── Main ─────────────────────────────────────────────────────
def list_images(img_dir: Path) -> list[Path]:
    valid = (".png", ".jpg", ".jpeg")
    return sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in valid and not p.name.endswith("_struct.png"))


def main():
    ap = argparse.ArgumentParser(description="实线箭头(3类)模型预测可视化 + 涂白擦除")
    ap.add_argument("--input", default=str(IMG_DIR), help="输入图片目录，默认 arrow_eraser/dataset/val/images")
    ap.add_argument("--output", default=str(OUTPUT_DIR), help="输出目录，默认 arrow_eraser/val_results/solid")
    ap.add_argument("--images", nargs="*", default=None, help="指定图片文件名列表（可多个）")
    ap.add_argument("--limit", type=int, default=0, help="最多处理 N 张（0=全部）")
    ap.add_argument("--refine", action="store_true", help="erase 额外做 HSV 精细擦除")
    args = ap.parse_args()

    out_dir = Path(args.output)
    pred_dir = out_dir / "preds"
    vis_dir = out_dir / "vis"
    for d in (out_dir, pred_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)

    img_dir = Path(args.input)
    if args.images:
        paths = [img_dir / n for n in args.images]
    else:
        paths = list_images(img_dir)
        if args.limit > 0:
            paths = paths[:args.limit]

    if not paths:
        print(f"⚠️ 没有找到图片：{img_dir}")
        return

    cuda = torch.cuda.is_available()
    print(f"Loading 3-class model from {MODEL_PATH} ... (cuda={cuda})")
    model = Unet(str(MODEL_PATH), num_classes=3, input_shape=[512, 512], cuda=cuda)
    print(f"Model loaded. {len(paths)} images to process.\n")

    n_ok = n_fail = 0
    for path in tqdm(paths, desc="solid arrow viz", unit="img"):
        if not path.exists():
            tqdm.write(f"  !! missing: {path.name}")
            n_fail += 1
            continue
        stem = path.stem
        try:
            img_rgb = imread_rgb(path)
            cls = classify(model, img_rgb)

            # 预测 class map（彩色，直观）
            imwrite_bgr(pred_dir / f"{stem}_pred.png",
                        cv2.cvtColor(class_map_rgb(cls), cv2.COLOR_RGB2BGR))
            # 4 面板可视化
            imwrite_bgr(vis_dir / f"{stem}_vis.png", build_composite(img_rgb, cls))
            n_ok += 1
        except Exception as e:
            tqdm.write(f"  ❌ 处理失败 [{path.name}]: {e}")
            n_fail += 1

    print("\n" + "=" * 50)
    print(f"完成：成功 {n_ok} 张" + (f"，失败 {n_fail} 张" if n_fail else ""))
    print(f"预测 class map → {pred_dir}")
    print(f"可视化面板     → {vis_dir}")


if __name__ == "__main__":
    main()
