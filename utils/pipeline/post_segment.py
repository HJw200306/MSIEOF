"""后分类模块（UNet++ 用作"是否是 2D 分子"分类器）。

背景
----
fragment_cleanup.py 对 extract.py 产生的候选图进行连通域后处理，
输出 crop_dataset_denoised/ 中的清理后候选分子图片。
但 YOLO 会把一些"形似分子"的东西（三维分子球棍模型、表格、装置图等）
也判成 mol，污染下游 OCSR。在 YOLO 之外加一个二级过滤器，比再训
YOLO 更快、风险更低。

设计
----
重用已训好的 UNet++(resnet101, classes=2)。该模型本就是在"2D 分子骨架
像素"上做前/背景分割，对三维结构 / 表格 / 装置图基本判不出前景。
因此不再把它当像素清理器使用，而是改用其前景像素占比做二分类：

    fg_ratio = |UNet++ 前景 ∩ YOLO 多边形掩码| / |YOLO 多边形掩码|

    fg_ratio >= FG_RATIO_THRESHOLD   → 视为真正的 2D 分子，原图复制到输出
    fg_ratio <  FG_RATIO_THRESHOLD   → 视为误检（3D / 表 / 装置图…），丢弃

分母用 extract.py sidecar `*.meta.npz` 里 YOLO 实例分割写出的多边形掩码，
而不是整张 crop 的 H·W：YOLO 的多边形已经紧贴分子轮廓，能把 crop 上下
左右的留白和无关空白排除在分母外，避免"骨架很细 + 留白很多"的真分子
被压低 ratio。sidecar 缺失或尺寸不一致时回退为"整图占比"旧定义。

之所以不在 crop 上做"像素抹白"清理：当前 YOLOv11 + extract 流水线给出的
crop 已经足够干净（YOLO 实例分割效果显著），UNet++ 再做像素级裁剪反而
可能咬掉真分子的边角；只取占比这一个标量当门控，副作用最小。

输出
----
crop_dataset_segmented/<原名>.png  —— 仅"被接受为分子"的图片，原样复制
                                      （sidecar `*.meta.npz` 也一起复制，
                                       供更下游模块继续使用）
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from utils.base.paths import PROJECT_ROOT
MODEL_PATH = PROJECT_ROOT / "model" / "post_seg_unetpp.pth"
INPUT_DIR = PROJECT_ROOT / "crop_dataset_denoised"
OUTPUT_DIR = PROJECT_ROOT / "crop_dataset_segmented"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
ENCODER = "resnet101"
NUM_CLASSES = 2
SIZE_DIVISOR = 32  # UnetPlusPlus(resnet101) 要求 H, W 是 32 的倍数

# 分类阈值：UNet++ 前景像素在 YOLO 多边形掩码内的占比下限。
#   操作 = (UNet++ 前景 ∩ YOLO 多边形掩码) 像素数 / YOLO 多边形掩码像素数；
#   单位 = 无量纲比例 [0, 1]；
#   几何尺度 = "YOLO 圈出来的分子区域"内部统计量（不是整张 crop，
#              已经把 crop 的白边/留白排除在分母外）；
#   调大（如 0.05）→ 过滤更严，三维 / 表 / 装置图能更稳剔掉，但骨架被
#                    UNet++ 抠得不全或字符占比偏高的真分子也可能误杀；
#   调小（如 0.001）→ 召回率高，但 3D 球棍模型这类"YOLO 框内 UNet++ 也
#                     能捡到一些前景"的负样本可能漏掉。
#   当前 0.01 = 1%，用于筛选几乎没有2D分子结构前景的图片（如表格、装置图、3D模型等）；
#               如 sidecar `.meta.npz` 缺失则回退到
#               "整图占比"（沿用旧定义，阈值仍按此参数比较）。
FG_RATIO_THRESHOLD = 0.01


class _UnetPP(nn.Module):
    def __init__(self, num_classes: int, encoder_name: str):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


def _strip_prefix(sd: dict, prefix: str) -> dict:
    if not any(k.startswith(prefix) for k in sd):
        return sd
    return {k[len(prefix):]: v for k, v in sd.items()}


# UNet++ 分类模型按 device 缓存的进程内单例。
# 权重 261 MB，每跑一次 run() 都重读一次在批处理下无所谓，但服务化后就是
# 每个任务白等几秒的固定开销；同进程同 device 只应加载一次。
_MODEL_CACHE: Dict[torch.device, nn.Module] = {}


def _load_model(device: torch.device) -> nn.Module:
    """加载 UNet++ 分类模型；同 device 在进程内只加载一次，之后返回缓存实例。"""
    cached = _MODEL_CACHE.get(device)
    if cached is not None:
        return cached

    print(f"正在加载 UNet++ 分类模型 ({device}) ...")
    model = _UnetPP(num_classes=NUM_CLASSES, encoder_name=ENCODER)
    sd = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _strip_prefix(sd, "module.")
    model.load_state_dict(sd)
    model = model.to(device).eval()
    _MODEL_CACHE[device] = model
    return model


_normalizer = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


def _image_to_tensor(image_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    image = _normalizer(image=image_rgb)["image"]
    image = image.transpose(2, 0, 1).astype("float32")
    return torch.from_numpy(image).unsqueeze(0).to(device)


def _whole_image_inference(model: nn.Module, image_tensor: torch.Tensor) -> torch.Tensor:
    """整张图一次性推理；填充到 SIZE_DIVISOR 的倍数后推理，再裁回原尺寸。"""
    b, c, h0, w0 = image_tensor.shape
    pad_h = (SIZE_DIVISOR - h0 % SIZE_DIVISOR) % SIZE_DIVISOR
    pad_w = (SIZE_DIVISOR - w0 % SIZE_DIVISOR) % SIZE_DIVISOR
    if pad_h or pad_w:
        # PyTorch reflect padding requires every pad width to be smaller than
        # the corresponding input dimension. Very small candidates can violate
        # that constraint (for example, a 15 x 14 px crop padded to 32 x 32).
        # Replicate padding preserves the boundary value and keeps inference
        # defined without resizing or otherwise altering the candidate content.
        pad_mode = "reflect" if pad_h < h0 and pad_w < w0 else "replicate"
        image_tensor = F.pad(
            image_tensor, (0, pad_w, 0, pad_h), mode=pad_mode
        )
    with torch.no_grad():
        logits = model(image_tensor)
    return logits[:, :, :h0, :w0]


def _load_yolo_mask(img_path: Path, target_hw: Tuple[int, int]) -> Optional[np.ndarray]:
    """读取 extract.py 写的 sidecar `*.meta.npz` 中的 YOLO 多边形掩码。

    返回与 crop 同尺寸的二值 mask（uint8，{0,1}）；缺失或尺寸对不上时返回 None。
    """
    meta_path = img_path.parent / f"{img_path.stem}.meta.npz"
    if not meta_path.exists():
        return None
    try:
        with np.load(str(meta_path)) as data:
            if "mask" not in data.files:
                return None
            mask = np.asarray(data["mask"]).astype(np.uint8)
    except Exception:
        return None
    if mask.shape != target_hw:
        return None
    return (mask > 0).astype(np.uint8)


def _compute_fg_ratio(model: nn.Module, img_path: Path,
                      device: torch.device
                      ) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """对单张图跑 UNet++，返回 (fg_ratio, unet_fg, yolo_mask, img_rgb)。

    fg_ratio 的定义：
      - 若存在 sidecar YOLO 多边形掩码（extract.py 输出的 *.meta.npz['mask']），
        fg_ratio = |UNet++ 前景 ∩ YOLO 多边形| / |YOLO 多边形|；
        分母只覆盖"YOLO 圈出来的分子区域"，排除 crop 的白边/留白干扰。
      - 若 sidecar 缺失或尺寸不一致，回退为旧定义：
        fg_ratio = |UNet++ 前景| / (H·W)。
    yolo_mask：YOLO 多边形二值掩码，回退时为 None（仅用于可视化区分两种情况）；
    unet_fg / img_rgb：可视化用，分类失败时也返回（便于排查）；读图失败返回 (0.0, None, None, None)。
    """
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return 0.0, None, None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = _image_to_tensor(img_rgb, device)

    logits = _whole_image_inference(model, tensor)
    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    unet_fg = (pred == 1).astype(np.uint8)

    h, w = unet_fg.shape
    yolo_mask = _load_yolo_mask(img_path, target_hw=(h, w))

    if yolo_mask is not None:
        denom = int(yolo_mask.sum())
        numer = int(np.logical_and(unet_fg.astype(bool), yolo_mask.astype(bool)).sum())
        fg_ratio = float(numer) / float(denom) if denom > 0 else 0.0
    else:
        denom = h * w
        fg_ratio = float(unet_fg.sum()) / float(denom) if denom > 0 else 0.0

    return fg_ratio, unet_fg, yolo_mask, img_rgb


def classify_image(
    model: nn.Module,
    img_path: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float = FG_RATIO_THRESHOLD,
    visualize: bool = False,
    viz_dir: Optional[Path] = None,
) -> Tuple[bool, float, Optional[Path]]:
    """单张图：UNet++ 推理 → 计算前景占比 → 与阈值比较 → 通过则原样复制。

    viz_dir 为 None 时叠加图写到默认 vis/post_seg/{keep,reject}/；传入目录
    则改写到 <viz_dir>/{keep,reject}/（供 eval 流水线落到评估输出树）。

    Returns:
        (is_molecule, fg_ratio, out_path or None)
        is_molecule=True 时 out_path 为输出图片的目标路径。
    """
    fg_ratio, unet_fg, yolo_mask, img_rgb = _compute_fg_ratio(model, img_path, device)
    is_molecule = fg_ratio >= threshold

    out_path: Optional[Path] = None
    if is_molecule:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / img_path.name
        shutil.copy2(img_path, out_path)
        # 顺带复制 sidecar（如有），保持与 extract.py 一致的下游接口
        meta_src = img_path.parent / f"{img_path.stem}.meta.npz"
        if meta_src.exists():
            shutil.copy2(meta_src, output_dir / meta_src.name)

    if visualize and img_rgb is not None and unet_fg is not None:
        _save_classify_overlay(
            img_rgb=img_rgb,
            unet_fg=unet_fg,
            yolo_mask=yolo_mask,
            fg_ratio=fg_ratio,
            threshold=threshold,
            is_molecule=is_molecule,
            stem=img_path.stem,
            out_dir=viz_dir,
        )

    return is_molecule, fg_ratio, out_path


def _save_classify_overlay(
    img_rgb: np.ndarray,
    unet_fg: np.ndarray,
    yolo_mask: Optional[np.ndarray],
    fg_ratio: float,
    threshold: float,
    is_molecule: bool,
    stem: str,
    out_dir: Optional[Path] = None,
) -> None:
    """post_seg 分类叠加图：UNet++ 前景（绿）+ YOLO 多边形（蓝，若有）
    + 标题里写 ratio / threshold / 判定结果。

    根据判定结果分流到子目录（root 为 out_dir，None 时用默认 vis/post_seg/）：
        is_molecule=True  → <root>/keep/<stem>_overlay.png
        is_molecule=False → <root>/reject/<stem>_overlay.png
    """
    from utils.base.viz_utils import (
        COLOR_BBOX as COLOR_YOLO,
        COLOR_FG,
        VIZ_POST,
        save_overlay_figure,
    )

    overlays = [(unet_fg.astype(bool), COLOR_FG, 0.35)]
    legend_items = [("UNet++ fg", COLOR_FG)]
    if yolo_mask is not None:
        overlays.append((yolo_mask.astype(bool), COLOR_YOLO, 0.20))
        legend_items.append(("YOLO poly (denom)", COLOR_YOLO))

    verdict = "MOL" if is_molecule else "REJECT"
    root = out_dir if out_dir is not None else VIZ_POST
    subdir = root / ("keep" if is_molecule else "reject")
    subdir.mkdir(parents=True, exist_ok=True)
    save_overlay_figure(
        img_rgb=img_rgb,
        overlays=overlays,
        title=f"[post_cls] {stem}  fg_ratio={fg_ratio:.4f}  "
              f"thr={threshold:.4f}  → {verdict}",
        legend_items=legend_items,
        out_path=subdir / f"{stem}_overlay.png",
    )


def run(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    device: str = "cuda",
    threshold: float = FG_RATIO_THRESHOLD,
    visualize: bool = False,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        print(f"⚠️ 输入目录不存在: {input_dir}")
        return

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"⚠️ 输入目录无图片: {input_dir}")
        return

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = _load_model(dev)
    print(f"✓ 权重: {MODEL_PATH.name}  ({dev})")
    print(f"✓ 前景占比阈值: {threshold:.4f}\n")

    output_dir.mkdir(parents=True, exist_ok=True)
    n_in = 0
    n_pass = 0
    rejected: List[Tuple[str, float]] = []
    for f in tqdm(files, desc="后分类"):
        is_mol, ratio, _ = classify_image(
            model, f, output_dir, dev,
            threshold=threshold, visualize=visualize,
        )
        n_in += 1
        if is_mol:
            n_pass += 1
            tqdm.write(f"[KEEP ] {f.name}  fg_ratio={ratio:.4f}")
        else:
            rejected.append((f.name, ratio))
            tqdm.write(f"[REJECT] {f.name}  fg_ratio={ratio:.4f}")

    print("\n" + "=" * 50)
    print(f"输入图片:       {n_in}")
    print(f"判为分子（保留）: {n_pass}")
    print(f"判为非分子（剔除）: {len(rejected)}")
    if rejected:
        print("\n被剔除清单（fg_ratio 升序）：")
        for name, r in sorted(rejected, key=lambda x: x[1]):
            print(f"  {r:.4f}  {name}")
    print(f"\n👉 保留结果保存至: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="把 UNet++ 当二分类器：对 crop_dataset_denoised/ 中每张图算前景像素占比，"
                    "≥ 阈值视为 2D 分子并复制到 crop_dataset_segmented/，否则剔除。"
    )
    parser.add_argument("--input", default=str(INPUT_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=FG_RATIO_THRESHOLD,
                        help=f"前景占比阈值（默认 {FG_RATIO_THRESHOLD}）。"
                             "低于该值视为非分子结构，剔除。")
    parser.add_argument("--visualize", action="store_true",
                        help="生成 UNet++ 前景叠加图到 vis/post_seg/")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        device=args.device,
        threshold=args.threshold,
        visualize=args.visualize,
    )
