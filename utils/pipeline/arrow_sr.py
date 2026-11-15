import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image
from tqdm import tqdm

from utils.base.paths import PROJECT_ROOT

from hat.hat_predictor import HATPredictor  # noqa: E402
from utils.pipeline.resolution_classifier import classify_directory  # noqa: E402

HAT_MODEL_PATH = PROJECT_ROOT / "model" / "HAT_S_ftv5.pth"
UNIFIED_ARROW_MODEL_PATH = PROJECT_ROOT / "model" / "unified_arrow_4class.pth"
# 流程顺序：extract → fragment_cleanup → post_segment → arrow_sr → img2smiles。
# 本步骤读 post_segment 的输出；候选图在此前已经过 fragment_cleanup
# 连通域后处理和 U-Net++ 分子图像筛选。
# 在此基础上做超分 + 统一四分类箭头去除，结果再写回 crop_dataset_processed/。
CROP_DIR = PROJECT_ROOT / "crop_dataset_segmented"
OUTPUT_DIR = PROJECT_ROOT / "crop_dataset_processed"

UNIFIED_ENCODER = "resnet101"
UNIFIED_NUM_CLASSES = 4
UNIFIED_CLASS_NAMES = ["background", "solid_arrow", "dashed_arrow", "bond"]
UNIFIED_SIZE_DIVISOR = 32


# ImageNet 归一化（与 four_class/core.py 一致），模块级单例避免重复构造。
_UNIFIED_NORMALIZER = albu.Normalize(mean=(0.485, 0.456, 0.406),
                                     std=(0.229, 0.224, 0.225))


class _UnifiedArrowUNetPP(torch.nn.Module):
    """统一箭头分割网络：SMP U-Net++ / ResNet-101 / 四分类。"""

    def __init__(self):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=UNIFIED_ENCODER,
            encoder_weights=None,
            in_channels=3,
            classes=UNIFIED_NUM_CLASSES,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


class ArrowSRProcessor:
    """封装 HAT 超分与统一四分类箭头去除。"""

    def __init__(
        self,
        hat_model_path: Path = HAT_MODEL_PATH,
        unified_arrow_model_path: Path = UNIFIED_ARROW_MODEL_PATH,
        device: str = "cuda",
        load_sr_model: bool = True,
    ):
        print("正在初始化模型...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sr_model = (
            HATPredictor(model_path=str(hat_model_path), device=device)
            if load_sr_model else None
        )
        if self.sr_model is not None:
            print("[OK] HAT 超分模型加载完成")
        self.unified_model = _UnifiedArrowUNetPP().to(self.device)
        checkpoint = torch.load(str(unified_arrow_model_path),
                                map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            classes = checkpoint.get("num_classes")
            names = checkpoint.get("class_names")
            if classes is not None and int(classes) != UNIFIED_NUM_CLASSES:
                raise ValueError(
                    f"{unified_arrow_model_path}: expected {UNIFIED_NUM_CLASSES} classes, "
                    f"found {classes}"
                )
            if names is not None and list(names) != UNIFIED_CLASS_NAMES:
                raise ValueError(
                    f"{unified_arrow_model_path}: unexpected class mapping {names}"
                )
            state = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
        else:
            state = checkpoint
        # 去掉 DataParallel 可能带的 module. 前缀。
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        output_key = next(
            (key for key in state if key.endswith("segmentation_head.0.weight")),
            None,
        )
        if output_key is None or int(state[output_key].shape[0]) != UNIFIED_NUM_CLASSES:
            found = None if output_key is None else int(state[output_key].shape[0])
            raise ValueError(
                f"{unified_arrow_model_path}: expected a four-class segmentation head, "
                f"found {found}"
            )
        # four_class/train.py 直接保存 SMP 模型，因此权重键没有额外的 ``model.`` 前缀。
        self.unified_model.model.load_state_dict(state, strict=True)
        self.unified_model.eval()
        print("[OK] 统一四分类箭头模型加载完成\n")

    def predict_masks(self, img_pil: Image.Image) -> dict[str, np.ndarray]:
        """运行统一四分类模型并返回原图尺寸的类别与二值掩膜。"""
        img_rgb = np.asarray(img_pil.convert("RGB"))
        h0, w0 = img_rgb.shape[:2]
        pad_h = (-h0) % UNIFIED_SIZE_DIVISOR
        pad_w = (-w0) % UNIFIED_SIZE_DIVISOR
        if pad_h or pad_w:
            img_input = cv2.copyMakeBorder(
                img_rgb, 0, pad_h, 0, pad_w,
                borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255),
            )
        else:
            img_input = img_rgb
        normed = _UNIFIED_NORMALIZER(image=img_input)["image"]
        tensor = torch.from_numpy(
            normed.transpose(2, 0, 1).astype(np.float32)
        ).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            pred = self.unified_model(tensor).argmax(dim=1)[0, :h0, :w0]
        classes = pred.cpu().numpy().astype(np.uint8)
        solid = classes == 1
        dashed = classes == 2
        bond = classes == 3
        return {
            "classes": classes,
            "solid": solid,
            "dashed": dashed,
            "arrow": solid | dashed,
            "bond": bond,
        }

    @staticmethod
    def _color_prediction(masks: dict[str, np.ndarray]) -> Image.Image:
        """生成兼容旧可视化接口的 RGB 类别图。"""
        h, w = masks["classes"].shape
        color = np.zeros((h, w, 3), dtype=np.uint8)
        color[masks["solid"]] = (255, 0, 0)
        color[masks["dashed"]] = (255, 140, 0)
        color[masks["bond"]] = (0, 128, 0)
        return Image.fromarray(color)

    def remove_arrows(self, img_pil: Image.Image) -> tuple[Image.Image, Image.Image]:
        """兼容旧接口；现在一次擦除统一模型识别的全部箭头。"""
        masks = self.predict_masks(img_pil)
        out = np.asarray(img_pil.convert("RGB")).copy()
        out[masks["arrow"] & ~masks["bond"]] = 255
        return Image.fromarray(out), self._color_prediction(masks)

    def remove_dashed_arrows(self, img_pil: Image.Image
                             ) -> tuple[Image.Image, np.ndarray]:
        """兼容旧接口；使用统一模型仅擦除虚线箭头类别。"""
        masks = self.predict_masks(img_pil)
        out = np.asarray(img_pil.convert("RGB")).copy()
        out[masks["dashed"] & ~masks["bond"]] = 255
        return Image.fromarray(out), masks["dashed"]

    def remove_arrows_protected(self, img_pil: Image.Image):
        """用统一四分类模型预测一次，合并两类箭头后按化学键优先擦除。"""
        rgb = np.asarray(img_pil.convert('RGB'))
        masks = self.predict_masks(img_pil)
        erased = masks["arrow"] & ~masks["bond"]
        out = rgb.copy()
        out[erased] = 255
        debug = {
            "solid_proposed": masks["solid"],
            "dashed_proposed": masks["dashed"],
            "bond": masks["bond"],
            "protected": masks["bond"],
            "solid_erased": masks["solid"] & erased,
            "dashed_erased": masks["dashed"] & erased,
            "erased": erased,
            "classes": masks["classes"],
            "version": "unified-four-class-argmax-arrow-merge-bond-priority-v1",
        }
        return Image.fromarray(out), debug

    def process_low_res(self, img_path: Path, viz_stem: Optional[str] = None) -> np.ndarray:
        """低清：SR → 统一四分类预测 → 化学键优先擦除。返回 BGR。"""
        if self.sr_model is None:
            raise RuntimeError("ArrowSRProcessor 初始化时未加载 HAT 超分模型")
        img_bgr = cv2.imread(str(img_path))
        img_sr_bgr = self.sr_model.predict(img_bgr)
        img_sr_pil = Image.fromarray(cv2.cvtColor(img_sr_bgr, cv2.COLOR_BGR2RGB))
        img_final_pil, guard = self.remove_arrows_protected(img_sr_pil)
        if viz_stem is not None:
            _save_sr_compare(img_bgr, img_sr_bgr, viz_stem)
            _save_protected_overlay(np.array(img_sr_pil), guard, viz_stem)
        return cv2.cvtColor(np.array(img_final_pil), cv2.COLOR_RGB2BGR)

    def process_high_res(self, img_path: Path, viz_stem: Optional[str] = None) -> np.ndarray:
        """高清：统一四分类预测并去除箭头。返回 BGR ndarray。"""
        img_pil = Image.open(img_path).convert("RGB")
        img_final_pil, guard = self.remove_arrows_protected(img_pil)
        if viz_stem is not None:
            _save_protected_overlay(np.array(img_pil), guard, viz_stem)
        return cv2.cvtColor(np.array(img_final_pil), cv2.COLOR_RGB2BGR)


def _save_protected_overlay(rgb, guard, stem):
    from utils.base.viz_utils import VIZ_ARROW, save_overlay_figure
    save_overlay_figure(img_rgb=rgb,
        overlays=[(guard['protected'],(0,100,255),.35),(guard['erased'],(255,80,0),.75)],
        title=f'{stem}: blue protected / orange erased',
        legend_items=[('protected',(0,100,255)),('erased',(255,80,0))],
        out_path=VIZ_ARROW / f'{stem}_protected.png')


def _save_sr_compare(orig_bgr: np.ndarray, sr_bgr: np.ndarray, stem: str) -> None:
    """SR 前后并排对比：左 = 原图最近邻放大到 SR 尺寸；右 = SR 结果。"""
    from utils.base.viz_utils import VIZ_SR, save_side_by_side

    h_sr, w_sr = sr_bgr.shape[:2]
    h0, w0 = orig_bgr.shape[:2]
    orig_up_bgr = cv2.resize(orig_bgr, (w_sr, h_sr), interpolation=cv2.INTER_NEAREST)
    left = cv2.cvtColor(orig_up_bgr, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2RGB)
    save_side_by_side(
        left_rgb=left,
        right_rgb=right,
        left_title=f"original ({w0}×{h0}) — nearest upsample",
        right_title=f"HAT SR ({w_sr}×{h_sr})",
        suptitle=f"[SR] {stem}  {w0}×{h0} → {w_sr}×{h_sr}  (×2)",
        out_path=VIZ_SR / f"{stem}_compare.png",
    )


def _save_arrow_overlay(before_rgb: np.ndarray, pred_rgb: np.ndarray,
                        dashed_mask: np.ndarray, stem: str) -> None:
    """去箭头叠加：在去（实线）箭头前的图上一次画出 arrow(红)+bond(蓝)+dashed(橙黄)。

    before_rgb 是"实线 eraser 还没处理"的图（low_res 用 SR 后图，high_res 用原图）;
    虚线 eraser 实际是在"实线 eraser 处理之后"的图上跑的, 但虚线箭头本身在像素
    位置上不会被实线 eraser 涂掉, 所以把虚线 mask 直接画在 before_rgb 上仍然
    对应正确的位置, 三种 mask 可以叠在同一张图里看。
    """
    from utils.base.viz_utils import (
        COLOR_ARROW,
        COLOR_ARROW_DASHED,
        COLOR_BOND,
        VIZ_ARROW,
        save_overlay_figure,
    )

    arrow_mask = (
        (pred_rgb[:, :, 0] == 255)
        & (pred_rgb[:, :, 1] == 0)
        & (pred_rgb[:, :, 2] == 0)
    )
    bond_mask = (
        (pred_rgb[:, :, 0] == 0)
        & (pred_rgb[:, :, 1] == 128)
        & (pred_rgb[:, :, 2] == 0)
    )
    save_overlay_figure(
        img_rgb=before_rgb,
        overlays=[
            (bond_mask, COLOR_BOND, 0.35),
            (arrow_mask, COLOR_ARROW, 0.55),
            (dashed_mask.astype(bool), COLOR_ARROW_DASHED, 0.55),
        ],
        title=(f"[arrow] {stem}  "
               f"arrow={int(arrow_mask.sum())}px  "
               f"bond={int(bond_mask.sum())}px  "
               f"dashed={int(dashed_mask.sum())}px"),
        legend_items=[("arrow (solid)", COLOR_ARROW),
                      ("bond", COLOR_BOND),
                      ("arrow (dashed)", COLOR_ARROW_DASHED)],
        out_path=VIZ_ARROW / f"{stem}_overlay.png",
    )


def run(
    crop_dir: Path = CROP_DIR,
    output_dir: Path = OUTPUT_DIR,
    low_threshold: float = 0.10,
    device: str = "cuda",
    classification: Optional[Sequence[dict]] = None,
    visualize: bool = False,
):
    """主流程：先分类，再分流处理。

    Args:
        classification: 可选，已有的分类结果 details 列表；
                        若为 None 则自动调用 classify_directory。
        visualize: 是否输出可视化到 vis/sr/ 与 vis/arrow/。
    """
    crop_dir = Path(crop_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if classification is None:
        summary = classify_directory(crop_dir, low_threshold=low_threshold)
        details = summary["details"]
    else:
        details = list(classification)

    if not details:
        print("⚠️ 没有可处理的图片。")
        return

    processor = ArrowSRProcessor(device=device)

    n_low = n_high = n_fail = 0
    t_low = t_high = 0.0
    failed: list[str] = []

    for item in tqdm(details, desc="超分/去箭头"):
        name = item["name"]
        label = item["label"]
        in_path = crop_dir / name
        out_path = output_dir / name
        if not in_path.exists():
            failed.append(name)
            n_fail += 1
            continue
        try:
            t0 = time.time()
            viz_stem = Path(name).stem if visualize else None
            if label == "低清":
                result = processor.process_low_res(in_path, viz_stem=viz_stem)
                t_low += time.time() - t0
                n_low += 1
            elif label == "高清":
                result = processor.process_high_res(in_path, viz_stem=viz_stem)
                t_high += time.time() - t0
                n_high += 1
            else:
                # 读取失败的跳过
                failed.append(name)
                n_fail += 1
                continue
            cv2.imwrite(str(out_path), result)
        except Exception as e:
            tqdm.write(f"  ❌ 处理失败 [{name}]: {e}")
            failed.append(name)
            n_fail += 1

    print("\n" + "=" * 50)
    print("处理完成 — 汇总")
    print("=" * 50)
    print(f"低清(先超分后去箭头): {n_low} 张  总耗时 {t_low:.1f}s"
          + (f"  均 {t_low / n_low:.3f}s" if n_low else ""))
    print(f"高清(直接去箭头)    : {n_high} 张  总耗时 {t_high:.1f}s"
          + (f"  均 {t_high / n_high:.3f}s" if n_high else ""))
    if n_fail:
        print(f"失败/跳过: {n_fail} 张 -> {failed}")
    print(f"\n👉 结果保存至: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="对 crop_dataset_segmented/（post_segment 输出）按高低清分流执行超分/去箭头")
    parser.add_argument("--input", default=str(CROP_DIR), help="待处理目录，默认 crop_dataset_segmented/")
    parser.add_argument("--output", default=str(OUTPUT_DIR), help="输出目录，默认 crop_dataset_processed/")
    parser.add_argument("--threshold", type=float, default=0.10, help="低清判定阈值")
    parser.add_argument("--device", default="cuda", help="cuda 或 cpu")
    parser.add_argument("--visualize", action="store_true", help="生成可视化到 vis/sr/ 与 vis/arrow/")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    run(
        crop_dir=Path(args.input),
        output_dir=Path(args.output),
        low_threshold=args.threshold,
        device=args.device,
        visualize=args.visualize,
    )
