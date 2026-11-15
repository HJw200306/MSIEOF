"""从 val 目录的整页图中找出「分子结构含箭头」或「分子结构低清」的页面。

流程：对每张整页图先用 YOLO 实例分割（复用 utils.pipeline.extract）裁出化学分子
结构，再对每个分子裁剪图分别做低清检测（ResolutionClassifier）和统一箭头检测
（四分类 U-Net++），只要某个分子命中（低清 或 含箭头），该整页即判定为命中。

输入：E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\val（默认）
YOLO 分割模型：model/YOLOv11_best.pt（复用 utils.pipeline.extract）
低清分类模型：model/classifier_fixed.pkl（sklearn 特征分类器，13 维，纯 CPU）
箭头模型：model/unified_arrow_4class.pth（U-Net++，4 类）
输出：lr_arrow/（命中的整页图副本，文件名不变）+ lr_arrow/normal/（均高清且无箭头的整页图副本）+ lr_arrow/no_mol/（未检出分子的整页图副本）+ lr_arrow/failed/（处理异常的整页图副本）+ lr_arrow/results.csv
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.base.paths import PROJECT_ROOT

from utils.pipeline.arrow_sr import ArrowSRProcessor, UNIFIED_ARROW_MODEL_PATH  # noqa: E402
from utils.pipeline.resolution_classifier import ResolutionClassifier, IMAGE_SUFFIXES  # noqa: E402
from utils.pipeline import extract as extract_mod  # noqa: E402

CLASSIFIER_PATH = PROJECT_ROOT / "model" / "classifier_fixed.pkl"
ARROW_MODEL_PATH = UNIFIED_ARROW_MODEL_PATH
INPUT_DIR = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\val")
OUTPUT_DIR = PROJECT_ROOT / "lr_arrow"
CROPS_DIR = OUTPUT_DIR / "_crops_tmp"  # extract 裁剪出的中间分子图
LOW_THRESHOLD = 0.10
ARROW_THRESHOLD = 100  # 实线箭头像素数阈值，超过此值判定为有箭头
YOLO_CONF = 0.25

class ArrowDetector:
    """实线/虚线统一箭头检测器。"""

    def __init__(self, model_path, device="cuda", arrow_threshold=ARROW_THRESHOLD):
        self.arrow_threshold = arrow_threshold
        self.processor = ArrowSRProcessor(
            unified_arrow_model_path=Path(model_path),
            device=device,
            load_sr_model=False,
        )

    def predict(self, img_path):
        """检测图片是否包含实线或虚线箭头。

        Args:
            img_path: 图片路径

        Returns:
            dict: {"has_arrow": bool, "arrow_pixels": int}
        """
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            return {"has_arrow": False, "arrow_pixels": 0}

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        masks = self.processor.predict_masks(img_pil)
        arrow_pixels = int(masks["arrow"].sum())

        return {
            "has_arrow": arrow_pixels > self.arrow_threshold,
            "arrow_pixels": arrow_pixels,
        }


def find_lowres_or_arrows(
    input_dir=INPUT_DIR,
    output_dir=OUTPUT_DIR,
    classifier_path=CLASSIFIER_PATH,
    arrow_model_path=ARROW_MODEL_PATH,
    low_threshold=LOW_THRESHOLD,
    arrow_threshold=ARROW_THRESHOLD,
    yolo_conf=YOLO_CONF,
    device="cuda",
    keep_crops=False,
) -> dict:
    """逐页提取分子结构，命中「低清或含箭头」的整页复制到输出目录（文件名不变）。

    Args:
        input_dir: 待检测的整页图目录。
        output_dir: 输出目录，命中的整页图平铺复制到此处；均高清且无箭头的整页图复制到 normal/，未检出分子的整页图复制到 no_mol/，处理异常的整页图复制到 failed/。
        classifier_path: 分辨率分类器 pickle 路径。
        arrow_model_path: 统一四分类箭头检测模型路径。
        low_threshold: 低清判定阈值，prob < 此值判低清。
        arrow_threshold: 实线与虚线箭头总像素数阈值，超过此值判定为有箭头。
        yolo_conf: YOLO 分子检测置信度阈值。
        device: 箭头检测设备（cuda/cpu）。
        keep_crops: 是否保留 extract 裁剪出的中间分子图（供调试）。

    Returns:
        {'hit', 'no_mol', 'normal', 'failed', 'details'} 汇总字典。
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        print(f"⚠️ 输入目录不存在: {input_dir}")
        return {"hit": 0, "no_mol": 0, "normal": 0, "failed": 0, "details": []}

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"⚠️ 目录为空（无图片）: {input_dir}")
        return {"hit": 0, "no_mol": 0, "normal": 0, "failed": 0, "details": []}

    if output_dir.resolve() == input_dir.resolve():
        print("❌ 输出目录不能与输入目录相同")
        return {"hit": 0, "no_mol": 0, "normal": 0, "failed": 0, "details": []}

    output_dir.mkdir(parents=True, exist_ok=True)
    normal_dir = output_dir / "normal"  # 均高清且无箭头的整页图归入此处
    normal_dir.mkdir(parents=True, exist_ok=True)
    no_mol_dir = output_dir / "no_mol"  # YOLO 未检出分子结构的整页图归入此处
    no_mol_dir.mkdir(parents=True, exist_ok=True)
    failed_dir = output_dir / "failed"  # 读图或 YOLO 处理异常的整页图归入此处
    failed_dir.mkdir(parents=True, exist_ok=True)
    # 每次运行前清空中间裁剪目录，避免残留旧文件混入。
    if CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR, ignore_errors=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # 初始化模型
    print(f"加载分辨率分类器: {classifier_path}")
    res_classifier = ResolutionClassifier(
        classifier_path=str(classifier_path),
        low_threshold=low_threshold,
    )
    print(f"加载统一四分类箭头检测模型: {arrow_model_path} (device={device})")
    arrow_detector = ArrowDetector(
        model_path=arrow_model_path,
        device=device,
        arrow_threshold=arrow_threshold,
    )
    print(f"加载 YOLO 分子分割模型（复用 utils.pipeline.extract，conf={yolo_conf}）")
    extract_mod._get_model()  # 提前加载 YOLO 模型（模块级懒加载单例），避免逐页重复加载

    details = []
    n_hit = n_no_mol = n_normal = n_failed = 0

    print(f"\n开始处理 {len(files)} 张整页图...")
    for idx, f in enumerate(files, 1):
        try:
            page = Image.open(f).convert("RGB")
        except Exception as e:
            shutil.copy2(f, failed_dir / f.name)  # 读取失败 → 归入 failed/
            n_failed += 1
            print(f"[{f.name}] 页面读取失败: {e} → 复制到 failed/")
            continue

        # 1. YOLO 提取分子结构裁剪图
        try:
            crops = extract_mod.extract_from_page(
                page,
                f.stem,
                CROPS_DIR,
                conf=yolo_conf,
                visualize=False,
                debug_dir=None,
            )
        except Exception as e:
            shutil.copy2(f, failed_dir / f.name)  # YOLO 提取失败 → 归入 failed/
            n_failed += 1
            print(f"[{f.name}] YOLO 提取失败: {e} → 复制到 failed/")
            continue

        if not crops:
            shutil.copy2(f, no_mol_dir / f.name)  # 未检出分子 → 归入 no_mol/
            n_no_mol += 1
            print(f"[{idx}/{len(files)}] {f.name} → 未检测到分子结构 → 复制到 no_mol/")
            continue

        # 2. 对每个分子裁剪图做低清 + 箭头检测
        hit_items = []  # (crop名, 是否低清, 是否箭头, prob, arrow_px)
        n_low = n_arrow = 0
        for crop in crops:
            low_res = res_classifier.predict(str(crop))
            if low_res.get("features") is None:
                continue
            arrow_res = arrow_detector.predict(str(crop))

            is_low = low_res["is_low_res"]
            has_arrow = arrow_res["has_arrow"]
            if is_low:
                n_low += 1
            if has_arrow:
                n_arrow += 1
            if is_low or has_arrow:
                hit_items.append((crop.name, is_low, has_arrow,
                                  low_res["prob"], arrow_res["arrow_pixels"]))

        # 3. 命中判定：任一分子「低清或含箭头」→ 整页命中
        if hit_items:
            shutil.copy2(f, output_dir / f.name)  # 文件名保持不变
            n_hit += 1
            hit_desc = ", ".join(
                f"{name}({'低清' if lo else ''}{'+' if lo and ar else ''}{'箭头' if ar else ''})"
                for name, lo, ar, _, _ in hit_items
            )
            print(f"[{idx}/{len(files)}] {f.name} → 命中（{len(hit_items)}/{len(crops)} 分子: {hit_desc}）")
            details.append({
                "page": f.name,
                "n_molecules": len(crops),
                "n_low": n_low,
                "n_arrow": n_arrow,
                "hit_details": hit_desc,
            })
        else:
            shutil.copy2(f, normal_dir / f.name)  # 均高清且无箭头 → 归入 normal/
            n_normal += 1
            print(f"[{idx}/{len(files)}] {f.name} → 正常（{len(crops)} 分子均高清且无箭头）→ 复制到 normal/")

    # 写 CSV（仅命中页面）
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "page", "n_molecules", "n_low", "n_arrow", "hit_details"
        ])
        writer.writeheader()
        writer.writerows(details)

    print("=" * 70)
    print(f"命中（含低清或箭头分子）: {n_hit} 页 → 复制到 {output_dir}")
    print(f"未检测到分子结构: {n_no_mol} 页 → 复制到 {no_mol_dir}")
    print(f"正常（均高清且无箭头）: {n_normal} 页 → 复制到 {normal_dir}")
    if n_failed:
        print(f"失败: {n_failed} 页 → 复制到 {failed_dir}")
    print(f"清单: {csv_path}")

    # 清理中间裁剪图
    if not keep_crops and CROPS_DIR.exists():
        shutil.rmtree(CROPS_DIR, ignore_errors=True)
        print("已清理中间裁剪目录 _crops_tmp/（可用 --keep-crops 保留）")
    elif keep_crops:
        print(f"中间分子裁剪图保留在: {CROPS_DIR}")

    return {"hit": n_hit, "no_mol": n_no_mol, "normal": n_normal, "failed": n_failed, "details": details}


def main():
    parser = argparse.ArgumentParser(description="从整页图中找出「分子结构低清或含箭头」的页面")
    parser.add_argument("--input", default=str(INPUT_DIR),
                        help=f"待检测的整页图目录，默认 {INPUT_DIR}")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help="输出目录，默认项目根目录/lr_arrow/")
    parser.add_argument("--res-model", default=str(CLASSIFIER_PATH),
                        help="分辨率分类器模型，默认 model/classifier_fixed.pkl")
    parser.add_argument("--arrow-model", default=str(ARROW_MODEL_PATH),
                        help="统一四分类箭头检测模型，默认 model/unified_arrow_4class.pth")
    parser.add_argument("--res-threshold", type=float, default=LOW_THRESHOLD,
                        help="低清判定阈值（高清概率低于此值判低清），默认 0.10")
    parser.add_argument("--arrow-threshold", type=int, default=ARROW_THRESHOLD,
                        help="实线箭头像素数阈值（超过此值判定为有箭头），默认 100")
    parser.add_argument("--yolo-conf", type=float, default=YOLO_CONF,
                        help="YOLO 分子检测置信度阈值，默认 0.25")
    parser.add_argument("--device", default="cuda",
                        help="箭头检测设备 cuda/cpu，默认 cuda")
    parser.add_argument("--keep-crops", action="store_true",
                        help="保留 extract 裁剪出的中间分子图（默认删除）")
    args = parser.parse_args()

    find_lowres_or_arrows(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        classifier_path=Path(args.res_model),
        arrow_model_path=Path(args.arrow_model),
        low_threshold=args.res_threshold,
        arrow_threshold=args.arrow_threshold,
        yolo_conf=args.yolo_conf,
        device=args.device,
        keep_crops=args.keep_crops,
    )


if __name__ == "__main__":
    main()
