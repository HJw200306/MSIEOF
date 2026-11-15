"""从语义分割标注（LabelMe格式）生成标注器输入数据集。

工作流：
1. 读取 json/PageToSMILES/*.json 中的分子polygon标注
2. 对每个分子实例：
   - 根据polygon裁剪图像（背景抹白）
   - 运行MolScribe预标注图结构
   - 保存裁剪图 + 预标注
3. 生成标注器数据结构（manifest + mapping）

输出：
  data/seg_page2smiles/
    ├── manifest.jsonl     # 标注器索引文件
    ├── images/            # 裁剪后的分子图像
    ├── annotations/       # MolScribe预标注
    └── mapping.json       # 实例ID → 原始标注映射
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import cv2
import numpy as np
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============ 配置区域 - 可以根据需要修改 ============
# 输入：分割标注数据
SEG_ROOT = Path(os.environ.get(
    "MARKUSH_SEG_ROOT",
    r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal",
))
JSON_DIR = SEG_ROOT / "json" / "PageToSMILES"
# PageToSMILES = 100 页子集（与 images/val 下同名文件内容完全一致）。
# json/labels/masks 已镜像到同名 PageToSMILES 目录，json 与 image 一一配套。
IMAGE_DIR = SEG_ROOT / "images" / "PageToSMILES"

# 输出：标注器数据
OUTPUT_ROOT = PROJECT_ROOT / "markush_annotator" / "data" / "seg_page2smiles"
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_ANNOTATIONS = OUTPUT_ROOT / "annotations"
OUTPUT_MANIFEST = OUTPUT_ROOT / "manifest.jsonl"
OUTPUT_MAPPING = OUTPUT_ROOT / "mapping.json"

# MolScribe设置
DEVICE = "auto"  # auto 会优先使用 CUDA，不可用时自动回退到 CPU
BATCH_SIZE = 16  # MolScribe批处理大小，根据GPU显存调整（8/16/32）
                 # 显存 < 8GB: 建议 4-8
                 # 显存 8-12GB: 建议 8-16
                 # 显存 > 12GB: 可以尝试 16-32
# ===================================================


def polygon_to_mask(points: List[List[float]], shape: Tuple[int, int]) -> np.ndarray:
    """将polygon坐标转换为二值掩膜。"""
    mask = np.zeros(shape, dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def crop_molecule_from_polygon(
    image: np.ndarray,
    points: List[List[float]],
    pad_pixels: int = 10
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """根据polygon裁剪分子图像，背景抹白。

    Returns:
        cropped_img: 裁剪后的图像（背景为白色）
        bbox: (x1, y1, x2, y2) 裁剪框坐标
    """
    h, w = image.shape[:2]

    # 生成掩膜
    mask = polygon_to_mask(points, (h, w))

    # 计算bbox
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None, None

    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1

    # 扩边
    x1 = max(0, x1 - pad_pixels)
    y1 = max(0, y1 - pad_pixels)
    x2 = min(w, x2 + pad_pixels)
    y2 = min(h, y2 + pad_pixels)

    # 裁剪
    crop_img = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]

    # 背景抹白
    crop_img[crop_mask == 0] = 255

    return crop_img, (x1, y1, x2, y2)


def clear_dir_contents(directory: Path) -> int:
    """清空目录内的内容（保留目录本身），返回删除的条目数。"""
    if not directory.exists():
        return 0
    n = 0
    for p in directory.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        n += 1
    return n


def sort_by_reading_order(
    entries: List[Tuple[int, List[List[float]]]]
) -> List[int]:
    """按阅读顺序排序：先从上到下分行，行内再从左到右。

    Args:
        entries: [(shape_index, polygon_points), ...]
    Returns:
        排序后的 shape_index 列表
    """
    boxes = []  # (shape_index, x1, y1, x2, y2)
    for idx, pts in entries:
        p = np.array(pts, dtype=float)
        x1, y1 = p.min(axis=0)
        x2, y2 = p.max(axis=0)
        boxes.append((idx, x1, y1, x2, y2))

    heights = [b[4] - b[2] for b in boxes]
    tol = max(10.0, 0.5 * float(np.median(heights)))  # 分行阈值

    # 先按中心 y 升序，再按阈值归并成行
    boxes.sort(key=lambda b: b[2] + b[4])
    rows: List[List[tuple]] = []
    for b in boxes:
        cy = (b[2] + b[4]) / 2.0
        if rows:
            row_cy = float(np.mean([(r[2] + r[4]) / 2.0 for r in rows[-1]]))
            if abs(cy - row_cy) <= tol:
                rows[-1].append(b)
                continue
        rows.append([b])

    # 行内按中心 x 升序
    ordered: List[int] = []
    for row in rows:
        row.sort(key=lambda b: b[1] + b[3])
        ordered.extend(b[0] for b in row)
    return ordered


def collect_crops_from_json(
    json_path: Path,
    image_dir: Path,
    start_index: int
) -> List[Dict[str, Any]]:
    """从单个LabelMe标注文件收集所有裁剪图像（不运行MolScribe）。

    同一页内的 mol 按阅读顺序（从上到下分行、行内从左到右）编号。

    Returns:
        List of crop metadata (without MolScribe predictions)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 读取对应图像
    image_filename = json_path.stem + ".png"
    image_path = image_dir / image_filename

    if not image_path.exists():
        print(f"[WARN] Image not found: {image_path}")
        return []

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"[WARN] Failed to read image: {image_path}")
        return []

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    shapes = data.get("shapes", [])

    # 收集所有 mol 多边形（保留原始 shape_index）
    mol_entries: List[Tuple[int, List[List[float]]]] = []
    for shape_idx, shape in enumerate(shapes):
        if shape.get("label") != "mol":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        mol_entries.append((shape_idx, points))

    if not mol_entries:
        return []

    points_by_idx = dict(mol_entries)
    ordered_idx = sort_by_reading_order(mol_entries)

    crops = []
    for shape_idx in ordered_idx:
        points = points_by_idx[shape_idx]

        # 裁剪分子
        crop_img, bbox = crop_molecule_from_polygon(image_rgb, points)
        if crop_img is None:
            continue

        # 生成实例ID（按阅读顺序递增）
        instance_id = start_index + len(crops)

        crops.append({
            "instance_id": instance_id,
            "source_file": json_path.name,
            "source_image": image_filename,
            "source_stem": json_path.stem,
            "shape_index": shape_idx,
            "bbox": bbox,
            "polygon": points,
            "crop_image": crop_img
        })

    return crops


def _resolve_device(requested: str) -> str:
    """选择可用设备，避免无 CUDA 的机器直接启动失败。"""
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if requested == "cuda" and not cuda_available:
        print("[WARN] CUDA 不可用，已自动切换到 CPU")
    return "cuda" if cuda_available else "cpu"


def _configure_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure:
                reconfigure(encoding="utf-8")


def main() -> int:
    """主流程：处理所有分割标注并生成标注器数据。"""

    _configure_console()

    parser = argparse.ArgumentParser(description="从 LabelMe 分割标注生成 Markush 标注器数据")
    parser.add_argument("--seg-root", type=Path, default=SEG_ROOT,
                        help="数据集根目录；也可设置 MARKUSH_SEG_ROOT")
    parser.add_argument("--json-dir", type=Path,
                        help="LabelMe JSON 目录（默认: <seg-root>/json/PageToSMILES）")
    parser.add_argument("--image-dir", type=Path,
                        help="原始图片目录（默认: <seg-root>/images/PageToSMILES）")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT,
                        help="标注器数据输出目录")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--check", action="store_true", help="仅检查路径和依赖，不生成数据")
    args, _ = parser.parse_known_args()

    seg_root = args.seg_root.expanduser().resolve()
    json_dir = (args.json_dir or seg_root / "json" / "PageToSMILES").expanduser().resolve()
    image_dir = (args.image_dir or seg_root / "images" / "PageToSMILES").expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    output_images = output_root / "images"
    output_annotations = output_root / "annotations"
    output_manifest = output_root / "manifest.jsonl"
    output_mapping = output_root / "mapping.json"

    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")

    print("=" * 70)
    print("步骤1: 从分割标注生成标注器数据")
    print("=" * 70)

    # 检查输入目录
    print(f"\n检查输入目录...")
    print(f"  JSON目录: {json_dir}")
    print(f"  图像目录: {image_dir}")

    if not json_dir.is_dir():
        print(f"\n[ERROR] JSON目录不存在: {json_dir}")
        print("请检查路径配置！")
        return 2

    if not image_dir.is_dir():
        print(f"\n[ERROR] 图像目录不存在: {image_dir}")
        print("请检查路径配置！")
        return 2

    try:
        from utils.pipeline.img2smiles import Img2SmilesProcessor
    except Exception as exc:
        print(f"\n[ERROR] 无法导入 MolScribe 处理器: {exc}")
        return 3

    json_files = sorted(json_dir.glob("*.json"))
    image_count = sum(1 for _ in image_dir.glob("*.png"))
    print(f"  找到 JSON {len(json_files)} 个，PNG {image_count} 张")
    if not json_files:
        print("[ERROR] JSON 目录中没有 .json 文件")
        return 2
    if args.check:
        print("[OK] 输入路径和运行依赖检查通过")
        return 0

    # 创建输出目录
    print(f"\n创建输出目录...")
    output_root.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_root}")

    # JSON_DIR 与 IMAGE_DIR 已一一配套（json/PageToSMILES 仅含这 100 页），
    # 无需再做文件名交集过滤。
    print(f"\n找到 {len(json_files)} 个标注文件"
          f"（{image_dir.name} 共 {image_count} 页，"
          f"{json_dir.parent.name}/{json_dir.name} 共 {len(json_files)} 个）")

    # 步骤1: 收集所有裁剪图像（不运行MolScribe）
    print(f"\n步骤1/2: 裁剪所有分子图像...")
    all_crops = []
    for json_file in tqdm(json_files, desc="裁剪图像"):
        crops = collect_crops_from_json(
            json_file,
            image_dir,
            start_index=len(all_crops)
        )
        all_crops.extend(crops)

    print(f"  收集到 {len(all_crops)} 个分子实例")
    if not all_crops:
        print("[ERROR] 未从输入标注中找到有效的 mol polygon")
        return 2

    device = _resolve_device(args.device)
    print(f"\n初始化 MolScribe (设备: {device})...")
    try:
        processor = Img2SmilesProcessor(device=device)
        print("  ✓ MolScribe 初始化成功")
    except Exception as e:
        print(f"  ✗ MolScribe 初始化失败: {e}")
        print("  提示: 可用 --device cpu 强制使用 CPU")
        return 3

    # 步骤2: 批量运行MolScribe预标注
    print(f"\n步骤2/2: MolScribe批量预标注 (batch_size={args.batch_size})...")

    all_results = []
    mapping_dict = {}

    # 按batch_size分批处理
    for batch_start in tqdm(range(0, len(all_crops), args.batch_size), desc="MolScribe推理"):
        batch_crops = all_crops[batch_start:batch_start + args.batch_size]

        # 提取图像
        batch_images = [crop["crop_image"] for crop in batch_crops]

        # 批量预测
        try:
            batch_preds = processor.predict_batch(batch_images, batch_size=args.batch_size)
        except Exception as e:
            print(f"\n[ERROR] Batch prediction failed: {e}")
            # 回退到逐个处理
            batch_preds = []
            for img in batch_images:
                try:
                    pred = processor.predict_batch([img], batch_size=1)[0]
                except:
                    pred = {"atoms": [], "bonds": [], "smiles": "", "confidence": 0.0}
                batch_preds.append(pred)

        # 组装结果
        for crop, pred in zip(batch_crops, batch_preds):
            instance_id = crop["instance_id"]

            result = {
                "instance_id": instance_id,
                "manifest": {
                    "index": instance_id,
                    "source": crop["source_stem"],
                    "shape_index": crop["shape_index"],
                    "image_size": [crop["crop_image"].shape[1], crop["crop_image"].shape[0]],
                    "smi": pred.get("smiles", ""),
                    "reason": "manual_review",
                    "review_status": "pending"
                },
                "mapping": {
                    "source_file": crop["source_file"],
                    "source_image": crop["source_image"],
                    "shape_index": crop["shape_index"],
                    "bbox": crop["bbox"],
                    "polygon": crop["polygon"],
                    "crop_filename": f"{instance_id:06d}.png"
                },
                "annotation": {
                    "index": instance_id,
                    "atoms": pred.get("atoms", []),
                    "bonds": pred.get("bonds", []),
                    "image_size": [crop["crop_image"].shape[1], crop["crop_image"].shape[0]],
                    "smi": pred.get("smiles", ""),
                    "confidence": pred.get("confidence", 0.0),
                    "sgroups": [],
                    "brackets": []
                },
                "crop_image": crop["crop_image"]
            }

            all_results.append(result)

    # 步骤3: 保存所有结果
    # 先清空旧的 images/annotations，避免上一轮残留污染（不碰 corrected_annotations）
    print(f"\n清理旧输出...")
    n_img = clear_dir_contents(output_images)
    n_ann = clear_dir_contents(output_annotations)
    print(f"  已删除: images {n_img} 个, annotations {n_ann} 个")

    print(f"\n保存结果...")
    with open(output_manifest, 'w', encoding='utf-8') as manifest_f:
        for res in all_results:
            instance_id = res["instance_id"]

            # 保存裁剪图像
            crop_path = output_images / f"{instance_id:06d}.png"
            cv2.imwrite(str(crop_path), cv2.cvtColor(res["crop_image"], cv2.COLOR_RGB2BGR))

            # 保存预标注
            anno_path = output_annotations / f"{instance_id:06d}.json"
            with open(anno_path, 'w', encoding='utf-8') as f:
                json.dump(res["annotation"], f, ensure_ascii=False, indent=2)

            # 写入manifest
            manifest_f.write(json.dumps(res["manifest"], ensure_ascii=False) + "\n")

            # 记录mapping
            mapping_dict[f"{instance_id:06d}"] = res["mapping"]

    # 保存mapping
    with open(output_mapping, 'w', encoding='utf-8') as f:
        json.dump(mapping_dict, f, ensure_ascii=False, indent=2)

    # 统计报告
    print("\n" + "=" * 70)
    print("✓ 步骤1完成！")
    print("=" * 70)
    print(f"  输入文件:     {len(json_files)} 个")
    print(f"  分子实例:     {len(all_results)} 个")
    print(f"  输出目录:     {output_root}")
    print(f"  manifest:     {output_manifest}")
    print(f"  mapping:      {output_mapping}")
    print("=" * 70)
    return 0
    print("\n下一步:")
    print("  1. 在PyCharm中打开 markush_annotator/run.py")
    print("  2. 修改启动参数（见run.py顶部配置区）")
    print("  3. 直接运行 run.py 启动标注器")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
