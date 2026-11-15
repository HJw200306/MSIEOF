"""将精修后的图结构标注回写到原始LabelMe格式。

工作流：
1. 读取 corrected_annotations/*.json（精修后的图结构）
2. 读取 mapping.json（实例→原始标注映射）
3. 从图结构生成SMILES（使用RDKit）
4. 将atoms、bonds、smiles回写到原始JSON的shape中
5. 生成导出报告

输出格式（扩展LabelMe）：
{
  "shapes": [
    {
      "label": "mol",
      "points": [[x, y], ...],
      "shape_type": "polygon",
      "atoms": [...],      // 精修后的
      "bonds": [...],      // 精修后的
      "smiles": "..."      // 从图结构生成
    }
  ]
}
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict

# 添加项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from markush_annotator.backend.data_manager import graph_to_smiles


# 路径配置
ANNOTATOR_ROOT = PROJECT_ROOT / "markush_annotator" / "data" / "seg_page2smiles"
CORRECTED_DIR = ANNOTATOR_ROOT / "corrected_annotations"
ANNOTATIONS_DIR = ANNOTATOR_ROOT / "annotations"
MAPPING_FILE = ANNOTATOR_ROOT / "mapping.json"

SEG_ROOT = Path(os.environ.get(
    "MARKUSH_SEG_ROOT",
    r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal",
))
SOURCE_JSON_DIR = SEG_ROOT / "json" / "PageToSMILES"
# 注意：Windows 文件名不区分大小写，不能只用大小写区分（PageTOSMILES == PageToSMILES）。
OUTPUT_JSON_DIR = SEG_ROOT / "json" / "PageToSMILES_approved"

# 只导出"审核通过"的页面，输出到独立的 PageTOSMILES 目录：
#   1) review_status.jsonl 中该页所有结构均为 approved 的页面
#   2) 额外纳入下列已人工复核的箭头页（source_image 文件名，见 arrow_pages 数据包）
FILTER_APPROVED_PAGES = True
EXTRA_SOURCE_IMAGES = {
    "JNP_2024_01_2-13_006.png",
    "JNP_2024_02_252-265_003.png",
    "JNP_2024_02_266-275_004.png",
    "JNP_2024_02_297-303_002.png",
    "JNP_2024_02_365-370_003.png",
    "JNP_2024_02_396-403_002.png",
    "JNP_2024_04_1059-1066_004.png",
    "JNP_2024_04_733-742_006.png",
    "JNP_2024_04_861-868_001.png",
    "Phyt_2024_217_113926_006.png",
}
REVIEW_STATUS_FILE = ANNOTATOR_ROOT / "review_status.jsonl"


def _configure_console() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure:
                reconfigure(encoding="utf-8")


def load_mapping() -> Dict[str, Dict[str, Any]]:
    """加载实例ID到原始标注的映射。"""
    if not MAPPING_FILE.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

    with open(MAPPING_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_corrected_annotation(instance_id: str) -> Dict[str, Any]:
    """加载精修后的标注数据。"""
    anno_path = CORRECTED_DIR / f"{instance_id}.json"
    if not anno_path.exists():
        return None

    with open(anno_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_annotation(instance_id: str) -> Dict[str, Any]:
    """加载 MolScribe 预标注（annotations/），作为 corrected 缺失时的兜底。"""
    anno_path = ANNOTATIONS_DIR / f"{instance_id}.json"
    if not anno_path.exists():
        return None

    with open(anno_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def collect_annotations_by_source(
    mapping: Dict[str, Dict[str, Any]]
) -> Dict[str, List[tuple]]:
    """按原始文件分组收集所有标注。

    Returns:
        {source_file: [(instance_id, shape_index, annotation_data), ...]}
    """
    grouped = defaultdict(list)

    n_fallback = 0
    for instance_id, meta in mapping.items():
        source_file = meta["source_file"]
        shape_index = meta["shape_index"]

        # 加载精修后的标注；缺失时回退到 MolScribe 预标注
        anno_data = load_corrected_annotation(instance_id)
        if anno_data is None:
            anno_data = load_annotation(instance_id)
            if anno_data is None:
                print(f"[SKIP] No annotation for {instance_id}")
                continue
            n_fallback += 1

        grouped[source_file].append((instance_id, shape_index, anno_data))

    print(f"  (fallback 到 annotations/ 的实例数: {n_fallback})")
    return grouped


def load_selected_source_files(mapping: Dict[str, Dict[str, Any]]) -> set:
    """选出需要导出的原始标注文件（source_file）。

    规则：
      1) review_status.jsonl 中该页（source_image）所有结构均为 approved；
      2) EXTRA_SOURCE_IMAGES 中列出的已复核箭头页。

    Returns:
        允许导出的 source_file 集合
    """
    status: Dict[int, str] = {}
    with open(REVIEW_STATUS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            status[int(obj["index"])] = obj["review_status"]

    # 页面 -> {结构状态}, 页面 -> {source_file}
    page_status = defaultdict(set)
    page_sources = defaultdict(set)
    for instance_id, meta in mapping.items():
        page = meta["source_image"]
        page_status[page].add(status.get(int(instance_id), "?"))
        page_sources[page].add(meta["source_file"])

    selected_pages = {p for p, sts in page_status.items() if sts == {"approved"}}
    selected_pages |= EXTRA_SOURCE_IMAGES

    selected_files = set()
    for page in selected_pages:
        selected_files |= page_sources.get(page, set())

    print(f"  审核通过页: {len([p for p, s in page_status.items() if s == {'approved'}])}"
          f" + 箭头页: {len(EXTRA_SOURCE_IMAGES)}"
          f" -> 选中 {len(selected_pages)} 个源页面 / {len(selected_files)} 个标注文件")
    return selected_files


def merge_annotation_to_labelme(
    source_json_path: Path,
    annotations: List[tuple],
    output_json_path: Path
) -> Dict[str, int]:
    """将图结构标注合并到原始LabelMe格式。

    Returns:
        统计信息 {"updated": N, "skipped": N}
    """
    # 读取原始标注
    with open(source_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    stats = {"updated": 0, "skipped": 0}

    # 按shape_index排序
    annotations.sort(key=lambda x: x[1])

    # 更新每个shape
    for instance_id, shape_index, anno_data in annotations:
        if shape_index >= len(data["shapes"]):
            print(f"[WARN] Shape index {shape_index} out of range in {source_json_path.name}")
            stats["skipped"] += 1
            continue

        shape = data["shapes"][shape_index]

        # 提取图结构
        atoms = anno_data.get("atoms", [])
        bonds = anno_data.get("bonds", [])

        # 从图结构生成SMILES
        smiles = graph_to_smiles(atoms, bonds, anno_data.get("brackets", []))

        # 更新shape（添加图结构字段）
        shape["atoms"] = atoms
        shape["bonds"] = bonds
        shape["smiles"] = smiles

        stats["updated"] += 1

    # 保存到输出目录
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return stats


def main() -> int:
    """主流程：将所有精修标注回写到原始格式。"""

    _configure_console()

    global ANNOTATOR_ROOT, CORRECTED_DIR, ANNOTATIONS_DIR, MAPPING_FILE
    global SOURCE_JSON_DIR, OUTPUT_JSON_DIR, REVIEW_STATUS_FILE, FILTER_APPROVED_PAGES

    parser = argparse.ArgumentParser(description="将图结构标注回写到 LabelMe JSON")
    parser.add_argument("--data-dir", type=Path, default=ANNOTATOR_ROOT,
                        help="标注器数据目录")
    parser.add_argument("--seg-root", type=Path, default=SEG_ROOT,
                        help="原始分割数据根目录；也可设置 MARKUSH_SEG_ROOT")
    parser.add_argument("--source-json-dir", type=Path,
                        help="原始 LabelMe JSON 目录")
    parser.add_argument("--output-dir", type=Path,
                        help="导出目录（默认与源目录同级的 PageToSMILES_approved）")
    parser.add_argument("--all-pages", action="store_true",
                        help="导出全部页面，不按 review_status 筛选")
    parser.add_argument("--check", action="store_true", help="仅检查输入，不写文件")
    args, _ = parser.parse_known_args()

    ANNOTATOR_ROOT = args.data_dir.expanduser().resolve()
    CORRECTED_DIR = ANNOTATOR_ROOT / "corrected_annotations"
    ANNOTATIONS_DIR = ANNOTATOR_ROOT / "annotations"
    MAPPING_FILE = ANNOTATOR_ROOT / "mapping.json"
    REVIEW_STATUS_FILE = ANNOTATOR_ROOT / "review_status.jsonl"

    seg_root = args.seg_root.expanduser().resolve()
    SOURCE_JSON_DIR = (
        args.source_json_dir or seg_root / "json" / "PageToSMILES"
    ).expanduser().resolve()
    OUTPUT_JSON_DIR = (
        args.output_dir or SOURCE_JSON_DIR.parent / "PageToSMILES_approved"
    ).expanduser().resolve()
    FILTER_APPROVED_PAGES = not args.all_pages

    print("=" * 70)
    print("步骤3: 图结构标注导出工具")
    print("=" * 70)

    # 检查目录
    print(f"\n检查输入目录...")
    print(f"  精修标注: {CORRECTED_DIR}")
    print(f"  映射文件: {MAPPING_FILE}")
    print(f"  原始标注: {SOURCE_JSON_DIR}")

    if not MAPPING_FILE.is_file():
        print(f"\n[ERROR] 映射文件不存在: {MAPPING_FILE}")
        return 2
    if not SOURCE_JSON_DIR.is_dir():
        print(f"\n[ERROR] 原始标注目录不存在: {SOURCE_JSON_DIR}")
        return 2
    if not ANNOTATIONS_DIR.is_dir() and not CORRECTED_DIR.is_dir():
        print(f"\n[ERROR] annotations 和 corrected_annotations 均不存在: {ANNOTATOR_ROOT}")
        return 2
    if FILTER_APPROVED_PAGES and not REVIEW_STATUS_FILE.is_file():
        print(f"\n[ERROR] 审核状态文件不存在: {REVIEW_STATUS_FILE}")
        print("可补齐该文件，或使用 --all-pages 导出全部页面")
        return 2
    if OUTPUT_JSON_DIR == SOURCE_JSON_DIR:
        print("\n[ERROR] 输出目录不能与原始 JSON 目录相同，以免覆盖源数据")
        return 2

    # 统计已标注文件
    corrected_files = list(CORRECTED_DIR.glob("*.json")) if CORRECTED_DIR.is_dir() else []
    if len(corrected_files) == 0:
        print("  [WARN] 没有精修标注，将使用 annotations/ 中的预标注")
    else:
        print(f"  找到 {len(corrected_files)} 个精修标注文件")

    # 加载映射
    print(f"\n加载映射文件: {MAPPING_FILE}")
    mapping = load_mapping()
    print(f"  找到 {len(mapping)} 个实例映射")
    if args.check:
        print("[OK] 映射、标注、源 JSON 和审核状态检查通过")
        return 0

    # 按原始文件分组
    print("\n收集精修后的标注...")
    grouped = collect_annotations_by_source(mapping)
    print(f"  涉及 {len(grouped)} 个原始文件")

    # 只保留审核通过的页面
    if FILTER_APPROVED_PAGES:
        print("\n筛选审核通过的页面...")
        selected_files = load_selected_source_files(mapping)
        grouped = {sf: annos for sf, annos in grouped.items() if sf in selected_files}
        print(f"  过滤后: {len(grouped)} 个原始文件")

    # 统计
    total_updated = 0
    total_skipped = 0

    # 处理每个原始文件
    print("\n回写标注到原始格式...")
    for source_file, annotations in grouped.items():
        source_path = SOURCE_JSON_DIR / source_file
        output_path = OUTPUT_JSON_DIR / source_file

        if not source_path.exists():
            print(f"[WARN] Source file not found: {source_path}")
            continue

        stats = merge_annotation_to_labelme(source_path, annotations, output_path)
        total_updated += stats["updated"]
        total_skipped += stats["skipped"]

        print(f"  {source_file}: {stats['updated']} updated, {stats['skipped']} skipped")

    # 最终报告
    print("\n" + "=" * 70)
    print("✓ 步骤3完成！")
    print("=" * 70)
    return 0
    print(f"  输入目录:     {CORRECTED_DIR}")
    print(f"  输出目录:     {OUTPUT_JSON_DIR}")
    print(f"  已更新:       {total_updated} 个分子")
    print(f"  已跳过:       {total_skipped} 个分子")
    print("=" * 70)

    # 提示
    print("\n输出格式说明：")
    print("  每个shape新增字段：")
    print("    - atoms:  精修后的原子列表")
    print("    - bonds:  精修后的化学键列表")
    print("    - smiles: 从图结构生成的SMILES (使用RDKit)")
    print("\n数据标注流程全部完成！")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())
