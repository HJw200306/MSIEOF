"""根据人工标注 GT 框生成 MolScribe 输入裁剪图。

裁剪窗口在人工紧包围框四周增加少量上下文，但实例清单中的检测框仍记录原始
GT 框。因此后续 IoU 匹配必然是一一对应，这一基线只衡量 MolScribe 识别性能，
不包含自动检测/分割造成的漏检。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

from PIL import Image, ImageDraw, ImageFilter

from utils.evaluation.table5_common import (load_page_checkpoints, read_jsonl,
                                            validate_dataset, write_jsonl,
                                            write_page_checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="表5人工标注裁剪图生成器")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="LabelMe人工标注JSON目录，用于多边形实例掩膜")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=20,
                        help="人工紧包围框四周保留的页面上下文像素，默认20")
    parser.add_argument("--mask-margin", type=int, default=3,
                        help="人工多边形掩膜向外保留的像素，防止边缘线条被截断，默认3")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.padding < 0:
        parser.error("--padding 不能小于0")
    if args.mask_margin < 0:
        parser.error("--mask-margin 不能小于0")

    pages = read_jsonl(args.gt)
    if args.max_pages:
        pages = pages[:args.max_pages]
    validate_dataset(pages, allow_subset=bool(args.max_pages))
    completed = load_page_checkpoints(args.output) if args.resume else {}
    crops_root = args.output.parent / "crops"
    for page_no, page in enumerate(pages, 1):
        if page["page_id"] in completed:
            print(f"[{page_no}/{len(pages)}] resume {page['page_id']}", flush=True)
            continue
        page_path = args.images / page["page_file"]
        if not page_path.is_file():
            raise FileNotFoundError(page_path)
        annotation_path = args.annotations / f"{page['page_id']}.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(annotation_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        polygons = [shape.get("points", []) for shape in annotation.get("shapes", [])
                    if shape.get("label") == "mol"]
        if len(polygons) != len(page.get("molecules", [])):
            raise ValueError(
                f"人工多边形与GT数量不一致: {page['page_id']} "
                f"polygon={len(polygons)} GT={len(page.get('molecules', []))}")
        with Image.open(page_path) as source:
            image = source.convert("RGB")
            width, height = image.size
            instances = []
            for index, molecule in enumerate(page.get("molecules", [])):
                raw = [float(x) for x in molecule["bbox_xyxy"]]
                x1 = max(0, int(raw[0]) - args.padding)
                y1 = max(0, int(raw[1]) - args.padding)
                x2 = min(width, int(raw[2] + 0.999999) + args.padding)
                y2 = min(height, int(raw[3] + 0.999999) + args.padding)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(f"无效GT框: {page['page_id']} {molecule.get('molecule_id')} {raw}")
                crop_file = crops_root / page["page_id"] / f"{page['page_id']}_{index:03d}.png"
                crop_file.parent.mkdir(parents=True, exist_ok=True)
                crop = image.crop((x1, y1, x2, y2))
                local_polygon = [(float(x) - x1, float(y) - y1)
                                 for x, y in polygons[index]]
                if len(local_polygon) < 3:
                    raise ValueError(
                        f"无效人工多边形: {page['page_id']} {molecule.get('molecule_id')}")
                mask = Image.new("L", crop.size, 0)
                ImageDraw.Draw(mask).polygon(local_polygon, fill=255)
                if args.mask_margin:
                    size = args.mask_margin * 2 + 1
                    mask = mask.filter(ImageFilter.MaxFilter(size))
                white = Image.new("RGB", crop.size, "white")
                Image.composite(crop, white, mask).save(crop_file)
                instances.append({
                    "bbox": raw,
                    "crop_bbox_xyxy": [x1, y1, x2, y2],
                    "crop_file": str(crop_file.resolve()),
                    "gt_molecule_id": molecule.get("molecule_id"),
                    "source": "manual_annotation_polygon",
                    "padding": args.padding,
                    "mask_margin": args.mask_margin,
                })
        row = {"page_id": page["page_id"], "page_file": page["page_file"],
               "instances": instances}
        completed[page["page_id"]] = row
        write_page_checkpoint(args.output, row)
        print(f"[{page_no}/{len(pages)}] {page['page_id']}: {len(instances)}", flush=True)
    rows = [completed[page["page_id"]] for page in pages]
    write_jsonl(args.output, rows)
    print(f"人工标注裁剪完成: {len(rows)} 页 -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
