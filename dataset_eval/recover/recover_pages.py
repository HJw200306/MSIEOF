# -*- coding: utf-8 -*-
"""
按 benchmark seg/boxes 标注，渲染对应 PDF 整页 PNG，并产出两套标注：
  1) YOLO 归一化 bbox txt（适配 YOLOv11 实例分割/检测训练或评测）
  2) mask PNG（与页面同尺寸，前景=255，背景=0；适配 UNet++ 二分类分割）

读取：
  E:/其他文章及数据集/PatCID_chemical_structures_in_patent_documents/
      benchmark_D2C-{RND,UNI}/data/seg/boxes/<office>/<stem>.json
  PDF 路径优先级（见 make_pdf_download_list.find_existing 的逻辑）：
      pdfs_benchmark/<office>/<pub_id>.pdf
      pdfs/<pub_id>.pdf
      pdfs/<legacy_subdir>/<pub_id>.pdf  (cnipa/uspto/epo/jpo/kipo)
      restored/<office>/<pub_id>.pdf

输出：
  restored/<subset>/<office>/<stem>.png            完整页面图
  restored/<subset>/<office>/<stem>.yolo.txt       YOLO bbox（class cx cy w h，归一化）
  restored/<subset>/<office>/<stem>.mask.png       二值 mask（页面同尺寸，bool→0/255）

类别约定：当前所有 seg 标签都属于「化学结构」目标，统一记为 class 0。
若以后需要区分 Markush / Clean，可改 CLS_MAP 表。

DPI=300 与 benchmark 的 original_width/height=2480×3509 严格对齐（A4@300DPI）。
对于非 A4 PDF（少量），用 image.size 与 original_width/height 之差做一次缩放，再写
出实际像素坐标。YOLO 归一化时再除以 image.size。

依赖：pip install pdf2image pillow numpy ； Windows 上还需要 poppler 可执行文件。
若 poppler 不在 PATH，可在命令行加：--poppler "<poppler\bin>" 或在环境变量中设置 POPPLER_PATH。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw

try:
    from pdf2image import convert_from_path
except ImportError as e:
    print("[FATAL] pdf2image not installed. Run: pip install pdf2image pillow numpy", file=sys.stderr)
    raise

sys.path.insert(0, str(Path(__file__).parent))
from decode_filename import decode_stem  # noqa: E402


BENCHMARK_ROOT = Path(r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents")
PDF_ROOT = BENCHMARK_ROOT / "pdfs_benchmark"
LEGACY_PDF_ROOT = BENCHMARK_ROOT / "pdfs"
RESTORED_ROOT = BENCHMARK_ROOT / "restored"
RESOLVED_MAPPING_CSV = BENCHMARK_ROOT / "pdf_resolved_mapping.csv"

OFFICE_TO_LEGACY = {"cn": "cnipa", "us": "uspto", "eu": "epo", "jp": "jpo", "kr": "kipo"}

# input_publication_id -> local PDF path (filled lazily on first lookup)
_RESOLVED_MAP: Optional[dict[str, Path]] = None


def _load_resolved_map() -> dict[str, Path]:
    global _RESOLVED_MAP
    if _RESOLVED_MAP is not None:
        return _RESOLVED_MAP
    table: dict[str, Path] = {}
    if RESOLVED_MAPPING_CSV.exists():
        with RESOLVED_MAPPING_CSV.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                key = (row.get("input_publication_id") or "").strip()
                local = (row.get("local_pdf_path") or "").strip()
                if key and local:
                    table[key] = Path(local)
    _RESOLVED_MAP = table
    return table

# 所有 seg 标签都视作 class 0 的化学结构；保留映射方便未来扩展
CLS_MAP = {
    "Mol1": 0, "Mol2": 0, "Mol3": 0, "Mol4": 0, "Mol5": 0,
    "markush_structure": 0, "standard_molecule": 0,
}


@dataclass
class RecoverStat:
    total: int = 0
    rendered: int = 0
    no_pdf: int = 0
    pdf_open_fail: int = 0
    page_missing: int = 0
    no_box: int = 0
    other_error: int = 0


def find_pdf(pub_id: str, office: str) -> Optional[Path]:
    candidates = [
        PDF_ROOT / office / f"{pub_id}.pdf",
        LEGACY_PDF_ROOT / f"{pub_id}.pdf",
        LEGACY_PDF_ROOT / OFFICE_TO_LEGACY.get(office, office) / f"{pub_id}.pdf",
        RESTORED_ROOT / office / f"{pub_id}.pdf",
    ]
    for p in candidates:
        if p.exists():
            return p

    resolved = _load_resolved_map().get(pub_id)
    if resolved and resolved.exists():
        return resolved
    return None


def to_pixel_box(value: dict, w: int, h: int) -> Optional[tuple[float, float, float, float]]:
    """Label Studio rectanglelabels: x,y,width,height 是百分比（0-100），
    相对左上角；返回 (x_min, y_min, x_max, y_max) 像素值。"""
    if "x" in value and "width" in value:
        x = value["x"] / 100.0 * w
        y = value["y"] / 100.0 * h
        ww = value["width"] / 100.0 * w
        hh = value["height"] / 100.0 * h
        return (x, y, x + ww, y + hh)
    pts = value.get("points")
    if pts:
        xs = [p[0] / 100.0 * w for p in pts]
        ys = [p[1] / 100.0 * h for p in pts]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def polygon_pixels(value: dict, w: int, h: int) -> Optional[list[tuple[float, float]]]:
    pts = value.get("points")
    if not pts:
        return None
    return [(p[0] / 100.0 * w, p[1] / 100.0 * h) for p in pts]


def render_one(
    json_path: Path,
    subset: str,
    office: str,
    dpi: int,
    poppler_path: Optional[str],
    overwrite: bool,
    write_visualization: bool,
    stat: RecoverStat,
) -> None:
    stem = json_path.stem
    info = decode_stem(stem)
    if info is None:
        stat.other_error += 1
        print(f"[SKIP] decode fail: {stem}")
        return

    pub_id = info["publication_id"]
    page_no = info["page"]

    pdf_path = find_pdf(pub_id, office)
    if pdf_path is None:
        stat.no_pdf += 1
        return

    out_dir = RESTORED_ROOT / subset / office
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}.png"
    out_yolo = out_dir / f"{stem}.yolo.txt"
    out_mask = out_dir / f"{stem}.mask.png"

    if (not overwrite) and out_png.exists() and out_yolo.exists() and out_mask.exists():
        stat.rendered += 1
        return

    try:
        pages = convert_from_path(
            str(pdf_path), dpi=dpi, first_page=page_no, last_page=page_no,
            poppler_path=poppler_path,
        )
    except Exception:
        stat.pdf_open_fail += 1
        print(f"[ERR ] PDF open: {pdf_path.name}\n{traceback.format_exc()}")
        return
    if not pages:
        stat.page_missing += 1
        print(f"[ERR ] page {page_no} missing: {pdf_path.name}")
        return

    img = pages[0]
    W, H = img.size

    raw = json.loads(json_path.read_text(encoding="utf-8"))
    results = raw.get("result") or []
    if not results:
        stat.no_box += 1
        # 仍然保留页面图，便于检查
        img.save(out_png)
        out_yolo.write_text("", encoding="utf-8")
        Image.new("L", (W, H), 0).save(out_mask)
        return

    yolo_lines: list[str] = []
    mask = Image.new("L", (W, H), 0)
    mdraw = ImageDraw.Draw(mask)
    vis = img.copy() if write_visualization else None
    vdraw = ImageDraw.Draw(vis, "RGBA") if write_visualization else None

    for r in results:
        v = r.get("value") or {}
        label_list = v.get("rectanglelabels") or v.get("polygonlabels") or []
        label = label_list[0] if label_list else ""
        cls = CLS_MAP.get(label, 0)

        # bbox 像素 (YOLO 用)
        bbox = to_pixel_box(v, W, H)
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            yolo_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # mask: 多边形优先填多边形，否则填 bbox
        poly = polygon_pixels(v, W, H)
        if poly is not None and len(poly) >= 3:
            mdraw.polygon(poly, fill=255)
            if vdraw is not None:
                vdraw.polygon(poly, outline=(255, 0, 0, 255), fill=(255, 0, 0, 60))
        elif bbox is not None:
            mdraw.rectangle(bbox, fill=255)
            if vdraw is not None:
                vdraw.rectangle(bbox, outline=(255, 0, 0, 255), width=4)

    img.save(out_png)
    out_yolo.write_text("\n".join(yolo_lines), encoding="utf-8")
    mask.save(out_mask)
    if write_visualization and vis is not None:
        (out_dir / f"{stem}.vis.png").parent.mkdir(parents=True, exist_ok=True)
        vis.save(out_dir / f"{stem}.vis.png")
    stat.rendered += 1


def iter_seg_jsons(subsets: Iterable[str], offices: Iterable[str]) -> Iterable[tuple[Path, str, str]]:
    for subset in subsets:
        root = BENCHMARK_ROOT / subset / "data" / "seg" / "boxes"
        if not root.exists():
            continue
        for office in offices:
            sub = root / office
            if not sub.exists():
                continue
            for jp in sorted(sub.glob("*.json")):
                yield jp, subset, office


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+",
                    default=["benchmark_D2C-RND", "benchmark_D2C-UNI"])
    ap.add_argument("--offices", nargs="+", default=["cn", "us", "eu", "jp", "kr"])
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--poppler", default=os.environ.get("POPPLER_PATH"),
                    help="poppler 的 bin 目录（Windows 下若 poppler 不在 PATH 必须给出）")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--vis", action="store_true", help="同时输出带框可视化 PNG")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个，0 表示不限制")
    args = ap.parse_args()

    stat = RecoverStat()
    todo = list(iter_seg_jsons(args.subsets, args.offices))
    if args.limit:
        todo = todo[: args.limit]
    stat.total = len(todo)

    for i, (jp, subset, office) in enumerate(todo, 1):
        render_one(jp, subset, office, args.dpi, args.poppler,
                   args.overwrite, args.vis, stat)
        if i % 25 == 0 or i == len(todo):
            print(f"[{i}/{len(todo)}] rendered={stat.rendered} "
                  f"no_pdf={stat.no_pdf} pdf_fail={stat.pdf_open_fail} "
                  f"no_box={stat.no_box} err={stat.other_error}")

    print("\n=== Summary ===")
    print(f"total           : {stat.total}")
    print(f"rendered ok     : {stat.rendered}")
    print(f"no pdf          : {stat.no_pdf}")
    print(f"pdf open fail   : {stat.pdf_open_fail}")
    print(f"page missing    : {stat.page_missing}")
    print(f"no box (still ok): {stat.no_box}")
    print(f"decode error    : {stat.other_error}")
    print(f"\nOutput root: {RESTORED_ROOT}")


if __name__ == "__main__":
    main()
