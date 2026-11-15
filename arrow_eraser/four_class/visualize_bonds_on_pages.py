"""Visualize four-class model predictions on PageToSMILES page images.

The production four-class model is run in contextual tiles so patent pages can
be processed without resizing or exhausting GPU memory.  All saved masks and
overlays retain the source page resolution.

Class colors (RGB): solid arrow=red, dashed arrow=orange, chemical bond=green.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.pipeline.arrow_sr import (  # noqa: E402
    ArrowSRProcessor,
    UNIFIED_CLASS_NAMES,
    _UNIFIED_NORMALIZER,
)


DEFAULT_INPUT = Path(
    r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal"
    r"\images\PageToSMILES_approved"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "PageToSMILES_approved_bond_visualization"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# RGB colors.  Green is deliberately bright so thin chemical bonds remain visible.
COLOR_SOLID = np.array([235, 45, 45], dtype=np.float32)
COLOR_DASHED = np.array([255, 145, 0], dtype=np.float32)
COLOR_BOND = np.array([0, 210, 70], dtype=np.float32)
PALETTE = [0, 0, 0, 235, 45, 45, 255, 145, 0, 0, 210, 70] + [0] * (256 * 3 - 12)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize chemical bonds predicted by the production four-class model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path,
                        default=PROJECT_ROOT / "model" / "unified_arrow_4class.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only the first N images; 0 processes all images.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_files(root: Path) -> list[Path]:
    return sorted(
        (p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p.relative_to(root)).lower(),
    )


def _starts(length: int, core: int) -> list[int]:
    return list(range(0, length, core)) or [0]


def predict_page(processor: ArrowSRProcessor, rgb: np.ndarray, tile_size: int,
                 context: int, batch_size: int) -> np.ndarray:
    """Return an original-resolution class map using non-overlapping core tiles.

    Each core receives surrounding context before inference.  Only the core is
    copied into the result, which removes seams without averaging class logits.
    """
    if context < 0 or tile_size <= 2 * context:
        raise ValueError("tile-size must be larger than 2 * context")
    if tile_size % 32:
        raise ValueError("tile-size must be divisible by 32")
    core = tile_size - 2 * context
    height, width = rgb.shape[:2]
    result = np.zeros((height, width), dtype=np.uint8)
    jobs: list[tuple[int, int, int, int]] = []
    for y0 in _starts(height, core):
        for x0 in _starts(width, core):
            jobs.append((y0, min(y0 + core, height), x0, min(x0 + core, width)))

    model = processor.unified_model
    device = processor.device
    for offset in range(0, len(jobs), batch_size):
        batch_jobs = jobs[offset:offset + batch_size]
        tensors: list[np.ndarray] = []
        crop_meta: list[tuple[int, int, int, int, int, int]] = []
        for y0, y1, x0, x1 in batch_jobs:
            sy0, sy1 = max(0, y0 - context), min(height, y1 + context)
            sx0, sx1 = max(0, x0 - context), min(width, x1 + context)
            crop = rgb[sy0:sy1, sx0:sx1]
            canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            oy, ox = y0 - sy0, x0 - sx0
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            normed = _UNIFIED_NORMALIZER(image=canvas)["image"]
            tensors.append(normed.transpose(2, 0, 1).astype(np.float32))
            crop_meta.append((y0, y1, x0, x1, oy, ox))
        tensor = torch.from_numpy(np.stack(tensors)).to(device)
        with torch.inference_mode():
            pred = model(tensor).argmax(dim=1).cpu().numpy().astype(np.uint8)
        for classes, (y0, y1, x0, x1, oy, ox) in zip(pred, crop_meta):
            result[y0:y1, x0:x1] = classes[oy:oy + y1 - y0, ox:ox + x1 - x0]
    return result


def make_overlay(rgb: np.ndarray, classes: np.ndarray, alpha: float,
                 bonds_only: bool) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    selections = [(classes == 3, COLOR_BOND)] if bonds_only else [
        (classes == 1, COLOR_SOLID),
        (classes == 2, COLOR_DASHED),
        (classes == 3, COLOR_BOND),
    ]
    for mask, color in selections:
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def save_outputs(source: Path, relative: Path, output: Path, classes: np.ndarray,
                 rgb: np.ndarray, alpha: float) -> dict[str, object]:
    stem = relative.with_suffix("")
    class_path = output / "class_masks" / stem.with_suffix(".png")
    bond_path = output / "bond_masks" / stem.with_suffix(".png")
    overlay_path = output / "overlays" / stem.with_suffix(".png")
    bond_overlay_path = output / "bond_overlays" / stem.with_suffix(".png")
    for path in (class_path, bond_path, overlay_path, bond_overlay_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    class_image = Image.fromarray(classes, mode="P")
    class_image.putpalette(PALETTE)
    class_image.save(class_path)
    Image.fromarray(((classes == 3) * 255).astype(np.uint8)).save(bond_path)
    Image.fromarray(make_overlay(rgb, classes, alpha, bonds_only=False)).save(overlay_path)
    Image.fromarray(make_overlay(rgb, classes, alpha, bonds_only=True)).save(bond_overlay_path)

    counts = np.bincount(classes.ravel(), minlength=4)
    return {
        "source": str(source),
        "relative_path": relative.as_posix(),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "background_pixels": int(counts[0]),
        "solid_arrow_pixels": int(counts[1]),
        "dashed_arrow_pixels": int(counts[2]),
        "bond_pixels": int(counts[3]),
        "class_mask": class_path.relative_to(output).as_posix(),
        "bond_mask": bond_path.relative_to(output).as_posix(),
        "overlay": overlay_path.relative_to(output).as_posix(),
        "bond_overlay": bond_overlay_path.relative_to(output).as_posix(),
    }


def write_index(output: Path, records: list[dict[str, object]], metadata: dict) -> None:
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "pages": records}, handle,
                  ensure_ascii=False, indent=2)
    fields = list(records[0]) if records else ["source"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    cards = []
    for record in records:
        bond_url = quote(str(record["bond_overlay"]), safe="/")
        all_url = quote(str(record["overlay"]), safe="/")
        title = html.escape(str(record["relative_path"]))
        cards.append(
            f'<article><h2>{title}</h2><p>bond pixels: {record["bond_pixels"]:,} · '
            f'solid arrow: {record["solid_arrow_pixels"]:,} · dashed arrow: '
            f'{record["dashed_arrow_pixels"]:,}</p><a href="{bond_url}">'
            f'<img loading="lazy" src="{bond_url}" alt="{title}"></a>'
            f'<p><a href="{bond_url}">化学键叠加原图</a> · '
            f'<a href="{all_url}">四分类完整叠加图</a></p></article>'
        )
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>统一四分类模型：化学键识别可视化</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f5f6f7;color:#17202a}
header{position:sticky;top:0;background:#fff;padding:14px 18px;border-radius:8px;z-index:2;box-shadow:0 2px 10px #0002}
.legend b{display:inline-block;width:14px;height:14px;margin:0 5px 0 16px;vertical-align:-2px}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;margin-top:20px}
article{background:white;padding:14px;border-radius:8px;box-shadow:0 1px 5px #0002;overflow:hidden}
h2{font-size:15px;word-break:break-all}p{font-size:13px}img{width:100%;height:520px;object-fit:contain;background:#eee}
</style></head><body><header><h1>统一四分类模型：真实化学键预测</h1>
<p>绿色＝化学键；红色＝实线箭头；橙色＝虚线箭头。点击图片查看原始分辨率。</p>
<p class="legend"><b style="background:#00d246"></b>化学键 <b style="background:#eb2d2d"></b>实线箭头 <b style="background:#ff9100"></b>虚线箭头</p>
</header><main>""" + "\n".join(cards) + "</main></body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    files = image_files(args.input)
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"No supported images found under: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    processor = ArrowSRProcessor(
        unified_arrow_model_path=args.model,
        device=args.device,
        load_sr_model=False,
    )
    records: list[dict[str, object]] = []
    start = time.time()
    for index, source in enumerate(files, 1):
        relative = source.relative_to(args.input)
        expected = args.output / "bond_overlays" / relative.with_suffix(".png")
        if expected.exists() and not args.overwrite:
            print(f"[{index}/{len(files)}] skip existing: {relative}", flush=True)
            continue
        with Image.open(source) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        classes = predict_page(processor, rgb, args.tile_size, args.context, args.batch_size)
        record = save_outputs(source, relative, args.output, classes, rgb, args.alpha)
        records.append(record)
        print(f"[{index}/{len(files)}] {relative}: bond={record['bond_pixels']:,}px", flush=True)

    # Include earlier records when resuming so the index remains complete.
    old_summary = args.output / "summary.json"
    if old_summary.exists() and not args.overwrite:
        try:
            previous = json.loads(old_summary.read_text(encoding="utf-8")).get("pages", [])
            by_name = {str(item["relative_path"]): item for item in previous}
            by_name.update({str(item["relative_path"]): item for item in records})
            records = [by_name[key] for key in sorted(by_name)]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    metadata = {
        "model": str(args.model.resolve()),
        "class_names": UNIFIED_CLASS_NAMES,
        "input": str(args.input.resolve()),
        "tile_size": args.tile_size,
        "context": args.context,
        "batch_size": args.batch_size,
        "overlay_alpha": args.alpha,
        "processed_pages": len(records),
        "elapsed_seconds_this_run": round(time.time() - start, 3),
    }
    write_index(args.output, records, metadata)
    print(f"Done: {len(records)} pages; index={args.output / 'index.html'}")


if __name__ == "__main__":
    main()
