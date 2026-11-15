"""Unified four-class typed post-processing experiment on approved molecule crops.

The production model is not changed.  This script compares raw argmax removal,
the historical solid/dashed post-processors, and probability-aware typed
post-processing with a soft chemical-bond guard.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from arrow_eraser.common.visualize_dual_union_on_pages import (  # noqa: E402
    DEFAULT_INPUT,
    DEFAULT_PAGE_STEMS,
    _dilate_boolean,
    annotated_crop_boxes,
    find_gt_mask,
    image_files,
    load_gt_bond,
    original_crop_montage,
    original_crop_panel,
)
from utils.pipeline.arrow_sr import (  # noqa: E402
    ArrowSRProcessor,
    UNIFIED_ARROW_MODEL_PATH,
    UNIFIED_SIZE_DIVISOR,
    _UNIFIED_NORMALIZER,
)


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "unified_typed_postprocess_experiment"
RECOMMENDED_STRATEGY = "legacy_typed_v4_25"
COLORS = {
    "erase": np.array([190, 40, 235], dtype=np.float32),
    "correct_guard": np.array([0, 190, 70], dtype=np.float32),
    "wrong_guard": np.array([235, 45, 45], dtype=np.float32),
    "gt_damage": np.array([35, 105, 255], dtype=np.float32),
}


STRATEGIES = {
    "baseline_raw": {"kind": "baseline"},
    "legacy_typed_v3_25": {
        "kind": "legacy", "dashed_strategy": "v3_25", "solid_halo_iterations": 3,
    },
    "legacy_typed_v4_25": {
        "kind": "legacy", "dashed_strategy": "v4_25", "solid_halo_iterations": 3,
    },
    "typed_conservative": {
        "kind": "probability", "solid_threshold": 0.40,
        "dashed_threshold": 0.40, "bond_threshold": 0.30,
        "close_kernel": 3,
    },
    "typed_balanced": {
        "kind": "probability", "solid_threshold": 0.30,
        "dashed_threshold": 0.30, "bond_threshold": 0.30,
        "close_kernel": 3,
    },
    "typed_recall": {
        "kind": "probability", "solid_threshold": 0.20,
        "dashed_threshold": 0.25, "bond_threshold": 0.25,
        "close_kernel": 5,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare typed post-processing strategies for the unified model."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--gt-json-dir", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=UNIFIED_ARROW_MODEL_PATH)
    parser.add_argument("--pages", nargs="*", default=DEFAULT_PAGE_STEMS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gt-tolerance", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.72)
    return parser.parse_args()


@torch.inference_mode()
def predict_probabilities(
    processor: ArrowSRProcessor, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = rgb.shape[:2]
    pad_h, pad_w = (-height) % UNIFIED_SIZE_DIVISOR, (-width) % UNIFIED_SIZE_DIVISOR
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(
            rgb, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    else:
        image = rgb
    normalized = _UNIFIED_NORMALIZER(image=image)["image"]
    tensor = torch.from_numpy(
        normalized.transpose(2, 0, 1).astype(np.float32)
    ).unsqueeze(0).to(processor.device)
    logits = processor.unified_model(tensor)[0, :, :height, :width]
    probabilities = logits.softmax(dim=0).cpu().numpy().astype(np.float32)
    classes = probabilities.argmax(axis=0).astype(np.uint8)
    return classes, probabilities


def historical_solid(
    rgb: np.ndarray, raw: np.ndarray, halo_iterations: int = 3
) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    halo = cv2.dilate(
        raw.astype(np.uint8), kernel, iterations=halo_iterations
    ).astype(bool)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    return raw | (halo & (saturation > 30))


def historical_dashed(
    raw: np.ndarray, probability: np.ndarray, strategy: str
) -> np.ndarray:
    expanded = raw | (probability > 0.25)
    if strategy == "v3_25":
        return expanded
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(expanded.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(closed, dilate_kernel, iterations=1).astype(bool)


def strategy_masks(
    rgb: np.ndarray,
    classes: np.ndarray,
    probabilities: np.ndarray,
    config: dict[str, object],
) -> dict[str, np.ndarray]:
    raw_solid, raw_dashed, raw_bond = classes == 1, classes == 2, classes == 3
    kind = str(config["kind"])
    if kind == "baseline":
        solid, dashed, guard = raw_solid, raw_dashed, raw_bond
    elif kind == "legacy":
        solid = historical_solid(
            rgb, raw_solid, int(config["solid_halo_iterations"])
        )
        dashed = historical_dashed(
            raw_dashed, probabilities[2], str(config["dashed_strategy"])
        )
        guard = raw_bond
    else:
        ink = np.min(rgb, axis=2) < 235
        halo_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        solid_halo = cv2.dilate(raw_solid.astype(np.uint8), halo_kernel, iterations=1).astype(bool)
        solid = raw_solid | (
            solid_halo & ink & (probabilities[1] >= float(config["solid_threshold"]))
        )
        dashed_seed = raw_dashed | (
            ink & (probabilities[2] >= float(config["dashed_threshold"]))
        )
        close_size = int(config["close_kernel"])
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)
        )
        dashed_closed = cv2.morphologyEx(
            dashed_seed.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        )
        dashed = cv2.dilate(dashed_closed, halo_kernel, iterations=1).astype(bool) & ink
        raw_guard = cv2.dilate(raw_bond.astype(np.uint8), halo_kernel, iterations=1).astype(bool)
        soft_guard = ink & (probabilities[3] >= float(config["bond_threshold"]))
        guard = raw_guard | soft_guard
    candidate = solid | dashed
    erased = candidate & ~guard
    return {
        "solid": solid,
        "dashed": dashed,
        "candidate": candidate,
        "guard": guard,
        "erased": erased,
    }


def measure(
    masks: dict[str, np.ndarray], gt: np.ndarray, valid: np.ndarray, tolerance: float
) -> dict[str, int | float]:
    tolerated_gt = _dilate_boolean(gt & valid, tolerance)
    guard = masks["guard"] & valid
    erased = masks["erased"] & valid
    protected = masks["candidate"] & guard & valid
    correct_guard = guard & tolerated_gt
    wrong_guard = guard & ~tolerated_gt
    correct_protected = protected & tolerated_gt
    wrong_protected = protected & ~tolerated_gt
    gt_damage = erased & gt & valid
    outside_erased = erased & ~tolerated_gt
    precision = 100.0 * int(correct_guard.sum()) / int(guard.sum()) if guard.any() else 0.0
    gt_coverage = 100.0 * int((gt & _dilate_boolean(guard, tolerance)).sum()) / int(gt.sum()) if gt.any() else 0.0
    f1 = 2 * precision * gt_coverage / (precision + gt_coverage) if precision + gt_coverage else 0.0
    return {
        "candidate_pixels": int(masks["candidate"].sum()),
        "erased_pixels": int(erased.sum()),
        "solid_erased_pixels": int((masks["solid"] & ~guard & valid).sum()),
        "dashed_erased_pixels": int((masks["dashed"] & ~guard & valid).sum()),
        "guard_pixels": int(guard.sum()),
        "guard_correct_pixels": int(correct_guard.sum()),
        "guard_wrong_pixels": int(wrong_guard.sum()),
        "guard_precision_percent": precision,
        "guard_gt_coverage_percent": gt_coverage,
        "guard_f1_percent": f1,
        "correctly_protected_pixels": int(correct_protected.sum()),
        "wrongly_protected_pixels": int(wrong_protected.sum()),
        "gt_damage_pixels": int(gt_damage.sum()),
        "outside_gt_erased_pixels": int(outside_erased.sum()),
    }


def add_metrics(rows: list[dict[str, int | float]]) -> dict[str, int | float]:
    count_fields = [
        "candidate_pixels", "erased_pixels", "solid_erased_pixels",
        "dashed_erased_pixels", "guard_pixels", "guard_correct_pixels",
        "guard_wrong_pixels", "correctly_protected_pixels",
        "wrongly_protected_pixels", "gt_damage_pixels", "outside_gt_erased_pixels",
    ]
    result = {field: sum(int(row[field]) for row in rows) for field in count_fields}
    guard = int(result["guard_pixels"])
    result["guard_precision_percent"] = (
        100.0 * int(result["guard_correct_pixels"]) / guard if guard else 0.0
    )
    # GT coverage and F1 are recomputed from per-crop GT totals by the caller.
    return result


def blend(rgb: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color * alpha


def result_panel(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    gt: np.ndarray,
    valid: np.ndarray,
    metrics: dict[str, int | float],
    tolerance: float,
    crop_index: int,
    alpha: float,
    width: int = 420,
    height: int = 300,
) -> np.ndarray:
    tolerated_gt = _dilate_boolean(gt & valid, tolerance)
    correct_guard = masks["guard"] & tolerated_gt & valid
    wrong_guard = masks["guard"] & ~tolerated_gt & valid
    gt_damage = masks["erased"] & gt & valid
    safe_erase = masks["erased"] & ~gt_damage & valid
    # Show the real output image: erased pixels are white before the diagnostic
    # colours are overlaid.  This makes residual arrows and bond breaks visible.
    cleaned = rgb.copy()
    cleaned[masks["erased"] & valid] = 255
    visual = cleaned.astype(np.float32)
    blend(visual, safe_erase, COLORS["erase"], alpha)
    blend(visual, correct_guard, COLORS["correct_guard"], alpha)
    blend(visual, wrong_guard, COLORS["wrong_guard"], alpha)
    blend(visual, gt_damage, COLORS["gt_damage"], alpha)
    visual = np.clip(visual, 0, 255).astype(np.uint8)
    panel = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(panel, f"crop {crop_index:02d}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (25, 25, 25), 1, cv2.LINE_AA)
    text = (
        f"erase {metrics['erased_pixels']:,} | bond damage {metrics['gt_damage_pixels']:,} | "
        f"wrong protect {metrics['wrongly_protected_pixels']:,}"
    )
    cv2.putText(panel, text, (10, 41), cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (55, 55, 55), 1, cv2.LINE_AA)
    top, bottom = 53, height - 8
    scale = min((width - 16) / visual.shape[1], (bottom - top) / visual.shape[0])
    size = (max(1, round(visual.shape[1] * scale)), max(1, round(visual.shape[0] * scale)))
    resized = cv2.resize(visual, size, interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA)
    x, y = (width - size[0]) // 2, top + (bottom - top - size[1]) // 2
    panel[y:y + size[1], x:x + size[0]] = resized
    cv2.rectangle(panel, (x, y), (x + size[0] - 1, y + size[1] - 1), (200, 200, 200), 1)
    return panel


def cleaned_panel(
    rgb: np.ndarray,
    masks: dict[str, np.ndarray],
    valid: np.ndarray,
    metrics: dict[str, int | float],
    crop_index: int,
    width: int = 420,
    height: int = 300,
) -> np.ndarray:
    """Render the actual white-filled removal result without mask overlays."""
    cleaned = rgb.copy()
    cleaned[masks["erased"] & valid] = 255
    panel = np.full((height, width, 3), 255, np.uint8)
    cv2.putText(panel, f"crop {crop_index:02d}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.50, (25, 25, 25), 1, cv2.LINE_AA)
    text = (
        f"removed {metrics['erased_pixels']:,} | "
        f"GT foreground hit {metrics['gt_damage_pixels']:,}"
    )
    cv2.putText(panel, text, (10, 41), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (55, 55, 55), 1, cv2.LINE_AA)
    top, bottom = 53, height - 8
    scale = min((width - 16) / cleaned.shape[1], (bottom - top) / cleaned.shape[0])
    size = (max(1, round(cleaned.shape[1] * scale)),
            max(1, round(cleaned.shape[0] * scale)))
    resized = cv2.resize(
        cleaned, size, interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    )
    x, y = (width - size[0]) // 2, top + (bottom - top - size[1]) // 2
    panel[y:y + size[1], x:x + size[0]] = resized
    cv2.rectangle(panel, (x, y), (x + size[0] - 1, y + size[1] - 1),
                  (200, 200, 200), 1)
    return panel


def montage(
    strategy: str,
    page_groups: list[dict[str, object]],
    summary: dict[str, object],
    diagnostic: bool = True,
) -> np.ndarray:
    columns, panel_w, panel_h, gap, margin = 4, 420, 300, 14, 24
    header_h, page_header_h = 130, 42
    width = margin * 2 + columns * panel_w + (columns - 1) * gap
    group_heights = []
    for group in page_groups:
        rows = (len(group["panels"]) + columns - 1) // columns
        group_heights.append(page_header_h + rows * panel_h + max(0, rows - 1) * gap + gap)
    canvas = np.full((header_h + sum(group_heights) + margin, width, 3), 245, np.uint8)
    view_name = "diagnostic overlay" if diagnostic else "actual cleaned result"
    cv2.putText(canvas, f"Unified typed postprocess: {strategy} ({view_name})", (margin, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 20, 20), 2, cv2.LINE_AA)
    legend = (
        "white=actual removal  purple=safe erase  green=correct bond guard  red=wrong guard  blue=GT structure erased"
        if diagnostic else
        "No colour overlay: white pixels are the actual removal result; inspect arrow residue and broken bonds directly"
    )
    cv2.putText(canvas, legend,
                (margin, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (45, 45, 45), 1, cv2.LINE_AA)
    line = (
        f"erased={summary['erased_pixels']:,}  GT damage={summary['gt_damage_pixels']:,}  "
        f"correct protect={summary['correctly_protected_pixels']:,}  "
        f"wrong protect={summary['wrongly_protected_pixels']:,}"
    )
    cv2.putText(canvas, line, (margin, 99), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (25, 25, 25), 1, cv2.LINE_AA)
    y = header_h
    for group, group_h in zip(page_groups, group_heights):
        cv2.putText(canvas, f"{group['page']}   crops={len(group['panels'])}",
                    (margin, y + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (25, 25, 25), 1, cv2.LINE_AA)
        y += page_header_h
        for index, panel in enumerate(group["panels"]):
            row, column = divmod(index, columns)
            x, py = margin + column * (panel_w + gap), y + row * (panel_h + gap)
            canvas[py:py + panel_h, x:x + panel_w] = panel
        y += group_h - page_header_h
    return canvas


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    gt_dir = args.gt_dir or args.input.parents[1] / "masks" / args.input.name
    json_dir = args.gt_json_dir or args.input.parents[1] / "json" / args.input.name
    if not gt_dir.is_dir() or not json_dir.is_dir():
        raise FileNotFoundError(f"GT directories missing: masks={gt_dir}; json={json_dir}")
    files_by_stem = {path.stem: path for path in image_files(args.input)}
    missing = [Path(name).stem for name in args.pages if Path(name).stem not in files_by_stem]
    if missing:
        raise FileNotFoundError(f"Missing pages: {missing}")
    files = [files_by_stem[Path(name).stem] for name in args.pages]
    args.output.mkdir(parents=True, exist_ok=True)

    processor = ArrowSRProcessor(
        unified_arrow_model_path=args.model, device=args.device, load_sr_model=False
    )
    raw_pages: list[dict[str, object]] = []
    original_groups: list[dict[str, object]] = []
    start = time.time()
    for page_index, source in enumerate(files, 1):
        relative = source.relative_to(args.input)
        with Image.open(source) as opened:
            page_rgb = np.asarray(opened.convert("RGB"))
        gt_path = find_gt_mask(gt_dir, relative)
        page_gt, page_valid, _ = load_gt_bond(gt_path, page_rgb.shape[:2], {2, 3})
        json_path = json_dir / relative.with_suffix(".json")
        boxes = annotated_crop_boxes(json_path, page_rgb.shape[:2])
        crops, original_panels = [], []
        for crop_index, item in enumerate(boxes):
            x0, y0, x1, y1 = item["bbox"]
            rgb = page_rgb[y0:y1, x0:x1].copy()
            classes, probabilities = predict_probabilities(processor, rgb)
            crops.append({
                "index": crop_index,
                "rgb": rgb,
                "gt": page_gt[y0:y1, x0:x1],
                "valid": page_valid[y0:y1, x0:x1],
                "classes": classes,
                "probabilities": probabilities,
            })
            original_panels.append(original_crop_panel(rgb, crop_index))
        raw_pages.append({"page": source.stem, "crops": crops})
        original_groups.append({"page": source.stem, "panels": original_panels})
        print(f"[{page_index}/{len(files)}] {source.name}: crops={len(crops)}", flush=True)

    Image.fromarray(original_crop_montage(original_groups)).save(args.output / "cropped_originals.png")
    summary_rows, page_rows, crop_rows = [], [], []
    baseline_outside = None
    for strategy, config in STRATEGIES.items():
        strategy_pages, cleaned_pages, all_metrics = [], [], []
        for page in raw_pages:
            panels, cleaned_panels, page_metrics = [], [], []
            for crop in page["crops"]:
                masks = strategy_masks(
                    crop["rgb"], crop["classes"], crop["probabilities"], config
                )
                metrics = measure(masks, crop["gt"], crop["valid"], args.gt_tolerance)
                panels.append(result_panel(
                    crop["rgb"], masks, crop["gt"], crop["valid"], metrics,
                    args.gt_tolerance, int(crop["index"]), args.alpha,
                ))
                cleaned_panels.append(cleaned_panel(
                    crop["rgb"], masks, crop["valid"], metrics, int(crop["index"])
                ))
                crop_rows.append({
                    "strategy": strategy, "page": page["page"],
                    "crop": crop["index"], **metrics,
                })
                page_metrics.append(metrics)
                all_metrics.append(metrics)
            page_total = add_metrics(page_metrics)
            page_rows.append({"strategy": strategy, "page": page["page"], **page_total})
            strategy_pages.append({"page": page["page"], "panels": panels})
            cleaned_pages.append({"page": page["page"], "panels": cleaned_panels})
        total = add_metrics(all_metrics)
        if baseline_outside is None:
            baseline_outside = max(1, int(total["outside_gt_erased_pixels"]))
        total["outside_erase_retention_percent"] = (
            100.0 * int(total["outside_gt_erased_pixels"]) / baseline_outside
        )
        summary_rows.append({"strategy": strategy, **config, **total})
        Image.fromarray(montage(strategy, strategy_pages, total)).save(
            args.output / f"visualization_{strategy}.png"
        )
        Image.fromarray(montage(strategy, cleaned_pages, total, diagnostic=False)).save(
            args.output / f"cleaned_{strategy}.png"
        )
        if strategy == RECOMMENDED_STRATEGY:
            Image.fromarray(montage(strategy, strategy_pages, total)).save(
                args.output / "best_version_diagnostic.png"
            )
            Image.fromarray(
                montage(strategy, cleaned_pages, total, diagnostic=False)
            ).save(args.output / "best_version_cleaned.png")
        print(
            f"{strategy}: erased={total['erased_pixels']:,}; "
            f"GT damage={total['gt_damage_pixels']:,}; "
            f"correct protect={total['correctly_protected_pixels']:,}; "
            f"wrong protect={total['wrongly_protected_pixels']:,}", flush=True
        )

    for filename, rows in (
        ("strategy_summary.csv", summary_rows),
        ("per_page_metrics.csv", page_rows),
        ("per_crop_metrics.csv", crop_rows),
    ):
        with (args.output / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
            writer.writeheader()
            writer.writerows(rows)
    (args.output / "experiment_config.json").write_text(
        json.dumps({
            "model": str(args.model.resolve()), "gt_tolerance": args.gt_tolerance,
            "pages": [path.stem for path in files], "strategies": STRATEGIES,
            "recommended_strategy": RECOMMENDED_STRATEGY,
            "recommendation_basis": (
                "Best visual balance on the selected 19 approved crops: three-iteration "
                "solid HSV halo, dashed v4_25 close+dilate, and unified-model bond protection."
            ),
            "elapsed_seconds": round(time.time() - start, 3),
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"Done: {args.output}", flush=True)


if __name__ == "__main__":
    main()
