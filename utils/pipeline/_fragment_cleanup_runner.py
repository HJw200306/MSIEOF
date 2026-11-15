"""Private page/candidate orchestration for the fragment-cleanup stage.

The official stage and public import path are
:mod:`utils.pipeline.fragment_cleanup`.  This private module keeps filesystem,
sidecar and same-page candidate handling separate from the geometric cleanup
algorithm.

This module replaces the former area/distance-only CC filter. The current
algorithm compares each ink component with the target's original YOLO polygon,
the other predictions on the same page, and page-level long-line evidence. It
therefore operates on all candidates from one page together.

The reusable geometric rules live in :mod:`utils.fragment_cleanup`. No GT,
SMILES, filename allowlist, or manually drawn ROI is used for a decision.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from utils.base.paths import PROJECT_ROOT

from utils.pipeline.fragment_cleanup import PARAMETERS, clean_crop, page_lines


INPUT_DIR = PROJECT_ROOT / "crop_dataset"
OUTPUT_DIR = PROJECT_ROOT / "crop_dataset_denoised"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class FragmentCleanupContextError(RuntimeError):
    """Raised when page/neighbor geometry required by the new cleanup is absent."""


def _meta_path(image_path: Path) -> Path:
    return image_path.with_suffix(".meta.npz")


def _load_geometry(candidate: Dict, page_hw: Tuple[int, int]) -> Dict:
    image_path = Path(candidate["crop_path"])
    meta_path = _meta_path(image_path)
    if not meta_path.exists():
        raise FragmentCleanupContextError(f"CC sidecar missing: {meta_path}")

    page_h, page_w = page_hw
    with np.load(meta_path) as data:
        if "polygon" not in data.files:
            raise FragmentCleanupContextError(
                f"original YOLO polygon missing from sidecar: {meta_path}")
        polygon = np.asarray(data["polygon"], dtype=np.float32)
        if "exp_bbox" in candidate:
            exp_bbox = np.rint(candidate["exp_bbox"]).astype(int)
        elif "bbox_norm" in data.files:
            exp_bbox = np.rint(
                data["bbox_norm"] * [page_w, page_h, page_w, page_h]
            ).astype(int)
        else:
            raise FragmentCleanupContextError(
                f"expanded page bbox missing for candidate: {image_path}")

        if "raw_bbox" in candidate:
            raw_bbox = np.rint(candidate["raw_bbox"]).astype(int)
        elif "bbox_raw_norm" in data.files:
            raw_bbox = np.rint(
                data["bbox_raw_norm"] * [page_w, page_h, page_w, page_h]
            ).astype(int)
        else:
            raise FragmentCleanupContextError(
                f"raw YOLO bbox missing for candidate: {image_path}")

    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise FragmentCleanupContextError(
            f"invalid original YOLO polygon in sidecar: {meta_path}")
    x1, y1, x2, y2 = map(int, exp_bbox)
    if not (0 <= x1 < x2 <= page_w and 0 <= y1 < y2 <= page_h):
        raise FragmentCleanupContextError(
            f"expanded bbox outside page for {image_path}: {exp_bbox.tolist()}")

    return {
        "name": image_path.name,
        "image_path": image_path,
        "meta_path": meta_path,
        "polygon": polygon,
        "exp_bbox": exp_bbox,
        "raw_bbox": raw_bbox,
    }


def _save_overlay(rgb: np.ndarray, erase: np.ndarray, destination: Path) -> None:
    overlay = rgb.copy()
    overlay[erase] = [226, 39, 55]
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(destination)


def cleanup_candidates_impl(
    candidates: List[Dict],
    page_rgb: np.ndarray,
    output_dir: Path,
    visualize: bool = False,
) -> List[Dict]:
    """Clean all candidates from one page with cross-instance evidence.

    ``candidates`` must contain ``crop_path`` plus page-coordinate ``raw_bbox``
    and ``exp_bbox`` values, or sidecars containing ``bbox_raw_norm`` and
    ``bbox_norm``. Every sidecar must retain the original YOLO ``polygon``.

    The list and dictionaries are updated in place for compatibility with the
    existing Page-to-SMILES evaluators. Candidate count and bounding boxes never
    change; only selected crop pixels are replaced with white.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not candidates:
        return candidates

    page = np.asarray(page_rgb)
    if page.ndim != 3 or page.shape[2] != 3:
        raise FragmentCleanupContextError(
            f"page_rgb must be an HxWx3 RGB array, got {page.shape}")
    page_h, page_w = page.shape[:2]
    line_mask = page_lines(page)
    geometry = [_load_geometry(candidate, (page_h, page_w))
                for candidate in candidates]

    for candidate, current in zip(candidates, geometry):
        with Image.open(current["image_path"]) as image:
            rgb = np.asarray(image.convert("RGB"))
        x1, y1, x2, y2 = map(int, current["exp_bbox"])
        if rgb.shape[:2] != (y2 - y1, x2 - x1):
            raise FragmentCleanupContextError(
                f"crop/bbox size mismatch for {current['image_path']}: "
                f"crop={rgb.shape[:2]}, bbox={(y2-y1, x2-x1)}")

        neighbors = []
        for other in geometry:
            if other is current:
                continue
            ox1, oy1, _, _ = map(int, other["exp_bbox"])
            neighbors.append({
                "name": other["name"],
                "raw_bbox": other["raw_bbox"].tolist(),
                "polygon": other["polygon"] + [ox1 - x1, oy1 - y1],
            })

        cleaned, erase, evidence = clean_crop(
            rgb,
            current["polygon"],
            current["raw_bbox"].tolist(),
            neighbors,
            line_mask[y1:y2, x1:x2],
        )
        if not np.array_equal(cleaned[~erase], rgb[~erase]):
            raise AssertionError("fragment cleanup changed a retained pixel")
        if erase.any() and not np.all(cleaned[erase] == 255):
            raise AssertionError("fragment cleanup replacement is not pure white")

        out_image = output_dir / current["image_path"].name
        out_meta = output_dir / current["meta_path"].name
        Image.fromarray(cleaned).save(out_image)
        shutil.copy2(current["meta_path"], out_meta)
        if visualize:
            _save_overlay(
                rgb, erase,
                output_dir / "overlays" / current["image_path"].name,
            )

        changed = bool(erase.any())
        candidate["crop_path"] = out_image
        candidate["cleanup_changed"] = changed
        candidate["cleanup_removed_pixels"] = int(erase.sum())
        candidate["cleanup_removed_components"] = len(
            evidence.get("removed_groups", []))
        candidate["cleanup_review_components"] = len(
            evidence.get("review_groups", []))
        candidate["cleanup_evidence"] = evidence

    return candidates


def cleanup_image_impl(
    img_path: Path,
    output_dir: Path,
    *,
    own_polygon=None,
    raw_bbox=None,
    neighbor_predictions: Optional[Sequence[Dict]] = None,
    page_line_mask=None,
    visualize: bool = False,
):
    """Clean one crop when the caller explicitly supplies its page context.

    Production page pipelines should use :func:`cleanup_candidates_impl`. The old
    area/distance-only single-image fallback has intentionally been removed;
    silently running without neighbor context would change the validated rules.
    """
    img_path = Path(img_path)
    meta_path = _meta_path(img_path)
    if own_polygon is None and meta_path.exists():
        with np.load(meta_path) as data:
            if "polygon" in data.files:
                own_polygon = np.asarray(data["polygon"], dtype=np.float32)
            if raw_bbox is None and {
                    "bbox_raw_norm", "page_size"}.issubset(data.files):
                page_h, page_w = map(int, data["page_size"])
                raw_bbox = np.rint(
                    data["bbox_raw_norm"]
                    * [page_w, page_h, page_w, page_h]
                ).astype(int).tolist()
    if own_polygon is None or raw_bbox is None or neighbor_predictions is None:
        raise FragmentCleanupContextError(
            "The updated CC algorithm requires the original target polygon, "
            "raw bbox, and all same-page neighbor predictions. Use "
            "cleanup_candidates(candidates, page_rgb, output_dir).")

    with Image.open(img_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    cleaned, erase, evidence = clean_crop(
        rgb, own_polygon, raw_bbox, list(neighbor_predictions), page_line_mask)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_image = output_dir / img_path.name
    Image.fromarray(cleaned).save(out_image)
    if meta_path.exists():
        shutil.copy2(meta_path, output_dir / meta_path.name)
    if visualize:
        _save_overlay(rgb, erase, output_dir / "overlays" / img_path.name)
    info = {
        **evidence,
        "sidecar_missing": not meta_path.exists(),
        "removed_pixels": int(erase.sum()),
    }
    return bool(erase.any()), info


def _page_stem(crop_path: Path) -> str:
    match = re.match(r"^(?P<page>.+)_mol\d+$", crop_path.stem)
    if not match:
        raise FragmentCleanupContextError(
            f"cannot infer page stem from crop name: {crop_path.name}")
    return match.group("page")


def _find_page(page_dir: Path, stem: str) -> Path:
    matches = [
        path for path in page_dir.iterdir()
        if path.is_file() and path.stem == stem
        and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if len(matches) != 1:
        raise FragmentCleanupContextError(
            f"expected one source page for {stem}, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply page-aware fragment cleanup to extracted crops")
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    parser.add_argument("--page-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    crop_paths = sorted(
        path for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    by_page: Dict[str, List[Dict]] = {}
    for crop_path in crop_paths:
        by_page.setdefault(_page_stem(crop_path), []).append(
            {"crop_path": crop_path})

    total = changed = 0
    for stem, candidates in sorted(by_page.items()):
        page_path = _find_page(args.page_dir, stem)
        with Image.open(page_path) as image:
            page_rgb = np.asarray(image.convert("RGB"))
        cleanup_candidates_impl(
            candidates, page_rgb, args.output, visualize=args.visualize)
        total += len(candidates)
        changed += sum(
            bool(candidate["cleanup_changed"]) for candidate in candidates)
    print(f"fragment cleanup complete: {changed}/{total} crops changed")


if __name__ == "__main__":
    main()

