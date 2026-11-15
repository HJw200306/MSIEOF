"""Audit image/mask pairing and label values for the organized datasets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def audit_pair_dir(images: Path, masks: Path) -> dict:
    image_files = {p.stem: p for p in images.iterdir() if p.is_file()}
    mask_files = {p.stem: p for p in masks.iterdir() if p.is_file()}
    labels: Counter[int] = Counter()
    size_mismatches = []
    for stem in sorted(image_files.keys() & mask_files.keys()):
        with Image.open(image_files[stem]) as image, Image.open(mask_files[stem]) as mask:
            if image.size != mask.size:
                size_mismatches.append(stem)
            labels.update(int(v) for v in np.unique(np.asarray(mask)))
    return {
        "images": len(image_files),
        "masks": len(mask_files),
        "pairs": len(image_files.keys() & mask_files.keys()),
        "missing_masks": sorted(image_files.keys() - mask_files.keys()),
        "missing_images": sorted(mask_files.keys() - image_files.keys()),
        "label_presence_by_file": dict(sorted(labels.items())),
        "size_mismatches": size_mismatches,
    }


def main() -> None:
    report = {
        "solid_legacy_binary": audit_pair_dir(
            ROOT / "solid/dataset/VOCdevkit/VOC2007/JPEGImages",
            ROOT / "solid/dataset/VOCdevkit/VOC2007/SegmentationClass",
        ),
        "dashed": {},
    }
    dashed_root = ROOT / "dashed/dataset/DashedArrow-Seg"
    for split in ("train", "val", "val_synthetic"):
        report["dashed"][split] = audit_pair_dir(
            dashed_root / split / "images", dashed_root / split / "masks"
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
