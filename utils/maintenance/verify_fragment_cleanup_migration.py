"""Verify the utils implementation against all 563 frozen cleanup outputs."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from utils.base.paths import PROJECT_ROOT as ROOT

from utils.pipeline.fragment_cleanup import PARAMETERS, clean_crop, page_lines


REFERENCE = ROOT / "outputs" / "cc_fragment_cleanup_approved50_20261020"


def rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def main() -> None:
    started = time.perf_counter()
    manifest = json.loads(
        (REFERENCE / "manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["source"])
    if manifest["parameters"] != PARAMETERS:
        raise AssertionError("utils parameters differ from the frozen experiment")
    saved = json.loads(
        (REFERENCE / "results.json").read_text(encoding="utf-8"))["samples"]
    saved_by_name = {row["crop_name"]: row for row in saved}
    rows = json.loads((source / "crops.json").read_text(encoding="utf-8"))
    if len(rows) != len(saved_by_name) or len(rows) != 563:
        raise AssertionError("expected the frozen 563-crop batch")

    by_page = defaultdict(list)
    for original in rows:
        row = original.copy()
        crop_path = source / row["cc_image"]
        with np.load(crop_path.with_suffix(".meta.npz")) as data:
            page_h, page_w = map(int, data["page_size"])
            row["bbox"] = np.rint(
                data["bbox_norm"]
                * [page_w, page_h, page_w, page_h]
            ).astype(int)
            row["raw_bbox"] = np.rint(
                data["bbox_raw_norm"]
                * [page_w, page_h, page_w, page_h]
            ).astype(int)
            row["polygon"] = np.asarray(data["polygon"])
            row["polygon_page"] = row["polygon"] + row["bbox"][:2]
        by_page[row["page_id"]].append(row)

    mismatches = []
    pixel_matches = mask_matches = evidence_matches = 0
    for page_id, page_rows in sorted(by_page.items()):
        page = rgb(source / "inputs" / "pages" / f"{page_id}.png")
        lines = page_lines(page)
        for row in page_rows:
            name = row["crop_name"]
            target = saved_by_name[name]
            before = rgb(source / row["cc_image"])
            x1, y1, x2, y2 = map(int, row["bbox"])
            neighbors = [
                {
                    "name": other["crop_name"],
                    "raw_bbox": other["raw_bbox"].tolist(),
                    "polygon": other["polygon_page"] - [x1, y1],
                }
                for other in page_rows if other["crop_name"] != name
            ]
            after, erased, evidence = clean_crop(
                before,
                row["polygon"],
                row["raw_bbox"].tolist(),
                neighbors,
                lines[y1:y2, x1:x2],
            )
            stored_after = rgb(REFERENCE / target["after"])
            with Image.open(REFERENCE / target["erase_mask"]) as image:
                stored_mask = np.asarray(image.convert("L")) > 0
            pixel_ok = np.array_equal(after, stored_after)
            mask_ok = np.array_equal(erased, stored_mask)
            evidence_ok = all(
                target[key] == value for key, value in evidence.items())
            pixel_matches += int(pixel_ok)
            mask_matches += int(mask_ok)
            evidence_matches += int(evidence_ok)
            if not (pixel_ok and mask_ok and evidence_ok):
                mismatches.append({
                    "crop_name": name,
                    "after_pixels_equal": pixel_ok,
                    "erase_mask_equal": mask_ok,
                    "evidence_equal": evidence_ok,
                })

    report = {
        "passed": not mismatches,
        "scope": "utils.fragment_cleanup versus frozen 563-crop outputs",
        "crops_checked": len(rows),
        "pages_with_crops": len(by_page),
        "after_pixel_identical_count": pixel_matches,
        "erase_mask_identical_count": mask_matches,
        "evidence_identical_count": evidence_matches,
        "mismatches": mismatches,
        "seconds": time.perf_counter() - started,
    }
    destination = REFERENCE / "utils_migration_verification.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise AssertionError(f"migration differs; see {destination}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

