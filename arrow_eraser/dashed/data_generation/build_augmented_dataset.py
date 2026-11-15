"""Build a traceable train-only augmentation of DashedArrow-Seg."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
DASHED_ROOT = HERE.parent
DEFAULT_SOURCE = DASHED_ROOT / "dataset" / "DashedArrow-Seg"
DEFAULT_OUTPUT = DASHED_ROOT / "dataset" / "DashedArrow-Seg-Augmented"
VALID_VALUES = {0, 1, 2, 255}
AUGMENTATIONS = ("environment", "style", "lowres")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def jpeg_round_trip(rgb: np.ndarray, quality: int) -> np.ndarray:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def environment_variant(image: np.ndarray, rng: np.random.Generator):
    height, width = image.shape[:2]
    work = image.astype(np.float32) / 255.0
    gamma = float(rng.uniform(0.82, 1.18))
    gains = rng.uniform(0.92, 1.08, size=3).astype(np.float32)
    work = np.power(np.clip(work, 0, 1), gamma) * gains
    axis = str(rng.choice(["horizontal", "vertical", "diagonal"]))
    strength = float(rng.uniform(-0.07, 0.07))
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(-1, 1, height, dtype=np.float32)[:, None]
    gradient = x if axis == "horizontal" else y if axis == "vertical" else (x + y) / 2
    work += gradient[..., None] * strength
    noise_sigma = float(rng.uniform(0.003, 0.018))
    work += rng.normal(0, noise_sigma, size=work.shape).astype(np.float32)
    output = np.clip(work * 255, 0, 255).astype(np.uint8)
    blur_sigma = 0.0
    if rng.random() < 0.45:
        blur_sigma = float(rng.uniform(0.25, 0.75))
        output = cv2.GaussianBlur(output, (3, 3), blur_sigma)
    quality = int(rng.integers(55, 93))
    output = jpeg_round_trip(output, quality)
    return output, {
        "gamma": round(gamma, 5),
        "channel_gains": [round(float(v), 5) for v in gains],
        "gradient_axis": axis,
        "gradient_strength": round(strength, 5),
        "noise_sigma_01": round(noise_sigma, 5),
        "blur_sigma": round(blur_sigma, 5),
        "jpeg_quality": quality,
    }


def style_variant(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    palettes = np.asarray(
        [(18, 18, 18), (75, 75, 75), (170, 35, 35), (35, 65, 160),
         (38, 125, 78), (135, 60, 145)], dtype=np.float32
    )
    color = palettes[int(rng.integers(0, len(palettes)))]
    opacity = float(rng.uniform(0.68, 0.96))
    output = image.astype(np.float32)
    arrow = mask == 1
    output[arrow] = output[arrow] * (1 - opacity) + color * opacity
    global_gamma = float(rng.uniform(0.9, 1.1))
    output = np.power(np.clip(output / 255.0, 0, 1), global_gamma) * 255
    return np.clip(output, 0, 255).astype(np.uint8), {
        "arrow_rgb": [int(v) for v in color],
        "arrow_opacity": round(opacity, 5),
        "global_gamma": round(global_gamma, 5),
        "geometry_changed": False,
    }


def visibility_metrics(image: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    arrow = mask == 1
    background = mask == 0
    if not np.any(arrow) or not np.any(background):
        return 0.0, 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    background_level = float(np.median(gray[background]))
    contrast = np.abs(gray[arrow] - background_level)
    return float(contrast.mean()), float(np.mean(contrast >= 10.0))


def lowres_variant(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    height, width = image.shape[:2]
    original_arrow_pixels = int(np.count_nonzero(mask == 1))
    attempts = [float(rng.uniform(0.50, 0.85)), 0.65, 0.75, 0.85]
    selected = None
    for factor in attempts:
        small_width = max(2, round(width * factor))
        small_height = max(2, round(height * factor))
        small_image = cv2.resize(image, (small_width, small_height), interpolation=cv2.INTER_AREA)
        small_mask = cv2.resize(mask, (small_width, small_height), interpolation=cv2.INTER_NEAREST)
        up_interpolation = int(rng.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]))
        restored_image = cv2.resize(small_image, (width, height), interpolation=up_interpolation)
        restored_mask = cv2.resize(small_mask, (width, height), interpolation=cv2.INTER_NEAREST)
        arrow_pixels = int(np.count_nonzero(restored_mask == 1))
        mean_contrast, visible_ratio = visibility_metrics(restored_image, restored_mask)
        retained_ratio = arrow_pixels / max(1, original_arrow_pixels)
        selected = (factor, up_interpolation, restored_image, restored_mask,
                    mean_contrast, visible_ratio, retained_ratio)
        if arrow_pixels and retained_ratio >= 0.60 and mean_contrast >= 10 and visible_ratio >= 0.35:
            break
    factor, interpolation, restored_image, restored_mask, mean_contrast, visible_ratio, retained_ratio = selected
    quality = int(rng.integers(65, 96))
    restored_image = jpeg_round_trip(restored_image, quality)
    mean_contrast, visible_ratio = visibility_metrics(restored_image, restored_mask)
    interpolation_name = {
        cv2.INTER_LINEAR: "linear", cv2.INTER_CUBIC: "cubic", cv2.INTER_LANCZOS4: "lanczos4"
    }[interpolation]
    return restored_image, restored_mask, {
        "scale_factor": round(float(factor), 5),
        "down_interpolation": "area",
        "up_interpolation": interpolation_name,
        "jpeg_quality": quality,
        "arrow_pixel_retained_ratio": round(float(retained_ratio), 5),
        "arrow_mean_contrast": round(float(mean_contrast), 5),
        "arrow_visible_ratio_at_contrast_10": round(float(visible_ratio), 5),
    }


def source_files(source: Path, split: str, sample_id: str) -> tuple[Path, Path]:
    images = list((source / split / "images").glob(f"{sample_id}.*"))
    mask = source / split / "masks" / f"{sample_id}.png"
    if len(images) != 1 or not mask.is_file():
        raise FileNotFoundError(f"Cannot resolve pair: {split}/{sample_id}")
    return images[0], mask


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args) -> list[dict[str, object]]:
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    source_rows = list(csv.DictReader((source / "manifest.csv").open("r", encoding="utf-8-sig")))
    ids = set()
    output_rows: list[dict[str, object]] = []
    for split in ("train", "val", "val_synthetic"):
        (output / split / "images").mkdir(parents=True)
        (output / split / "masks").mkdir(parents=True)

    train_index = 0
    for row in source_rows:
        sample_id, split = row["id"], row["split"]
        if sample_id in ids:
            raise ValueError(f"Duplicate source id: {sample_id}")
        ids.add(sample_id)
        image_path, mask_path = source_files(source, split, sample_id)
        destination_image = output / split / "images" / image_path.name
        destination_mask = output / split / "masks" / mask_path.name
        shutil.copy2(image_path, destination_image)
        shutil.copy2(mask_path, destination_mask)
        base = {
            "id": sample_id, "split": split, "kind": row["kind"],
            "source": row["source"], "group": row["group"],
            "parent_id": sample_id, "augmentation": "original",
            "augmentation_seed": "", "augmentation_params": "{}",
            "image": str(destination_image.relative_to(output)),
            "mask": str(destination_mask.relative_to(output)),
            "image_sha256": sha256(destination_image), "mask_sha256": sha256(destination_mask),
        }
        output_rows.append(base)
        if split != "train":
            continue

        image, mask = read_rgb(image_path), read_mask(mask_path)
        values = {int(value) for value in np.unique(mask)}
        if image.shape[:2] != mask.shape or not values <= VALID_VALUES:
            raise ValueError(f"Invalid source pair: {sample_id}")
        for offset, augmentation in enumerate(AUGMENTATIONS):
            variant_seed = args.seed + train_index * 101 + offset
            rng = np.random.default_rng(variant_seed)
            variant_mask = mask.copy()
            if augmentation == "environment":
                variant_image, params = environment_variant(image, rng)
            elif augmentation == "style":
                variant_image, params = style_variant(image, mask, rng)
            else:
                variant_image, variant_mask, params = lowres_variant(image, mask, rng)
            variant_id = f"aug_{augmentation}_{train_index:04d}"
            variant_image_path = output / "train" / "images" / f"{variant_id}.png"
            variant_mask_path = output / "train" / "masks" / f"{variant_id}.png"
            save_png(variant_image_path, variant_image)
            save_png(variant_mask_path, variant_mask)
            output_rows.append({
                "id": variant_id, "split": "train", "kind": row["kind"],
                "source": "derived_augmentation", "group": row["group"],
                "parent_id": sample_id, "augmentation": augmentation,
                "augmentation_seed": variant_seed,
                "augmentation_params": json.dumps(params, ensure_ascii=False, separators=(",", ":")),
                "image": str(variant_image_path.relative_to(output)),
                "mask": str(variant_mask_path.relative_to(output)),
                "image_sha256": sha256(variant_image_path), "mask_sha256": sha256(variant_mask_path),
            })
        train_index += 1
        if train_index % 50 == 0:
            print(f"augmented {train_index}/529 training parents", flush=True)
    write_manifest(output / "manifest.csv", output_rows)
    return output_rows


def validate(source: Path, output: Path, rows: list[dict[str, object]]) -> dict[str, object]:
    errors: list[str] = []
    ids = Counter(str(row["id"]) for row in rows)
    duplicate_ids = sorted(key for key, count in ids.items() if count > 1)
    if duplicate_ids:
        errors.append(f"duplicate ids: {duplicate_ids[:10]}")
    split_counts: Counter[str] = Counter()
    augmentation_counts: Counter[str] = Counter()
    kind_counts: Counter[str] = Counter()
    class_pixels: Counter[int] = Counter()
    groups: dict[str, set[str]] = defaultdict(set)
    image_hashes: dict[str, set[str]] = defaultdict(set)
    # Compare decoded label pixels here, not PNG file bytes. Saving an identical
    # label array with a different encoder/compression level may change its file
    # SHA-256 while leaving every semantic label unchanged.
    original_masks: dict[str, str] = {}
    lowres_metrics = []
    for row in rows:
        sample_id, split = str(row["id"]), str(row["split"])
        image_path, mask_path = output / str(row["image"]), output / str(row["mask"])
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"missing pair: {sample_id}")
            continue
        image, mask = read_rgb(image_path), read_mask(mask_path)
        if image.shape[:2] != mask.shape:
            errors.append(f"shape mismatch: {sample_id}")
        values = {int(value) for value in np.unique(mask)}
        if not values <= VALID_VALUES:
            errors.append(f"invalid mask values: {sample_id} {sorted(values)}")
        if not np.any(mask == 1):
            errors.append(f"empty dashed arrow: {sample_id}")
        if not np.any(mask == 2):
            errors.append(f"empty bond: {sample_id}")
        for value in values:
            class_pixels[value] += int(np.count_nonzero(mask == value))
        digest = sha256(image_path)
        if digest != row["image_sha256"] or sha256(mask_path) != row["mask_sha256"]:
            errors.append(f"manifest hash mismatch: {sample_id}")
        split_counts[split] += 1
        augmentation_counts[str(row["augmentation"])] += 1
        kind_counts[f"{split}:{row['kind']}:{row['augmentation']}"] += 1
        groups[str(row["group"])].add(split)
        image_hashes[split].add(digest)
        mask_pixel_hash = hashlib.sha256(mask.tobytes()).hexdigest()
        if row["augmentation"] == "original":
            original_masks[sample_id] = mask_pixel_hash
            if split != "train":
                source_image, source_mask = source_files(source, split, sample_id)
                if sha256(source_image) != digest or sha256(source_mask) != row["mask_sha256"]:
                    errors.append(f"held-out original changed: {sample_id}")
        elif row["augmentation"] in {"environment", "style"}:
            if mask_pixel_hash != original_masks.get(str(row["parent_id"])):
                errors.append(f"photometric augmentation changed mask: {sample_id}")
        elif row["augmentation"] == "lowres":
            params = json.loads(str(row["augmentation_params"]))
            lowres_metrics.append(params)

    group_overlap = sorted(group for group, splits in groups.items() if "train" in splits and len(splits) > 1)
    if group_overlap:
        errors.append(f"train/held-out group overlap: {group_overlap[:10]}")
    hash_overlap = {
        "train_val": len(image_hashes["train"] & image_hashes["val"]),
        "train_val_synthetic": len(image_hashes["train"] & image_hashes["val_synthetic"]),
    }
    if any(hash_overlap.values()):
        errors.append(f"train/held-out image hash overlap: {hash_overlap}")
    report = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_dataset": str(source.resolve()),
        "dataset": str(output.resolve()),
        "samples": len(rows),
        "split_counts": dict(split_counts),
        "augmentation_counts": dict(augmentation_counts),
        "kind_counts": dict(kind_counts),
        "class_pixels": {str(value): class_pixels[value] for value in (0, 1, 2, 255)},
        "train_heldout_group_overlap": group_overlap,
        "train_heldout_exact_image_hash_overlap": hash_overlap,
        "lowres_visibility": {
            "samples": len(lowres_metrics),
            "scale_min": min(p["scale_factor"] for p in lowres_metrics),
            "scale_max": max(p["scale_factor"] for p in lowres_metrics),
            "minimum_arrow_pixel_retained_ratio": min(p["arrow_pixel_retained_ratio"] for p in lowres_metrics),
            "minimum_arrow_mean_contrast": min(p["arrow_mean_contrast"] for p in lowres_metrics),
            "minimum_arrow_visible_ratio": min(p["arrow_visible_ratio_at_contrast_10"] for p in lowres_metrics),
        },
        "errors": errors,
        "passed": not errors,
    }
    (output / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build train-only augmented DashedArrow-Seg")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing output and refresh its reports without rebuilding images.",
    )
    args = parser.parse_args()
    if args.validate_only:
        rows = list(csv.DictReader((args.output / "manifest.csv").open("r", encoding="utf-8-sig")))
    else:
        rows = build(args)
    report = validate(args.source.resolve(), args.output.resolve(), rows)
    source_info = json.loads((args.source / "dataset.json").read_text(encoding="utf-8"))
    dataset_info = {
        "schema_version": 1,
        "name": "DashedArrow-Seg-Augmented",
        "classes": source_info["classes"],
        "ignore_index": source_info["ignore_index"],
        "source_dataset": str(args.source.resolve()),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seed": args.seed,
        "augmentation_scope": "train only; val and val_synthetic copied byte-for-byte",
        "variants_per_training_parent": list(AUGMENTATIONS),
        "counts": report["split_counts"],
        "passed_quality_checks": report["passed"],
    }
    (args.output / "dataset.json").write_text(
        json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    if not report["passed"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(report["lowres_visibility"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
