"""Visualize the archived dual-model union on annotated molecule crops.

Historical fusion policy:
  arrow = solid-model arrow OR dashed-model arrow
  bond  = solid-model bond  OR dashed-model bond
  erase = arrow AND NOT bond

Approved JSON bounding boxes are applied before inference, so page text and
headers never enter either arrow model.  The solid arrow uses its historical
HSV-halo refinement; the dashed arrow uses the selected historical v3/v4
strategy (v4_25 by default).  Both arrow predictions and both bond predictions
are fused, with bond-union priority.  The only final artifact is one montage of
processed crops with bond overlap and false-positive counts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nets.unet import Unet as VggUnet  # noqa: E402


DEFAULT_INPUT = Path(
    r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal"
    r"\images\PageToSMILES_approved"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "selected_dual_model_crop_visualization"
DEFAULT_SOLID = PROJECT_ROOT / "model" / "legacy_arrow_models" / "misp_weights_no_freeze.pth"
DEFAULT_DASHED = PROJECT_ROOT / "model" / "legacy_arrow_models" / "dashed_arrow_model.pth"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
DEFAULT_PAGE_STEMS = [
    "JNP_2024_04_861-868_001",
    "JNP_2024_02_396-403_002",
    "JNP_2024_02_266-275_004",
    "JNP_2024_01_2-13_006",
]

COLOR_SOLID = np.array([235, 45, 45], dtype=np.float32)
COLOR_DASHED = np.array([255, 145, 0], dtype=np.float32)
COLOR_BOND = np.array([0, 210, 70], dtype=np.float32)
COLOR_ERASE = np.array([190, 40, 235], dtype=np.float32)
COLOR_TP = np.array([0, 210, 70], dtype=np.float32)
COLOR_FP = np.array([235, 45, 45], dtype=np.float32)
COLOR_FN = np.array([40, 110, 255], dtype=np.float32)
PALETTE = [0, 0, 0, 190, 40, 235, 0, 210, 70] + [0] * (256 * 3 - 9)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize dual-model arrow removal on approved molecule crops."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gt-dir", type=Path, default=None,
        help=("Expert pixel-level GT mask directory. By default it is inferred as "
              "<dataset>/masks/<input-folder-name>."),
    )
    parser.add_argument(
        "--gt-json-dir", type=Path, default=None,
        help=("Approved LabelMe JSON directory. By default it is inferred as "
              "<dataset>/json/<input-folder-name>."),
    )
    parser.add_argument(
        "--pages", nargs="*", default=DEFAULT_PAGE_STEMS,
        help="Page stems to process. The default is the four requested JNP pages.",
    )
    parser.add_argument("--solid-model", type=Path, default=DEFAULT_SOLID)
    parser.add_argument("--dashed-model", type=Path, default=DEFAULT_DASHED)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.72)
    parser.add_argument("--gt-bond-values", default="2,3",
                        help="Bond class values for indexed non-binary GT masks.")
    parser.add_argument("--gt-tolerance", type=float, default=1.0,
                        help="Euclidean pixel tolerance around expert GT (default: 1.0).")
    parser.add_argument(
        "--tolerance-sweep", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
        help="Evaluate an inclusive tolerance range after one shared model inference.",
    )
    parser.add_argument(
        "--dashed-postprocess",
        choices=("v3_25", "v3_30", "v4_25", "v4_30"),
        default="v4_25",
        help="Historical dashed-arrow strategy; v4_25 is the default selected result.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    else:
        state = checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    return checkpoint if isinstance(checkpoint, dict) else {}, state


class DualUnionPredictor:
    def __init__(self, solid_path: Path, dashed_path: Path, device_name: str):
        if not solid_path.is_file() or not dashed_path.is_file():
            raise FileNotFoundError(f"Missing model: solid={solid_path}; dashed={dashed_path}")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        self.device = torch.device(device_name)

        _, solid_state = load_state(solid_path)
        if "final.weight" not in solid_state or int(solid_state["final.weight"].shape[0]) != 3:
            raise ValueError("Solid checkpoint must be VGG-U-Net with three output classes")
        self.solid = VggUnet(num_classes=3, pretrained=False, backbone="vgg")
        self.solid.load_state_dict(solid_state, strict=True)
        self.solid.to(self.device).eval()

        dashed_meta, dashed_state = load_state(dashed_path)
        dashed_state = {key.removeprefix("model."): value for key, value in dashed_state.items()}
        head = next((key for key in dashed_state if key.endswith("segmentation_head.0.weight")), None)
        if head is None or int(dashed_state[head].shape[0]) != 3:
            raise ValueError("Dashed checkpoint must be U-Net++ with three output classes")
        names = dashed_meta.get("class_names")
        if names is not None and list(names) != ["background", "dashed_arrow", "bond"]:
            raise ValueError(f"Unexpected dashed class mapping: {names}")
        encoder = dashed_meta.get("encoder", "resnet101")
        self.dashed = smp.UnetPlusPlus(
            encoder_name=encoder, encoder_weights=None, in_channels=3, classes=3
        )
        self.dashed.load_state_dict(dashed_state, strict=True)
        self.dashed.to(self.device).eval()

    @torch.inference_mode()
    def predict_batch(
        self, canvases: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        solid_input = torch.from_numpy(
            (canvases.astype(np.float32) / 255.0).transpose(0, 3, 1, 2).copy()
        ).to(self.device)
        solid = self.solid(solid_input).argmax(dim=1).cpu().numpy().astype(np.uint8)
        dashed_array = (canvases.astype(np.float32) / 255.0 - MEAN) / STD
        dashed_input = torch.from_numpy(
            dashed_array.transpose(0, 3, 1, 2).copy()
        ).to(self.device)
        dashed_logits = self.dashed(dashed_input)
        dashed = dashed_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        dashed_arrow_probability = (
            dashed_logits.softmax(dim=1)[:, 1].cpu().numpy().astype(np.float32)
        )
        return solid, dashed, dashed_arrow_probability


def image_files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*")
         if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: str(path.relative_to(root)).lower(),
    )


def predict_page(predictor: DualUnionPredictor, rgb: np.ndarray, tile_size: int,
                 context: int, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if context < 0 or tile_size <= 2 * context:
        raise ValueError("tile-size must be larger than 2 * context")
    if tile_size % 32:
        raise ValueError("tile-size must be divisible by 32")
    core = tile_size - 2 * context
    height, width = rgb.shape[:2]
    solid_page = np.zeros((height, width), dtype=np.uint8)
    dashed_page = np.zeros((height, width), dtype=np.uint8)
    dashed_probability_page = np.zeros((height, width), dtype=np.float32)
    jobs = [
        (y0, min(y0 + core, height), x0, min(x0 + core, width))
        for y0 in range(0, height, core)
        for x0 in range(0, width, core)
    ]
    for offset in range(0, len(jobs), batch_size):
        batch_jobs = jobs[offset:offset + batch_size]
        canvases, meta = [], []
        for y0, y1, x0, x1 in batch_jobs:
            sy0, sy1 = max(0, y0 - context), min(height, y1 + context)
            sx0, sx1 = max(0, x0 - context), min(width, x1 + context)
            crop = rgb[sy0:sy1, sx0:sx1]
            canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            canvas[:crop.shape[0], :crop.shape[1]] = crop
            canvases.append(canvas)
            meta.append((y0, y1, x0, x1, y0 - sy0, x0 - sx0))
        solid_batch, dashed_batch, probability_batch = predictor.predict_batch(np.stack(canvases))
        for solid, dashed, probability, (y0, y1, x0, x1, oy, ox) in zip(
                solid_batch, dashed_batch, probability_batch, meta):
            h, w = y1 - y0, x1 - x0
            solid_page[y0:y1, x0:x1] = solid[oy:oy + h, ox:ox + w]
            dashed_page[y0:y1, x0:x1] = dashed[oy:oy + h, ox:ox + w]
            dashed_probability_page[y0:y1, x0:x1] = probability[oy:oy + h, ox:ox + w]
    return solid_page, dashed_page, dashed_probability_page


def fuse(solid: np.ndarray, dashed: np.ndarray) -> dict[str, np.ndarray]:
    solid_arrow = solid == 1
    dashed_arrow = dashed == 1
    arrow_union = solid_arrow | dashed_arrow
    solid_bond = solid == 2
    dashed_bond = dashed == 2
    bond_union = solid_bond | dashed_bond
    erased = arrow_union & ~bond_union
    classes = np.zeros(solid.shape, dtype=np.uint8)
    classes[arrow_union] = 1
    classes[bond_union] = 2
    return {
        "solid_arrow": solid_arrow,
        "dashed_arrow": dashed_arrow,
        "arrow_union": arrow_union,
        "solid_bond": solid_bond,
        "dashed_bond": dashed_bond,
        "bond_union": bond_union,
        "erased": erased,
        "classes": classes,
    }


def refine_solid_arrow(rgb: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Historical solid-arrow refinement: include colourful pixels in a 3x3 halo."""
    arrow = classes == 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    halo = cv2.dilate(arrow.astype(np.uint8), kernel, iterations=3).astype(bool)
    saturation = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1]
    return arrow | (halo & (saturation > 30))


def refine_dashed_arrow(
    classes: np.ndarray, arrow_probability: np.ndarray, strategy: str
) -> np.ndarray:
    """Historical v3/v4 strategies; bond-union protection is applied afterwards."""
    threshold = 0.25 if strategy.endswith("25") else 0.30
    expanded = (classes == 1) | (arrow_probability > threshold)
    if strategy.startswith("v3"):
        return expanded
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(expanded.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.dilate(closed, dilate_kernel, iterations=1).astype(bool)


def find_gt_mask(gt_dir: Path, relative: Path) -> Path:
    direct_candidates = [
        gt_dir / relative.with_suffix(".png"),
        gt_dir / f"{relative.stem}.png",
    ]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate
    matches = sorted(
        path for path in gt_dir.rglob(f"{relative.stem}.*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple GT masks found for {relative.stem}: "
            + ", ".join(str(path) for path in matches)
        )
    raise FileNotFoundError(
        f"Missing independent GT mask for {relative.stem}. Expected "
        f"{gt_dir / (relative.stem + '.png')}"
    )


def load_gt_bond(
    path: Path, expected_shape: tuple[int, int], bond_values: set[int]
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load binary, indexed or RGB GT without treating model predictions as GT."""
    with Image.open(path) as opened:
        array = np.asarray(opened)
    encoding: str
    if array.ndim == 3:
        rgb = array[..., :3]
        if np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2]):
            array = rgb[..., 0]
        else:
            rgb_i16 = rgb.astype(np.int16)
            gt_bond = ((rgb_i16[..., 1] > rgb_i16[..., 0] + 20)
                       & (rgb_i16[..., 1] > rgb_i16[..., 2] + 20))
            valid = np.ones(gt_bond.shape, dtype=bool)
            encoding = "rgb-green-bond"
            if gt_bond.shape != expected_shape:
                raise ValueError(
                    f"GT shape {gt_bond.shape} != source shape {expected_shape}: {path}"
                )
            return gt_bond, valid, encoding
    if array.ndim != 2:
        raise ValueError(f"Unsupported GT mask shape {array.shape}: {path}")
    unique = set(int(value) for value in np.unique(array))
    if unique <= {0, 1}:
        gt_bond = array == 1
        valid = np.ones(array.shape, dtype=bool)
        encoding = "binary-0-1"
    elif unique <= {0, 255}:
        gt_bond = array == 255
        valid = np.ones(array.shape, dtype=bool)
        encoding = "binary-0-255"
    elif unique - {0} and min(unique - {0}) >= 128:
        gt_bond = array >= 128
        valid = np.ones(array.shape, dtype=bool)
        encoding = "antialiased-binary-0-128to255"
    else:
        gt_bond = np.isin(array, list(bond_values))
        valid = array != 255
        encoding = f"indexed-bond-{','.join(map(str, sorted(bond_values)))}-ignore-255"
    if gt_bond.shape != expected_shape:
        raise ValueError(f"GT shape {gt_bond.shape} != source shape {expected_shape}: {path}")
    return gt_bond, valid, encoding


def rasterize_approved_json_bonds(
    json_path: Path, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str]:
    """Create page-level bond pixels from approved crop-local atom/bond annotations.

    The JSON gives bond topology and atom centres rather than a dense bond mask.
    A corridor is drawn for every annotated bond and intersected with real page
    ink, so text/background inside molecular bounding boxes is not labelled as a
    bond merely because it lies in the crop.
    """
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    height, width = rgb.shape[:2]
    corridor = np.zeros((height, width), dtype=np.uint8)
    annotated_bonds = 0
    for shape in payload.get("shapes", []):
        atoms = shape.get("atoms") or []
        bonds = shape.get("bonds") or []
        evaluation = shape.get("evaluation") or {}
        bbox = evaluation.get("bbox_xyxy")
        if not atoms or not bonds or not bbox or len(bbox) != 4:
            continue
        x0, y0 = float(bbox[0]), float(bbox[1])
        for bond in bonds:
            if isinstance(bond, dict):
                endpoints = bond.get("endpoint_atoms") or [
                    bond.get("begin_atom", bond.get("u")),
                    bond.get("end_atom", bond.get("v")),
                ]
            else:
                endpoints = bond[:2]
            if len(endpoints) != 2 or endpoints[0] is None or endpoints[1] is None:
                continue
            first, second = int(endpoints[0]), int(endpoints[1])
            if first >= len(atoms) or second >= len(atoms) or first < 0 or second < 0:
                continue
            p0 = np.array([x0 + float(atoms[first]["x"]),
                           y0 + float(atoms[first]["y"])], dtype=np.float32)
            p1 = np.array([x0 + float(atoms[second]["x"]),
                           y0 + float(atoms[second]["y"])], dtype=np.float32)
            length = float(np.linalg.norm(p1 - p0))
            if length < 1.0:
                continue
            # Trim label centres at bond endpoints while retaining unlabeled-carbon strokes.
            trim = min(2.0, length * 0.08)
            direction = (p1 - p0) / length
            q0 = np.rint(p0 + direction * trim).astype(int)
            q1 = np.rint(p1 - direction * trim).astype(int)
            thickness = max(7, min(11, int(round(length * 0.42))))
            cv2.line(corridor, tuple(q0), tuple(q1), 1, thickness=thickness,
                     lineType=cv2.LINE_8)
            annotated_bonds += 1
    if annotated_bonds == 0:
        raise ValueError(f"No usable atom/bond annotations found in: {json_path}")
    # Include black and coloured structure strokes but reject the white page.
    ink = np.min(rgb, axis=2) < 235
    gt_bond = corridor.astype(bool) & ink
    if not gt_bond.any():
        raise ValueError(f"Approved bond annotations produced an empty GT mask: {json_path}")
    valid = np.ones((height, width), dtype=bool)
    encoding = f"approved-json-graph-x-page-ink-v1;annotated_bonds={annotated_bonds}"
    return gt_bond, valid, encoding


def _dilate_boolean(mask: np.ndarray, tolerance: float) -> np.ndarray:
    if tolerance <= 0:
        return mask
    if not mask.any():
        return mask.copy()
    distance = cv2.distanceTransform(
        (~mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    return distance <= float(tolerance) + 1e-6


def bond_metrics(
    predicted: np.ndarray, gt: np.ndarray, valid: np.ndarray, tolerance: float
) -> dict[str, float | int]:
    predicted = predicted & valid
    gt = gt & valid
    tolerated_gt = _dilate_boolean(gt, tolerance)
    overlap = predicted & tolerated_gt
    false_positive = predicted & ~tolerated_gt
    gt_covered = gt & _dilate_boolean(predicted, tolerance)
    predicted_pixels = int(predicted.sum())
    gt_pixels = int(gt.sum())
    overlap_pixels = int(overlap.sum())
    false_pixels = int(false_positive.sum())
    return {
        "predicted_bond_pixels": predicted_pixels,
        "gt_bond_pixels": gt_pixels,
        "bond_overlap_pixels": overlap_pixels,
        "bond_overlap_pred_percent": 100.0 * overlap_pixels / predicted_pixels if predicted_pixels else 0.0,
        "gt_covered_pixels": int(gt_covered.sum()),
        "gt_coverage_percent": 100.0 * int(gt_covered.sum()) / gt_pixels if gt_pixels else 0.0,
        "false_bond_pixels": false_pixels,
        "false_bond_pred_percent": 100.0 * false_pixels / predicted_pixels if predicted_pixels else 0.0,
        "gt_tolerance_pixels": tolerance,
    }


def model_overlay(
    rgb: np.ndarray,
    erase_mask: np.ndarray,
    predicted_bond: np.ndarray,
    gt_bond: np.ndarray,
    valid: np.ndarray,
    alpha: float,
    tolerance: float,
) -> np.ndarray:
    tolerated_gt = _dilate_boolean(gt_bond & valid, tolerance)
    true_bond = predicted_bond & tolerated_gt & valid
    false_bond = predicted_bond & ~tolerated_gt & valid
    return overlay(rgb, [
        (erase_mask, COLOR_ERASE),
        (true_bond, COLOR_TP),
        (false_bond, COLOR_FP),
    ], alpha)


def annotated_panel(image: np.ndarray, title: str, metrics: dict[str, float | int]) -> np.ndarray:
    banner_height = 96
    panel = np.full((image.shape[0] + banner_height, image.shape[1], 3), 255, np.uint8)
    panel[banner_height:] = image
    cv2.putText(panel, title, (18, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.70,
                (25, 25, 25), 2, cv2.LINE_AA)
    line = (
        f"GT bond overlap: {metrics['bond_overlap_pixels']:,} px "
        f"({metrics['bond_overlap_pred_percent']:.2f}% of prediction) | "
        f"wrong bond: {metrics['false_bond_pixels']:,} px "
        f"({metrics['false_bond_pred_percent']:.2f}% of prediction)"
    )
    cv2.putText(panel, line, (18, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (25, 25, 25), 1, cv2.LINE_AA)
    cv2.putText(panel, "purple=arrow erase  green=correct bond  red=wrong bond",
                (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 70, 70), 1, cv2.LINE_AA)
    return panel


def overlay(rgb: np.ndarray, layers: list[tuple[np.ndarray, np.ndarray]], alpha: float) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    for mask, color in layers:
        result[mask] = result[mask] * (1.0 - alpha) + color * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def save_outputs(
    source: Path,
    relative: Path,
    output: Path,
    rgb: np.ndarray,
    solid: np.ndarray,
    dashed: np.ndarray,
    dashed_probability: np.ndarray,
    gt_bond: np.ndarray,
    gt_valid: np.ndarray,
    gt_path: Path,
    gt_encoding: str,
    dashed_postprocess: str,
    alpha: float,
) -> dict[str, object]:
    stem = relative.with_suffix("")
    paths = {
        "comparison": output / "final_comparisons" / stem.with_suffix(".png"),
        "solid_overlay": output / "solid_model_overlays" / stem.with_suffix(".png"),
        "dashed_overlay": output / "dashed_model_overlays" / stem.with_suffix(".png"),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    solid_bond = solid == 2
    dashed_bond = dashed == 2
    bond_union = solid_bond | dashed_bond
    solid_arrow = refine_solid_arrow(rgb, solid)
    dashed_arrow = refine_dashed_arrow(dashed, dashed_probability, dashed_postprocess)
    solid_erase = solid_arrow & ~bond_union
    dashed_erase = dashed_arrow & ~bond_union

    solid_metrics = bond_metrics(solid_bond, gt_bond, gt_valid, 0)
    dashed_metrics = bond_metrics(dashed_bond, gt_bond, gt_valid, 0)
    solid_image = model_overlay(
        rgb, solid_erase, solid_bond, gt_bond, gt_valid, alpha, 0
    )
    dashed_image = model_overlay(
        rgb, dashed_erase, dashed_bond, gt_bond, gt_valid, alpha, 0
    )
    solid_panel = annotated_panel(solid_image, "Legacy solid-arrow model", solid_metrics)
    dashed_panel = annotated_panel(
        dashed_image,
        f"Legacy dashed-arrow model + {dashed_postprocess.replace('_', '.')} postprocess",
        dashed_metrics,
    )
    comparison = np.hstack([solid_panel, dashed_panel])

    Image.fromarray(solid_panel).save(paths["solid_overlay"])
    Image.fromarray(dashed_panel).save(paths["dashed_overlay"])
    Image.fromarray(comparison).save(paths["comparison"])

    return {
        "source": str(source),
        "relative_path": relative.as_posix(),
        "width": int(rgb.shape[1]),
        "height": int(rgb.shape[0]),
        "gt_path": str(gt_path),
        "gt_encoding": gt_encoding,
        "solid_arrow_erased_pixels": int(solid_erase.sum()),
        "dashed_arrow_erased_pixels": int(dashed_erase.sum()),
        **{f"solid_{key}": value for key, value in solid_metrics.items()},
        **{f"dashed_{key}": value for key, value in dashed_metrics.items()},
        **{name: path.relative_to(output).as_posix() for name, path in paths.items()},
    }


def write_index(output: Path, records: list[dict[str, object]], metadata: dict) -> None:
    (output / "summary.json").write_text(
        json.dumps({"metadata": metadata, "pages": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    fields = list(records[0]) if records else ["source"]
    with (output / "summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    cards = []
    for record in records:
        image_url = quote(str(record["comparison"]), safe="/")
        title = html.escape(str(record["relative_path"]))
        cards.append(
            f'<article><h2>{title}</h2>'
            f'<p>实线：GT键重合 {record["solid_bond_overlap_pixels"]:,} px '
            f'({record["solid_bond_overlap_pred_percent"]:.2f}%)；误报 '
            f'{record["solid_false_bond_pixels"]:,} px '
            f'({record["solid_false_bond_pred_percent"]:.2f}%)</p>'
            f'<p>虚线：GT键重合 {record["dashed_bond_overlap_pixels"]:,} px '
            f'({record["dashed_bond_overlap_pred_percent"]:.2f}%)；误报 '
            f'{record["dashed_false_bond_pixels"]:,} px '
            f'({record["dashed_false_bond_pred_percent"]:.2f}%)</p>'
            f'<a href="{image_url}"><img loading="lazy" src="{image_url}" alt="{title}"></a>'
            f'<p><a href="{image_url}">打开原始分辨率对照图</a></p></article>'
        )
    document = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>旧双模型化学键 GT 对照</title><style>
body{font-family:Arial,"Microsoft YaHei",sans-serif;margin:24px;background:#f5f6f7;color:#17202a}
header{position:sticky;top:0;background:white;padding:14px 18px;border-radius:8px;z-index:2;box-shadow:0 2px 10px #0002}
main{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:18px;margin-top:20px}
article{background:white;padding:14px;border-radius:8px;box-shadow:0 1px 5px #0002;overflow:hidden}
h2{font-size:15px;word-break:break-all}p{font-size:13px}img{width:100%;height:520px;object-fit:contain;background:#eee}
b{display:inline-block;width:14px;height:14px;margin:0 5px 0 16px;vertical-align:-2px}
</style></head><body><header><h1>旧实线/虚线模型：化学键 GT 对照</h1>
<p>紫＝最终箭头擦除区；绿＝与 GT 重合的化学键；红＝实际不是化学键的误报。</p>
</header><main>""" + "\n".join(cards) + "</main></body></html>"
    (output / "index.html").write_text(document, encoding="utf-8")


def annotated_crop_boxes(json_path: Path, page_shape: tuple[int, int]) -> list[dict[str, object]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    page_height, page_width = page_shape
    boxes: list[dict[str, object]] = []
    for order, shape in enumerate(payload.get("shapes", [])):
        evaluation = shape.get("evaluation") or {}
        bbox = evaluation.get("bbox_xyxy")
        if bbox is None:
            points = shape.get("points") or []
            if not points:
                continue
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            bbox = [min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10]
        x0 = max(0, int(np.floor(float(bbox[0]))))
        y0 = max(0, int(np.floor(float(bbox[1]))))
        x1 = min(page_width, int(np.ceil(float(bbox[2]))))
        y1 = min(page_height, int(np.ceil(float(bbox[3]))))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append({
            "bbox": (x0, y0, x1, y1),
            "shape_index": int(shape.get("shape_index", order)),
            "instance_id": str(shape.get("instance_id", "")),
        })
    boxes.sort(key=lambda item: (int(item["shape_index"]), str(item["instance_id"])))
    if not boxes:
        raise ValueError(f"No usable annotated crop boxes found in: {json_path}")
    return boxes


def combine_metric_rows(rows: list[dict[str, float | int]]) -> dict[str, float | int]:
    predicted = sum(int(row["predicted_bond_pixels"]) for row in rows)
    gt = sum(int(row["gt_bond_pixels"]) for row in rows)
    overlap_pixels = sum(int(row["bond_overlap_pixels"]) for row in rows)
    false_pixels = sum(int(row["false_bond_pixels"]) for row in rows)
    return {
        "predicted_bond_pixels": predicted,
        "gt_bond_pixels": gt,
        "bond_overlap_pixels": overlap_pixels,
        "bond_overlap_pred_percent": 100.0 * overlap_pixels / predicted if predicted else 0.0,
        "gt_covered_pixels": sum(int(row["gt_covered_pixels"]) for row in rows),
        "gt_coverage_percent": (
            100.0 * sum(int(row["gt_covered_pixels"]) for row in rows) / gt if gt else 0.0
        ),
        "false_bond_pixels": false_pixels,
        "false_bond_pred_percent": 100.0 * false_pixels / predicted if predicted else 0.0,
    }


def bond_strategy_row(
    page: str, crops: int, strategy: str, metrics: dict[str, float | int]
) -> dict[str, float | int | str]:
    precision = float(metrics["bond_overlap_pred_percent"])
    coverage = float(metrics["gt_coverage_percent"])
    f1 = 2.0 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return {
        "page": page,
        "crops": crops,
        "strategy": strategy,
        "gt_pixels": metrics["gt_bond_pixels"],
        "predicted_bond_pixels": metrics["predicted_bond_pixels"],
        "overlap_pixels": metrics["bond_overlap_pixels"],
        "wrong_pixels": metrics["false_bond_pixels"],
        "precision_percent": precision,
        "wrong_percent": metrics["false_bond_pred_percent"],
        "gt_coverage_percent": coverage,
        "f1_percent": f1,
    }


def crop_result_panel(
    visualization: np.ndarray,
    crop_index: int,
    metrics: dict[str, float | int],
    width: int = 420,
    height: int = 300,
) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(panel, f"crop {crop_index:02d}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (25, 25, 25), 1, cv2.LINE_AA)
    metric_text = (
        f"overlap {metrics['bond_overlap_pixels']:,}px "
        f"({metrics['bond_overlap_pred_percent']:.1f}% pred) | "
        f"wrong {metrics['false_bond_pixels']:,}px "
        f"({metrics['false_bond_pred_percent']:.1f}% pred)"
    )
    cv2.putText(panel, metric_text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.37, (55, 55, 55), 1, cv2.LINE_AA)
    area_top, area_bottom = 54, height - 8
    available_width = width - 16
    available_height = area_bottom - area_top
    scale = min(available_width / visualization.shape[1],
                available_height / visualization.shape[0])
    resized_width = max(1, int(round(visualization.shape[1] * scale)))
    resized_height = max(1, int(round(visualization.shape[0] * scale)))
    interpolation = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(visualization, (resized_width, resized_height),
                         interpolation=interpolation)
    x = (width - resized_width) // 2
    y = area_top + (available_height - resized_height) // 2
    panel[y:y + resized_height, x:x + resized_width] = resized
    cv2.rectangle(panel, (x, y), (x + resized_width - 1, y + resized_height - 1),
                  (200, 200, 200), 1)
    return panel


def original_crop_panel(
    crop: np.ndarray, crop_index: int, width: int = 420, height: int = 270
) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(panel, f"crop {crop_index:02d}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (25, 25, 25), 1, cv2.LINE_AA)
    area_top, area_bottom = 34, height - 8
    available_width = width - 16
    available_height = area_bottom - area_top
    scale = min(available_width / crop.shape[1], available_height / crop.shape[0])
    resized_width = max(1, int(round(crop.shape[1] * scale)))
    resized_height = max(1, int(round(crop.shape[0] * scale)))
    interpolation = cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(crop, (resized_width, resized_height), interpolation=interpolation)
    x = (width - resized_width) // 2
    y = area_top + (available_height - resized_height) // 2
    panel[y:y + resized_height, x:x + resized_width] = resized
    cv2.rectangle(panel, (x, y), (x + resized_width - 1, y + resized_height - 1),
                  (200, 200, 200), 1)
    return panel


def original_crop_montage(page_groups: list[dict[str, object]]) -> np.ndarray:
    columns = 4
    panel_width, panel_height = 420, 270
    gap, margin = 14, 24
    header_height, page_header_height = 72, 42
    canvas_width = margin * 2 + columns * panel_width + (columns - 1) * gap
    group_heights = []
    for group in page_groups:
        rows = (len(group["panels"]) + columns - 1) // columns
        group_heights.append(page_header_height + rows * panel_height + max(0, rows - 1) * gap + gap)
    canvas_height = header_height + sum(group_heights) + margin
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Original annotated molecule crops (before arrow removal)",
                (margin, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.88,
                (20, 20, 20), 2, cv2.LINE_AA)
    y = header_height
    for group, group_height in zip(page_groups, group_heights):
        cv2.putText(canvas, f"{group['page']}   crops={len(group['panels'])}",
                    (margin, y + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (25, 25, 25), 1, cv2.LINE_AA)
        y += page_header_height
        for index, panel in enumerate(group["panels"]):
            row, column = divmod(index, columns)
            x = margin + column * (panel_width + gap)
            py = y + row * (panel_height + gap)
            canvas[py:py + panel_height, x:x + panel_width] = panel
        y += group_height - page_header_height
    return canvas


def final_crop_montage(
    page_groups: list[dict[str, object]], overall: dict[str, float | int]
) -> np.ndarray:
    columns = 4
    panel_width, panel_height = 420, 300
    gap, margin = 14, 24
    header_height, page_header_height = 125, 42
    canvas_width = margin * 2 + columns * panel_width + (columns - 1) * gap
    group_heights = []
    for group in page_groups:
        rows = (len(group["panels"]) + columns - 1) // columns
        group_heights.append(page_header_height + rows * panel_height + max(0, rows - 1) * gap + gap)
    canvas_height = header_height + sum(group_heights) + margin
    canvas = np.full((canvas_height, canvas_width, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Dual-model union after annotated molecule cropping",
                (margin, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                "purple=erased arrow   green=correct bond (prediction AND GT)   red=wrong bond prediction",
                (margin, 67), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (45, 45, 45), 1, cv2.LINE_AA)
    summary = (
        f"Bond overlap with GT: {overall['bond_overlap_pixels']:,} px "
        f"({overall['bond_overlap_pred_percent']:.2f}% of predicted bond)   |   "
        f"Wrong bond prediction: {overall['false_bond_pixels']:,} px "
        f"({overall['false_bond_pred_percent']:.2f}% of predicted bond)"
    )
    cv2.putText(canvas, summary, (margin, 99), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (25, 25, 25), 1, cv2.LINE_AA)
    y = header_height
    for group, group_height in zip(page_groups, group_heights):
        page_metrics = group["metrics"]
        page_text = (
            f"{group['page']}   crops={len(group['panels'])}   "
            f"overlap={page_metrics['bond_overlap_pixels']:,}px   "
            f"wrong={page_metrics['false_bond_pixels']:,}px"
        )
        cv2.putText(canvas, page_text, (margin, y + 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.56, (25, 25, 25), 1, cv2.LINE_AA)
        y += page_header_height
        for index, panel in enumerate(group["panels"]):
            row, column = divmod(index, columns)
            x = margin + column * (panel_width + gap)
            py = y + row * (panel_height + gap)
            canvas[py:py + panel_height, x:x + panel_width] = panel
        y += group_height - page_header_height
    return canvas


def main() -> None:
    args = parse_args()
    if not args.input.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {args.input}")
    gt_dir = args.gt_dir or args.input.parents[1] / "masks" / args.input.name
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"Expert pixel GT directory does not exist: {gt_dir}")
    if args.gt_tolerance < 0:
        raise ValueError("--gt-tolerance must be >= 0")
    gt_json_dir = args.gt_json_dir or args.input.parents[1] / "json" / args.input.name
    if not gt_json_dir.is_dir():
        raise FileNotFoundError(
            f"Approved annotation directory does not exist: {gt_json_dir}\n"
            "The JSON files are required for molecule crop boxes; set --gt-json-dir."
        )
    all_files = image_files(args.input)
    if args.pages:
        requested = {Path(name).stem for name in args.pages}
        by_stem = {path.stem: path for path in all_files}
        missing_pages = sorted(requested - set(by_stem))
        if missing_pages:
            raise FileNotFoundError(
                "Missing requested source pages under input: " + ", ".join(missing_pages)
            )
        files = [by_stem[Path(name).stem] for name in args.pages]
    else:
        files = all_files
    if args.limit:
        files = files[:args.limit]
    if not files:
        raise RuntimeError(f"No images found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading archived solid and dashed models...", flush=True)
    predictor = DualUnionPredictor(args.solid_model, args.dashed_model, args.device)
    print("Models loaded.", flush=True)
    try:
        gt_bond_values = {int(value.strip()) for value in args.gt_bond_values.split(",")}
    except ValueError as error:
        raise ValueError("--gt-bond-values must be comma-separated integers") from error
    if args.tolerance_sweep:
        sweep_start, sweep_stop, sweep_step = args.tolerance_sweep
        if sweep_start < 0 or sweep_stop < sweep_start or sweep_step <= 0:
            raise ValueError("--tolerance-sweep requires 0 <= START <= STOP and STEP > 0")
        count = int(np.floor((sweep_stop - sweep_start) / sweep_step + 1e-9))
        tolerances = [round(sweep_start + index * sweep_step, 8)
                      for index in range(count + 1)]
        if tolerances[-1] < sweep_stop - 1e-8:
            tolerances.append(float(sweep_stop))
    else:
        tolerances = [float(args.gt_tolerance)]
    start = time.time()
    raw_page_groups: list[dict[str, object]] = []
    original_page_groups: list[dict[str, object]] = []
    for index, source in enumerate(files, 1):
        relative = source.relative_to(args.input)
        with Image.open(source) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        gt_json_path = gt_json_dir / relative.with_suffix(".json")
        if not gt_json_path.is_file():
            raise FileNotFoundError(f"Missing approved annotation JSON: {gt_json_path}")
        gt_path = find_gt_mask(gt_dir, relative)
        page_gt_bond, page_gt_valid, _ = load_gt_bond(
            gt_path, rgb.shape[:2], gt_bond_values
        )
        boxes = annotated_crop_boxes(gt_json_path, rgb.shape[:2])
        raw_crops: list[dict[str, object]] = []
        original_panels: list[np.ndarray] = []
        for crop_number, box_info in enumerate(boxes):
            x0, y0, x1, y1 = box_info["bbox"]
            crop = rgb[y0:y1, x0:x1].copy()
            original_panels.append(original_crop_panel(crop, crop_number))
            crop_gt = page_gt_bond[y0:y1, x0:x1]
            crop_valid = page_gt_valid[y0:y1, x0:x1]
            solid, dashed, dashed_probability = predict_page(
                predictor, crop, args.tile_size, args.context, args.batch_size
            )
            solid_bond = solid == 2
            dashed_bond = dashed == 2
            bond_union = solid_bond | dashed_bond
            solid_arrow = refine_solid_arrow(crop, solid)
            dashed_arrow = refine_dashed_arrow(
                dashed, dashed_probability, args.dashed_postprocess
            )
            arrow_union = solid_arrow | dashed_arrow
            erase_mask = arrow_union & ~bond_union
            cleaned = crop.copy()
            cleaned[erase_mask] = 255
            raw_crops.append({
                "crop_number": crop_number,
                "cleaned": cleaned,
                "erase_mask": erase_mask,
                "solid_bond": solid_bond,
                "dashed_bond": dashed_bond,
                "bond_union": bond_union,
                "gt": crop_gt,
                "valid": crop_valid,
            })
        raw_page_groups.append({"page": source.stem, "crops": raw_crops})
        original_page_groups.append({"page": source.stem, "panels": original_panels})
        print(
            f"[{index}/{len(files)}] {relative}: inferred crops={len(raw_crops)}",
            flush=True,
        )

    originals_path = args.output / "cropped_originals.png"
    Image.fromarray(original_crop_montage(original_page_groups)).save(originals_path)
    sweep_rows: list[dict[str, float | int | str]] = []
    for tolerance in tolerances:
        page_groups: list[dict[str, object]] = []
        all_metric_rows: list[dict[str, float | int]] = []
        per_model_rows: list[dict[str, float | int | str]] = []
        strategy_rows: list[dict[str, float | int | str]] = []
        all_solid_rows: list[dict[str, float | int]] = []
        all_dashed_rows: list[dict[str, float | int]] = []
        for raw_group in raw_page_groups:
            panels: list[np.ndarray] = []
            page_metric_rows: list[dict[str, float | int]] = []
            page_solid_rows: list[dict[str, float | int]] = []
            page_dashed_rows: list[dict[str, float | int]] = []
            for raw_crop in raw_group["crops"]:
                metrics = bond_metrics(
                    raw_crop["bond_union"], raw_crop["gt"], raw_crop["valid"], tolerance
                )
                solid_metrics = bond_metrics(
                    raw_crop["solid_bond"], raw_crop["gt"], raw_crop["valid"], tolerance
                )
                dashed_metrics = bond_metrics(
                    raw_crop["dashed_bond"], raw_crop["gt"], raw_crop["valid"], tolerance
                )
                visualization = model_overlay(
                    raw_crop["cleaned"], raw_crop["erase_mask"], raw_crop["bond_union"],
                    raw_crop["gt"], raw_crop["valid"], args.alpha, tolerance,
                )
                panels.append(crop_result_panel(
                    visualization, int(raw_crop["crop_number"]), metrics
                ))
                page_metric_rows.append(metrics)
                page_solid_rows.append(solid_metrics)
                page_dashed_rows.append(dashed_metrics)
                all_metric_rows.append(metrics)
                all_solid_rows.append(solid_metrics)
                all_dashed_rows.append(dashed_metrics)
            solid_page = combine_metric_rows(page_solid_rows)
            dashed_page = combine_metric_rows(page_dashed_rows)
            union_page = combine_metric_rows(page_metric_rows)
            page_groups.append({
                "page": raw_group["page"],
                "panels": panels,
                "metrics": union_page,
            })
            page_name = str(raw_group["page"])
            crop_count = len(raw_group["crops"])
            strategy_rows.extend([
                bond_strategy_row(page_name, crop_count, "solid_bond", solid_page),
                bond_strategy_row(page_name, crop_count, "dashed_bond", dashed_page),
                bond_strategy_row(page_name, crop_count, "bond_union", union_page),
            ])
            per_model_rows.append({
                "page": raw_group["page"],
                "crops": len(raw_group["crops"]),
                "gt_pixels": solid_page["gt_bond_pixels"],
                "solid_predicted": solid_page["predicted_bond_pixels"],
                "solid_overlap": solid_page["bond_overlap_pixels"],
                "solid_wrong": solid_page["false_bond_pixels"],
                "solid_precision_percent": solid_page["bond_overlap_pred_percent"],
                "solid_false_percent": solid_page["false_bond_pred_percent"],
                "solid_gt_coverage_percent": solid_page["gt_coverage_percent"],
                "dashed_predicted": dashed_page["predicted_bond_pixels"],
                "dashed_overlap": dashed_page["bond_overlap_pixels"],
                "dashed_wrong": dashed_page["false_bond_pixels"],
                "dashed_precision_percent": dashed_page["bond_overlap_pred_percent"],
                "dashed_false_percent": dashed_page["false_bond_pred_percent"],
                "dashed_gt_coverage_percent": dashed_page["gt_coverage_percent"],
            })
        overall = combine_metric_rows(all_metric_rows)
        solid_overall = combine_metric_rows(all_solid_rows)
        dashed_overall = combine_metric_rows(all_dashed_rows)
        total_crops = sum(len(group["crops"]) for group in raw_page_groups)
        strategy_rows.extend([
            bond_strategy_row("TOTAL", total_crops, "solid_bond", solid_overall),
            bond_strategy_row("TOTAL", total_crops, "dashed_bond", dashed_overall),
            bond_strategy_row("TOTAL", total_crops, "bond_union", overall),
        ])
        per_model_rows.append({
            "page": "TOTAL",
            "crops": total_crops,
            "gt_pixels": solid_overall["gt_bond_pixels"],
            "solid_predicted": solid_overall["predicted_bond_pixels"],
            "solid_overlap": solid_overall["bond_overlap_pixels"],
            "solid_wrong": solid_overall["false_bond_pixels"],
            "solid_precision_percent": solid_overall["bond_overlap_pred_percent"],
            "solid_false_percent": solid_overall["false_bond_pred_percent"],
            "solid_gt_coverage_percent": solid_overall["gt_coverage_percent"],
            "dashed_predicted": dashed_overall["predicted_bond_pixels"],
            "dashed_overlap": dashed_overall["bond_overlap_pixels"],
            "dashed_wrong": dashed_overall["false_bond_pixels"],
            "dashed_precision_percent": dashed_overall["bond_overlap_pred_percent"],
            "dashed_false_percent": dashed_overall["false_bond_pred_percent"],
            "dashed_gt_coverage_percent": dashed_overall["gt_coverage_percent"],
        })
        montage = final_crop_montage(page_groups, overall)
        tolerance_tag = f"{tolerance:g}".replace(".", "p")
        final_path = args.output / f"final_visualization_tolerance_{tolerance_tag}px.png"
        Image.fromarray(montage).save(final_path)
        per_model_path = args.output / f"per_model_bond_metrics_tolerance_{tolerance_tag}px.csv"
        with per_model_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_model_rows[0]))
            writer.writeheader()
            writer.writerows(per_model_rows)
        comparison_path = args.output / f"bond_strategy_comparison_tolerance_{tolerance_tag}px.csv"
        with comparison_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(strategy_rows[0]))
            writer.writeheader()
            writer.writerows(strategy_rows)
        sweep_rows.append({
            "tolerance_pixels": tolerance,
            "predicted_bond_pixels": overall["predicted_bond_pixels"],
            "matched_bond_pixels": overall["bond_overlap_pixels"],
            "matched_prediction_percent": overall["bond_overlap_pred_percent"],
            "wrong_bond_pixels": overall["false_bond_pixels"],
            "wrong_prediction_percent": overall["false_bond_pred_percent"],
            "visualization": final_path.name,
        })
        print(
            f"tolerance={tolerance:g}px: matched={overall['bond_overlap_pixels']:,}; "
            f"wrong={overall['false_bond_pixels']:,} "
            f"({overall['false_bond_pred_percent']:.4f}%)", flush=True
        )
    if len(sweep_rows) > 1:
        sweep_path = args.output / "tolerance_sweep.csv"
        with sweep_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0]))
            writer.writeheader()
            writer.writerows(sweep_rows)
    print(
        f"Done: pages={len(raw_page_groups)}; "
        f"crops={sum(len(group['crops']) for group in raw_page_groups)}; "
        f"tolerances={len(tolerances)}; elapsed={time.time() - start:.2f}s; "
        f"originals={originals_path}", flush=True
    )


if __name__ == "__main__":
    main()
