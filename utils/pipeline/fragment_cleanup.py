"""GT-free cleanup of neighboring and clipped fragments in molecule crops.

The decision uses only crop pixels, the target's original YOLO polygon, other
predictions on the same page, and optional page-level straight-line evidence.
It never uses a filename, annotation, SMILES, or manually drawn region.
Ambiguous components are preserved and reported for review.

This is the reusable production copy of the algorithm frozen in
``outputs/cc_fragment_cleanup_approved50_20261020/cleanup_algorithm.py``.
"""
from __future__ import annotations

import cv2
import numpy as np


PARAMETERS = {
    "close_radius": 1,
    "neighbor_coverage_min": 0.90,
    "target_coverage_max_for_neighbor": 0.50,
    "coverage_advantage_min": 0.50,
    "duplicate_box_iou_max": 0.25,
    "duplicate_box_containment_max": 0.60,
    "neighbor_component_area_ratio_max": 0.50,
    "external_target_coverage_max": 0.02,
    "external_component_area_ratio_max": 0.75,
    "clipped_edge_distance_px": 1,
    "core_coverage_min": 0.60,
    "core_gap_fraction_of_crop_short_side": 0.025,
    "core_gap_min_px": 3.0,
    "page_line_min_length_fraction": 0.08,
    "page_line_min_length_px": 80,
}


def polygon_mask(shape, polygon):
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask


def box_relation(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter), inter / min(aa, bb)


def page_lines(rgb):
    """Find long axial page runs as evidence, not an unconditional erase."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    h, w = ink.shape
    masks = []
    for horizontal, dim in ((True, w), (False, h)):
        length = max(
            PARAMETERS["page_line_min_length_px"],
            round(dim * PARAMETERS["page_line_min_length_fraction"]),
        )
        bridge = (3, 1) if horizontal else (1, 3)
        axis = (length, 1) if horizontal else (1, length)
        work = cv2.morphologyEx(
            ink, cv2.MORPH_CLOSE, np.ones(bridge[::-1], np.uint8))
        lines = cv2.morphologyEx(
            work, cv2.MORPH_OPEN, np.ones(axis[::-1], np.uint8))
        masks.append(cv2.dilate(lines, np.ones((3, 3), np.uint8)))
    return np.maximum(*masks)


def clean_crop(rgb, own_polygon, raw_bbox, neighbor_predictions,
               page_line_mask=None):
    """Return cleaned RGB pixels, exact erasure mask, and geometric evidence."""
    h, w = rgb.shape[:2]
    own = polygon_mask((h, w), own_polygon)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    threshold, ink = cv2.threshold(
        gray, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    close_size = 2 * PARAMETERS["close_radius"] + 1
    work = cv2.morphologyEx(
        ink,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_size, close_size)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    empty = {
        "components": [],
        "removed_groups": [],
        "review_groups": [],
        "main_cc": None,
        "threshold": float(threshold),
    }
    if n <= 1:
        return rgb.copy(), np.zeros((h, w), bool), empty

    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    main_area = int(stats[main, cv2.CC_STAT_AREA])
    neighbors = []
    for pred in neighbor_predictions:
        iou, containment = box_relation(raw_bbox, pred["raw_bbox"])
        if (iou > PARAMETERS["duplicate_box_iou_max"]
                or containment > PARAMETERS["duplicate_box_containment_max"]):
            continue
        mask = polygon_mask((h, w), pred["polygon"])
        if mask.any():
            neighbors.append((pred["name"], mask))

    records = []
    core = np.zeros((h, w), bool)
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        area = len(xs)
        cov = float(own[ys, xs].mean())
        foreign_cov, foreign_name = 0.0, None
        for name, mask in neighbors:
            support = float(mask[ys, xs].mean())
            if support > foreign_cov:
                foreign_cov, foreign_name = support, name
        x, y, bw, bh = map(int, stats[i, :4])
        line_cov = (
            float(page_line_mask[ys, xs].mean())
            if page_line_mask is not None else 0.0
        )
        record = {
            "cc": i,
            "bbox_xyxy": [x, y, x + bw, y + bh],
            "closed_area": area,
            "ink_pixels": int(ink[ys, xs].sum()),
            "area_ratio": area / main_area,
            "target_coverage": cov,
            "neighbor_coverage": foreign_cov,
            "neighbor": foreign_name,
            "edge_distance": min(x, y, w - x - bw, h - y - bh),
            "page_line_coverage": line_cov,
            "action": "keep",
            "reason": "preserved",
        }
        records.append(record)
        if i == main or cov >= PARAMETERS["core_coverage_min"]:
            core[ys, xs] = True

    core_distance = cv2.distanceTransform(
        (~core).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    min_gap = max(
        PARAMETERS["core_gap_min_px"],
        PARAMETERS["core_gap_fraction_of_crop_short_side"] * min(h, w),
    )
    remove_ids = []
    for record in records:
        i = record["cc"]
        cc = labels == i
        record["distance_to_supported_core_px"] = float(
            core_distance[cc].min())
        if i == main:
            record["reason"] = "main_component_protected"
            continue
        neighbor_owned = (
            record["neighbor_coverage"] >= PARAMETERS["neighbor_coverage_min"]
            and record["target_coverage"]
            <= PARAMETERS["target_coverage_max_for_neighbor"]
            and record["neighbor_coverage"] - record["target_coverage"]
            >= PARAMETERS["coverage_advantage_min"]
            and record["area_ratio"]
            <= PARAMETERS["neighbor_component_area_ratio_max"]
        )
        clipped_external = (
            record["target_coverage"]
            <= PARAMETERS["external_target_coverage_max"]
            and record["edge_distance"]
            <= PARAMETERS["clipped_edge_distance_px"]
            and record["area_ratio"]
            <= PARAMETERS["external_component_area_ratio_max"]
            and record["distance_to_supported_core_px"] >= min_gap
        )
        if neighbor_owned:
            record.update(
                action="remove", reason="neighbor_instance_support")
        elif clipped_external:
            reason = (
                "external_clipped_component_with_page_line"
                if record["page_line_coverage"] >= 0.20
                else "external_clipped_component"
            )
            record.update(action="remove", reason=reason)
        elif record["target_coverage"] < 0.50 and record["ink_pixels"] >= 6:
            record.update(
                action="review", reason="ambiguous_external_component_preserved")
        if record["action"] == "remove":
            remove_ids.append(i)

    removed_cc = np.isin(labels, remove_ids)
    line_erase = np.zeros((h, w), bool)
    if page_line_mask is not None:
        candidate = ((page_line_mask > 0) & (ink > 0)).astype(np.uint8)
        nc, line_labels, line_stats, _ = cv2.connectedComponentsWithStats(
            candidate, 8)
        for j in range(1, nc):
            component = line_labels == j
            x, y, bw, bh, area = map(int, line_stats[j])
            horizontal = bw >= 0.70 * w and bh <= max(4, 0.06 * h)
            vertical = bh >= 0.70 * h and bw <= max(4, 0.06 * w)
            near_edge = (
                min(x, y, w - x - bw, h - y - bh) <= 0.12 * min(h, w))
            if not (horizontal or vertical) or not near_edge:
                continue
            if float(own[component].mean()) > PARAMETERS[
                    "external_target_coverage_max"]:
                continue
            if core_distance[component].min() < min_gap:
                continue
            if np.all(removed_cc[component]):
                continue
            line_erase |= component
            records.append({
                "cc": f"line_{j}",
                "bbox_xyxy": [x, y, x + bw, y + bh],
                "ink_pixels": area,
                "action": "remove",
                "reason": "page_supported_external_border_line",
                "target_coverage": float(own[component].mean()),
            })

    erase_seed = removed_cc | line_erase
    erase = np.zeros((h, w), bool)
    if erase_seed.any():
        dt_remove = cv2.distanceTransform(
            (~erase_seed).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        kept_ink = (ink > 0) & ~erase_seed
        dt_keep = cv2.distanceTransform(
            (~kept_ink).astype(np.uint8),
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )
        nonwhite = np.any(rgb < 255, axis=2)
        erase = (dt_remove <= 1.5) & (dt_remove < dt_keep) & nonwhite

    result = rgb.copy()
    result[erase] = 255
    return result, erase, {
        "threshold": float(threshold),
        "main_cc": main,
        "main_area": main_area,
        "minimum_external_gap_px": min_gap,
        "components": records,
        "removed_groups": [
            record for record in records if record["action"] == "remove"],
        "review_groups": [
            record for record in records if record["action"] == "review"],
    }


def cleanup_candidates(candidates, page_rgb, output_dir, visualize=False):
    """Run the current page-aware fragment-cleanup stage.

    This is the public pipeline entry.  Detailed page/candidate orchestration
    lives in a private implementation module; callers should import only this
    function from :mod:`utils.pipeline.fragment_cleanup`.
    """
    from utils.pipeline._fragment_cleanup_runner import cleanup_candidates_impl

    return cleanup_candidates_impl(
        candidates, page_rgb, output_dir, visualize=visualize)


def main():
    """Command-line entry for the fragment-cleanup stage."""
    from utils.pipeline._fragment_cleanup_runner import main as runner_main

    runner_main()


if __name__ == "__main__":
    main()

