"""表5四方法评测共享的数据、匹配与评分逻辑。

评分函数直接复用 ``utils.experiments.ablation_v2``，保证 CEM/GEM 与消融
实验完全相同；本模块只负责数据校验、IoU 贪心匹配和结果汇总。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

EXPECTED_PAGES = 50
EXPECTED_GT = 562
IOU_THRESHOLD = 0.5


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                    encoding="utf-8")
    temp.replace(path)


def load_page_checkpoints(output: Path) -> Dict[str, Dict]:
    """读取最终清单和逐页检查点，后者优先。"""
    output = Path(output)
    completed: Dict[str, Dict] = {}
    if output.is_file():
        try:
            completed.update({row["page_id"]: row for row in read_jsonl(output)})
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            pass
    checkpoint_dir = output.parent / "pages"
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.glob("*.json"):
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
                completed[row["page_id"]] = row
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                continue
    return completed


def write_page_checkpoint(output: Path, row: Dict) -> None:
    """原子写入单页结果，供中断后恢复。"""
    write_json(Path(output).parent / "pages" / f"{row['page_id']}.json", row)


def validate_dataset(rows: List[Dict], allow_subset: bool = False) -> Tuple[int, int]:
    pages = len(rows)
    n_gt = sum(len(row.get("molecules", [])) for row in rows)
    if not allow_subset and (pages != EXPECTED_PAGES or n_gt != EXPECTED_GT):
        raise ValueError(
            f"表5数据集必须是 {EXPECTED_PAGES} 页 / {EXPECTED_GT} GT，"
            f"当前读取到 {pages} 页 / {n_gt} GT")
    return pages, n_gt


def compute_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def greedy_match(pred_boxes, gt_boxes, threshold: float = IOU_THRESHOLD):
    matrix = [[compute_iou(pred, gt) for gt in gt_boxes] for pred in pred_boxes]
    used_pred, used_gt, matches = set(), set(), []
    while True:
        best, best_i, best_j = 0.0, -1, -1
        for i in range(len(pred_boxes)):
            if i in used_pred:
                continue
            for j in range(len(gt_boxes)):
                if j not in used_gt and matrix[i][j] > best:
                    best, best_i, best_j = float(matrix[i][j]), i, j
        if best < threshold:
            break
        matches.append((best_i, best_j, best))
        used_pred.add(best_i)
        used_gt.add(best_j)
    return (matches,
            [i for i in range(len(pred_boxes)) if i not in used_pred],
            [j for j in range(len(gt_boxes)) if j not in used_gt])


def score_manifest(method: str, gt_rows: List[Dict], manifest_rows: List[Dict]) -> Dict:
    """按 ``ablation_v2.legacy_compare`` 对带 SMILES 的实例清单评分。"""
    # 延迟导入，分割专用环境不需要 RDKit、MolScribe 或箭头模型依赖。
    from utils.experiments.ablation_v2 import LEGACY_SCORING_VERSION, legacy_compare

    by_page = {row["page_id"]: row for row in manifest_rows}
    total_gt = total_pred = total_matched = 0
    n_valid = n_cem = n_gem = 0
    per_page = []
    for gt_page in gt_rows:
        page_id = gt_page["page_id"]
        gt_mols = gt_page.get("molecules", [])
        predictions = by_page.get(page_id, {}).get("instances", [])
        matches, unmatched_pred, unmatched_gt = greedy_match(
            [x["bbox"] for x in predictions],
            [x["bbox_xyxy"] for x in gt_mols])
        details = []
        for pred_i, gt_i, iou in matches:
            score = legacy_compare(gt_mols[gt_i]["smiles"],
                                   predictions[pred_i].get("pred_smiles", ""))
            n_valid += int(score["scoring_valid"])
            n_cem += int(score["cem"])
            n_gem += int(score["gem"])
            details.append({
                "pred_idx": pred_i, "gt_idx": gt_i, "iou": iou,
                "gt_molecule_id": gt_mols[gt_i].get("molecule_id"),
                "pred_smiles": predictions[pred_i].get("pred_smiles", ""),
                "canonical_match": score["cem"], "graph_match": score["gem"],
                "pred_valid": score["scoring_valid"],
            })
        total_gt += len(gt_mols)
        total_pred += len(predictions)
        total_matched += len(matches)
        per_page.append({
            "page_id": page_id, "page_file": gt_page["page_file"],
            "n_ground_truth": len(gt_mols), "n_predictions": len(predictions),
            "n_matched": len(matches), "n_false_positives": len(unmatched_pred),
            "n_false_negatives": len(unmatched_gt), "matched_results": details,
        })
    summary = {
        "method": method, "dataset": "PJChemPageToSMILES",
        "n_pages": len(gt_rows), "n_ground_truth_total": total_gt,
        "n_predictions_total": total_pred, "n_matched": total_matched,
        "n_valid": n_valid, "n_cem": n_cem, "n_gem": n_gem,
        "Validity": n_valid / total_matched if total_matched else 0.0,
        "CEM": n_cem / total_gt if total_gt else 0.0,
        "GEM": n_gem / total_gt if total_gt else 0.0,
        "scoring_version": LEGACY_SCORING_VERSION,
        "cem_gem_denominator": "all_ground_truth",
        "iou_threshold": IOU_THRESHOLD,
    }
    return {"summary": summary, "pages": per_page}
