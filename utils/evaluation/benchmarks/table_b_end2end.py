"""表 B：端到端 page→SMILES 对比（DECIMER vs Ours）。

把 DECIMER 作为完整 OCSR 系统，在 ``PageToSMILES_approved``（49 页 / 469 分子）
上跑 page→SMILES，与 Ours（表 A ``full`` 配置）用同一套识别指标横向对比。

DECIMER 端到端 = DECIMER-Seg（Mask R-CNN 分割，``get_expanded_masks``）
            → 每个结构 mask bbox 裁剪 → DECIMER OCSR（``predict_SMILES``）。
Ours 端到端 = YOLO+洪水填充+候选筛选 → 图像优化 → MolScribe（复用表 A full）。

口径说明
--------
DECIMER 只输出字符串 SMILES：超原子/缩写已展开，R 基团记作 ``[X]``/``[R]``/``[Rn]``/
``[Y]``/``[Z]``（tokenizer pattern ``R([0-9]*)|X([0-9]*)|Y([0-9]*)|Z([0-9]*)``）。
GT 是 CXSMILES：超原子/R 基团统一 ``*`` + ``|$...$|`` label。因此对比前把
DECIMER 的 R 基团 token 还原为裸 ``*``（``decimer_preprocess``），再走与表 A 完全
相同的 ``utils.base.smiles_canon.canonicalize``（label 保留）算 CEM。含超原子/缩写的
分子 DECIMER 因展开结构 + 无法还原 label，CEM 天然为 0 —— 这正是端到端对比要
暴露的差异。

用法
----
  python -m utils.evaluation.benchmarks.table_b_end2end --max-pages 2
  python -m utils.evaluation.benchmarks.table_b_end2end
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.base.paths import PROJECT_ROOT

SEG_ROOT = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal")
TEST_SET_ROOT = SEG_ROOT / "images" / "PageToSMILES_approved"
TEST_SET_JSONL = SEG_ROOT / "PageToSMILES_approved.jsonl"
LABEL_TABLE_PATH = SEG_ROOT / "label_table.json"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation_results_tableB_e2e"
OURS_SUMMARY = PROJECT_ROOT / "evaluation_results_ablation" / "full" / "summary.json"

IOU_THRESHOLD = 0.5
MORGAN_RADIUS = 2

# DECIMER R 基团 token：R/X/Y/Z 开头、后面是数字（可再跟字母后缀）或直接 ]。
# 形式如 [R]/[R1]/[R4e]/[R15a]/[X]/[Y]/[Z]，统一还原为 *。
# 负向先行 (?![a-z]) 排除 [Re]/[Rh]/[Rn]/[Ru]/[Ra]/[Rb]/[Xe] 等真实元素
# （R 后直接跟字母 = 元素，跟数字 = R 基团）。
_RGROUP_RE = re.compile(r"\[([RXYZ])(?![a-z])([0-9a-z]*)\]")


def decimer_preprocess(smiles: str) -> str:
    """把 DECIMER 的 R 基团 placeholder 还原为裸 ``*``，与 GT 的 dummy 原子对齐。"""
    if not smiles:
        return ""
    return _RGROUP_RE.sub("*", smiles)


def load_gt_pages(max_pages: int = None) -> List[Dict]:
    pages = []
    with open(TEST_SET_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
            if max_pages and len(pages) >= max_pages:
                break
    return pages


def compute_iou(box1, box2) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    inter = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def greedy_match(pred_boxes, gt_boxes, iou_threshold: float = IOU_THRESHOLD):
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    iou = [[compute_iou(p, g) for g in gt_boxes] for p in pred_boxes]
    matched, used_p, used_g = [], set(), set()
    while True:
        best, bi, bj = 0.0, -1, -1
        for i in range(n_pred):
            if i in used_p:
                continue
            for j in range(n_gt):
                if j in used_g:
                    continue
                if iou[i][j] > best:
                    best, bi, bj = iou[i][j], i, j
        if best >= iou_threshold:
            matched.append((bi, bj, best)); used_p.add(bi); used_g.add(bj)
        else:
            break
    unmatched_p = [i for i in range(n_pred) if i not in used_p]
    unmatched_g = [j for j in range(n_gt) if j not in used_g]
    return matched, unmatched_p, unmatched_g


def run_decimer(max_pages: int = None) -> List[Dict]:
    """DECIMER 端到端：每页分割 + OCSR + 匹配 + 识别指标。"""
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

    from decimer_segmentation import get_expanded_masks
    from DECIMER import predict_SMILES
    from utils.base.smiles_canon import canonicalize, tanimoto

    with open(LABEL_TABLE_PATH, "r", encoding="utf-8") as f:
        label_table = json.load(f)

    pages = load_gt_pages(max_pages)
    per_page = []

    with tempfile.TemporaryDirectory(prefix="decimer_e2e_") as tmp:
        tmp = Path(tmp)
        for idx, page in enumerate(pages, 1):
            page_path = TEST_SET_ROOT / page["page_file"]
            img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[{idx}] 读图失败 {page['page_file']}")
                continue

            masks = get_expanded_masks(img)
            if masks is None or masks.ndim != 3 or masks.shape[2] == 0:
                masks = np.zeros((img.shape[0], img.shape[1], 0), dtype=bool)

            pred_boxes, pred_smiles = [], []
            for i in range(masks.shape[2]):
                m = masks[:, :, i].astype(bool)
                ys, xs = np.nonzero(m)
                if xs.size == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                crop = img[y1:y2, x1:x2].copy()
                crop[~m[y1:y2, x1:x2]] = 255  # mask 外涂白，隔离单结构
                crop_path = tmp / f"{page['page_id']}_{i}.png"
                cv2.imwrite(str(crop_path), crop)
                try:
                    raw = predict_SMILES(str(crop_path))
                except Exception:
                    raw = ""
                pred_boxes.append([x1, y1, x2, y2])
                pred_smiles.append(decimer_preprocess(raw))

            gt_mols = page["molecules"]
            gt_boxes = [m["bbox_xyxy"] for m in gt_mols]
            matched, up, ug = greedy_match(pred_boxes, gt_boxes)

            matched_results = []
            for pi, gi, iou in matched:
                pred_s = pred_smiles[pi]
                gt_s = gt_mols[gi]["smiles"]
                pred_canon = canonicalize(pred_s, label_table)
                gt_canon = canonicalize(gt_s, label_table)
                pred_valid = pred_canon is not None
                gt_valid = gt_canon is not None
                canonical_match = pred_valid and gt_valid and pred_canon == gt_canon
                t = tanimoto(pred_s, gt_s, radius=MORGAN_RADIUS) \
                    if (pred_valid and gt_valid) else None
                matched_results.append({
                    "pred_idx": pi, "gt_idx": gi, "iou": iou,
                    "pred_smiles": pred_s,
                    "pred_canonical": pred_canon,
                    "gt_smiles": gt_s,
                    "gt_canonical": gt_canon,
                    "pred_valid": pred_valid, "gt_valid": gt_valid,
                    "canonical_match": canonical_match,
                    "tanimoto": t,
                    "gt_molecule_id": gt_mols[gi]["molecule_id"],
                })

            per_page.append({
                "page_id": page["page_id"],
                "page_file": page["page_file"],
                "n_gt": len(gt_mols),
                "n_pred": len(pred_boxes),
                "n_matched": len(matched),
                "n_fp": len(up),
                "n_fn": len(ug),
                "matched_results": matched_results,
            })
            print(f"[{idx}/{len(pages)}] {page['page_file']}  "
                  f"GT={len(gt_mols)}  Pred={len(pred_boxes)}  Matched={len(matched)}")

    return per_page


def aggregate(per_page: List[Dict]) -> Dict:
    n_gt = sum(p["n_gt"] for p in per_page)
    n_pred = sum(p["n_pred"] for p in per_page)
    n_matched = sum(p["n_matched"] for p in per_page)
    all_matched = [m for p in per_page for m in p["matched_results"]]

    if n_matched > 0:
        validity = sum(1 for m in all_matched if m["pred_valid"]) / n_matched
        canonical_em = sum(1 for m in all_matched if m["canonical_match"]) / n_matched
        t_scores = [m["tanimoto"] for m in all_matched if m["tanimoto"] is not None]
        tanimoto_mean = float(np.mean(t_scores)) if t_scores else 0.0
    else:
        validity = canonical_em = tanimoto_mean = 0.0

    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    tanimoto_sum = sum((m["tanimoto"] or 0.0) for m in all_matched)
    tanimoto_all_gt = tanimoto_sum / n_gt if n_gt else 0.0

    return {
        "n_pages": len(per_page),
        "n_ground_truth_total": n_gt,
        "n_predictions_total": n_pred,
        "n_matched": n_matched,
        "n_false_positives": sum(p["n_fp"] for p in per_page),
        "n_false_negatives": sum(p["n_fn"] for p in per_page),
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "coverage": recall,
        "validity": validity,
        "canonical_smiles_em": canonical_em,
        "tanimoto_similarity_mean": tanimoto_mean,
        "tanimoto_at_all_gt": tanimoto_all_gt,
    }


def load_ours() -> Dict:
    with open(OURS_SUMMARY, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="表B 端到端 page→SMILES 对比")
    ap.add_argument("--max-pages", type=int, default=None)
    args = ap.parse_args()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    per_page = run_decimer(args.max_pages)
    agg = aggregate(per_page)
    (OUTPUT_ROOT / "decimer").mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ROOT / "decimer" / "per_page.json", "w", encoding="utf-8") as f:
        json.dump(per_page, f, indent=2, ensure_ascii=False)
    with open(OUTPUT_ROOT / "decimer" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)

    ours = load_ours()
    rows = [
        ("Ours (full)", ours),
        ("DECIMER", agg),
    ]
    print("\n" + "=" * 78)
    print("表 B：端到端 page→SMILES 对比（同一 GT、同一 IoU@0.5 匹配、同一 CEM 口径）")
    print("=" * 78)
    header = (f"{'方法':<14}{'GT':>5}{'Pred':>6}{'Mat':>6}{'P':>7}{'R':>7}{'F1':>7}"
              f"{'Cov':>7}{'Val':>7}{'CEM':>7}{'Tan':>8}{'T@allGT':>9}")
    print(header)
    print("-" * 78)
    for label, s in rows:
        print(f"{label:<14}{s['n_ground_truth_total']:>5}{s['n_predictions_total']:>6}"
              f"{s['n_matched']:>6}{s['detection_precision']:>7.1%}"
              f"{s['detection_recall']:>7.1%}{s['detection_f1']:>7.1%}"
              f"{s['coverage']:>7.1%}{s['validity']:>7.1%}"
              f"{s['canonical_smiles_em']:>7.1%}{s['tanimoto_similarity_mean']:>8.3f}"
              f"{s['tanimoto_at_all_gt']:>9.4f}")
    print("=" * 78)
    print(f"→ 结果在 {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
