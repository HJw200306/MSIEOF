"""表 B：三系统检测级对比（Ours / DECIMER-Seg / YoDe-Seg）。

把三个"分子结构定位"系统放到同一数据集（PageToSMILES_approved，49 页 / 469 分子）
上，用同一套 IoU 贪心匹配与检测指标对比。三者的共同可比层是**检测/分割**：
DECIMER-Image-Segmentation（Mask R-CNN）与 YoDe-Segmentation（YOLOv5+DeepLabv3）
都只做结构定位、不产出 SMILES，因此表 B 用检测层指标（P / R / F1 / Coverage）。

- Ours：直接复用表 A 的 ``full`` 配置检测结果（同一 GT、同一 IoU 阈值）。
- DECIMER-Seg：``decimer_segmentation.get_expanded_masks``（TensorFlow，CPU）。
- YoDe-Seg：仓库 ``YoDe-Segmentation`` 的检测输出（PyTorch）。

方法间进程隔离（TF 与 torch 不同进程），每个 ``--method`` 单独跑、结果落盘 JSON，
最后 ``all`` 聚合对比，避免同进程同时 import tensorflow 与 torch。

用法
----
  python -m utils.evaluation.benchmarks.table_b_detection --method decimer --max-pages 2
  python -m utils.evaluation.benchmarks.table_b_detection --method all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from utils.base.paths import PROJECT_ROOT

SEG_ROOT = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal")
TEST_SET_ROOT = SEG_ROOT / "images" / "PageToSMILES_approved"
TEST_SET_JSONL = SEG_ROOT / "PageToSMILES_approved.jsonl"
OUTPUT_ROOT = PROJECT_ROOT / "evaluation_results_tableB"

IOU_THRESHOLD = 0.5


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
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
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


def detection_metrics(n_gt, n_pred, n_matched) -> Dict:
    precision = n_matched / n_pred if n_pred else 0.0
    recall = n_matched / n_gt if n_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n_ground_truth": n_gt,
        "n_predictions": n_pred,
        "n_matched": n_matched,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "coverage": recall,
    }


def _page_boxes(page_data: Dict) -> List[List[float]]:
    return [m["bbox_xyxy"] for m in page_data["molecules"]]


# --------------------------------------------------------------------------
# DECIMER-Seg（TensorFlow，CPU）
# --------------------------------------------------------------------------
def run_decimer(max_pages: int = None) -> List[Dict]:
    import cv2
    import numpy as np
    from dataset_eval.eval_msirost import predict_decimer

    pages = load_gt_pages(max_pages)
    per_page = []
    for page in pages:
        page_path = TEST_SET_ROOT / page["page_file"]
        insts = predict_decimer.predict_decimer_for_page(page_path)
        pred_boxes = [list(i["bbox"]) for i in insts]
        gt_boxes = _page_boxes(page)
        matched, up, ug = greedy_match(pred_boxes, gt_boxes)
        per_page.append({
            "page_id": page["page_id"],
            "n_gt": len(gt_boxes),
            "n_pred": len(pred_boxes),
            "n_matched": len(matched),
            "n_fp": len(up),
            "n_fn": len(ug),
        })
    return per_page


# --------------------------------------------------------------------------
# YoDe-Seg（PyTorch）—— 检测 bbox，占位待仓库/权重就绪后填充
# --------------------------------------------------------------------------
def run_yode(max_pages: int = None) -> List[Dict]:
    raise NotImplementedError("YoDe-Seg 适配器待仓库/权重就绪后实现")


# --------------------------------------------------------------------------
# 聚合 + 对比
# --------------------------------------------------------------------------
def aggregate(per_page: List[Dict]) -> Dict:
    n_gt = sum(p["n_gt"] for p in per_page)
    n_pred = sum(p["n_pred"] for p in per_page)
    n_matched = sum(p["n_matched"] for p in per_page)
    return detection_metrics(n_gt, n_pred, n_matched)


def load_ours_detection() -> Dict:
    """从表 A 的 full 配置 summary 复用检测层结果。"""
    summary_path = PROJECT_ROOT / "evaluation_results_ablation" / "full" / "summary.json"
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    return {
        "n_ground_truth": s["n_ground_truth_total"],
        "n_predictions": s["n_predictions_total"],
        "n_matched": s["n_matched"],
        "precision": s["detection_precision"],
        "recall": s["detection_recall"],
        "f1": s["detection_f1"],
        "coverage": s["coverage"],
    }


def run_method(method: str, max_pages: int = None) -> Dict:
    if method == "decimer":
        per_page = run_decimer(max_pages)
    elif method == "yode":
        per_page = run_yode(max_pages)
    elif method == "ours":
        return load_ours_detection()
    else:
        raise ValueError(method)

    agg = aggregate(per_page)
    out = OUTPUT_ROOT / method
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_page.json", "w", encoding="utf-8") as f:
        json.dump(per_page, f, indent=2, ensure_ascii=False)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    return agg


def run_all(max_pages: int = None) -> None:
    summaries = {}
    for method in ("ours", "decimer", "yode"):
        out = OUTPUT_ROOT / method / "summary.json"
        if out.exists():
            with open(out, "r", encoding="utf-8") as f:
                summaries[method] = json.load(f)
            print(f"[skip] {method} 已有结果，复用")
            continue
        if method == "ours":
            summaries["ours"] = load_ours_detection()
            (OUTPUT_ROOT / "ours").mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_ROOT / "ours" / "summary.json", "w", encoding="utf-8") as f:
                json.dump(summaries["ours"], f, indent=2, ensure_ascii=False)
            print("[ours] 复用表 A full 检测结果")
            continue
        print(f"\n=== 开始方法: {method} ===")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--method", method]
        if max_pages:
            cmd += ["--max-pages", str(max_pages)]
        subprocess.run(cmd, check=True)
        with open(out, "r", encoding="utf-8") as f:
            summaries[method] = json.load(f)

    print("\n" + "=" * 60)
    print("表 B：三系统检测级对比")
    print("=" * 60)
    header = f"{'方法':<16}{'GT':>5}{'Pred':>6}{'Matched':>8}{'P':>7}{'R':>7}{'F1':>7}{'Cov':>7}"
    print(header)
    print("-" * 60)
    for method, label in (("ours", "Ours"), ("decimer", "DECIMER-Seg"),
                          ("yode", "YoDe-Seg")):
        s = summaries.get(method)
        if not s:
            print(f"{label:<16}  (无结果)")
            continue
        print(f"{label:<16}{s['n_ground_truth']:>5}{s['n_predictions']:>6}"
              f"{s['n_matched']:>8}{s['precision']:>7.1%}{s['recall']:>7.1%}"
              f"{s['f1']:>7.1%}{s['coverage']:>7.1%}")

    with open(OUTPUT_ROOT / "tableB_detection.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n→ 结果在 {OUTPUT_ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="表 B 三系统检测级对比")
    parser.add_argument("--method", type=str, default="decimer",
                        choices=["ours", "decimer", "yode", "all"])
    parser.add_argument("--max-pages", type=int, default=None)
    args = parser.parse_args()

    if args.method == "all":
        run_all(args.max_pages)
    else:
        agg = run_method(args.method, args.max_pages)
        print(json.dumps(agg, indent=2, ensure_ascii=False))
