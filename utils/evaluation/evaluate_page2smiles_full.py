"""完整 MSIEOF 流程的端到端 Page-to-SMILES 评估（论文 4.2.2 节）。

与 evaluate_page2smiles.py（baseline：YOLO 检测 + bbox 裁切 + MolScribe）的区别在于
本脚本执行论文 3.2-3.4 节描述的完整流程：

    完整页面 PNG
      → YOLOv11m-seg 实例分割
      → 依据实例掩膜提取候选分子图（洪水填充，非分子像素抹白）
      → fragment_cleanup 连通域后处理（清除邻近分子残片）
      → U-Net++ 候选筛选（fg_ratio 门控，剔除非分子候选）
      → 图像质量判定（Gradient Boosting，13 维特征）
          低质量 → HAT ×2 超分辨率
          高质量 → 保持原图
      → 统一箭头去除（ResNet-101 U-Net++ 四类分割）
      → MolScribe 结构识别
      → RDKit 解析与规范化
      → 与真值框 IoU 匹配并计算指标

当前正式顺序：extract → fragment_cleanup → post_segment →
arrow_sr → img2smiles，与论文 3.2–3.4 节一致。

指标口径与 baseline 脚本保持一致（复用其指标函数），IoU 匹配使用 YOLO 多边形的
紧包围盒（未扩边），与 baseline 可直接对比；扩边 bbox 仅用作裁剪窗口。

参数说明：
  - iou_threshold: IoU 阈值，单位=IoU 比值，几何尺度=线性 (0-1)，调大使匹配更严格
  - conf_threshold: YOLO 置信度，单位=置信度，几何尺度=概率 (0-1)，调大减少假阳性
  - bbox_pad_ratio: 裁剪框外扩比例，单位=各自边长的倍数，几何尺度=线性
        （0.10 → 扩后面积约 1.44 倍），调大不易裁断但更易带入邻近分子
  - post_seg_threshold: U-Net++ 前景占比门控，单位=比例 (0-1)，几何尺度=YOLO 多边形内部，
        调大过滤更严但可能误杀真分子
  - low_res_threshold: 分辨率分类器判低清的概率上界，单位=概率 (0-1)，
        几何尺度=分类器输出的"高清"概率，调大则更多图像走超分
  - morgan_radius: Morgan 指纹半径，单位=跳数，几何尺度=指数级 (2→三原子片段)

使用方法：
  python evaluate_page2smiles_full.py --max-pages 10      # 快速验证
  python evaluate_page2smiles_full.py                     # 完整 290 页
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.base.paths import DATA_DIR, PROJECT_ROOT

# Windows 多进程兼容性修复：必须在导入 YOLO 之前设置
import multiprocessing  # noqa: E402

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import torch  # noqa: E402
from rdkit import RDLogger  # noqa: E402

from utils.pipeline import extract as _extract  # noqa: E402
from utils.pipeline import post_segment as _post_segment  # noqa: E402
from utils.pipeline.fragment_cleanup import cleanup_candidates  # noqa: E402
from utils.pipeline.arrow_sr import ArrowSRProcessor  # noqa: E402
from utils.pipeline.resolution_classifier import ResolutionClassifier, CLASSIFIER_PATH  # noqa: E402
from utils.pipeline.img2smiles import Img2SmilesProcessor  # noqa: E402

# 指标口径与 baseline 完全一致：直接复用，避免两套脚本的指标实现漂移
from utils.evaluation.evaluate_page2smiles import (  # noqa: E402
    canonicalize_smiles_safe,
    compute_iou,
    compute_summary_metrics,
    get_inchi_key,
    compute_tanimoto,
    get_optimal_batch_size,
    polygon_to_bbox,
    print_summary,
    save_matched_results_csv,
)

RDLogger.DisableLog("rdApp.*")

class FullPipelineEvaluator:
    """完整 MSIEOF 流程的 Page-to-SMILES 评估器。

    五个模型常驻显存，跨页面复用（YOLO / U-Net++ / HAT+两个箭头分割 / 分辨率分类器
    / MolScribe）。每页的中间产物写进临时目录，跑完即删。
    """

    def __init__(
        self,
        test_set_root: Path,
        iou_threshold: float = 0.5,
        conf_threshold: float = 0.25,
        bbox_pad_ratio: float = 0.10,
        post_seg_threshold: float = _post_segment.FG_RATIO_THRESHOLD,
        low_res_threshold: float = 0.10,
        morgan_radius: int = 2,
        device: str = "cuda",
        batch_size: Optional[int] = None,
    ):
        self.test_set_root = Path(test_set_root)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.bbox_pad_ratio = bbox_pad_ratio
        self.post_seg_threshold = post_seg_threshold
        self.low_res_threshold = low_res_threshold
        self.morgan_radius = morgan_radius

        print("加载 YOLOv11m-seg ...")
        self.yolo_model = _extract._get_model()

        print("加载 U-Net++ (候选筛选) ...")
        self.pseg_device = torch.device(
            "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        )
        self.pseg_model = _post_segment._load_model(self.pseg_device)

        print("加载 HAT + 箭头分割模型 ...")
        self.arrow_sr = ArrowSRProcessor(device=device)

        print("加载分辨率分类器 ...")
        self.res_classifier = ResolutionClassifier(
            classifier_path=str(CLASSIFIER_PATH), low_threshold=low_res_threshold
        )

        print("加载 MolScribe ...")
        self.img2smiles = Img2SmilesProcessor(device=device)

        self.batch_size = batch_size if batch_size is not None else get_optimal_batch_size()
        if batch_size is not None:
            print(f"✓ 使用指定 batch_size = {batch_size}")

        print("✓ 全部模型加载完成\n")

    def _extract_candidates(
        self, page_arr: np.ndarray, crops_dir: Path, page_stem: str
    ) -> List[Dict]:
        """阶段 1-2：YOLO 实例分割 → 依据实例掩膜提取候选分子图（洪水填充）。

        与 utils.pipeline.extract.extract_from_page 逐像素一致：整页 Otsu 二值化，每实例在扩边
        bbox 内以 raw polygon 为 seed、raw ∪ 页面黑像素为 host 做 8 邻接形态学重建，
        非重建区域抹白。同时写 sidecar .meta.npz —— denoise 阶段硬依赖它。

        Returns:
            [{"raw_bbox", "exp_bbox", "confidence", "crop_path"}, ...] 按阅读顺序
        """
        h, w = page_arr.shape[:2]
        results = self.yolo_model.predict(
            source=page_arr,
            conf=self.conf_threshold,
            classes=[_extract.MOL_CLASS_ID],
            verbose=False,
        )

        binary = _extract._page_binary(page_arr, close_r=3)

        raw_masks: List[np.ndarray] = []
        raw_bboxes: List[Tuple[int, int, int, int]] = []
        exp_bboxes: List[Tuple[int, int, int, int]] = []
        polygons: List[np.ndarray] = []
        confidences: List[float] = []

        for result in results:
            masks = result.masks
            if masks is None or len(masks) == 0:
                continue
            boxes = result.boxes
            for contour_xy, box in zip(masks.xy, boxes):
                if len(contour_xy) < 3:
                    continue
                mask_full = _extract._contour_to_mask(contour_xy, (h, w))
                ys, xs = np.where(mask_full > 0)
                if xs.size == 0:
                    continue
                x1, x2 = int(xs.min()), int(xs.max()) + 1
                y1, y2 = int(ys.min()), int(ys.max()) + 1
                raw = (x1, y1, x2, y2)
                raw_masks.append(mask_full)
                raw_bboxes.append(raw)
                exp_bboxes.append(
                    _extract._expand_bbox_per_axis(raw, self.bbox_pad_ratio, h, w)
                )
                polygons.append(np.asarray(contour_xy, dtype=np.float32))
                confidences.append(float(box.conf[0]) if box.conf is not None else 1.0)

        if not exp_bboxes:
            return []

        flood_local: List[np.ndarray] = []
        crops: List[np.ndarray] = []
        for raw_mask, bbox in zip(raw_masks, exp_bboxes):
            bx1, by1, bx2, by2 = bbox
            sub_raw = raw_mask[by1:by2, bx1:bx2]
            sub_black = binary[by1:by2, bx1:bx2]
            sub_host = cv2.bitwise_or(sub_raw, sub_black)
            sub_recon = _extract._morph_reconstruct(sub_raw, sub_host)
            flood_local.append(sub_recon)

            sub_img = page_arr[by1:by2, bx1:bx2].copy()
            sub_img[sub_recon == 0] = 255
            crops.append(sub_img)

        crops_dir.mkdir(parents=True, exist_ok=True)
        order = _extract._reading_order(exp_bboxes)
        candidates: List[Dict] = []
        for new_idx, old_idx in enumerate(order):
            crop_path = crops_dir / f"{page_stem}_mol{new_idx:03d}.png"
            Image.fromarray(crops[old_idx]).save(crop_path)

            bx1, by1, _, _ = exp_bboxes[old_idx]
            poly_local = (
                polygons[old_idx] - np.array([bx1, by1], dtype=np.float32)
            ).astype(np.float32)
            np.savez_compressed(
                crops_dir / f"{page_stem}_mol{new_idx:03d}.meta.npz",
                mask=flood_local[old_idx].astype(np.uint8),
                polygon=poly_local,
            )
            candidates.append({
                "raw_bbox": list(raw_bboxes[old_idx]),
                "exp_bbox": list(exp_bboxes[old_idx]),
                "confidence": confidences[old_idx],
                "crop_path": crop_path,
            })
        return candidates

    def _cleanup_fragments(
        self, candidates: List[Dict], out_dir: Path, page_rgb=None
    ) -> List[Dict]:
        """阶段3：按整页候选执行跨实例、边缘残片和长线清理。"""
        if page_rgb is None:
            raise RuntimeError("新版CC需要整页图像和同页全部候选，不能退化为单图处理")
        return cleanup_candidates(candidates, page_rgb, out_dir, visualize=False)

    def _filter_candidates(
        self, candidates: List[Dict], keep_dir: Path
    ) -> List[Dict]:
        """阶段 4：U-Net++ 候选筛选。fg_ratio < 阈值的候选被剔除，不进入后续。

        分母是 sidecar 里的 YOLO 多边形区域（post_segment.py:_compute_fg_ratio）。
        本阶段是全流程唯一会减少候选数量的环节。
        """
        kept: List[Dict] = []
        for cand in candidates:
            is_mol, fg_ratio, out_path = _post_segment.classify_image(
                self.pseg_model,
                cand["crop_path"],
                keep_dir,
                self.pseg_device,
                threshold=self.post_seg_threshold,
                visualize=False,
            )
            cand["fg_ratio"] = float(fg_ratio)
            cand["kept"] = bool(is_mol)
            if is_mol and out_path is not None:
                cand["crop_path"] = out_path
                kept.append(cand)
        return kept

    def _optimize_candidates(self, candidates: List[Dict]) -> List[np.ndarray]:
        """阶段 5-6：选择性超分辨率 + 箭头去除。

        分辨率分类器判低清 → HAT ×2 超分 → 统一四分类去箭头；
        判高清 → 直接统一四分类去箭头（箭头去除无条件执行）。

        Returns:
            与 candidates 等长的 RGB ndarray 列表（MolScribe 要求 RGB）
        """
        optimized: List[np.ndarray] = []
        for cand in candidates:
            crop_path = cand["crop_path"]
            verdict = self.res_classifier.predict(str(crop_path))
            is_low = bool(verdict.get("is_low_res", False))
            cand["res_prob"] = float(verdict.get("prob", 0.0))
            cand["is_low_res"] = is_low

            if is_low:
                out_bgr = self.arrow_sr.process_low_res(crop_path, viz_stem=None)
            else:
                out_bgr = self.arrow_sr.process_high_res(crop_path, viz_stem=None)

            optimized.append(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB))
        return optimized

    def match_predictions_to_ground_truth(
        self, predictions: List[Dict], ground_truths: List[Dict]
    ) -> Tuple[List, List, List]:
        """贪心 IoU 匹配（与 baseline 脚本同口径）。

        预测几何用 raw_bbox（YOLO 多边形紧包围盒，未扩边），保证与 baseline 可比：
        扩边 bbox 只是裁剪窗口，不代表模型对分子范围的判断。
        """
        n_pred, n_gt = len(predictions), len(ground_truths)
        iou_matrix = np.zeros((n_pred, n_gt))
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truths):
                iou_matrix[i, j] = compute_iou(pred["bbox"], gt["bbox_xyxy"])

        matched_pairs: List[Tuple[int, int, float]] = []
        used_pred: set = set()
        used_gt: set = set()
        while True:
            max_iou, max_i, max_j = 0.0, -1, -1
            for i in range(n_pred):
                if i in used_pred:
                    continue
                for j in range(n_gt):
                    if j in used_gt:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou, max_i, max_j = iou_matrix[i, j], i, j
            if max_iou >= self.iou_threshold:
                matched_pairs.append((max_i, max_j, max_iou))
                used_pred.add(max_i)
                used_gt.add(max_j)
            else:
                break

        unmatched_preds = [i for i in range(n_pred) if i not in used_pred]
        unmatched_gts = [j for j in range(n_gt) if j not in used_gt]
        return matched_pairs, unmatched_preds, unmatched_gts

    def _empty_page_result(self, page_data: Dict, n_detected: int = 0) -> Dict:
        return {
            "page_id": page_data["page_id"],
            "page_file": page_data["page_file"],
            "n_ground_truth": len(page_data["molecules"]),
            "n_detected": n_detected,
            "n_predictions": 0,
            "n_matched": 0,
            "n_false_positives": 0,
            "n_false_negatives": len(page_data["molecules"]),
            "n_rejected_by_filter": n_detected,
            "n_super_resolved": 0,
            "matched_results": [],
            "predictions": [],
            "ground_truths": page_data["molecules"],
        }

    def evaluate_page(self, page_data: Dict, work_dir: Path) -> Dict:
        """对单页跑完整 MSIEOF 流程并计算指标。

        work_dir 存放本页中间产物（crop / denoised / kept），由调用方负责清理。
        """
        page_file = self.test_set_root / page_data["page_file"]
        page_arr = np.array(Image.open(page_file).convert("RGB"))
        page_stem = Path(page_data["page_file"]).stem

        candidates = self._extract_candidates(
            page_arr, work_dir / "crop", page_stem
        )
        if not candidates:
            return self._empty_page_result(page_data, n_detected=0)
        n_detected = len(candidates)

        candidates = self._cleanup_fragments(
            candidates, work_dir / "fragment_cleaned", page_rgb=page_arr)
        candidates = self._filter_candidates(candidates, work_dir / "kept")
        if not candidates:
            return self._empty_page_result(page_data, n_detected=n_detected)

        optimized_rgb = self._optimize_candidates(candidates)

        pred_results = self.img2smiles.predict_batch(
            optimized_rgb, batch_size=self.batch_size
        )

        predictions: List[Dict] = []
        for cand, pred in zip(candidates, pred_results):
            raw_smiles = pred.get("smiles", "")
            smiles_canon, is_valid = canonicalize_smiles_safe(raw_smiles)
            predictions.append({
                "bbox": cand["raw_bbox"],
                "exp_bbox": cand["exp_bbox"],
                "confidence": cand["confidence"],
                "fg_ratio": cand["fg_ratio"],
                "is_low_res": cand["is_low_res"],
                "res_prob": cand["res_prob"],
                "cleanup_changed": cand["cleanup_changed"],
                "smiles_raw": raw_smiles,
                "smiles_canonical": smiles_canon,
                "is_valid": is_valid,
                "smiles_confidence": pred.get("confidence"),
                "n_atoms": len(pred.get("atoms", [])),
                "n_bonds": len(pred.get("bonds", [])),
            })

        ground_truths = page_data["molecules"]
        matched_pairs, unmatched_preds, unmatched_gts = (
            self.match_predictions_to_ground_truth(predictions, ground_truths)
        )

        matched_results = []
        for pred_idx, gt_idx, iou in matched_pairs:
            pred = predictions[pred_idx]
            gt = ground_truths[gt_idx]
            gt_canon, gt_valid = canonicalize_smiles_safe(gt["smiles"])

            canonical_match = (
                pred["smiles_canonical"] == gt_canon
                if (pred["is_valid"] and gt_valid)
                else False
            )
            pred_inchi = get_inchi_key(pred["smiles_canonical"]) if pred["is_valid"] else None
            gt_inchi = get_inchi_key(gt_canon) if gt_valid else None
            graph_match = pred_inchi == gt_inchi and pred_inchi is not None

            tanimoto = None
            if pred["is_valid"] and gt_valid:
                tanimoto = compute_tanimoto(
                    pred["smiles_canonical"], gt_canon, radius=self.morgan_radius
                )

            matched_results.append({
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou,
                "pred_smiles": pred["smiles_canonical"],
                "gt_smiles": gt_canon,
                "pred_valid": pred["is_valid"],
                "gt_valid": gt_valid,
                "canonical_match": canonical_match,
                "graph_match": graph_match,
                "tanimoto": tanimoto,
                "pred_confidence": pred["confidence"],
                "gt_molecule_id": gt["molecule_id"],
                "is_low_res": pred["is_low_res"],
            })

        return {
            "page_id": page_data["page_id"],
            "page_file": page_data["page_file"],
            "n_ground_truth": len(ground_truths),
            "n_detected": n_detected,
            "n_predictions": len(predictions),
            "n_matched": len(matched_pairs),
            "n_false_positives": len(unmatched_preds),
            "n_false_negatives": len(unmatched_gts),
            "n_rejected_by_filter": n_detected - len(predictions),
            "n_super_resolved": sum(1 for p in predictions if p["is_low_res"]),
            "matched_results": matched_results,
            "predictions": predictions,
            "ground_truths": ground_truths,
        }

def evaluate_dataset(
    test_set_jsonl: Path,
    test_set_root: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    bbox_pad_ratio: float = 0.10,
    post_seg_threshold: float = _post_segment.FG_RATIO_THRESHOLD,
    low_res_threshold: float = 0.10,
    morgan_radius: int = 2,
    max_pages: Optional[int] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None,
) -> Dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages: List[Dict] = []
    with open(test_set_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
            if max_pages and len(pages) >= max_pages:
                break

    n_mol = sum(len(p["molecules"]) for p in pages)
    print(f"✓ 加载测试集: {len(pages)} 页, {n_mol} 个分子结构\n")

    evaluator = FullPipelineEvaluator(
        test_set_root=test_set_root,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        bbox_pad_ratio=bbox_pad_ratio,
        post_seg_threshold=post_seg_threshold,
        low_res_threshold=low_res_threshold,
        morgan_radius=morgan_radius,
        device=device,
        batch_size=batch_size,
    )

    all_results: List[Dict] = []
    with tempfile.TemporaryDirectory(prefix="msieof_eval_") as tmp_root:
        tmp_root = Path(tmp_root)
        for page_data in tqdm(pages, desc="完整流程评估"):
            work_dir = tmp_root / Path(page_data["page_file"]).stem
            try:
                all_results.append(evaluator.evaluate_page(page_data, work_dir))
            except Exception as e:
                print(f"\n⚠️  页面 {page_data['page_file']} 评估失败: {e}")
                import traceback

                traceback.print_exc()
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = compute_summary_metrics(all_results, morgan_radius)
    summary["pipeline"] = "full_MSIEOF"
    summary["stage_order"] = (
        "extract(mask flood) -> fragment_cleanup(CC) -> unetpp_filter -> selective_SR -> "
        "arrow_removal -> molscribe -> rdkit"
    )
    summary["n_detected_total"] = sum(r.get("n_detected", 0) for r in all_results)
    summary["n_rejected_by_filter_total"] = sum(
        r.get("n_rejected_by_filter", 0) for r in all_results
    )
    summary["n_super_resolved_total"] = sum(
        r.get("n_super_resolved", 0) for r in all_results
    )
    summary["config"] = {
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "bbox_pad_ratio": bbox_pad_ratio,
        "post_seg_threshold": post_seg_threshold,
        "low_res_threshold": low_res_threshold,
        "morgan_radius": morgan_radius,
        "molscribe_batch_size": evaluator.batch_size,
    }

    with open(output_dir / "results_详细.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(output_dir / "summary_汇总.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    save_matched_results_csv(all_results, output_dir / "matched_results.csv")

    print_summary(summary)
    print("-" * 70)
    print("完整流程各阶段统计:")
    print(f"  YOLO 检出候选:        {summary['n_detected_total']}")
    print(f"  U-Net++ 剔除:         {summary['n_rejected_by_filter_total']}")
    print(f"  进入 MolScribe:       {summary['n_predictions_total']}")
    print(f"  其中触发超分辨率:     {summary['n_super_resolved_total']}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="完整 MSIEOF 流程的端到端 Page-to-SMILES 评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-set-root", type=str,
                        default=str(DATA_DIR / "MolMole_Patent300"),
                        help="测试集根目录")
    parser.add_argument("--test-set-jsonl", type=str, default=None,
                        help="测试集JSONL (默认 <root>/01_MolMole_Patent300/derived/pages_ordinary_complete.jsonl)")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "evaluation_results_full"),
                        help="评估结果输出目录")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU匹配阈值，单位=IoU比值，几何尺度=线性 (0-1)，调大更严格")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="YOLO置信度，单位=置信度，几何尺度=概率 (0-1)，调大减少假阳性")
    parser.add_argument("--bbox-pad-ratio", type=float, default=0.10,
                        help="裁剪框外扩比例，单位=各自边长的倍数，几何尺度=线性")
    parser.add_argument("--post-seg-threshold", type=float,
                        default=_post_segment.FG_RATIO_THRESHOLD,
                        help="U-Net++前景占比门控，单位=比例(0-1)，几何尺度=YOLO多边形内部")
    parser.add_argument("--low-res-threshold", type=float, default=0.10,
                        help="判低清的概率上界，单位=概率(0-1)，调大则更多图走超分")
    parser.add_argument("--morgan-radius", type=int, default=2,
                        help="Morgan指纹半径，单位=跳数，几何尺度=指数级 (2→三原子片段)")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="最多评估页数 (快速测试用)")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="MolScribe批处理大小 (None=自动检测)")

    args = parser.parse_args()

    test_set_root = Path(args.test_set_root)
    test_set_jsonl = (
        Path(args.test_set_jsonl)
        if args.test_set_jsonl
        else test_set_root / "01_MolMole_Patent300" / "derived" / "pages_ordinary_complete.jsonl"
    )
    if not test_set_jsonl.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_set_jsonl}")

    print("=" * 70)
    print("完整 MSIEOF 流程端到端评估")
    print("流程: YOLOv11m-seg → 掩膜提取 → 连通域处理 → U-Net++筛选")
    print("      → 选择性超分 → 箭头去除 → MolScribe → RDKit")
    print("=" * 70)
    print(f"  测试集:           {test_set_jsonl.name}")
    print(f"  输出目录:         {args.output_dir}")
    print(f"  IoU阈值:          {args.iou_threshold}")
    print(f"  YOLO置信度:       {args.conf_threshold}")
    print(f"  裁剪外扩比例:     {args.bbox_pad_ratio}")
    print(f"  候选筛选阈值:     {args.post_seg_threshold}")
    print(f"  低清判定阈值:     {args.low_res_threshold}")
    print(f"  Morgan半径:       {args.morgan_radius}")
    print(f"  推理设备:         {args.device}")
    if args.max_pages:
        print(f"  限制页数:         {args.max_pages} (快速测试)")
    print("=" * 70 + "\n")

    start = time.time()
    summary = evaluate_dataset(
        test_set_jsonl=test_set_jsonl,
        test_set_root=test_set_root,
        output_dir=Path(args.output_dir),
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        bbox_pad_ratio=args.bbox_pad_ratio,
        post_seg_threshold=args.post_seg_threshold,
        low_res_threshold=args.low_res_threshold,
        morgan_radius=args.morgan_radius,
        max_pages=args.max_pages,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"\n✓ 评估完成！总耗时: {(time.time() - start) / 60:.1f} 分钟")
    print(f"✓ 结果已保存到: {Path(args.output_dir).absolute()}")
