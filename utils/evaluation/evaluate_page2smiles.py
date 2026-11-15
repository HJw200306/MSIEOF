"""端到端 Page-to-SMILES 评估脚本

测试集：MolMole Patent300 (290页, 2,360个结构)
评估指标：
  - Validity: 预测SMILES的有效性 (能否被RDKit解析)
  - Canonical SMILES EM: 规范化SMILES的完全匹配率
  - Graph Exact Match (GEM): 分子图结构完全匹配率 (通过InChI比较)
  - Tanimoto Similarity: 基于Morgan指纹的平均相似度

流程：
  1. 读取测试集JSONL文件 (pages_ordinary_complete.jsonl)
  2. 对每张整页图像运行完整pipeline:
     - YOLO检测获取bbox坐标
     - 裁剪并处理每个分子区域
     - MolScribe预测SMILES
  3. 通过IoU匹配预测框与真值框 (阈值默认0.5)
  4. 计算各项指标并生成详细报告

参数说明：
  - iou_threshold: IoU阈值，单位=IoU比值，几何尺度=线性 (0-1)，调大使匹配更严格
  - morgan_radius: Morgan指纹半径，单位=跳数，几何尺度=指数级增长捕获范围 (1跳→直接邻居, 2跳→三原子片段, 3跳→五原子片段)

使用方法：
  python evaluate_page2smiles.py --test-set-root "C:/Users/Admin/Desktop/测试集" \\
                                  --output-dir "./evaluation_results" \\
                                  --max-pages 10  # 快速测试用
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
import time
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit import RDLogger

# Windows 多进程兼容性修复：必须在导入 YOLO 之前设置
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # 已经设置过

# YOLO
from ultralytics import YOLO

# PyTorch for CUDA detection
import torch

# 本地模块
from utils.pipeline.img2smiles import Img2SmilesProcessor
from utils.base.paths import DATA_DIR, PROJECT_ROOT

# 关闭RDKit警告
RDLogger.DisableLog("rdApp.*")

MODEL_PATH = PROJECT_ROOT / "model" / "YOLOv11_best.pt"
MOL_CLASS_ID = 0  # 分子类别ID


def get_optimal_batch_size() -> int:
    """根据CUDA性能自动选择MolScribe批处理大小

    检测GPU显存并推荐合适的batch size：
    - >= 16 GB: batch_size = 32
    - >= 10 GB: batch_size = 24
    - >= 6 GB:  batch_size = 16
    - < 6 GB:   batch_size = 8
    - CPU:      batch_size = 4

    Returns:
        推荐的batch size
    """
    if not torch.cuda.is_available():
        print("⚠️  未检测到CUDA，使用CPU模式，batch_size=4")
        return 4

    try:
        # 获取GPU显存信息（GB）
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_name = torch.cuda.get_device_name(0)

        print(f"✓ 检测到GPU: {gpu_name}")
        print(f"✓ 显存: {gpu_memory_gb:.1f} GB")

        # 根据显存大小推荐batch size
        if gpu_memory_gb >= 16:
            batch_size = 32
            print(f"✓ 推荐batch_size = {batch_size} (大显存优化)")
        elif gpu_memory_gb >= 10:
            batch_size = 24
            print(f"✓ 推荐batch_size = {batch_size} (标准配置)")
        elif gpu_memory_gb >= 6:
            batch_size = 16
            print(f"✓ 推荐batch_size = {batch_size} (中等显存)")
        else:
            batch_size = 8
            print(f"✓ 推荐batch_size = {batch_size} (小显存保守配置)")

        return batch_size

    except Exception as e:
        print(f"⚠️  GPU信息获取失败: {e}，使用默认batch_size=16")
        return 16


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """计算两个边界框的IoU (Intersection over Union)

    Args:
        box1, box2: [x1, y1, x2, y2] 格式，左上角和右下角坐标

    Returns:
        IoU值 (0-1之间)
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def canonicalize_smiles_safe(smiles: str) -> Tuple[Optional[str], bool]:
    """安全地规范化SMILES

    Returns:
        (canonical_smiles, is_valid)
    """
    if not smiles or not isinstance(smiles, str):
        return None, False

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, False
        # 使用isomeric SMILES保留立体信息
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        return canonical, True
    except:
        return None, False


def get_inchi_key(smiles: str) -> Optional[str]:
    """从SMILES生成InChIKey用于图结构比较"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchiKey(mol)
    except:
        return None


def compute_tanimoto(smiles1: str, smiles2: str, radius: int = 2) -> Optional[float]:
    """计算两个SMILES的Tanimoto相似度 (基于Morgan指纹)

    参数：
      - radius: Morgan指纹半径，单位=跳数，几何尺度=指数级 (2→三原子片段)
      - nBits: 比特向量长度 (2048)

    Returns:
        Tanimoto相似度 (0-1之间)，解析失败返回None
    """
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return None

        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=radius, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=radius, nBits=2048)

        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return None


def polygon_to_bbox(polygon: np.ndarray) -> List[float]:
    """将多边形转换为紧包围矩形框

    Args:
        polygon: Nx2数组，每行是(x, y)坐标

    Returns:
        [x1, y1, x2, y2] 格式的bbox
    """
    if len(polygon) == 0:
        return [0, 0, 0, 0]

    xs = polygon[:, 0]
    ys = polygon[:, 1]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    return [x1, y1, x2, y2]


def crop_molecule_region(page_arr: np.ndarray, bbox: List[float],
                         padding_ratio: float = 0.1) -> np.ndarray:
    """从整页图像中裁剪分子区域，带适当padding

    Args:
        page_arr: 整页图像 (numpy array)
        bbox: [x1, y1, x2, y2]
        padding_ratio: 边界扩展比例，单位=各自边长的倍数，几何尺度=线性

    Returns:
        裁剪后的RGB图像
    """
    h, w = page_arr.shape[:2]
    x1, y1, x2, y2 = bbox

    # 计算padding
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * padding_ratio)
    pad_y = int(bh * padding_ratio)

    # 扩展bbox
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))

    # 裁剪
    crop = page_arr[y1:y2, x1:x2]

    return crop


class PageToSMILESEvaluator:
    """Page-to-SMILES端到端评估器"""

    def __init__(self,
                 test_set_root: Path,
                 model_path: Path = MODEL_PATH,
                 iou_threshold: float = 0.5,
                 conf_threshold: float = 0.25,
                 morgan_radius: int = 2,
                 device: str = "cuda",
                 batch_size: Optional[int] = None):
        """
        Args:
            test_set_root: 测试集根目录
            model_path: YOLO模型路径
            iou_threshold: IoU匹配阈值，单位=IoU比值，几何尺度=线性 (0-1)
            conf_threshold: YOLO置信度阈值，单位=置信度，几何尺度=概率 (0-1)
            morgan_radius: Morgan指纹半径，单位=跳数，几何尺度=指数级
            device: 推理设备 (cuda/cpu)
            batch_size: MolScribe批处理大小，None=自动检测
        """
        self.test_set_root = Path(test_set_root)
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.morgan_radius = morgan_radius

        # 加载YOLO模型
        print(f"加载YOLO模型: {model_path}")
        self.yolo_model = YOLO(str(model_path))

        # 加载MolScribe模型
        print(f"加载MolScribe模型...")
        self.img2smiles_processor = Img2SmilesProcessor(device=device)

        # 自动检测或使用指定的batch size
        if batch_size is None:
            self.batch_size = get_optimal_batch_size()
        else:
            self.batch_size = batch_size
            print(f"✓ 使用指定batch_size = {batch_size}")

        print("✓ 模型加载完成\n")

    def detect_molecules(self, page_arr: np.ndarray) -> List[Dict]:
        """使用YOLO检测页面中的分子结构

        Returns:
            检测结果列表，每个元素包含 bbox, polygon, confidence
        """
        results = self.yolo_model.predict(
            source=page_arr,
            conf=self.conf_threshold,
            classes=[MOL_CLASS_ID],
            verbose=False
        )

        detections = []
        for result in results:
            masks = result.masks
            if masks is None or len(masks) == 0:
                continue

            boxes = result.boxes
            for idx, (polygon_xy, box) in enumerate(zip(masks.xy, boxes)):
                if len(polygon_xy) < 3:
                    continue

                bbox = polygon_to_bbox(polygon_xy)

                detections.append({
                    "bbox": bbox,  # [x1, y1, x2, y2]
                    "polygon": polygon_xy,
                    "confidence": float(box.conf[0]) if box.conf is not None else 1.0
                })

        return detections

    def predict_smiles(self, crop_rgb: np.ndarray) -> Dict:
        """对裁剪的分子图像预测SMILES

        Returns:
            包含 smiles_raw, smiles_canonical, is_valid, confidence 的字典
        """
        preds = self.img2smiles_processor.predict_batch([crop_rgb], batch_size=1)
        pred = preds[0]

        raw_smiles = pred.get("smiles", "")
        smiles_canon, is_valid = canonicalize_smiles_safe(raw_smiles)

        return {
            "smiles_raw": raw_smiles,
            "smiles_canonical": smiles_canon,
            "is_valid": is_valid,
            "confidence": pred.get("confidence"),
            "n_atoms": len(pred.get("atoms", [])),
            "n_bonds": len(pred.get("bonds", []))
        }

    def match_predictions_to_ground_truth(self,
                                          predictions: List[Dict],
                                          ground_truths: List[Dict]) -> Tuple[List, List, List]:
        """通过IoU匹配预测与真值

        Returns:
            (matched_pairs, unmatched_preds, unmatched_gts)
            - matched_pairs: [(pred_idx, gt_idx, iou), ...]
            - unmatched_preds: [pred_idx, ...]
            - unmatched_gts: [gt_idx, ...]
        """
        n_pred = len(predictions)
        n_gt = len(ground_truths)

        # 计算IoU矩阵
        iou_matrix = np.zeros((n_pred, n_gt))
        for i, pred in enumerate(predictions):
            pred_bbox = pred["bbox"]
            for j, gt in enumerate(ground_truths):
                gt_bbox = gt["bbox_xyxy"]
                iou_matrix[i, j] = compute_iou(pred_bbox, gt_bbox)

        # 贪心匹配：每次选择IoU最大的配对
        matched_pairs = []
        matched_pred_indices = set()
        matched_gt_indices = set()

        while True:
            max_iou = 0.0
            max_i, max_j = -1, -1

            for i in range(n_pred):
                if i in matched_pred_indices:
                    continue
                for j in range(n_gt):
                    if j in matched_gt_indices:
                        continue
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_i, max_j = i, j

            if max_iou >= self.iou_threshold:
                matched_pairs.append((max_i, max_j, max_iou))
                matched_pred_indices.add(max_i)
                matched_gt_indices.add(max_j)
            else:
                break

        unmatched_preds = [i for i in range(n_pred) if i not in matched_pred_indices]
        unmatched_gts = [j for j in range(n_gt) if j not in matched_gt_indices]

        return matched_pairs, unmatched_preds, unmatched_gts

    def evaluate_page(self, page_data: Dict) -> Dict:
        """评估单个页面（批量推理优化版）

        Args:
            page_data: 从JSONL读取的单页数据

        Returns:
            评估结果字典
        """
        page_file = self.test_set_root / page_data["page_file"]

        # 读取整页图像
        page_img = Image.open(page_file).convert("RGB")
        page_arr = np.array(page_img)

        # Step 1: 检测分子结构
        detections = self.detect_molecules(page_arr)

        if len(detections) == 0:
            # 没有检测到分子，直接返回空结果
            return {
                "page_id": page_data["page_id"],
                "page_file": page_data["page_file"],
                "n_ground_truth": len(page_data["molecules"]),
                "n_predictions": 0,
                "n_matched": 0,
                "n_false_positives": 0,
                "n_false_negatives": len(page_data["molecules"]),
                "matched_results": [],
                "predictions": [],
                "ground_truths": page_data["molecules"]
            }

        # Step 2: 批量裁剪所有分子区域
        crops = []
        valid_detection_indices = []
        for idx, det in enumerate(detections):
            crop = crop_molecule_region(page_arr, det["bbox"], padding_ratio=0.1)
            if crop.size > 0:
                crops.append(crop)
                valid_detection_indices.append(idx)

        if len(crops) == 0:
            # 所有裁剪都失败，返回空结果
            return {
                "page_id": page_data["page_id"],
                "page_file": page_data["page_file"],
                "n_ground_truth": len(page_data["molecules"]),
                "n_predictions": 0,
                "n_matched": 0,
                "n_false_positives": 0,
                "n_false_negatives": len(page_data["molecules"]),
                "matched_results": [],
                "predictions": [],
                "ground_truths": page_data["molecules"]
            }

        # Step 3: 批量推理SMILES（关键优化：一次处理所有crops）
        pred_results = self.img2smiles_processor.predict_batch(
            crops,
            batch_size=self.batch_size
        )

        # Step 4: 组装预测结果
        predictions = []
        for det_idx, pred_result in zip(valid_detection_indices, pred_results):
            det = detections[det_idx]

            raw_smiles = pred_result.get("smiles", "")
            smiles_canon, is_valid = canonicalize_smiles_safe(raw_smiles)

            predictions.append({
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "smiles_raw": raw_smiles,
                "smiles_canonical": smiles_canon,
                "is_valid": is_valid,
                "smiles_confidence": pred_result.get("confidence"),
                "n_atoms": len(pred_result.get("atoms", [])),
                "n_bonds": len(pred_result.get("bonds", []))
            })

        # Step 5: 匹配预测与真值
        ground_truths = page_data["molecules"]
        matched_pairs, unmatched_preds, unmatched_gts = self.match_predictions_to_ground_truth(
            predictions, ground_truths
        )

        # Step 6: 计算匹配对的指标
        matched_results = []
        for pred_idx, gt_idx, iou in matched_pairs:
            pred = predictions[pred_idx]
            gt = ground_truths[gt_idx]

            gt_smiles_canon, gt_valid = canonicalize_smiles_safe(gt["smiles"])

            # 计算各项指标
            canonical_match = (pred["smiles_canonical"] == gt_smiles_canon) if (pred["is_valid"] and gt_valid) else False

            # Graph Exact Match: 通过InChIKey比较
            pred_inchi = get_inchi_key(pred["smiles_canonical"]) if pred["is_valid"] else None
            gt_inchi = get_inchi_key(gt_smiles_canon) if gt_valid else None
            graph_match = (pred_inchi == gt_inchi and pred_inchi is not None)

            # Tanimoto相似度
            tanimoto = None
            if pred["is_valid"] and gt_valid:
                tanimoto = compute_tanimoto(
                    pred["smiles_canonical"],
                    gt_smiles_canon,
                    radius=self.morgan_radius
                )

            matched_results.append({
                "pred_idx": pred_idx,
                "gt_idx": gt_idx,
                "iou": iou,
                "pred_smiles": pred["smiles_canonical"],
                "gt_smiles": gt_smiles_canon,
                "pred_valid": pred["is_valid"],
                "gt_valid": gt_valid,
                "canonical_match": canonical_match,
                "graph_match": graph_match,
                "tanimoto": tanimoto,
                "pred_confidence": pred["confidence"],
                "gt_molecule_id": gt["molecule_id"]
            })

        return {
            "page_id": page_data["page_id"],
            "page_file": page_data["page_file"],
            "n_ground_truth": len(ground_truths),
            "n_predictions": len(predictions),
            "n_matched": len(matched_pairs),
            "n_false_positives": len(unmatched_preds),
            "n_false_negatives": len(unmatched_gts),
            "matched_results": matched_results,
            "predictions": predictions,
            "ground_truths": ground_truths
        }


def evaluate_dataset(
    test_set_jsonl: Path,
    test_set_root: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
    morgan_radius: int = 2,
    max_pages: Optional[int] = None,
    device: str = "cuda",
    batch_size: Optional[int] = None
) -> Dict:
    """评估整个测试集

    Args:
        test_set_jsonl: 测试集JSONL文件路径
        test_set_root: 测试集根目录
        output_dir: 输出目录
        iou_threshold: IoU匹配阈值，单位=IoU比值，几何尺度=线性 (0-1)
        conf_threshold: YOLO置信度阈值，单位=置信度，几何尺度=概率 (0-1)
        morgan_radius: Morgan指纹半径，单位=跳数，几何尺度=指数级
        max_pages: 最多评估页数 (用于快速测试)
        device: 推理设备
        batch_size: MolScribe批处理大小，None=自动检测

    Returns:
        汇总评估结果
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 读取测试集
    pages = []
    with open(test_set_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
            if max_pages and len(pages) >= max_pages:
                break

    print(f"✓ 加载测试集: {len(pages)} 页, {sum(len(p['molecules']) for p in pages)} 个分子结构\n")

    # 初始化评估器
    evaluator = PageToSMILESEvaluator(
        test_set_root=test_set_root,
        iou_threshold=iou_threshold,
        conf_threshold=conf_threshold,
        morgan_radius=morgan_radius,
        device=device,
        batch_size=batch_size
    )

    # 逐页评估
    all_results = []
    for page_data in tqdm(pages, desc="评估进度"):
        try:
            result = evaluator.evaluate_page(page_data)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠️  页面 {page_data['page_file']} 评估失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算汇总指标
    summary = compute_summary_metrics(all_results, morgan_radius)

    # 保存详细结果
    with open(output_dir / "results_详细.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 保存汇总结果
    with open(output_dir / "summary_汇总.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 保存CSV格式的匹配结果 (方便Excel分析)
    save_matched_results_csv(all_results, output_dir / "matched_results.csv")

    # 打印汇总报告
    print_summary(summary)

    return summary


def compute_summary_metrics(results: List[Dict], morgan_radius: int) -> Dict:
    """计算汇总评估指标"""

    # 收集所有匹配对的结果
    all_matched = []
    for page_result in results:
        all_matched.extend(page_result["matched_results"])

    n_pages = len(results)
    n_gt_total = sum(r["n_ground_truth"] for r in results)
    n_pred_total = sum(r["n_predictions"] for r in results)
    n_matched = len(all_matched)
    n_fp_total = sum(r["n_false_positives"] for r in results)
    n_fn_total = sum(r["n_false_negatives"] for r in results)

    # 计算指标 (仅针对匹配的预测)
    if n_matched > 0:
        # Validity: 预测中有效的比例
        n_valid = sum(1 for m in all_matched if m["pred_valid"])
        validity = n_valid / n_matched

        # Canonical SMILES EM: 规范化SMILES完全匹配
        n_canonical_match = sum(1 for m in all_matched if m["canonical_match"])
        canonical_em = n_canonical_match / n_matched

        # Graph Exact Match: 图结构完全匹配 (InChIKey)
        n_graph_match = sum(1 for m in all_matched if m["graph_match"])
        graph_em = n_graph_match / n_matched

        # Tanimoto Similarity: 平均相似度
        tanimoto_scores = [m["tanimoto"] for m in all_matched if m["tanimoto"] is not None]
        tanimoto_mean = np.mean(tanimoto_scores) if tanimoto_scores else 0.0
        tanimoto_std = np.std(tanimoto_scores) if tanimoto_scores else 0.0
    else:
        validity = 0.0
        canonical_em = 0.0
        graph_em = 0.0
        tanimoto_mean = 0.0
        tanimoto_std = 0.0

    # 检测层面的指标
    precision = n_matched / n_pred_total if n_pred_total > 0 else 0.0
    recall = n_matched / n_gt_total if n_gt_total > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_pages": n_pages,
        "n_ground_truth_total": n_gt_total,
        "n_predictions_total": n_pred_total,
        "n_matched": n_matched,
        "n_false_positives": n_fp_total,
        "n_false_negatives": n_fn_total,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": f1,
        "validity": validity,
        "canonical_smiles_em": canonical_em,
        "graph_exact_match_gem": graph_em,
        "tanimoto_similarity_mean": tanimoto_mean,
        "tanimoto_similarity_std": tanimoto_std,
        "morgan_radius": morgan_radius,
        "per_page_results": [
            {
                "page_file": r["page_file"],
                "n_gt": r["n_ground_truth"],
                "n_pred": r["n_predictions"],
                "n_matched": r["n_matched"]
            }
            for r in results
        ]
    }


def save_matched_results_csv(results: List[Dict], csv_path: Path):
    """保存匹配结果为CSV文件"""

    rows = []
    for page_result in results:
        page_file = page_result["page_file"]
        for m in page_result["matched_results"]:
            rows.append({
                "page_file": page_file,
                "gt_molecule_id": m["gt_molecule_id"],
                "iou": f"{m['iou']:.4f}",
                "pred_valid": m["pred_valid"],
                "canonical_match": m["canonical_match"],
                "graph_match": m["graph_match"],
                "tanimoto": f"{m['tanimoto']:.4f}" if m["tanimoto"] is not None else "N/A",
                "pred_smiles": m["pred_smiles"] or "",
                "gt_smiles": m["gt_smiles"] or "",
                "pred_confidence": f"{m['pred_confidence']:.4f}" if m["pred_confidence"] else "N/A"
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def print_summary(summary: Dict):
    """打印评估摘要"""
    print("\n" + "="*70)
    print("端到端 Page-to-SMILES 评估结果")
    print("="*70)
    print(f"测试页数:           {summary['n_pages']}")
    print(f"真值结构总数:       {summary['n_ground_truth_total']}")
    print(f"预测结构总数:       {summary['n_predictions_total']}")
    print(f"成功匹配:           {summary['n_matched']}")
    print(f"假阳性 (FP):        {summary['n_false_positives']}")
    print(f"假阴性 (FN):        {summary['n_false_negatives']}")
    print("-"*70)
    print("检测层面指标:")
    print(f"  Precision:        {summary['detection_precision']:.2%}")
    print(f"  Recall:           {summary['detection_recall']:.2%}")
    print(f"  F1:               {summary['detection_f1']:.2%}")
    print("-"*70)
    print("SMILES识别指标 (基于成功匹配的结构):")
    print(f"  Validity (有效性):              {summary['validity']:.2%}")
    print(f"  Canonical SMILES EM (完全匹配): {summary['canonical_smiles_em']:.2%}")
    print(f"  Graph Exact Match (GEM):        {summary['graph_exact_match_gem']:.2%}")
    print(f"  Tanimoto Similarity (平均):     {summary['tanimoto_similarity_mean']:.4f} ± {summary['tanimoto_similarity_std']:.4f}")
    print(f"  Morgan指纹半径:                 {summary['morgan_radius']}")
    print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="端到端 Page-to-SMILES 评估 (MolMole Patent300)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--test-set-root",
        type=str,
        default=str(DATA_DIR / "MolMole_Patent300"),
        help="测试集根目录"
    )
    parser.add_argument(
        "--test-set-jsonl",
        type=str,
        default=None,
        help="测试集JSONL文件 (默认: <root>/01_MolMole_Patent300/derived/pages_ordinary_complete.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "evaluation_results"),
        help="评估结果输出目录"
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU匹配阈值，单位=IoU比值，几何尺度=线性 (0-1)，调大使匹配更严格"
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="YOLO置信度阈值，单位=置信度，几何尺度=概率 (0-1)，调大减少假阳性"
    )
    parser.add_argument(
        "--morgan-radius",
        type=int,
        default=2,
        help="Morgan指纹半径，单位=跳数，几何尺度=指数级 (2→三原子片段)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="最多评估页数 (用于快速测试，如 --max-pages 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="MolScribe批处理大小 (None=自动检测，推荐16-32)"
    )

    args = parser.parse_args()

    test_set_root = Path(args.test_set_root)

    if args.test_set_jsonl:
        test_set_jsonl = Path(args.test_set_jsonl)
    else:
        test_set_jsonl = test_set_root / "01_MolMole_Patent300" / "derived" / "pages_ordinary_complete.jsonl"

    if not test_set_jsonl.exists():
        raise FileNotFoundError(f"测试集文件不存在: {test_set_jsonl}")

    print("="*70)
    print("配置信息:")
    print(f"  测试集根目录:   {test_set_root}")
    print(f"  测试集文件:     {test_set_jsonl.name}")
    print(f"  输出目录:       {args.output_dir}")
    print(f"  IoU阈值:        {args.iou_threshold}")
    print(f"  YOLO置信度:     {args.conf_threshold}")
    print(f"  Morgan半径:     {args.morgan_radius}")
    print(f"  推理设备:       {args.device}")
    if args.batch_size:
        print(f"  Batch Size:     {args.batch_size} (手动指定)")
    else:
        print(f"  Batch Size:     自动检测")
    if args.max_pages:
        print(f"  限制页数:       {args.max_pages} (快速测试模式)")
    print("="*70 + "\n")

    start_time = time.time()

    summary = evaluate_dataset(
        test_set_jsonl=test_set_jsonl,
        test_set_root=test_set_root,
        output_dir=Path(args.output_dir),
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold,
        morgan_radius=args.morgan_radius,
        max_pages=args.max_pages,
        device=args.device,
        batch_size=args.batch_size
    )

    elapsed = time.time() - start_time
    print(f"\n✓ 评估完成！总耗时: {elapsed/60:.1f} 分钟")
    print(f"✓ 详细结果已保存到: {Path(args.output_dir).absolute()}")
