# -*- coding: utf-8 -*-
"""
OBB 标注数据管理与 AI 预推理服务 (OBB Manager)
负责 data/real_obb_finetune/ 数据集的元数据读取、标注管理、YOLO 格式导出及模型实时预推理。
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKUSH_RECO_ROOT = Path(os.environ["MARKUSH_RECO_ROOT"]) if os.environ.get("MARKUSH_RECO_ROOT") else None

sys.path.insert(0, str(PROJECT_ROOT))
if MARKUSH_RECO_ROOT and MARKUSH_RECO_ROOT.exists() and str(MARKUSH_RECO_ROOT) not in sys.path:
    sys.path.insert(0, str(MARKUSH_RECO_ROOT))

DEFAULT_DATASET_DIR = (
    MARKUSH_RECO_ROOT / "data" / "real_obb_finetune"
    if MARKUSH_RECO_ROOT and MARKUSH_RECO_ROOT.exists()
    else PROJECT_ROOT / "data" / "real_obb_finetune"
)


CLASS_NAMES = {
    0: "text",
    1: "l_bracket",
    2: "r_bracket",
    3: "bracket_sub",
    4: "wave_ap",
    5: "dash_ap",
}

CLASS_COLORS = {
    0: "#eab308",  # text: yellow / gold
    1: "#00e5ff",  # l_bracket: cyan
    2: "#2979ff",  # r_bracket: blue
    3: "#ff9100",  # bracket_sub: orange / amber
    4: "#ec4899",  # wave_ap: pink / magenta
    5: "#10b981",  # dash_ap: emerald / green
}


class OBBManager:
    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.img_dir = self.dataset_dir / "images"
        self.manifest_path = self.dataset_dir / "manifest.jsonl"
        self.annotations_dir = self.dataset_dir / "annotations"
        self.yolo_dir = self.dataset_dir / "yolo_labels"
        self.status_file = self.dataset_dir / "review_status.jsonl"

        self._manifest_cache: List[Dict[str, Any]] = []
        self._id_to_sample: Dict[str, Dict[str, Any]] = {}
        self._status_cache: Dict[str, str] = {}
        self._detector = None
        self._initialized = False

        self._ensure_dirs()
        self._reload_manifest()

    def _ensure_dirs(self):
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.yolo_dir.mkdir(parents=True, exist_ok=True)

    def _reload_manifest(self):
        self._manifest_cache = []
        self._id_to_sample = {}
        self._status_cache = {}

        if self.status_file.exists():
            with open(self.status_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            self._status_cache[d["id"]] = d.get("status", "reviewed")
                        except Exception:
                            pass

        if self.manifest_path.exists():
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            sid = str(d["id"])
                            # 标注状态
                            ann_p = self.annotations_dir / f"{sid}.json"
                            if sid in self._status_cache:
                                d["status"] = self._status_cache[sid]
                            elif ann_p.exists():
                                d["status"] = "reviewed"
                            else:
                                d["status"] = "pending"

                            # 检查已有标注框数
                            d["box_count"] = 0
                            if ann_p.exists():
                                try:
                                    with open(ann_p, "r", encoding="utf-8") as af:
                                        ad = json.load(af)
                                        d["box_count"] = len(ad.get("boxes", []))
                                except Exception:
                                    pass

                            self._manifest_cache.append(d)
                            self._id_to_sample[sid] = d
                        except Exception:
                            pass
            self._initialized = True

    def get_dataset_info(self) -> Dict[str, Any]:
        """返回数据集统计信息"""
        if not self._initialized:
            self._reload_manifest()

        total = len(self._manifest_cache)
        reviewed_cnt = sum(1 for s in self._manifest_cache if s.get("status") == "reviewed")
        pending_cnt = total - reviewed_cnt

        sources = {}
        features = {}
        for s in self._manifest_cache:
            src = s.get("source", "other")
            sources[src] = sources.get(src, 0) + 1
            for f in s.get("features", []):
                features[f] = features.get(f, 0) + 1

        return {
            "total": total,
            "reviewed": reviewed_cnt,
            "pending": pending_cnt,
            "sources": sources,
            "features": features,
            "classes": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        }

    def list_samples(
        self,
        source: str = "all",
        feature: str = "all",
        status: str = "all",
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        if not self._initialized:
            self._reload_manifest()

        filtered = self._manifest_cache

        if source != "all":
            filtered = [s for s in filtered if s.get("source") == source]

        if feature != "all":
            filtered = [s for s in filtered if feature in s.get("features", [])]

        if status != "all":
            filtered = [s for s in filtered if s.get("status") == status]

        if keyword:
            kw = keyword.lower().strip()
            filtered = [
                s for s in filtered
                if kw in str(s.get("id", "")).lower()
                or kw in str(s.get("smiles", "")).lower()
                or kw in str(s.get("original_index", "")).lower()
            ]

        total_matching = len(filtered)
        paged = filtered[offset : offset + limit]

        return {
            "total": total_matching,
            "limit": limit,
            "offset": offset,
            "items": paged,
        }

    def get_sample_detail(self, sample_id: str, auto_predict: bool = False) -> Optional[Dict[str, Any]]:
        """获取单个样本详情与当前 OBB 标注"""
        if not self._initialized:
            self._reload_manifest()

        sample = self._id_to_sample.get(str(sample_id))
        if not sample:
            # 尝试通过原始 index 查询
            for s in self._manifest_cache:
                if str(s.get("original_index")) == str(sample_id):
                    sample = s
                    break
        if not sample:
            return None

        sid = sample["id"]
        fname = sample.get("filename", f"{sid}.png")
        img_p = self.img_dir / fname

        # 读取已有标注
        ann_p = self.annotations_dir / f"{sid}.json"
        boxes = []
        is_preannotated = False

        if ann_p.exists():
            try:
                with open(ann_p, "r", encoding="utf-8") as f:
                    ann_data = json.load(f)
                    boxes = ann_data.get("boxes", [])
                    is_preannotated = ann_data.get("annotator", "").startswith("unified_obb_det")
            except Exception as e:
                print(f"[WARN] Failed to read {ann_p}: {e}")
        elif auto_predict and img_p.exists():
            # 自动进行 AI 预标注
            try:
                boxes = self.predict_sample(sid)
                is_preannotated = True
            except Exception as e:
                print(f"[WARN] Auto prediction failed for {sid}: {e}")

        # 检查图像尺寸
        img_size = sample.get("image_size")
        if not img_size and img_p.exists():
            im = cv2.imread(str(img_p))
            if im is not None:
                img_size = [im.shape[1], im.shape[0]]

        return {
            "id": sid,
            "source": sample.get("source", "unknown"),
            "original_index": sample.get("original_index"),
            "filename": fname,
            "image_size": img_size or [800, 600],
            "features": sample.get("features", []),
            "smiles": sample.get("smiles", ""),
            "status": sample.get("status", "pending"),
            "boxes": boxes,
            "is_preannotated": is_preannotated,
            "classes": CLASS_NAMES,
            "class_colors": CLASS_COLORS,
        }

    def get_image_bytes(self, sample_id: str) -> Optional[bytes]:
        sample = self._id_to_sample.get(str(sample_id))
        if not sample:
            for s in self._manifest_cache:
                if str(s.get("original_index")) == str(sample_id):
                    sample = s
                    break
        if not sample:
            return None

        fname = sample.get("filename", f"{sample['id']}.png")
        img_p = self.img_dir / fname
        if not img_p.exists():
            return None

        with open(img_p, "rb") as f:
            return f.read()

    def _get_detector(self):
        if self._detector is None:
            from markushreco.models import model_loader
            self._detector = model_loader.get_unified_obb_det()
        return self._detector

    def predict_sample(self, sample_id: str, conf: float = 0.18) -> List[Dict[str, Any]]:
        """调用 unified_obb_det 深度模型对单张图像进行实时推理"""
        sample = self._id_to_sample.get(str(sample_id))
        if not sample:
            raise ValueError(f"Sample not found: {sample_id}")

        fname = sample.get("filename", f"{sample['id']}.png")
        img_p = self.img_dir / fname
        if not img_p.exists():
            raise FileNotFoundError(f"Image not found: {img_p}")

        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            raise ValueError(f"Failed to read image: {img_p}")

        det = self._get_detector()
        model = getattr(det, "model_a", None)
        if model is None:
            model = det.model

        res = model.predict(img_bgr, conf=conf, verbose=False)[0]
        boxes = []
        if res.obb is not None and len(res.obb) > 0:
            cls_arr = res.obb.cls.cpu().numpy().astype(int)
            conf_arr = res.obb.conf.cpu().numpy().astype(float)
            corners_arr = res.obb.xyxyxyxy.cpu().numpy().astype(float)  # (N, 4, 2)

            for i, (c_id, score, corners) in enumerate(zip(cls_arr, conf_arr, corners_arr)):
                boxes.append({
                    "id": f"box_{i+1}",
                    "class_id": int(c_id),
                    "class_name": CLASS_NAMES.get(int(c_id), f"cls_{c_id}"),
                    "corners": corners.tolist(),
                    "conf": round(float(score), 4),
                })
        return boxes

    def save_annotation(
        self,
        sample_id: str,
        boxes: List[Dict[str, Any]],
        image_size: Optional[List[int]] = None,
        status: str = "reviewed",
        annotator: str = "human_review",
    ) -> bool:
        """保存 OBB 标注至 JSON 并同步输出 YOLO OBB .txt"""
        sample = self._id_to_sample.get(str(sample_id))
        if not sample:
            for s in self._manifest_cache:
                if str(s.get("original_index")) == str(sample_id):
                    sample = s
                    break
        if not sample:
            return False

        sid = sample["id"]
        if not image_size or len(image_size) != 2:
            image_size = sample.get("image_size", [800, 600])

        w, h = image_size[0], image_size[1]

        # 1. 保存 JSON
        ann_data = {
            "id": sid,
            "source": sample.get("source"),
            "original_index": sample.get("original_index"),
            "image_size": [w, h],
            "boxes": boxes,
            "annotator": annotator,
            "status": status,
        }
        ann_path = self.annotations_dir / f"{sid}.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(ann_data, f, ensure_ascii=False, indent=2)

        # 2. 导出 Ultralytics YOLO OBB 标签 (.txt)
        # 格式: class_id x1 y1 x2 y2 x3 y4 (归一化至 0~1)
        yolo_path = self.yolo_dir / f"{sid}.txt"
        yolo_lines = []
        for b in boxes:
            c_id = int(b["class_id"])
            pts = np.asarray(b["corners"], dtype=float)
            norm_pts = pts.copy()
            norm_pts[:, 0] = np.clip(norm_pts[:, 0] / max(w, 1), 0.0, 1.0)
            norm_pts[:, 1] = np.clip(norm_pts[:, 1] / max(h, 1), 0.0, 1.0)
            coords_str = " ".join(f"{coord:.6f}" for coord in norm_pts.reshape(-1))
            yolo_lines.append(f"{c_id} {coords_str}")

        with open(yolo_path, "w", encoding="utf-8") as f:
            f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        # 3. 维护 review_status.jsonl
        self._status_cache[sid] = status
        sample["status"] = status
        sample["box_count"] = len(boxes)

        # 追加或更新 status_file
        status_entries = {}
        if self.status_file.exists():
            with open(self.status_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            status_entries[d["id"]] = d
                        except Exception:
                            pass
        status_entries[sid] = {"id": sid, "status": status, "box_count": len(boxes)}
        with open(self.status_file, "w", encoding="utf-8") as f:
            for d in status_entries.values():
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

        return True
