import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# 将当前项目与可选的 MarkushReco 挂载至 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MARKUSH_RECO_ROOT = os.environ.get("MARKUSH_RECO_ROOT", "")

sys.path.insert(0, str(PROJECT_ROOT))
if MARKUSH_RECO_ROOT and os.path.exists(MARKUSH_RECO_ROOT) and MARKUSH_RECO_ROOT not in sys.path:
    sys.path.insert(0, MARKUSH_RECO_ROOT)


_MOLSCRIBE_PARENT = os.environ.get("MOLSCRIBE_PARENT", "")
if _MOLSCRIBE_PARENT and os.path.exists(_MOLSCRIBE_PARENT) and _MOLSCRIBE_PARENT not in sys.path:
    sys.path.insert(0, _MOLSCRIBE_PARENT)


class AlignService:
    def __init__(self):
        self._aligner = None
        self._det = None
        self._rec = None
        self._initialized_aligner = False
        self._initialized_ocr = False

    def _init_aligner(self):
        if self._initialized_aligner:
            return
        try:
            from markushreco.postprocess.skeleton_aligner import SkeletonAligner
            self._aligner = SkeletonAligner()
            self._initialized_aligner = True
        except Exception as e:
            print(f"[WARN] SkeletonAligner not loaded: {e}")
            self._initialized_aligner = True

    def _init_ocr(self):
        if self._initialized_ocr:
            return
        try:
            from markushreco.models import model_loader
            self._det = model_loader.get_unified_obb_det()
            self._rec = model_loader.get_text_rec()
            self._initialized_ocr = True
        except Exception as e:
            print(f"[WARN] OCR models not loaded: {e}")
            self._initialized_ocr = True

    def align_atoms(
        self,
        atoms: List[Dict[str, Any]],
        bonds: List[Any],
        image_shape: tuple,
        image: Optional[np.ndarray] = None
    ) -> List[Dict[str, Any]]:
        """
        自动对齐原子坐标至图像骨架。
        如果模型未加载或失败，保持原坐标返回。
        """
        self._init_aligner()
        if not self._aligner or image is None or not atoms:
            return atoms

        try:
            h, w = image_shape[:2]
            # 归一化输入
            norm_atoms = []
            for a in atoms:
                norm_atoms.append({
                    "atom_symbol": a.get("symbol", "C"),
                    "x": a.get("x", 0.0) / max(w, 1),
                    "y": a.get("y", 0.0) / max(h, 1),
                })
            
            norm_bonds = []
            for b in bonds:
                if isinstance(b, list) and len(b) >= 2:
                    norm_bonds.append({"endpoint_atoms": [b[0], b[1]], "bond_type": "single"})

            # 执行对齐
            aligned = self._aligner.align(norm_atoms, norm_bonds, None, [])
            
            # 还原像素坐标
            result = []
            for i, a in enumerate(atoms):
                item = dict(a)
                if i < len(aligned):
                    item["x"] = round(aligned[i]["x"] * w, 1)
                    item["y"] = round(aligned[i]["y"] * h, 1)
                result.append(item)
            return result
        except Exception as e:
            print(f"[WARN] Align failed, keeping raw coords: {e}")
            return atoms

    def ocr_suggest(
        self,
        atoms: List[Dict[str, Any]],
        image: Optional[np.ndarray]
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        运行 OCR 文本检测与识别，对位置落在文本框内的 * 或 _AP 原子匹配识别结果。
        如果无 OCR 框，保留 _AP。
        返回 (更新后的 atoms 列表, 更新的原子数量)。
        """
        self._init_ocr()
        if not self._det or not self._rec or image is None or not atoms:
            return atoms, 0

        try:
            import cv2
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            from markushreco.markush.substituent_variation import _run_ocr, _clean_ocr_text
            ocr_boxes = _run_ocr(image, self._det, self._rec)
            if not ocr_boxes:
                return atoms, 0

            updated_count = 0
            new_atoms = [dict(a) for a in atoms]

            for i, a in enumerate(new_atoms):
                px = a.get("x", 0.0)
                py = a.get("y", 0.0)
                sym = a.get("symbol", "")
                label = a.get("label")

                # 优先匹配通配符 / 连接点 / 未明确标签的节点
                is_candidate = (
                    sym in ("*", "_AP") or
                    label in ("_AP", None, "") or
                    a.get("attachment_point")
                )

                if not is_candidate:
                    continue

                # 查找包含该原子坐标的 OCR 框 (容差 ±10px)
                hits = [(b, t) for b, t in ocr_boxes
                        if b[0] - 10 <= px <= b[2] + 10 and b[1] - 10 <= py <= b[3] + 10]
                if not hits:
                    continue

                # 取中心最近的文本框
                b0, t0 = min(hits, key=lambda bt:
                             ((bt[0][0] + bt[0][2]) / 2.0 - px) ** 2
                             + ((bt[0][1] + bt[0][3]) / 2.0 - py) ** 2)

                cleaned_text = _clean_ocr_text(t0).strip().strip("[]").strip()
                # 至少含一个字母 (超原子/R 基团标签)
                if cleaned_text and any(ch.isalpha() for ch in cleaned_text):
                    a["label"] = cleaned_text
                    a["attachment_point"] = False
                    a["recovered_from_ap"] = True
                    if a.get("symbol") == "_AP":
                        a["symbol"] = "*"
                    updated_count += 1

            return new_atoms, updated_count
        except Exception as e:
            print(f"[WARN] OCR suggest failed: {e}")
            return atoms, 0
