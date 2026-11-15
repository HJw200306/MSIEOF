import os
import json
import io
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np

# repository/markush_annotator/backend/data_manager.py -> repository
BASE_DIR = Path(__file__).resolve().parents[2]

# 优先使用环境变量或 MolParser 特殊案例纠错数据集，其次回退至历史 full_gate 与 pilot_uspto
CUSTOM_DATA_DIR = os.environ.get("ANNOTATOR_DATA_DIR")
DEFAULT_MOLPARSER_REVIEW = BASE_DIR / "experiments" / "molparser_manual_review"
DEFAULT_FULL_GATE = BASE_DIR / "experiments" / "pilot_uspto_batch" / "full_gate"

if CUSTOM_DATA_DIR and Path(CUSTOM_DATA_DIR).exists():
    DATA_ROOT = Path(CUSTOM_DATA_DIR)
    DEFAULT_MANIFEST = DATA_ROOT / "manifest.jsonl"  # 数据根目录下直接用 manifest.jsonl
    DEFAULT_ANNOTATIONS_DIR = DATA_ROOT / "annotations"
    DEFAULT_CORRECTED_DIR = DATA_ROOT / "corrected_annotations"
    DEFAULT_STATUS_FILE = DATA_ROOT / "review_status.jsonl"  # 直接在根目录
    DEFAULT_IMAGE_DIR = DATA_ROOT / "images"
elif DEFAULT_MOLPARSER_REVIEW.exists():
    DEFAULT_MANIFEST = DEFAULT_MOLPARSER_REVIEW / "manual_review" / "manifest.jsonl"
    DEFAULT_ANNOTATIONS_DIR = DEFAULT_MOLPARSER_REVIEW / "annotations"
    DEFAULT_CORRECTED_DIR = DEFAULT_MOLPARSER_REVIEW / "corrected_annotations"
    DEFAULT_STATUS_FILE = DEFAULT_MOLPARSER_REVIEW / "manual_review" / "review_status.jsonl"
    DEFAULT_IMAGE_DIR = DEFAULT_MOLPARSER_REVIEW / "images"
elif DEFAULT_FULL_GATE.exists():
    DEFAULT_MANIFEST = DEFAULT_FULL_GATE / "manual_review" / "manifest.jsonl"
    DEFAULT_ANNOTATIONS_DIR = DEFAULT_FULL_GATE / "annotations"
    DEFAULT_CORRECTED_DIR = DEFAULT_FULL_GATE / "corrected_annotations"
    DEFAULT_STATUS_FILE = DEFAULT_FULL_GATE / "manual_review" / "review_status.jsonl"
    DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "uspto_hf"
else:
    DEFAULT_MANIFEST = BASE_DIR / "experiments" / "pilot_uspto" / "manual_review" / "manifest.jsonl"
    DEFAULT_ANNOTATIONS_DIR = BASE_DIR / "experiments" / "pilot_uspto" / "annotations"
    DEFAULT_CORRECTED_DIR = BASE_DIR / "experiments" / "pilot_uspto" / "corrected_annotations"
    DEFAULT_STATUS_FILE = BASE_DIR / "experiments" / "pilot_uspto" / "manual_review" / "review_status.jsonl"
    DEFAULT_IMAGE_DIR = BASE_DIR / "data" / "uspto_hf"

# 标准大类原因定义
CATEGORY_DEFS = [
    {
        "key": "case_b_brackets",
        "label": "Markush 括号",
        "en": "Markush Brackets",
        "icon": "🔲",
        "description": "聚合物或重复单元化学括号 / 浮动连接基团 (case_b_brackets)",
    },
    {
        "key": "stereo_bond",
        "label": "立体键结构",
        "en": "Stereo Bonds",
        "icon": "🧪",
        "description": "手性中心与楔形/虚线立体分子键 (stereo_bond)",
    },
    {
        "key": "superatom",
        "label": "超原子/保护基",
        "en": "Superatoms",
        "icon": "🏷️",
        "description": "缩写基团、保护基与 Markush 变量原子 (superatom)",
    },
    {
        "key": "atom_no_text",
        "label": "原子无文本",
        "en": "Atom No Text",
        "icon": "📝",
        "description": "未检测到字符的隐式或缺失原子 (如 atom_no_text:1 ~ 6)",
    },
    {
        "key": "bond_len_outlier",
        "label": "键长异常",
        "en": "Bond Length Outlier",
        "icon": "📏",
        "description": "骨架比例畸变或过长/过短化学键 (如 bond_len_outlier:1/14)",
    },
    {
        "key": "ap_replaced",
        "label": "附着点替换",
        "en": "AP Replaced",
        "icon": "🔗",
        "description": "Markush 附着点 / AP 结构基团替换",
    },
    {
        "key": "atom_overlap",
        "label": "原子重叠",
        "en": "Atom Overlap",
        "icon": "💥",
        "description": "原子坐标重叠或间距过近 (如 atom_overlap:0.0px)",
    },
    {
        "key": "cc_chain_replaced",
        "label": "碳链超原子",
        "en": "Carbon Chain Replaced",
        "icon": "⛓️",
        "description": "多碳烷基直链缩写替换",
    },
    {
        "key": "composite",
        "label": "复合特殊结构",
        "en": "Composite Reasons",
        "icon": "🔀",
        "description": "同时存在多项特殊结构 (如 括号+立体键+超原子)",
    },
    {
        "key": "other",
        "label": "其他分类",
        "en": "Other Reasons",
        "icon": "❓",
        "description": "未单独归类的审核原因",
    },
]


def parse_reason_category(raw_reason: str) -> str:
    """将包含具体像素或数量的细粒度原因归类到标准大类"""
    if not raw_reason or raw_reason == "unknown":
        return "other"
    raw_reason = str(raw_reason).strip()

    # 包含 '+' 的复合原因
    parts = [p.strip() for p in raw_reason.split("+") if p.strip()]
    if len(parts) > 1:
        return "composite"

    p = parts[0] if parts else raw_reason
    if "bracket" in p or p.startswith("case_b_brackets"):
        return "case_b_brackets"
    if "stereo" in p:
        return "stereo_bond"
    if "super" in p or "superatom" in p:
        return "superatom"
    if p.startswith("atom_no_text"):
        return "atom_no_text"
    if p.startswith("bond_len_outlier"):
        return "bond_len_outlier"
    if p.startswith("ap_replaced") or p.startswith("ap_"):
        return "ap_replaced"
    if p.startswith("atom_overlap"):
        return "atom_overlap"
    if p.startswith("bond_overlap"):
        return "bond_overlap"
    if p.startswith("cc_chain_replaced") or "cc_chain" in p:
        return "cc_chain_replaced"
    return "other"


def get_reason_display_info(raw_reason: str) -> Dict[str, Any]:
    """生成友好的人类可读原因展示信息 (包含大类图标、中文名和细粒度参数)"""
    cat_key = parse_reason_category(raw_reason)
    cat_meta = next((c for c in CATEGORY_DEFS if c["key"] == cat_key), None)
    if not cat_meta:
        cat_meta = {"key": cat_key, "label": cat_key, "en": cat_key, "icon": "🏷️"}

    # 提取细分参数 (例如 atom_no_text:2 -> 2个; atom_overlap:0.0px -> 0.0px)
    detail_str = ""
    if ":" in raw_reason:
        parts = raw_reason.split("+")
        detail_items = []
        for part in parts:
            if ":" in part:
                detail_items.append(part.split(":", 1)[1])
        if detail_items:
            detail_str = ", ".join(detail_items)

    display_badge = f"{cat_meta['icon']} {cat_meta['label']}"
    if detail_str:
        display_badge += f" ({detail_str})"

    return {
        "category": cat_key,
        "category_label": cat_meta["label"],
        "category_icon": cat_meta["icon"],
        "raw_reason": raw_reason,
        "detail": detail_str,
        "display_badge": display_badge,
    }


# 周期表标准单质元素集合 (排除易与有机缩写/保护基混淆的金属与放射性元素)
STANDARD_ELEMENTS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Th", "Pa", "U", "Np", "Pu"
}

# 明确强制判定为超原子/保护基/官能团/Markush 变量的符号集合
FORCED_SUPERATOMS = {
    "Ac", "Ts", "Ms", "Tf", "Ph", "Bn", "Bz", "Tol", "Mes", "Naph",
    "Boc", "Fmoc", "Cbz", "Piv", "TBDMS", "TMS", "TIPS", "THP", "Tr", "PMB",
    "Me", "Et", "Pr", "iPr", "nPr", "cPr", "Bu", "iBu", "nBu", "tBu", "cHex",
    "OMe", "OEt", "OPh", "SMe", "NH2", "NMe2", "NO2", "SO2", "SO3H", "COOH",
    "CO2H", "COOMe", "COOEt", "CONH2", "CHO", "CN", "CF3", "OCF3", "SF5",
    "CH3", "H3C", "CH2", "C2H5", "CH2CH3", "HO3S", "SO2NH2", "O2S", "SOOO",
    "R", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11", "R12",
    "X", "Y", "Z", "Ar", "Het", "Hal", "_AP", "*"
}


def normalize_atom_dict(a: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一原子数据结构的颗粒度规范:
    1. 若已有 label: symbol 恒为 '*'，label 保持原值
    2. 若 symbol 为 '*' 或 '_AP': symbol 恒为 '*'，label 缺省为 '_AP'
    3. 若 symbol 属于超原子/基团 (如 SO2, SO3H, CF3, Me, Et, R1...) 或非标准单质元素/包含数字:
       symbol 规范为 '*'，label 规范为原 symbol 字符串
    4. 其余标准化学元素: symbol 保持标准大小写，label 为 None
    """
    raw_sym = str(a.get("symbol") or a.get("atom_symbol") or "C").strip()
    raw_lab = a.get("label") or a.get("atom_label")
    if isinstance(raw_lab, str):
        raw_lab = raw_lab.strip()
    if not raw_lab:
        raw_lab = None

    # 如果已有 label
    if raw_lab:
        a["symbol"] = "*"
        a["label"] = raw_lab
        return a

    # 如果无 label，检查 symbol 是否是通配符
    if raw_sym in ("*", "_AP"):
        a["symbol"] = "*"
        a["label"] = "_AP"
        return a

    # 检查是否为 R 基团模式 (如 R1, R2, X, Y, Z, 1*, 2*)
    is_rgroup = False
    if raw_sym.startswith("R") and (len(raw_sym) == 1 or raw_sym[1:].isdigit() or raw_sym[1:] in "abcdef"):
        is_rgroup = True
    elif raw_sym.endswith("*") and raw_sym[:-1].isdigit():
        raw_sym = "R" + raw_sym[:-1]
        is_rgroup = True

    # 判断是否为超原子
    if (
        is_rgroup
        or raw_sym in FORCED_SUPERATOMS
        or any(ch.isdigit() for ch in raw_sym)
        or len(raw_sym) > 2
        or raw_sym not in STANDARD_ELEMENTS
    ):
        a["symbol"] = "*"
        a["label"] = raw_sym
    else:
        # 标准元素: 保证规范大小写 (如 Cl, Br, Si)
        norm_sym = raw_sym.capitalize() if len(raw_sym) <= 2 else raw_sym
        a["symbol"] = norm_sym
        a["label"] = None

    return a


def graph_to_smiles(
    atoms: List[Dict[str, Any]],
    bonds: List[Any],
    brackets: Optional[List[Any]] = None
) -> str:
    """使用 RDKit 将 2D 图结构转换为标准 CXSMILES (含 |$...$| 扩展段)"""
    try:
        from rdkit import Chem
    except ImportError:
        print("[WARN] RDKit is not installed in the environment.")
        return ""

    if not atoms:
        return ""

    try:
        rw = Chem.RWMol()
        atom_idx_map = {}

        for i, a in enumerate(atoms):
            norm_a = dict(a)
            normalize_atom_dict(norm_a)

            sym = norm_a.get("symbol", "C")
            label = norm_a.get("label")

            if sym == "*" or label:
                atom = Chem.Atom(0)  # Dummy atom (*)
                atom_label = str(label) if label else "_AP"
                atom.SetProp("atomLabel", atom_label)
            else:
                try:
                    atom = Chem.Atom(sym)
                except Exception:
                    atom = Chem.Atom(0)
                    if sym != "C":
                        atom.SetProp("atomLabel", sym)

            idx = rw.AddAtom(atom)
            atom_idx_map[i] = idx

        bond_type_map = {
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
            4: Chem.BondType.AROMATIC,
            5: Chem.BondType.SINGLE,  # Wedge
            6: Chem.BondType.SINGLE,  # Hash Dash
            7: Chem.BondType.UNSPECIFIED,  # Wave -> SMILES '~' (不确定/连接点)
        }

        for b in bonds:
            if isinstance(b, list) and len(b) >= 2:
                u, v = b[0], b[1]
                order = b[2] if len(b) > 2 else 1
            elif isinstance(b, dict):
                u = b.get("endpoint_atoms", [0, 1])[0] if "endpoint_atoms" in b else b.get("u", 0)
                v = b.get("endpoint_atoms", [0, 1])[1] if "endpoint_atoms" in b else b.get("v", 1)
                order = b.get("bond_type", 1)
                if isinstance(order, str):
                    t_map = {"single": 1, "double": 2, "triple": 3, "aromatic": 4, "wedge": 5, "dash": 6}
                    order = t_map.get(order.lower(), 1)
            else:
                continue

            try:
                u, v, order = int(u), int(v), int(order)
            except (TypeError, ValueError):
                continue

            if u in atom_idx_map and v in atom_idx_map and u != v:
                b_type = bond_type_map.get(order, Chem.BondType.SINGLE)
                existing_bond = rw.GetBondBetweenAtoms(atom_idx_map[u], atom_idx_map[v])
                if not existing_bond:
                    rw.AddBond(atom_idx_map[u], atom_idx_map[v], b_type)
                    if order == 4:
                        rw.GetAtomWithIdx(atom_idx_map[u]).SetIsAromatic(True)
                        rw.GetAtomWithIdx(atom_idx_map[v]).SetIsAromatic(True)

        mol = rw.GetMol()
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY)
        except Exception:
            pass

        try:
            smi = Chem.MolToCXSmiles(mol)
        except Exception:
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = ""

        return smi
    except Exception as e:
        print(f"[WARN] graph_to_smiles conversion failed: {e}")
        return ""


def convert_polygons_to_brackets(polygons: List[List[List[float]]]) -> List[Dict[str, Any]]:
    if not polygons:
        return []
    
    brackets = []
    n = len(polygons)
    
    if n % 3 == 0:
        for i in range(0, n, 3):
            gid = (i // 3) + 1
            brackets.append({
                "id": i,
                "group_id": gid,
                "type": "left",
                "polygon": polygons[i]
            })
            brackets.append({
                "id": i + 1,
                "group_id": gid,
                "type": "right",
                "polygon": polygons[i + 1]
            })
            brackets.append({
                "id": i + 2,
                "group_id": gid,
                "type": "subscript",
                "polygon": polygons[i + 2],
                "text": "n" if gid == 1 else "m"
            })
    elif n % 2 == 0:
        for i in range(0, n, 2):
            gid = (i // 2) + 1
            brackets.append({
                "id": i,
                "group_id": gid,
                "type": "left",
                "polygon": polygons[i]
            })
            brackets.append({
                "id": i + 1,
                "group_id": gid,
                "type": "right",
                "polygon": polygons[i + 1]
            })
    else:
        for i, poly in enumerate(polygons):
            t = "left" if i == 0 else ("right" if i == 1 else "subscript")
            brackets.append({
                "id": i,
                "group_id": 1,
                "type": t,
                "polygon": poly,
                "text": "n" if t == "subscript" else None
            })
            
    return brackets


class DataManager:
    def __init__(
        self,
        manifest_path: Optional[Path] = None,
        image_root: Optional[Path] = None,
        annotations_dir: Optional[Path] = None,
        corrected_dir: Optional[Path] = None,
        status_file: Optional[Path] = None,
    ):
        self.manifest_path = Path(manifest_path or DEFAULT_MANIFEST)
        self.image_root = Path(image_root or DEFAULT_IMAGE_DIR)
        self.annotations_dir = Path(annotations_dir or DEFAULT_ANNOTATIONS_DIR)
        self.corrected_dir = Path(corrected_dir or DEFAULT_CORRECTED_DIR)
        self.status_file = Path(status_file or DEFAULT_STATUS_FILE)
        
        self.corrected_dir.mkdir(parents=True, exist_ok=True)
        if self.status_file.parent:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            
        self.samples: List[Dict[str, Any]] = []
        self.sample_index_map: Dict[int, int] = {}  # index -> list index
        self.statuses: Dict[int, Dict[str, Any]] = {}
        self.raw_edges: Dict[int, List[List[int]]] = {}
        self.raw_edges_cache_file = self.manifest_path.parent / "raw_edges_cache.json"
        
        self._load_status()
        self._load_manifest()
        self._load_raw_edges()

    def _load_raw_edges(self):
        """加载或构建原始 ground-truth edges 字典以恢复化学键类型 (单/双/三/芳香/楔形)"""
        self.raw_edges.clear()
        if self.raw_edges_cache_file.exists():
            try:
                with open(self.raw_edges_cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.raw_edges = {int(k): v for k, v in data.items()}
                return
            except Exception:
                pass

        raw_manifest_candidates = [
            self.image_root / "manifest.jsonl",
            BASE_DIR / "data" / "uspto_hf" / "manifest.jsonl",
        ]
        raw_manifest = next((p for p in raw_manifest_candidates if p.exists()), None)
        if not raw_manifest:
            return

        needed_indices = set(s["index"] for s in self.samples)
        if not needed_indices:
            return

        try:
            with open(raw_manifest, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    d = json.loads(line)
                    idx = d.get("index")
                    if idx in needed_indices:
                        edges = d.get("edges", [])
                        if edges:
                            self.raw_edges[idx] = edges

            if self.raw_edges:
                with open(self.raw_edges_cache_file, "w", encoding="utf-8") as f:
                    json.dump(self.raw_edges, f)
        except Exception as e:
            print(f"[WARN] Failed to load raw edges: {e}")

    def _load_status(self):
        """加载审核进度状态"""
        self.statuses.clear()
        if self.status_file.exists():
            with open(self.status_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record = json.loads(line)
                            self.statuses[record["index"]] = record
                        except Exception:
                            pass

    def _save_status(self, index: int, status: str, notes: str = ""):
        """更新并写回单条状态"""
        record = {
            "index": index,
            "review_status": status,
            "review_notes": notes,
        }
        self.statuses[index] = record
        
        with open(self.status_file, "w", encoding="utf-8") as f:
            for rec in self.statuses.values():
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_manifest(self):
        """加载人工审核池 manifest.jsonl"""
        self.samples.clear()
        self.sample_index_map.clear()
        
        if not self.manifest_path.exists():
            print(f"[WARN] Manifest not found: {self.manifest_path}")
            return

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = item["index"]
                
                # 注入已纠错/审核状态
                if idx in self.statuses:
                    item["review_status"] = self.statuses[idx].get("review_status", "pending")
                    item["review_notes"] = self.statuses[idx].get("review_notes", "")
                else:
                    item["review_status"] = "pending"
                    item["review_notes"] = ""

                # 检查是否存在已保存的 corrected_annotations
                corrected_file = self.corrected_dir / f"{idx:06d}.json"
                item["has_corrected"] = corrected_file.exists()

                self.samples.append(item)
                self.sample_index_map[idx] = len(self.samples) - 1

        print(f"[INFO] Loaded {len(self.samples)} samples from {self.manifest_path}")

    def get_manifest_info(self) -> Dict[str, Any]:
        """获取数据集概览与统计信息 (按大类归纳聚合)"""
        total = len(self.samples)
        
        # 统计各大类数量
        cat_counts_map = {}
        for s in self.samples:
            raw_r = s.get("reason", "unknown")
            cat = parse_reason_category(raw_r)
            cat_counts_map[cat] = cat_counts_map.get(cat, 0) + 1

        categories_list = []
        for d in CATEGORY_DEFS:
            k = d["key"]
            cnt = cat_counts_map.get(k, 0)
            if cnt > 0 or k == "atom_no_text":
                categories_list.append({
                    "key": k,
                    "label": d["label"],
                    "en": d["en"],
                    "icon": d["icon"],
                    "count": cnt,
                    "description": d["description"],
                })

        status_counts = {"pending": 0, "approved": 0, "discarded": 0}
        corrected_count = 0
        for s in self.samples:
            st = s.get("review_status", "pending")
            status_counts[st] = status_counts.get(st, 0) + 1
            if s.get("has_corrected"):
                corrected_count += 1

        return {
            "total_count": total,
            "categories": categories_list,
            "reasons": sorted(list(set(s.get("reason", "unknown") for s in self.samples))),
            "status_counts": status_counts,
            "manifest_path": str(self.manifest_path),
            "corrected_count": corrected_count,
        }

    def list_samples(
        self,
        reason: Optional[str] = None,
        status: Optional[str] = None,
        keyword: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """按条件筛选样本列表 (支持大类原因与模糊匹配)"""
        filtered = self.samples

        if reason and reason != "all":
            def match_reason(s_reason: str, filter_key: str) -> bool:
                if not s_reason:
                    return filter_key == "other"
                if s_reason == filter_key:
                    return True
                # 检查主分类
                if parse_reason_category(s_reason) == filter_key:
                    return True
                # 检查复合分类中是否包含此大类
                if "+" in s_reason:
                    parts = [p.strip() for p in s_reason.split("+")]
                    for p in parts:
                        if parse_reason_category(p) == filter_key:
                            return True
                return False

            filtered = [s for s in filtered if match_reason(s.get("reason", ""), reason)]

        if status and status != "all":
            filtered = [s for s in filtered if s.get("review_status") == status]

        if keyword:
            kw = keyword.lower()
            filtered = [
                s for s in filtered
                if kw in str(s.get("index")) or kw in s.get("smi", "").lower() or kw in s.get("reason", "").lower()
            ]

        total_filtered = len(filtered)
        items = filtered[offset : offset + limit]

        brief_items = []
        for s in items:
            raw_r = s.get("reason", "")
            brief_items.append({
                "index": s["index"],
                "smi": s.get("smi", ""),
                "reason": raw_r,
                "reason_info": get_reason_display_info(raw_r),
                "review_status": s.get("review_status", "pending"),
                "has_corrected": s.get("has_corrected", False),
                "crop": s.get("crop", "") or s.get("full_image", ""),
            })

        return {
            "total": total_filtered,
            "items": brief_items,
            "offset": offset,
            "limit": limit,
        }

    def resolve_image_path(self, raw_rel: str) -> Optional[Path]:
        """多策略解析原图实际存储路径"""
        if not raw_rel:
            return None
        candidates = [
            self.image_root / raw_rel,
            self.image_root / Path(raw_rel).name,
            self.image_root / "images" / Path(raw_rel).name,
            self.manifest_path.parent / raw_rel,
            self.manifest_path.parent / Path(raw_rel).name,
            self.manifest_path.parent.parent / "images" / Path(raw_rel).name,
            BASE_DIR / "experiments" / "molparser_manual_review" / "images" / Path(raw_rel).name,
            BASE_DIR / "data" / "uspto_hf" / raw_rel,
            BASE_DIR / raw_rel,
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def get_image_bytes(self, index: int) -> Optional[bytes]:
        """读取原图并编码为 PNG 字节流"""
        if index not in self.sample_index_map:
            return None
        sample = self.samples[self.sample_index_map[index]]
        raw_img = sample.get("image", "")
        img_path = self.resolve_image_path(raw_img)

        # 若原图不存在，优先寻找裁剪图
        if not img_path:
            crop_name = sample.get("crop", "") or sample.get("full_image", "") or f"{index:06d}.png"
            candidates = [
                DEFAULT_IMAGE_DIR / crop_name,  # ← 添加：优先使用当前数据目录的images/
                self.manifest_path.parent / crop_name,
                self.manifest_path.parent / "images" / crop_name,  # ← 添加：支持images子目录
                BASE_DIR / "experiments" / "pilot_uspto_batch" / "full_gate" / "manual_review" / crop_name,
                BASE_DIR / "experiments" / "pilot_uspto" / "manual_review" / crop_name,
            ]
            for c in candidates:
                if c.exists():
                    img_path = c
                    break

        if not img_path or not img_path.exists():
            print(f"[DEBUG] Image not found for index {index}")
            return None

        print(f"[DEBUG] Loading image for index {index}: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[DEBUG] Failed to read image: {img_path}")
            return None

        success, encoded = cv2.imencode(".png", img)
        if not success:
            return None
        return encoded.tobytes()

    def get_cv2_image(self, index: int) -> Optional[np.ndarray]:
        """读取原图为 cv2 BGR 矩阵"""
        if index not in self.sample_index_map:
            return None
        sample = self.samples[self.sample_index_map[index]]
        raw_img = sample.get("image", "")
        img_path = self.resolve_image_path(raw_img)
        if not img_path:
            crop_name = sample.get("crop", "") or sample.get("full_image", "") or f"{index:06d}.png"
            candidates = [
                self.manifest_path.parent / crop_name,
                BASE_DIR / "experiments" / "pilot_uspto_batch" / "full_gate" / "manual_review" / crop_name,
                BASE_DIR / "experiments" / "pilot_uspto" / "manual_review" / crop_name,
            ]
            for c in candidates:
                if c.exists():
                    img_path = c
                    break

        if not img_path or not img_path.exists():
            return None

        return cv2.imread(str(img_path))

    def get_sample_detail(self, index: int) -> Optional[Dict[str, Any]]:
        """获取单个样本的完整标注与原图元数据"""
        if index not in self.sample_index_map:
            return None
        sample = self.samples[self.sample_index_map[index]]
        
        # 优先读取已纠错文件，其次读取原始 pilot 标注 JSON
        corrected_file = self.corrected_dir / f"{index:06d}.json"
        raw_annotation_file = self.annotations_dir / f"{index:06d}.json"
        
        annotation_data = None
        source = "generated"

        if corrected_file.exists():
            with open(corrected_file, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            source = "corrected"
            print(f"[DEBUG] Loaded corrected annotation for index {index}: {len(annotation_data.get('atoms', []))} atoms")
        elif raw_annotation_file.exists():
            with open(raw_annotation_file, "r", encoding="utf-8") as f:
                annotation_data = json.load(f)
            source = "annotation"
            print(f"[DEBUG] Loaded raw annotation for index {index}: {len(annotation_data.get('atoms', []))} atoms")
        else:
            print(f"[DEBUG] No annotation file found for index {index}")
            print(f"[DEBUG]   Corrected: {corrected_file}")
            print(f"[DEBUG]   Raw: {raw_annotation_file}")
            
        # 读取原图尺寸
        img_path = self.resolve_image_path(sample.get("image", ""))
        if not img_path:
            crop_name = sample.get("crop", "") or sample.get("full_image", "") or f"{index:06d}.png"
            for c in [self.manifest_path.parent / crop_name, BASE_DIR / "experiments" / "pilot_uspto_batch" / "full_gate" / "manual_review" / crop_name]:
                if c.exists():
                    img_path = c
                    break

        width, height = 800, 600
        if img_path and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]

        # 优先读取纠错后的 esmiles，保证修改后的 SMILES 重新载入时正确展示
        raw_r = sample.get("reason", "")
        cur_smi = sample.get("smi", "")
        if annotation_data:
            cur_smi = annotation_data.get("esmiles") or annotation_data.get("smi") or cur_smi

        # 组装返回数据
        result = {
            "index": index,
            "smi": cur_smi,
            "reason": raw_r,
            "reason_info": get_reason_display_info(raw_r),
            "image_size": [width, height],
            "bracket_polygons": sample.get("bracket_polygons", []),
            "ap_atoms": sample.get("ap_atoms", []),
            "source": source,
            "review_status": sample.get("review_status", "pending"),
            "review_notes": sample.get("review_notes", ""),
            "atoms": [],
            "bonds": [],
            "sgroups": [],
            "brackets": [],
        }

        if annotation_data:
            # 如果annotation_data中有image_size，优先使用它（因为坐标是基于这个尺寸归一化的）
            if "image_size" in annotation_data and annotation_data["image_size"]:
                width, height = annotation_data["image_size"]
                print(f"[DEBUG] Using image_size from annotation_data: {width}x{height}")
            else:
                print(f"[DEBUG] Using default/detected image_size: {width}x{height}")

            # 转换atoms字段名：atom_symbol -> symbol (前端期望symbol字段)
            raw_atoms = annotation_data.get("atoms", [])
            result["atoms"] = []
            for a in raw_atoms:
                atom = dict(a)  # 复制原atom
                if "atom_symbol" in atom and "symbol" not in atom:
                    atom["symbol"] = atom.pop("atom_symbol")

                # 关键修复：将归一化坐标(0~1)转换为像素坐标
                if "x" in atom and "y" in atom:
                    original_x, original_y = atom["x"], atom["y"]
                    if 0 <= atom["x"] <= 1.0 and 0 <= atom["y"] <= 1.0:
                        # 坐标是归一化的，需要转换为像素
                        atom["x"] = atom["x"] * width
                        atom["y"] = atom["y"] * height
                        if len(result["atoms"]) == 0:  # 只打印第一个原子的转换
                            print(f"[DEBUG] Atom coordinate conversion: ({original_x:.4f}, {original_y:.4f}) -> ({atom['x']:.2f}, {atom['y']:.2f}) using size {width}x{height}")

                result["atoms"].append(atom)

            raw_bonds = annotation_data.get("bonds", [])
            print(f"[DEBUG] Populating result for index {index}: {len(result['atoms'])} atoms, {len(raw_bonds)} raw bonds")
            
            # 构建原始 ground truth edges 查找表: (u, v) -> bond_type
            e_map = {}
            raw_e = self.raw_edges.get(index, [])
            for e in raw_e:
                if len(e) >= 3:
                    e_map[(min(e[0], e[1]), max(e[0], e[1]))] = e[2]

            normalized_bonds = []
            for b in raw_bonds:
                if isinstance(b, list):
                    if len(b) >= 3:
                        u, v, t = b[0], b[1], b[2]
                        if t == 1:
                            t = e_map.get((min(u, v), max(u, v)), 1)
                    elif len(b) == 2:
                        u, v = b[0], b[1]
                        t = e_map.get((min(u, v), max(u, v)), 1)
                    else:
                        continue
                    normalized_bonds.append([u, v, t])
                elif isinstance(b, dict):
                    u = b.get("endpoint_atoms", [0, 1])[0] if "endpoint_atoms" in b else b.get("begin_atom", b.get("u", 0))
                    v = b.get("endpoint_atoms", [0, 1])[1] if "endpoint_atoms" in b else b.get("end_atom", b.get("v", 1))
                    t = b.get("bond_type", 1)
                    if isinstance(t, str):
                        t_map = {
                            "single": 1, "double": 2, "triple": 3, "aromatic": 4,
                            "wedge": 5, "dash": 6, "solid wedge": 5, "dashed wedge": 6
                        }
                        t = t_map.get(t.lower(), 1)
                    if t == 1:
                        t = e_map.get((min(u, v), max(u, v)), 1)
                    normalized_bonds.append([u, v, t])
            result["bonds"] = normalized_bonds
            result["sgroups"] = annotation_data.get("sgroups", [])
            if "brackets" in annotation_data and annotation_data["brackets"]:
                result["brackets"] = annotation_data["brackets"]
            # 使用annotation_data中的image_size（坐标已基于此尺寸转换）
            if "image_size" in annotation_data and annotation_data["image_size"]:
                result["image_size"] = annotation_data["image_size"]

        # 如果没有标注 brackets，则从检测多边形预初始化
        if not result["brackets"]:
            result["brackets"] = convert_polygons_to_brackets(sample.get("bracket_polygons", []))

        # 超原子与符号颗粒度规范化: 凡是超原子/R基团，symbol 必须为 '*' 且 label 为文本
        for a in result["atoms"]:
            normalize_atom_dict(a)

        # 若当前无有效 CXSMILES 或为旧版格式，根据规范化后的图结构实时计算权威 CXSMILES
        if result["atoms"]:
            gen_cx = graph_to_smiles(result["atoms"], result["bonds"], result["brackets"])
            if gen_cx:
                # 只要不是已人工纠错的自定义 SMILES，或者当前 SMILES 缺少 CXSMILES 扩展段，一律采用实时计算出的权威 CXSMILES
                if source != "corrected" or not cur_smi or "|" not in cur_smi:
                    result["smi"] = gen_cx
                else:
                    result["smi"] = cur_smi

        print(f"[DEBUG] Returning result for index {index}: {len(result['atoms'])} atoms, {len(result['bonds'])} bonds")
        return result

    def save_sample_annotation(self, index: int, data: Dict[str, Any]) -> bool:
        """保存人工纠错后的标注数据"""
        try:
            target_file = self.corrected_dir / f"{index:06d}.json"
            
            # 标准化数据结构
            atoms = data.get("atoms", [])
            for a in atoms:
                normalize_atom_dict(a)

            bonds = data.get("bonds", [])
            brackets = data.get("brackets", [])

            # 根据当前图结构自动计算最新权威 CXSMILES
            gen_cx = graph_to_smiles(atoms, bonds, brackets)
            saved_smi = gen_cx or data.get("esmiles") or data.get("smi", "")

            payload = {
                "index": index,
                "esmiles": saved_smi,
                "image_size": data.get("image_size", [800, 600]),
                "coverage": 1.0,
                "atoms": atoms,
                "bonds": bonds,
                "sgroups": data.get("sgroups", []),
                "brackets": brackets,
                "ocr_atoms": data.get("ocr_atoms", []),
                "superatom_merges": data.get("superatom_merges", [])
            }
            
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

            review_status = data.get("review_status", "approved")
            review_notes = data.get("review_notes", "")
            self._save_status(index, review_status, review_notes)
            
            if index in self.sample_index_map:
                s = self.samples[self.sample_index_map[index]]
                s["review_status"] = review_status
                s["review_notes"] = review_notes
                s["has_corrected"] = True
                if saved_smi:
                    s["smi"] = saved_smi

            return True
        except Exception as e:
            print(f"[ERR] Failed to save annotation for #{index}: {e}")
            return False
