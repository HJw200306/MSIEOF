"""Label 保留的 SMILES 规范化与图匹配（端到端 OCSR 评测用）。

背景
----
MolScribe 及其衍生标注把超原子 / Markush 变量 / R 基团编码为 `*`（RDKit dummy
原子），其身份（如 [CH2]、[R1]、[C@@H]）放在 CXSMILES 尾部的 `|$...$|` 扩展段里。

修复的两个 bug（均经实测确认）
----------------------------
1. `evaluate_page2smiles.canonicalize_smiles_safe` 用裸 `Chem.MolFromSmiles`，
   RDKit 会**静默丢弃** `|$...$|` 段：`[CH2]` 与 `[OH]` 两个不同超原子都退化成
   同一个裸 `*`，Canonical SMILES EM 分不清它们。
2. `evaluate_page2smiles.get_inchi_key` 对含 `*`（dummy 原子）的分子，
   `Chem.MolToInchiKey` 返回**空串 ''**（不报错）；旧逻辑
   `graph_match = pred_inchi == gt_inchi and pred_inchi is not None` 在两个都含
   超原子的分子上得到 `'' == ''` → 恒 True，Graph EM 对含超原子的分子系统性虚高。

方案
----
用 RDKit 的 `*` 同位素承载 label 身份：把每个 distinct label 确定性映射为一个
同位素号（实测 `[99999*]` 可 round-trip，超过 99999 会回绕，故上限取 99999），
再走 `Chem.MolToSmiles` 让 RDKit 在规范化的同时保留 label 差异。

映射用「排序全局表」`label_table`（label -> 1..N），由构建 JSONL 时收集所有 GT
label 生成并落盘，跨 GT/预测调用保持一致，保证无碰撞。未命中表的 label 退化为
crc32 % 99999 + 1（顺序无关，但可能碰撞，仅用于极小概率的越表预测）。

已知局限
--------
- Tanimoto 为骨架级：Morgan 指纹不区分 `*` 的同位素（实测 `[1*]C` 与 `[2*]C`
  Tanimoto=1.0），故超原子 label 不参与相似度，仅参与精确匹配指标。
- `parse_smiles` 只处理末尾单个 `|$...$|` 段（本项目真值即如此）。
"""

from __future__ import annotations

import zlib
from typing import Dict, List, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# 实测 RDKit 对 * 的同位素 round-trip 上限（[99999*] OK，[999999*] 回绕）。
_ISO_MAX = 99999


def split_cxsmiles(smiles: str) -> Tuple[str, Optional[List[str]]]:
    """拆 ``bare |$l0;l1;...$|`` -> ``(bare, [l0, l1, ...])``。

    无映射段返回 ``(smiles, None)``。空 label 保留为 ``''``（与原子一一对应）。
    """
    if not smiles or "|$" not in smiles:
        return smiles, None
    bare, mapping = smiles.split("|$", 1)
    if mapping.endswith("$|"):
        mapping = mapping[:-2]
    elif mapping.endswith("|"):
        mapping = mapping[:-1]
    # 兼容结尾残留的 '$'（如 ``|$...$|`` 被截断的情况）
    mapping = mapping.rstrip("$")
    labels = mapping.split(";") if mapping else []
    return bare, labels


def _mol_from_smiles(bare: str) -> Optional[Chem.Mol]:
    """解析 bare SMILES，容忍 GT 标注中的超价原子。

    GT 由标注器 ``graph_to_smiles``（``MolToCXSmiles``）生成，其中部分分子
    含超价 C/N/O（如 C=O=C 使 O 价=4），``MolFromSmiles`` 严格 sanitize 会拒收。
    此时退回 ``sanitize=False`` 解析 + 跳过价键检查的部分 sanitize，尽量保留。
    语法错误（括号不配对等）仍返回 None。
    """
    mol = Chem.MolFromSmiles(bare)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(bare, sanitize=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
        )
    except Exception:
        return None
    return mol


def parse_smiles(smiles: str) -> Tuple[Optional[Chem.Mol], Optional[List[str]]]:
    """解析 SMILES -> ``(mol, labels)``。

    ``labels[i]`` 是第 i 个原子的超原子 label（非超原子为 ``''``）。
    解析失败返回 ``(None, None)``。RDKit ``MolFromSmiles`` 保持原子输入序，
    故 ``labels`` 与 ``mol`` 的原子索引一一对应。
    """
    if not smiles:
        return None, None
    bare, labels = split_cxsmiles(smiles)
    mol = _mol_from_smiles(bare)
    if mol is None:
        return None, None
    if labels is None:
        labels = [""] * mol.GetNumAtoms()
    elif len(labels) < mol.GetNumAtoms():
        labels = labels + [""] * (mol.GetNumAtoms() - len(labels))
    labels = labels[: mol.GetNumAtoms()]
    return mol, labels


def build_label_table(smiles_list: List[str]) -> Dict[str, int]:
    """从一批 SMILES 收集 distinct label，排序后映射到同位素 1..N。

    排序保证确定性且无碰撞；N 远小于 99999，不会触及上限。
    """
    labels: set = set()
    for s in smiles_list:
        _, labs = parse_smiles(s)
        if labs:
            labels.update(l for l in labs if l)
    return {lab: i + 1 for i, lab in enumerate(sorted(labels))}


def _fallback_iso(label: str) -> int:
    """label 不在 label_table 时的顺序无关退化映射。"""
    return (zlib.crc32(label.encode("utf-8")) % (_ISO_MAX - 1)) + 1


def canonicalize(
    smiles: str,
    label_table: Optional[Dict[str, int]] = None,
    isomeric: bool = True,
) -> Optional[str]:
    """label 保留的规范 SMILES。解析失败返回 None。"""
    mol, labels = parse_smiles(smiles)
    if mol is None:
        return None
    table = label_table or {}
    for atom, lab in zip(mol.GetAtoms(), labels):
        if atom.GetAtomicNum() == 0 and lab:
            atom.SetIsotope(table.get(lab) or _fallback_iso(lab))
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def graph_exact_match(
    s1: str, s2: str, label_table: Optional[Dict[str, int]] = None
) -> bool:
    """label 保留的图精确匹配：规范化后字符串全等。"""
    c1 = canonicalize(s1, label_table)
    c2 = canonicalize(s2, label_table)
    return c1 is not None and c2 is not None and c1 == c2


def canonical_bare(smiles: str, isomeric: bool = False) -> Optional[str]:
    """骨架级规范 SMILES（丢弃 label），供人读 / CSV 展示。解析失败返回 None。"""
    mol, _ = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=isomeric)
    except Exception:
        return None


def tanimoto(s1: str, s2: str, radius: int = 2) -> Optional[float]:
    """Morgan 指纹 Tanimoto（骨架级，超原子 label 不参与）。"""
    m1 = parse_smiles(s1)[0]
    m2 = parse_smiles(s2)[0]
    if m1 is None or m2 is None:
        return None
    try:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(m1, radius, nBits=2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(m2, radius, nBits=2048)
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return None
