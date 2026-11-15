"""取代基异变 (Substituent Variation)。

与 mol_obb_gen 原有的 "只对 SMILES 中的 `*` 原子改写 _displayLabel" 不同, 这里允许
*任意* 非氢原子按概率被一个随机 R/X/Y 符号覆盖。RDKit 的 _displayLabel 机制只换原子
渲染出来的文本, 不会改变骨架价键, 所以等价于 Markush 图里的 "这个位置是变量"。

实际选择落子原子时遵循两个原则避免画面混乱:
    - 优先选 degree==1 的端点原子 (替换它最不会破坏环结构的可读性);
    - 不替换已经被 ap_variation 选中绘制连接点的原子 (`reserved_atoms`)。
    - 不替换芳香原子 (强行换成 R 会让芳香环看起来缺一个 ring atom)。

保留两个公开接口给上层:
    pick_substitution_atoms — 决策"哪些原子要被换"
    apply_substitutions     — 在 mol 上执行 SetProp + 返回 OCR 文本映射

这样未来的"重复变异 / 位置变异"也可以读 `pick_substitution_atoms` 的返回, 决定它们
该规避哪些原子。
"""

import random

from draw.mol_obb_gen import (
    make_r_symbol,
    make_conservative_dummy_symbol,
)
from draw.mol_draw import format_html_subscripts_superscripts


def pick_substitution_atoms(mol, *, prob=0.40, reserved=None,
                            max_total=None, only_terminal=False):
    """决定哪些原子要被替换为 R 基团符号。
    返回 list[int] (atom_idx).

    Args:
        mol: RDKit Mol (已 Compute2DCoords, 但未 DrawMolecule).
        prob: 单个候选原子被选中的独立概率。
        reserved: set[int], 必须跳过的原子 (例如 AP 已经占用的端点 C)。
        max_total: 上限; None = 不限。避免把整张图盖满 R。
        only_terminal: True 时只考虑 degree<=1 的端点。

    选中原则:
        - atomic_num != 0 (不是 RDKit dummy *)
        - 元素 != H
        - 不在芳香环里
        - 不在 reserved
    """
    reserved = reserved or set()
    cands = []
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if idx in reserved:
            continue
        if atom.GetAtomicNum() == 0:
            continue
        if atom.GetSymbol() == 'H':
            continue
        if atom.GetIsAromatic():
            continue
        if only_terminal and atom.GetDegree() > 1:
            continue
        cands.append(idx)

    random.shuffle(cands)
    picked = [i for i in cands if random.random() < prob]
    if max_total is not None and len(picked) > max_total:
        picked = picked[:max_total]
    return picked


def apply_substitutions(mol, atom_indices, *, show_r_sym_prob=0.6):
    """在 atom_indices 上 SetProp("_displayLabel", ...)。

    Args:
        mol: RDKit Mol
        atom_indices: 来自 pick_substitution_atoms 的列表
        show_r_sym_prob: 该原子被画成"R/X/Y 类符号"的概率;
                        剩余概率会用 make_conservative_dummy_symbol() 画成普通超原子
                        (CH3, OMe...) 以增加分布多样性。

    Returns:
        dict[int, str]  atom_idx -> 用 OCR plain text (无 HTML) 标注的真值。
                        用于和 mol_obb_gen 已有的 OCR 通路对齐。
    """
    ocr_truth = {}
    for idx in atom_indices:
        atom = mol.GetAtomWithIdx(idx)
        if random.random() < show_r_sym_prob:
            sym = make_r_symbol()
        else:
            sym = make_conservative_dummy_symbol()
        # make_r_symbol / dummy 都已经可能带 <sub><sup>; 没有标签时再走 format
        label = sym if '<' in sym else format_html_subscripts_superscripts(
            sym, random_sup_prob=0.12)
        atom.SetProp('_displayLabel', label)
        # OCR plain
        from draw.mol_obb_gen import html_to_ocr_plain  # 局部 import 避免循环
        ocr_truth[idx] = html_to_ocr_plain(label)
    return ocr_truth
