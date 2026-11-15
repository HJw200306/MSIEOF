"""化学结构辅助工具 (从 R_Groups_Segment/draw/mol_draw.py 提取)。"""
from rdkit import Chem
from rdkit.Chem import BRICS


def break_smile(smiles: str):
    """用 BRICS 规则把 SMILES 拆成片段, 返回长度在 (2, 15) 之间的子结构 SMILES 列表。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    processed_bonds = set()
    out = []
    bonds_info = BRICS.FindBRICSBonds(mol)
    for bond_info in bonds_info:
        a1, a2 = bond_info[0]
        sorted_pair = tuple(sorted((a1, a2)))
        if sorted_pair in processed_bonds:
            continue
        processed_bonds.add(sorted_pair)
        bond = mol.GetBondBetweenAtoms(a1, a2)
        if bond is None:
            continue
        bond_idx = bond.GetIdx()
        broken_mol = Chem.FragmentOnBonds(mol, [bond_idx], dummyLabels=[(0, 0)])
        frags = Chem.GetMolFrags(broken_mol, asMols=True)
        for frag in frags:
            smi = Chem.MolToSmiles(frag)
            if 2 < len(smi) < 15:
                out.append(smi)
    return out
