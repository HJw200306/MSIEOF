"""生成 MolScribe 兼容格式的 Markush 数据集。

输出 CSV 格式:
    file_path, SMILES, node_coords, edges

- file_path: 相对于 out_dir 的图像路径
- SMILES: 含 R/X/Y 等 Markush 变量原子的 SMILES 字符串
- node_coords: [[x, y], ...] 像素坐标列表（已缩放到图像尺寸），
               顺序与 SMILES 中原子顺序一致（通过 atom map 对齐）
- edges: [[u, v, bond_type], ...] 三元组列表
         bond_type: 1=单键 2=双键 3=三键 4=芳香键 5=楔形键(实) 6=楔形键(虚)

使用方法:
    python gen_markush_molscribe.py \
        --csv dataset/train_200k.csv \
        --out dataset/molscribe_markush \
        --train 500 --val 20 --workers 8
"""

import argparse
import multiprocessing as mp
import os
import random
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

cv2.setNumThreads(1)

from draw.markush.mol_markush_gen import draw_markush_obb, _gen_worker  # noqa: E402
from draw.mol_obb_gen import iter_smiles  # noqa: E402


# ── 键类型映射 (与 MolScribe dataset.py 保持一致) ──────────────────────────
# RDKit BondType → MolScribe bond_type
_BOND_TYPE_MAP = {
    Chem.BondType.SINGLE:    1,
    Chem.BondType.DOUBLE:    2,
    Chem.BondType.TRIPLE:    3,
    Chem.BondType.AROMATIC:  4,
}

# RDKit BondDir (楔形键) → MolScribe stereo bond_type
_STEREO_DIR_MAP = {
    Chem.BondDir.BEGINWEDGE:  5,  # 实楔形
    Chem.BondDir.BEGINDASH:   6,  # 虚楔形
}


def mol_to_molscribe_graph(mol, atom_coords_px):
    """从 RDKit mol + 像素坐标提取 MolScribe 格式的标签。

    atom_coords_px: dict {atom_idx: (x, y)}，draw_markush_obb 返回的 atom_coords。

    返回:
        smiles: str，原子 map 已清除的 SMILES（含 R/X/Y 等 atom label）
        coords: list of [x, y]，与原子顺序对齐
        edges:  list of [u, v, bond_type]
    """
    # RDKit 规范化 SMILES 会重排原子顺序，需要通过 atom map 对齐。
    # 先给每个原子打 atom map = atom.GetIdx() + 1 (>0)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)

    smi_mapped = Chem.MolToSmiles(mol)
    # 解析 mapped SMILES，得到规范原子顺序
    mol2 = Chem.MolFromSmiles(smi_mapped)
    if mol2 is None:
        return None, None, None

    # 建立 atom_map_num → 原始 atom_idx 的映射（用于查坐标）
    map_to_orig = {}
    for atom in mol.GetAtoms():
        map_to_orig[atom.GetAtomMapNum()] = atom.GetIdx()

    # 按 mol2 (规范顺序) 重建坐标和符号
    coords = []
    for atom in mol2.GetAtoms():
        amn = atom.GetAtomMapNum()
        orig_idx = map_to_orig.get(amn)
        if orig_idx is None or orig_idx not in atom_coords_px:
            return None, None, None
        x, y = atom_coords_px[orig_idx]
        coords.append([float(x), float(y)])

    # 清除 atom map，生成输出 SMILES
    for atom in mol2.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles = Chem.MolToSmiles(mol2)

    # 边列表
    n = mol2.GetNumAtoms()
    edges = []
    for bond in mol2.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        stereo = bond.GetBondDir()
        if stereo in _STEREO_DIR_MAP:
            t = _STEREO_DIR_MAP[stereo]
            edges.append([u, v, t])
            edges.append([v, u, 11 - t])
        else:
            t = _BOND_TYPE_MAP.get(bond.GetBondType(), 1)
            edges.append([u, v, t])
            edges.append([v, u, t])

    return smiles, coords, edges


def _worker_init():
    cv2.setNumThreads(1)
    seed = (os.getpid() * 2654435761) ^ int.from_bytes(os.urandom(4), 'little')
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)


def _gen_one(smi):
    """尝试生成一条 MolScribe 样本，返回 (img, smiles, coords, edges) 或 None。"""
    try:
        result = draw_markush_obb(smi)
    except Exception:
        return None
    if result is None:
        return None

    (cv_img, labels, _ocr, _overlays, _mask,
     _atom_boxes, _bond_obbs, _removed_bonds, _per_masks,
     mol, _atom_display_ocr, atom_coords) = result

    smiles, coords, edges = mol_to_molscribe_graph(mol, atom_coords)
    if smiles is None:
        return None

    return cv_img, smiles, coords, edges


def generate_dataset(out_dir, n_train, n_val, csv_path, workers=4, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    out_dir = Path(out_dir)
    splits = {'train': n_train, 'val': n_val}

    smiles_pool = iter_smiles(csv_path, limit=max(n_train, n_val) * 10)
    total = n_train + n_val
    if len(smiles_pool) < total:
        smiles_pool = (smiles_pool * (total // len(smiles_pool) + 1))[:total]

    pool = None
    if workers > 1:
        pool = mp.Pool(workers, initializer=_worker_init)

    try:
        smi_idx = 0
        for split, n in splits.items():
            img_dir = out_dir / split / 'images'
            img_dir.mkdir(parents=True, exist_ok=True)

            rows = []
            produced = 0
            attempts = 0
            max_attempts = n * 20

            def _feed():
                nonlocal smi_idx
                while attempts < max_attempts and produced < n:
                    yield smiles_pool[smi_idx % len(smiles_pool)]
                    smi_idx += 1

            if pool is not None:
                results_iter = pool.imap_unordered(_gen_one, _feed(), chunksize=4)
            else:
                results_iter = (_gen_one(s) for s in _feed())

            pbar = tqdm(total=n, desc=f'[{split}]', unit='img')
            for result in results_iter:
                attempts += 1
                if produced >= n:
                    break
                if result is None:
                    continue
                cv_img, smiles, coords, edges = result

                name = f"{split}_{produced:06d}.png"
                img_path = img_dir / name
                cv2.imwrite(str(img_path), cv_img)

                rel_path = f"{split}/images/{name}"
                rows.append({
                    'file_path': rel_path,
                    'SMILES':    smiles,
                    'node_coords': str(coords),
                    'edges':     str(edges),
                })
                produced += 1
                pbar.update(1)
            pbar.close()

            df = pd.DataFrame(rows, columns=['file_path', 'SMILES', 'node_coords', 'edges'])
            csv_out = out_dir / f'{split}.csv'
            df.to_csv(str(csv_out), index=False)
            print(f'[{split}] done. produced={produced} attempts={attempts} -> {csv_out}')

    finally:
        if pool is not None:
            pool.terminate()
            pool.join()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv',     default=r"C:\Users\Administrator\Desktop\WorkPlace\R_Groups_Segment\dataset/train_200k.csv",  help='SMILES 数据源 CSV')
    ap.add_argument('--out',     default=r"C:\Users\Administrator\Desktop\WorkPlace\R_Groups_Segment\dataset/molscribe",  help='输出目录')
    ap.add_argument('--train',   type=int, default=50)
    ap.add_argument('--val',     type=int, default=20)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--seed',    type=int, default=42)
    args = ap.parse_args()

    generate_dataset(args.out, args.train, args.val, args.csv,
                     workers=args.workers, seed=args.seed)
    print('done.')


if __name__ == '__main__':
    main()
