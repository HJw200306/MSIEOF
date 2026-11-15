"""从 PageToSMILES_approved 标注构建评测 JSONL + 全局 label_table。

用途
----
为统一消融评测准备单一数据集：把已人工审核通过（approved）的 LabelMe 标注
导出为一行一页的 JSONL，并同步生成 label 保留规范化所需的全局 ``label_table``。

输入
----
- ``json/PageToSMILES_approved/*.json``：49 页、479 个已审核分子结构
  （每个 shape 含 ``points``/``atoms``/``bonds``/``smiles``，SMILES 可能带
  CXSMILES ``|$...$|`` 超原子 label 段）。

输出
----
- ``PageToSMILES_approved.jsonl``：每行一个页面，见下方 schema。
- ``label_table.json``：``{label: isotope}``，由全量 GT label 排序生成，
  供 :mod:`utils.base.smiles_canon` 做无碰撞的 label→同位素映射。

JSONL schema（每行）
--------------------
.. code-block:: json

   {
     "page_id": "CN104230900B_006",          // 源文件名去扩展名
     "page_file": "CN104230900B_006.png",    // imagePath
     "molecules": [
       {
         "molecule_id": "CN104230900B_006_0", // 页内第 0 个 mol shape
         "bbox_xyxy": [x1, y1, x2, y2],       // polygon 紧包围框，整页坐标
         "smiles": "C*C1CC1.C~C |$;[CH2];;;;;$|",
         "atoms": [...],                      // 原样保留
         "bonds": [...]                       // 原样保留
       }
     ]
   }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# 添加项目路径
from utils.base.paths import PROJECT_ROOT

from utils.base.smiles_canon import build_label_table
from utils.base.chemical_graph import convert_graph
from utils.evaluation.evaluate_page2smiles import polygon_to_bbox

# 路径配置
SEG_ROOT = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal")
SOURCE_JSON_DIR = SEG_ROOT / "json" / "PageToSMILES_approved"
OUTPUT_JSONL = SEG_ROOT / "PageToSMILES_approved.jsonl"
OUTPUT_LABEL_TABLE = SEG_ROOT / "label_table.json"


def collect_pages(source_dir: Path) -> List[dict]:
    """读取所有 approved JSON，返回按文件名排序的页面记录（含原始 shape）。"""
    pages = []
    for json_path in sorted(source_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        page_id = json_path.stem
        page_file = data.get("imagePath", "")
        mols = []
        for shape in data.get("shapes", []):
            if shape.get("label") != "mol":
                continue
            points = np.asarray(shape.get("points", []), dtype=float)
            if points.size == 0:
                bbox = [0.0, 0.0, 0.0, 0.0]
            else:
                bbox = polygon_to_bbox(points)
            chemical = convert_graph(shape.get("atoms", []), shape.get("bonds", []), shape.get("brackets", []))
            mols.append({
                "molecule_id": f"{page_id}_{len(mols)}",
                "bbox_xyxy": bbox,
                "smiles": chemical["smiles"],
                "smiles_original": shape.get("smiles", ""),
                "chemistry": chemical,
                "atoms": chemical["atoms"],
                "bonds": shape.get("bonds", []),
            })

        pages.append({
            "page_id": page_id,
            "page_file": page_file,
            "molecules": mols,
        })
    return pages


def main() -> None:
    pages = collect_pages(SOURCE_JSON_DIR)
    all_smiles = [m["smiles"] for p in pages for m in p["molecules"] if m["smiles"]]

    n_mols = sum(len(p["molecules"]) for p in pages)
    n_labeled = sum(1 for s in all_smiles if "|$" in s)
    n_wavy = sum(1 for s in all_smiles if "~" in s)

    # 全局 label_table（排序生成，确定性、无碰撞）
    label_table = build_label_table(all_smiles)

    # 写 JSONL
    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for page in pages:
            f.write(json.dumps(page, ensure_ascii=False) + "\n")

    # 写 label_table
    with open(OUTPUT_LABEL_TABLE, "w", encoding="utf-8") as f:
        json.dump(label_table, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("PageToSMILES_approved JSONL 构建完成")
    print("=" * 60)
    print(f"  页面数:      {len(pages)}")
    print(f"  分子数:      {n_mols}")
    print(f"  含 label 段: {n_labeled} 个分子 (|$...$|)")
    print(f"  含波浪键:    {n_wavy} 个分子 (~)")
    print(f"  去重 label:  {len(label_table)} 个")
    print(f"  输出 JSONL:  {OUTPUT_JSONL}")
    print(f"  label_table: {OUTPUT_LABEL_TABLE}")


if __name__ == "__main__":
    main()
