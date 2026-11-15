"""修正 OCR 错读的 R 基团/元素标签文本, 让图与原图文字一致。

只改 corrected_annotations 里被错读的原子 symbol/label, 再用后端 graph_to_smiles
重新生成 esmiles。不动 review_status (保持 pending)。
用法: fix_labels.py [--write]
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles  # noqa: E402

ELEMENTS = {"H", "O", "Si", "N", "S", "P", "F", "Cl", "Br", "I"}

# idx -> {atom_index: 正确文本}
LABEL_FIX = {
    109: {1: "X23"},
    110: {1: "X23", 3: "R69"},
    116: {1: "H"},
    121: {6: "R70", 7: "R69"},
    122: {2: "R69", 8: "R70", 9: "R71"},
    123: {1: "R69", 8: "R70", 9: "R71"},
    124: {6: "R69"},
    125: {2: "R69", 8: "R70"},
    126: {1: "R69", 8: "R70"},
    127: {6: "R70", 7: "R71"},
    128: {6: "R70"},
    146: {1: "OR141", 2: "O", 3: "R142"},
    147: {0: "OR143", 2: "R142", 5: "O", 6: "R141"},
    148: {0: "R142", 1: "Si", 2: "R141", 3: "R143"},
    157: {4: "OMe"},
    191: {13: "H"},
    192: {9: "H"},
}


def main():
    write = "--write" in sys.argv
    for idx in sorted(LABEL_FIX):
        p = D / "corrected_annotations" / f"{idx:06d}.json"
        a = json.load(open(p, encoding="utf-8"))
        before = a.get("esmiles")
        for k, txt in LABEL_FIX[idx].items():
            at = a["atoms"][k]
            old = at.get("symbol")
            if txt in ELEMENTS:
                at["symbol"] = txt
                at["label"] = None
            else:
                at["symbol"] = "*"
                at["label"] = txt
            print(f"  [{idx}] atom {k}: {old!r} -> {txt!r}")
        a["esmiles"] = graph_to_smiles(a["atoms"], a["bonds"], a.get("brackets"))
        print(f"[{idx}] esmiles:\n   OLD {before}\n   NEW {a['esmiles']}")
        if write:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(a, f, ensure_ascii=False, indent=2)
    print("WRITTEN" if write else "DRY-RUN")


if __name__ == "__main__":
    main()
