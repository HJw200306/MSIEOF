"""给个别图补一条游离 C~C 片段(波浪线). 用法: add_sq.py"""
import json, sys
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles

# idx -> 新 C~C 的两个碳坐标(像素)
ADD = {
    137: [(108.6, 21.6), (117.5, 21.6)],
    140: [(28.0, 38.0), (24.0, 54.0)],
    143: [(126.5, 84.3), (132.1, 93.2)],
}

def main():
    write = "--write" in sys.argv
    for i, pts in ADD.items():
        p = D / "corrected_annotations" / f"{i:06d}.json"
        a = json.load(open(p, encoding="utf-8"))
        atoms = a["atoms"]; bonds = [list(b) for b in a["bonds"]]
        ia = len(atoms)
        for x, y in pts:
            atoms.append({"symbol": "C", "label": None, "x": x, "y": y})
        bonds.append([ia, ia + 1, 7])
        a["atoms"] = atoms; a["bonds"] = bonds
        a["esmiles"] = graph_to_smiles(atoms, bonds, a.get("brackets"))
        print(f"[{i}] +C~C -> {a['esmiles']}")
        if write:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(a, f, ensure_ascii=False, indent=2)
    print("WRITTEN" if write else "DRY-RUN")

if __name__ == "__main__":
    main()
