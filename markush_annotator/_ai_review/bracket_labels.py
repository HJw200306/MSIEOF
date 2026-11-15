"""把 corrected_annotations 里超原子(*)的 label 统一成 [内容] 形式."""
import json, sys
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles

def main():
    write = "--write" in sys.argv
    for i in range(110, 201):
        p = D / "corrected_annotations" / f"{i:06d}.json"
        if not p.exists():
            continue
        a = json.load(open(p, encoding="utf-8"))
        ch = False
        for x in a["atoms"]:
            lab = x.get("label")
            if x.get("symbol") == "*" and lab and not (lab.startswith("[") and lab.endswith("]")):
                x["label"] = f"[{lab}]"
                ch = True
        if not ch:
            continue
        a["esmiles"] = graph_to_smiles(a["atoms"], a["bonds"], a.get("brackets"))
        print(f"[{i}] {a['esmiles']}")
        if write:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(a, f, ensure_ascii=False, indent=2)
    print("WRITTEN" if write else "DRY-RUN")

if __name__ == "__main__":
    main()
