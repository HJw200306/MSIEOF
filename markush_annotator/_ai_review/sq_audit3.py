"""对比 备份(原AI版) 与 重建后: 波浪线占位*数 / C~C数 / 原子数."""
import json
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
BAK = D / "_bak_corrected_109_200"
def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None
print("idx  bak_na bak_star bak_wave | new_na new_C~C")
for i in range(109, 201):
    b = load(BAK/f"{i:06d}.json")
    n = load(D/"corrected_annotations"/f"{i:06d}.json")
    bs = sum(1 for a in (b["atoms"] if b else []) if a.get("symbol")=="*" and a.get("label") in (None,"_AP"))
    bw = sum(1 for x in (b["bonds"] if b else []) if x[2]==7)
    nw = sum(1 for x in (n["bonds"] if n else []) if x[2]==7)
    mark = "  <== star != C~C" if bs != nw else ""
    print(f"{i:3d}  {len(b['atoms']) if b else 0:4d}  {bs:3d}  {bw:3d} | {len(n['atoms']) if n else 0:4d}  {nw:3d}{mark}")
