"""sq_audit.py -> 每张图: raw 孤立 C-C 对数 / 波浪线占位*数 / 重建得到 C~C 对数."""
import json, sys
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"

def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None

for i in range(109, 201):
    raw = load(D/"annotations"/f"{i:06d}.json")
    cur = load(D/"corrected_annotations"/f"{i:06d}.json")
    # raw isolated C-C pairs
    at = raw["atoms"]; deg = {k:0 for k in range(len(at))}
    for b in raw["bonds"]:
        u,v = b["endpoint_atoms"]; deg[u]+=1; deg[v]+=1
    iso = 0
    for b in raw["bonds"]:
        u,v = b["endpoint_atoms"]
        if at[u]["atom_symbol"]=="C" and at[v]["atom_symbol"]=="C" and deg[u]==1 and deg[v]==1:
            iso += 1
    # squiggle * in corrected
    nstar = sum(1 for a in (cur["atoms"] if cur else [])
                if a.get("symbol")=="*" and a.get("label") in (None,"_AP"))
    # C~C in current corrected
    cc = sum(1 for b in (cur["bonds"] if cur else []) if b[2]==7)
    flag = "" if iso>0 else ("FALLBACK" if nstar else "")
    print(f"{i:3d}  raw_iso={iso}  star*={nstar}  wave_bonds={cc}  na={len(cur['atoms']) if cur else 0}  {flag}")
