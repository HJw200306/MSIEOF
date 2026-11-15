"""每张图: raw 里 孤立C-C对 / bond_type=='wave' / 其它非单键计数; 图内 wave_bonds."""
import json
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None
for i in range(109, 201):
    raw = load(D/"annotations"/f"{i:06d}.json")
    cur = load(D/"corrected_annotations"/f"{i:06d}.json")
    at = raw["atoms"]; deg = {k:0 for k in range(len(at))}
    for b in raw["bonds"]:
        u,v = b["endpoint_atoms"]; deg[u]+=1; deg[v]+=1
    iso = wave = 0
    for b in raw["bonds"]:
        u,v = b["endpoint_atoms"]
        if b.get("bond_type")=="wave": wave += 1
        elif at[u]["atom_symbol"]=="C" and at[v]["atom_symbol"]=="C" and deg[u]==1 and deg[v]==1:
            iso += 1
    cw = sum(1 for b in cur["bonds"] if b[2]==7)
    src = iso + wave
    mark = "" if cw == src else "  <== MISMATCH"
    print(f"{i:3d}  raw_iso={iso}  raw_wave={wave}  src={src}  graph_C~C={cw}  na={len(cur['atoms'])}{mark}")
