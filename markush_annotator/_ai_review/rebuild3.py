"""对 140/147/152 (raw 有重复原子) 以备份版为基, 把波浪线占位 * -> 真 C(接在原位置)+游离 C~C.
对 137/143 补一条游离 C~C. 其余不动."""
import json, sys
from pathlib import Path
ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
BAK = D / "_bak_corrected_109_200"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles
SQ_LABELS = {None, "_AP"}

def fix_from_bak(i):
    a = json.load(open(BAK / f"{i:06d}.json", encoding="utf-8"))
    atoms = [dict(x) for x in a["atoms"]]
    bonds = [list(b) for b in a["bonds"]]
    stars = [k for k, at in enumerate(atoms)
             if at.get("symbol") == "*" and at.get("label") in SQ_LABELS
             and any(b[2] == 7 and k in (b[0], b[1]) for b in bonds)]
    new_atoms = []
    extra_bonds = []          # (a,b) pairs into new_atoms
    drop = set(stars)
    remap = {}
    for k, at in enumerate(atoms):
        if k in drop:
            continue
        remap[k] = len(new_atoms)
        new_atoms.append(at)
    out_bonds = []
    for b in bonds:
        u, v, o = b
        if o == 7 and (u in drop or v in drop):
            continue
        if u in drop or v in drop:
            continue
        out_bonds.append([remap[u], remap[v], o])
    for k in stars:
        nb = None
        for b in bonds:
            if b[2] == 7 and k in (b[0], b[1]):
                nb = b[1] if b[0] == k else b[0]
                break
        if nb is None:
            continue
        sx, sy = atoms[k]["x"], atoms[k]["y"]
        nx, ny = atoms[nb]["x"], atoms[nb]["y"]
        # 真碳: 放在 * 原位置, 接到邻居
        ci = len(new_atoms)
        new_atoms.append({"symbol": "C", "label": None, "x": sx, "y": sy})
        out_bonds.append([remap[nb], ci, 1])
        # 游离 C~C: 在 邻居->* 之间
        ia = len(new_atoms)
        for t in (0.45, 0.75):
            new_atoms.append({"symbol": "C", "label": None,
                              "x": round(nx + (sx - nx) * t, 1),
                              "y": round(ny + (sy - ny) * t, 1)})
        out_bonds.append([ia, ia + 1, 7])
    return a, new_atoms, out_bonds

def add_sq(i, pts):
    p = D / "corrected_annotations" / f"{i:06d}.json"
    a = json.load(open(p, encoding="utf-8"))
    atoms = a["atoms"]; bonds = [list(b) for b in a["bonds"]]
    ia = len(atoms)
    for x, y in pts:
        atoms.append({"symbol": "C", "label": None, "x": x, "y": y})
    bonds.append([ia, ia + 1, 7])
    return a, atoms, bonds

def main():
    write = "--write" in sys.argv
    jobs = []
    for i in (140, 147, 152):
        a, at, bo = fix_from_bak(i)
        jobs.append((i, a, at, bo))
    for i, pts in {137: [(108.6, 21.6), (117.5, 21.6)],
                   143: [(126.5, 84.3), (132.1, 93.2)]}.items():
        a, at, bo = add_sq(i, pts)
        jobs.append((i, a, at, bo))
    for i, a, at, bo in jobs:
        esm = graph_to_smiles(at, bo, a.get("brackets"))
        print(f"[{i}] {a.get('esmiles')}  ->  {esm}")
        if write:
            out = {k: v for k, v in a.items() if k not in ("atoms", "bonds", "esmiles")}
            out["atoms"] = at; out["bonds"] = bo; out["esmiles"] = esm
            with open(D / "corrected_annotations" / f"{i:06d}.json", "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    print("WRITTEN" if write else "DRY-RUN")

if __name__ == "__main__":
    main()
