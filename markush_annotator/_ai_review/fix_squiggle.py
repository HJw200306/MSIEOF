"""把 corrected_annotations 里「波浪线的 * 占位原子」换成用户约定的 C~C 游离片段。

识别: symbol=='*' 且 label∈{None,'_AP'} 且至少有一条 order-7 键 -> 该原子是波浪线占位。
处理: 删掉该 *, 在原位置沿 邻居->* 方向放两个 C, 二者之间一条 order-7 键。
      片段与分子「不相连」(用户约定: 波浪线是游离的 C~C)。其余原子/键/标签不动。
review_status 不动。用法: fix_squiggle.py [--write] [idx ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles  # noqa: E402

SQ_LABELS = {None, "_AP"}


def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def fix(idx):
    p = D / "corrected_annotations" / f"{idx:06d}.json"
    a = load(p)
    if a is None:
        return None
    atoms = a["atoms"]
    bonds = [list(b) for b in a["bonds"]]

    sq = [k for k, at in enumerate(atoms)
          if at.get("symbol") == "*" and at.get("label") in SQ_LABELS
          and any(b[2] == 7 and k in (b[0], b[1]) for b in bonds)]

    newatoms = [dict(x) for x in atoms]
    got = {}   # squiggle atom -> (idx_a, idx_b) in newatoms
    for k in sq:
        nb = None
        for b in bonds:
            if b[2] == 7 and k in (b[0], b[1]):
                nb = b[1] if b[0] == k else b[0]
                break
        x0, y0 = atoms[nb]["x"], atoms[nb]["y"]
        x1, y1 = atoms[k]["x"], atoms[k]["y"]
        for t in (0.72, 1.08):
            newatoms.append({"symbol": "C", "label": None,
                             "x": round(x0 + (x1 - x0) * t, 1),
                             "y": round(y0 + (y1 - y0) * t, 1)})
        got[k] = (len(newatoms) - 2, len(newatoms) - 1)

    drop = set(sq)
    keep = [i for i in range(len(newatoms)) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    out_atoms = [newatoms[i] for i in keep]

    out_bonds = []
    for b in bonds:
        u, v, o = b[0], b[1], b[2]
        if u in drop or v in drop:
            continue                      # 波浪线键丢弃: 片段不与分子相连
        out_bonds.append([remap[u], remap[v], o])
    for k in sq:
        ua, ub = got[k]
        out_bonds.append([remap[ua], remap[ub], 7])

    out = {k: v for k, v in a.items() if k not in ("atoms", "bonds", "esmiles")}
    out["atoms"] = out_atoms
    out["bonds"] = out_bonds
    out["esmiles"] = graph_to_smiles(out_atoms, out_bonds, a.get("brackets"))
    return p, a.get("esmiles"), out


def main():
    write = "--write" in sys.argv
    args = [int(x) for x in sys.argv[1:] if not x.startswith("--")]
    idxs = args or list(range(109, 201))
    n = 0
    for i in idxs:
        r = fix(i)
        if r is None:
            continue
        p, old, out = r
        if old != out["esmiles"]:
            n += 1
            print(f"[{i}] {old}\n  -> {out['esmiles']}")
        if write:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"changed {n}  " + ("WRITTEN" if write else "DRY-RUN"))


if __name__ == "__main__":
    main()
