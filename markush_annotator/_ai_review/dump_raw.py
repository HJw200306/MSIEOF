"""Compact dump of raw annotation: atom index, symbol, norm coords, adjacency."""
import json
import sys
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")


def dump(i):
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    atoms = ann["atoms"]
    bonds = ann["bonds"]
    print(f"=== idx {i}  n_atoms={len(atoms)} n_bonds={len(bonds)} ===")
    adj = {k: [] for k in range(len(atoms))}
    for b in bonds:
        u, v = b["endpoint_atoms"]
        bt = b.get("bond_type", "single")
        adj[u].append((v, bt))
        adj[v].append((u, bt))
    for k, a in enumerate(atoms):
        sym = a.get("atom_symbol", "C")
        x, y = a["x"], a["y"]
        nb = ",".join(f"{v}:{t[:2]}" for v, t in sorted(adj[k]))
        print(f"  [{k:2d}] {sym:3s} x={x:.3f} y={y:.3f} conf={a.get('confidence',1):.2f} -> {nb}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        dump(int(a))
