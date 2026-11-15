"""Nearest-neighbour diff raw -> user corrected for 0-52, to learn edit conventions."""
import json
from collections import Counter
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")

sym_changes = Counter()
count_delta = Counter()
sample_notes = []


def atoms_of(p):
    a = json.load(open(p, encoding="utf-8"))
    return a.get("atoms", []), a.get("bonds", [])


for i in range(0, 53):
    rp = D / "annotations" / f"{i:06d}.json"
    cp = D / "corrected_annotations" / f"{i:06d}.json"
    if not cp.exists():
        continue
    ra, rb = atoms_of(rp)
    ca, cb = atoms_of(cp)
    count_delta[(len(ca) - len(ra), len(cb) - len(rb))] += 1
    used = set()
    for c in ca:
        best, bd = None, 9
        for k, r in enumerate(ra):
            if k in used:
                continue
            d = (r["x"] - c["x"]) ** 2 + (r["y"] - c["y"]) ** 2
            if d < bd:
                bd, best = d, k
        if best is None or bd > 0.02:
            sym_changes[("<NEW>", c.get("symbol"), c.get("label"))] += 1
            continue
        used.add(best)
        rs = ra[best].get("atom_symbol", "C")
        cs = c.get("symbol")
        cl = c.get("label")
        if rs != cs or cl:
            sym_changes[(rs, cs, cl)] += 1

print("=== atom-count delta (corrected - raw): bonds_delta too ===")
for k, v in sorted(count_delta.items()):
    print(f"  atoms{k[0]:+d} bonds{k[1]:+d} : {v} files")

print("\n=== symbol/label changes (raw_sym -> new_sym/label) ===")
for k, v in sym_changes.most_common():
    print(f"  {k} : {v}")
