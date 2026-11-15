"""启发式检测 MolScribe 把波浪线切成的假碳链。

信号: 由纯 C 组成的简单路径(2-3 个原子), 且该路径要么与主结构完全断开,
要么只在一端挂到主结构上。输出候选, 供人工看图确认。
"""
import json
import sys
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")


def components(n, bonds):
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in bonds:
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    comp = {}
    for k in range(n):
        comp.setdefault(find(k), []).append(k)
    return list(comp.values())


def analyze(i):
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    atoms = ann["atoms"]
    n = len(atoms)
    bonds = [(b["endpoint_atoms"][0], b["endpoint_atoms"][1]) for b in ann["bonds"]]
    adj = {k: [] for k in range(n)}
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)
    comps = components(n, bonds)
    comps.sort(key=len, reverse=True)
    lines = [f"=== {i}  n={n} nb={len(bonds)} ncomp={len(comps)} ==="]
    for ci, c in enumerate(comps):
        syms = {atoms[k].get("atom_symbol", "C") for k in c}
        degs = [len(adj[k]) for k in c]
        tag = "MAIN" if ci == 0 else f"frag{ci}"
        # 简单路径判定
        is_path = all(d <= 2 for d in degs) and len(c) >= 2
        cand = is_path and syms <= {"C"} and 2 <= len(c) <= 3
        mark = "  <== CANDIDATE" if cand else ""
        lines.append(f"  {tag}: atoms={sorted(c)} syms={sorted(syms)} degs={degs}{mark}")
    return "\n".join(lines)


if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) == 2 and "-" in args[1]:
        a, b = int(args[0]), int(args[1].split("-")[1] if False else args[1])
    for a in args:
        if "-" in a:
            lo, hi = a.split("-")
            for i in range(int(lo), int(hi) + 1):
                print(analyze(i))
        else:
            print(analyze(int(a)))
