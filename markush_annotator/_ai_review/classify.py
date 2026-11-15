"""给 109-200 每条原始标注分类: 波浪线在 MolScribe 输出里长什么样。

A = 一段孤立的纯C简单路径(2-3原子)  -> 整段就是波浪线
B = 主链末端一个悬空 C(度1)         -> 那个末端C疑似波浪线(需看图确认)
"""
import json
import sys
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")


def load(i):
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    atoms = ann["atoms"]
    n = len(atoms)
    bonds = [(b["endpoint_atoms"][0], b["endpoint_atoms"][1]) for b in ann["bonds"]]
    adj = {k: [] for k in range(n)}
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)
    return atoms, adj, n


def comps(n, adj):
    seen = [False] * n
    out = []
    for s in range(n):
        if seen[s]:
            continue
        st = [s]
        seen[s] = True
        c = []
        while st:
            x = st.pop()
            c.append(x)
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    st.append(y)
        out.append(sorted(c))
    out.sort(key=len, reverse=True)
    return out


def classify(i):
    atoms, adj, n = load(i)
    cs = comps(n, adj)
    main = cs[0]
    syms = [atoms[k].get("atom_symbol", "C") for k in range(n)]
    A = []
    for c in cs[1:]:
        d = [len(adj[k]) for k in c]
        if 2 <= len(c) <= 3 and all(syms[k] == "C" for k in c) and all(x <= 2 for x in d):
            A.append(c)
    B = []
    for k in main:
        if syms[k] == "C" and len(adj[k]) == 1:
            nb = adj[k][0]
            if syms[nb] != "C":
                B.append(k)
    cat = "A" if A else ("B" if B else "N")
    print(f"{i:3d} {cat}  A={A} B={B}  syms={syms}")


if __name__ == "__main__":
    for a in sys.argv[1:]:
        if "-" in a:
            lo, hi = a.split("-")
            for i in range(int(lo), int(hi) + 1):
                classify(i)
        else:
            classify(int(a))
