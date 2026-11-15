"""自动为 109-200 生成「波浪线」spec (支持片段两端各一条波浪线)。

经人工验证的规则:
  MolScribe 把一根波浪线画成 1-3 个"假碳", 贴片段开口端(最左/最右)。
  证据分两档:
    A) 存在孤立的小纯C连通块(2-3原子)  -> 波浪线 = 该块 + 紧挨主结构的那个末端假碳
    B) 无孤立块 -> 波浪线 = 开口端最小的纯C支链(切一刀就分离)
  例外(通式/聚合物/损坏)人工排除。
输出 {idx: {"waves":[{"fake":[...],"attach":k}], "note":..., "skip":bool}}
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
    return atoms, bonds, n


def comps_of(n, bonds):
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


def detect(i):
    atoms, bonds, n = load(i)
    if n < 2:
        return None
    S = [a.get("atom_symbol", "C") for a in atoms]
    deg = [0] * n
    nbr = [[] for _ in range(n)]
    for u, v in bonds:
        deg[u] += 1
        deg[v] += 1
        nbr[u].append(v)
        nbr[v].append(u)

    comps = comps_of(n, bonds)
    # "主结构" = 非碳原子最多(其次原子数最多)的连通块
    largest = max(comps, key=lambda c: (sum(1 for k in c if S[k] != "C"), len(c)))
    iso = [c for c in comps if c is not largest and 2 <= len(c) <= 3 and all(S[k] == "C" for k in c)]

    def dist(a, b):
        return (atoms[a]["x"] - atoms[b]["x"]) ** 2 + (atoms[a]["y"] - atoms[b]["y"]) ** 2

    waves = []
    used = set()
    if iso:
        for c in iso:
            # 主结构里离该孤立块最近的末端碳 = 同一根波浪线上的假碳
            cands = [k for k in largest if S[k] == "C" and deg[k] == 1]
            pend = min(cands, key=lambda k: min(dist(k, f) for f in c)) if cands else None
            fake = set(c)
            if pend is not None and (pend not in used):
                fake.add(pend)
                attach = nbr[pend][0]
            else:
                attach = min((k for k in range(n) if k not in fake), key=lambda k: min(dist(k, f) for f in fake))
            used |= fake
            waves.append({"fake": sorted(fake), "attach": attach})
    else:
        # 无孤立块: 只在最左端找最小的纯C支链 (右端的末端碳可能是真CH3, 不自动删)
        for extreme in (min(range(n), key=lambda k: atoms[k]["x"]),):
            if extreme in used or S[extreme] != "C":
                continue
            best = None
            for cut in bonds:
                sides = comps_of(n, [b for b in bonds if b != cut])
                for s in sides:
                    if extreme not in s or not (1 <= len(s) <= 3):
                        continue
                    if not all(S[k] == "C" for k in s):
                        continue
                    if len([k for k in range(n) if k not in s]) < 2:
                        continue
                    if best is None or len(s) < len(best[0]):
                        other = [k for k in range(n) if k not in s]
                        att = nbr[min(s, key=lambda k: min(dist(k, o) for o in other))] if s else []
                        att = [a for a in att if a not in s]
                        if att:
                            best = (s, att[0])
            if best:
                fake = set(best[0])
                used |= fake
                waves.append({"fake": sorted(fake), "attach": best[1]})

    if not waves:
        return None
    for w in waves:
        w["fake"] = [k for k in w["fake"] if k not in ()]  # noop keep
    return {"waves": waves}


if __name__ == "__main__":
    lo, hi = 109, 200
    if len(sys.argv) >= 3:
        lo, hi = int(sys.argv[1]), int(sys.argv[2])
    out, miss = {}, []
    for i in range(lo, hi + 1):
        d = detect(i)
        if d is None:
            miss.append(i)
        else:
            out[str(i)] = d
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print("NO-CANDIDATE:", miss, file=sys.stderr)
