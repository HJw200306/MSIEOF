"""按用户约定重建 corrected_annotations (109-200)。

约定 (从用户自己的 92/104-108 反推):
  * 波浪线 = 一对游离 C, 二者之间 order-7 键 (即 esmiles 里的 `.C~C`), 不与分子相连。
  * 元素符号照原样; R 基团文字 -> symbol='*', label=文字。
  * 坐标 = raw 归一化 * 图像尺寸。

算法:
  1) 以 annotations/ 的 raw 拓扑为基础 (它保留了真实原子 + 一条单独的 C-C 对代表波浪线)。
  2) 孤立 C-C 对 -> order 7。
  3) 非元素 token -> symbol/label, 尽量沿用 corrected_annotations 里已修正的 label(按位置配对)。
  4) 若 raw 没有孤立 C-C 对, 但 corrected 里有波浪线占位 *, 则在占位处补一对游离 C~C。
review_status 不动。
用法: rebuild_sq.py [--write] [idx ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
D = ROOT / "data" / "seg_page2smiles"
sys.path.insert(0, str(ROOT / "backend"))
from data_manager import graph_to_smiles  # noqa: E402
from PIL import Image  # noqa: E402

ELEMENTS = {"H", "O", "Si", "N", "S", "P", "F", "Cl", "Br", "I", "C", "B", "Se"}
ORDER = {"single": 1, "double": 2, "triple": 3, "aromatic": 4,
         "wedge": 5, "hash": 6, "wave": 7}
SQ_LABELS = {None, "_AP"}


def token_label(tok):
    t = tok.strip()
    return t[1:-1] if t.startswith("[") and t.endswith("]") else t


def load(p):
    return json.load(open(p, encoding="utf-8")) if p.exists() else None


def rebuild(idx):
    raw = load(D / "annotations" / f"{idx:06d}.json")
    cur = load(D / "corrected_annotations" / f"{idx:06d}.json")
    W, H = Image.open(D / "images" / f"{idx:06d}.png").size

    cpts = [(a.get("x", -9e9), a.get("y", -9e9), a.get("symbol"), a.get("label"))
            for a in cur["atoms"]] if cur else []

    def nearest(x, y, tol=9.0):
        best, bd = None, 1e9
        for cx, cy, sym, lab in cpts:
            d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
            if d < bd:
                bd, best = d, (sym, lab)
        return best if bd <= tol else None

    at = raw["atoms"]
    deg = {k: 0 for k in range(len(at))}
    for b in raw["bonds"]:
        u, v = b["endpoint_atoms"]
        deg[u] += 1
        deg[v] += 1
    iso = set()
    for b in raw["bonds"]:
        u, v = b["endpoint_atoms"]
        if (at[u]["atom_symbol"] == "C" and at[v]["atom_symbol"] == "C"
                and deg[u] == 1 and deg[v] == 1):
            iso.add(frozenset((u, v)))

    atoms, bonds = [], []
    for a in at:
        x = round(a["x"] * W, 1)
        y = round(a["y"] * H, 1)
        sym = a["atom_symbol"]
        if sym in ELEMENTS:
            atoms.append({"symbol": sym, "label": None, "x": x, "y": y})
        else:
            hit = nearest(x, y)
            if hit and hit[0] in ELEMENTS and hit[1] is None:
                atoms.append({"symbol": hit[0], "label": None, "x": x, "y": y})
            else:
                lab = (hit[1] if hit and hit[1] is not None else token_label(sym))
                atoms.append({"symbol": "*", "label": lab, "x": x, "y": y})
    for b in raw["bonds"]:
        u, v = b["endpoint_atoms"]
        o = ORDER.get(b.get("bond_type", "single"), 1)
        if frozenset((u, v)) in iso:
            o = 7
        bonds.append([u, v, o])

    # 补: raw 没有波浪线对, 但 corrected 标了波浪线占位 * 的位置
    if not iso and cur is not None:
        for k, a in enumerate(cur["atoms"]):
            if a.get("symbol") != "*" or a.get("label") not in SQ_LABELS:
                continue
            wv = [b for b in cur["bonds"] if b[2] == 7 and k in (b[0], b[1])]
            if not wv:
                continue
            b = wv[0]
            nb = b[1] if b[0] == k else b[0]
            qx, qy = cur["atoms"][nb]["x"], cur["atoms"][nb]["y"]
            px, py = a["x"], a["y"]
            ia = len(atoms)
            for t in (0.88, 1.16):
                atoms.append({"symbol": "C", "label": None,
                              "x": round(qx + (px - qx) * t, 1),
                              "y": round(qy + (py - qy) * t, 1)})
            bonds.append([ia, ia + 1, 7])

    return atoms, bonds


def main():
    write = "--write" in sys.argv
    args = [int(x) for x in sys.argv[1:] if not x.startswith("--")]
    idxs = args or list(range(109, 201))
    n = 0
    for i in idxs:
        atoms, bonds = rebuild(i)
        p = D / "corrected_annotations" / f"{i:06d}.json"
        cur = load(p)
        esm = graph_to_smiles(atoms, bonds, None)
        old = cur.get("esmiles") if cur else None
        if old != esm:
            n += 1
            print(f"[{i}] {old}\n  -> {esm}")
        if write:
            out = {k: v for k, v in (cur or {}).items()
                   if k not in ("atoms", "bonds", "esmiles")}
            out["atoms"] = atoms
            out["bonds"] = bonds
            out["esmiles"] = esm
            with open(p, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"changed {n}  " + ("WRITTEN" if write else "DRY-RUN"))


if __name__ == "__main__":
    main()
