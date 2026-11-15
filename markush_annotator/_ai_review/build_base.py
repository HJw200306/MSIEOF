"""raw -> corrected 基座 (202-400)。只做确定性的规范转换, 不做波浪线判定。

规则:
  元素原子     : symbol=元素符号, label=None  (含立体碳 [C@H]/[C@@H]/[C@]/[C@@] -> C;
                 芳香小写 c/o/n/s -> 大写; 显式 [H]/[N]/[O] -> H/N/O; 同位素 [18O]/[2H]/[3H])
  超原子/R基团 : symbol='*', label='[内容]'      (raw token 本就带括号, 原样保留)
  裸 *         : symbol='*', label='_AP'
坐标: raw 归一化 [0,1] -> 像素 = 归一化 * PNG 尺寸
键  : raw dict(endpoint_atoms,bond_type) -> [u, v, order]
"""
import json
import re
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
sys.path.insert(0, str(ROOT))
from backend.data_manager import graph_to_smiles  # noqa: E402

D = ROOT / "data" / "seg_page2smiles"

BOND_MAP = {
    "single": 1, "double": 2, "triple": 3, "aromatic": 4,
    "wedge": 5, "dash": 6, "solid wedge": 5, "dashed wedge": 6,
    "wave": 7, "wavy": 7,
}

# 元素符号 (常规元素, 不包含会被误判为 Markush 变量的短符号 Y/Z/Q/A/W/E/M 等)
ELEM2 = {
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar", "Ca", "Sc",
    "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se",
    "Br", "Kr", "Rb", "Sr", "Zr", "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "Th", "Pa", "U",
}
ELEM1 = {"H", "B", "C", "N", "O", "F", "P", "S", "K", "V", "I"}
ELEMENTS = ELEM1 | ELEM2

# 带方括号但明确是"画出来的元素字母"的白名单
BRACKET_ELEMS = {"H", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"}

STEREO_C = re.compile(r"^C@{1,2}H?$")
ISO_ELEM = re.compile(r"^(\d+)([A-Z][a-z]?)$")


def split_token(tok):
    """返回 (symbol, label, flag)。flag 标记需要人工看图确认的 token。"""
    t = str(tok).strip()
    if t == "*":
        return "*", "_AP", None
    if t == "_AP":
        return "*", "_AP", None

    bracketed = t.startswith("[") and t.endswith("]")
    inner = t[1:-1] if bracketed else t

    if not bracketed:
        # 裸 token: 元素 / 芳香小写
        if len(inner) == 1:
            up = inner.upper()
            if up in ELEMENTS:
                return up, None, None
        if len(inner) == 2:
            cand = inner[0].upper() + inner[1].lower()
            if cand in ELEMENTS:
                return cand, None, None
        return "*", f"[{inner}]", f"non-element bare token {t!r}"

    # 带方括号: 只有"画出来的元素字母 / 立体碳 / 同位素"当元素, 其余一律超原子
    if STEREO_C.match(inner):                       # [C@H] [C@@H] [C@] [C@@]
        return "C", None, None
    if inner in BRACKET_ELEMS:                      # [H] [N] [O] ...
        return inner, None, None
    m = ISO_ELEM.match(inner)                       # [18O] [2H] [3H]
    if m and m.group(1) and m.group(2) in BRACKET_ELEMS:
        return m.group(2), None, None

    return "*", t, None if _looks_clean(inner) else f"suspicious token {t!r}"


def _looks_clean(inner):
    """括号内是否有明显 OCR 垃圾 (括号不配对 / 只有标点数字)。"""
    if not inner:
        return False
    if inner.count("(") != inner.count(")"):
        return False
    if any(ch in inner for ch in ",;"):
        return False
    if not any(ch.isalpha() for ch in inner):
        return False
    return True


def load_raw(i):
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    W, H = Image.open(D / "images" / f"{i:06d}.png").size
    return ann, W, H


def base(i):
    ann, W, H = load_raw(i)
    atoms, flags = [], []
    for k, a in enumerate(ann["atoms"]):
        sym, lab, fl = split_token(a.get("atom_symbol", "C"))
        atoms.append({
            "symbol": sym, "label": lab,
            "x": round(a["x"] * W, 2), "y": round(a["y"] * H, 2),
        })
        if fl:
            flags.append((k, a.get("atom_symbol"), fl))
    bonds = []
    for b in ann["bonds"]:
        u, v = b["endpoint_atoms"]
        o = BOND_MAP.get(str(b.get("bond_type", "single")).lower(), 1)
        bonds.append([u, v, o])
    return atoms, bonds, (W, H), flags, ann


def payload(i):
    atoms, bonds, (W, H), flags, ann = base(i)
    esmi = graph_to_smiles(atoms, bonds, [])
    return {
        "index": i, "esmiles": esmi, "image_size": [W, H], "coverage": 1.0,
        "atoms": atoms, "bonds": bonds, "sgroups": [], "brackets": [],
        "ocr_atoms": [], "superatom_merges": [],
    }, flags


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    show_tokens = "--tokens" in sys.argv
    if len(args) == 1 and "-" in args[0]:
        lo, hi = (int(x) for x in args[0].split("-"))
    else:
        lo, hi = 202, 400
    writes = []
    for i in range(lo, hi + 1):
        p, flags = payload(i)
        writes.append((i, p, flags))
        if show_tokens:
            print(f"--- {i}  {p['image_size']}  na={len(p['atoms'])} nb={len(p['bonds'])}")
            print(f"    esmiles: {p['esmiles']}")
            for k, tok, fl in flags:
                print(f"    FLAG atom{k} {tok!r}: {fl}")
    print(f"built {len(writes)} payloads")
