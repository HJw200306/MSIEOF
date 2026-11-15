# -*- coding: utf-8 -*-
"""
将 benchmark_D2C-RND / benchmark_D2C-UNI 中的 seg / cls / rec 文件名
解码为 (office, publication_id, page, mol_index)。

命名约定（实测）：
  CN: CN1<14digits>CN0000<9digit pub><1digit><kind letter>PDFZH<8digit date>CN00<1ch>_<page>[_<mol>]
  US: <7~11digits>[kind]?_<page>[_<mol>]
  EP: EP<8digits>NW<kind>_<page>[_<mol>]
  JP: <kind letter>__<10digits>_<page>[_<mol>]
  KR: <13digits>_<page>[_<mol>]
"""

import re

CN_RE = re.compile(
    r"^CN1\d{14}CN0000(\d{9})\d([A-Z])PDFZH\d{8}CN00\S_(\d+)(?:_(\d+))?$"
)
US_RE = re.compile(
    r"^(?:US)?(\d{7,11})(A[12]|B[12]|S1|P[123])?_(\d+)(?:_(\d+))?$"
)
EP_RE = re.compile(
    r"^EP(\d{8})NW([A-Z]\d?)_(\d+)(?:_(\d+))?$"
)
JP_RE = re.compile(
    r"^([A-Z][A-Z0-9_]*)__(\d{10})_(\d+)(?:_(\d+))?$"
)
KR_RE = re.compile(
    r"^(\d{13})_(\d+)(?:_(\d+))?$"
)


def decode_stem(stem: str):
    m = CN_RE.match(stem)
    if m:
        return {
            "office": "cn",
            "publication_id": f"CN{m.group(1)}{m.group(2)}",
            "page": int(m.group(3)),
            "mol_index": int(m.group(4)) if m.group(4) else None,
        }
    m = EP_RE.match(stem)
    if m:
        return {
            "office": "eu",
            "publication_id": f"EP{m.group(1)}{m.group(2)}",
            "page": int(m.group(3)),
            "mol_index": int(m.group(4)) if m.group(4) else None,
        }
    m = JP_RE.match(stem)
    if m:
        return {
            "office": "jp",
            "publication_id": f"JP{m.group(2)}{m.group(1)}",
            "page": int(m.group(3)),
            "mol_index": int(m.group(4)) if m.group(4) else None,
        }
    m = KR_RE.match(stem)
    if m:
        return {
            "office": "kr",
            "publication_id": f"KR{m.group(1)}",
            "page": int(m.group(2)),
            "mol_index": int(m.group(3)) if m.group(3) else None,
        }
    m = US_RE.match(stem)
    if m:
        num = int(m.group(1))
        kind = m.group(2) or ""
        return {
            "office": "us",
            "publication_id": f"US{num}{kind}",
            "page": int(m.group(3)),
            "mol_index": int(m.group(4)) if m.group(4) else None,
        }
    return None


if __name__ == "__main__":
    samples = [
        "CN102012000365559CN00001036950760BPDFZH20160824CN00Z_5_0",
        "CN112011000056716CN00001032286530BPDFZH20160316CN00X_463_2",
        "04321368_5",
        "EP00103355NWB1_36",
        "A__2005263783_16",
        "1019950700951_11",
    ]
    for s in samples:
        print(s, "->", decode_stem(s))
