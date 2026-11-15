# -*- coding: utf-8 -*-
"""
按解码出的 publication_id 在本地 PDF 库中做"宽松匹配"。

匹配策略：
  1) 切分文件名为 (office_prefix, digits, kind)，digits 去掉前导零
  2) 用 (office, digits) 做主键，命中后再优先选 kind 相同的
  3) KR 特殊：benchmark 用 13 位申请号 "10-YYYY-NNNNNNN"，本地命名通常用
     "KR<YYYY><7位序号><kind>"，去掉前 "10" 再查一次

搜索目录（按顺序）：
  pdfs/                      (扁平，用户当前主要存放位置)
  pdfs_benchmark/<office>/   (新规范，可选)
  restored/<office>/         (回填的 PDF)
  PatCID_main_dataset/pdfs/<legacy>/ (主数据集附带的 PDF，若存在)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

BENCHMARK_ROOT = Path(r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents")

SEARCH_ROOTS = [
    BENCHMARK_ROOT / "pdfs",
    BENCHMARK_ROOT / "pdfs_benchmark",
    BENCHMARK_ROOT / "restored",
    BENCHMARK_ROOT / "PatCID_main_dataset" / "pdfs",
]

OFFICE_TO_LEGACY = {"cn": "cnipa", "us": "uspto", "eu": "epo", "jp": "jpo", "kr": "kipo"}

_PAT_RE = re.compile(r"^([A-Z]+)(\d+)([A-Z]\d*)?$")


def _split(name: str):
    """('EP3781311B1') -> ('EP', '3781311', 'B1') ；前导 0 去除。失败返回 None。"""
    m = _PAT_RE.match(name.upper())
    if not m:
        return None
    return (m.group(1), m.group(2).lstrip("0") or "0", m.group(3) or "")


_index: Optional[dict[tuple[str, str], list[Path]]] = None


def _build_index() -> dict[tuple[str, str], list[Path]]:
    idx: dict[tuple[str, str], list[Path]] = {}
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for pdf in root.rglob("*.pdf"):
            parts = _split(pdf.stem)
            if parts is None:
                continue
            office_pref, digits, _kind = parts
            idx.setdefault((office_pref, digits), []).append(pdf)
    return idx


def get_index() -> dict[tuple[str, str], list[Path]]:
    global _index
    if _index is None:
        _index = _build_index()
    return _index


def reset_index():
    """新文件下载后刷新索引。"""
    global _index
    _index = None


def find_pdf(publication_id: str, office: str = "") -> Optional[Path]:
    idx = get_index()
    parts = _split(publication_id)
    if parts is None:
        return None
    pref, digits, kind = parts

    hits = idx.get((pref, digits))
    if hits:
        if kind:
            for h in hits:
                p2 = _split(h.stem)
                if p2 and p2[2] == kind:
                    return h
        return hits[0]

    # KR: 13 位申请号 → 去掉前导 "10" 再试
    if pref == "KR" and digits.startswith("10") and len(digits) >= 11:
        sub_digits = digits[2:].lstrip("0") or "0"
        hits = idx.get(("KR", sub_digits))
        if hits:
            return hits[0]

    # EP/JP/CN/US 已经在主索引里做过前导零归一了，再无更多变体可猜
    return None


def stats():
    idx = get_index()
    by_office = {}
    total = 0
    for (pref, _), paths in idx.items():
        by_office[pref] = by_office.get(pref, 0) + len(paths)
        total += len(paths)
    return total, by_office


if __name__ == "__main__":
    total, by = stats()
    print(f"local PDFs: {total}")
    for k, v in sorted(by.items()):
        print(f"  {k:<4}: {v}")
    for pid in [
        "CN103695076B", "EP3781311B1", "EP03781311B1",
        "JP2005263783A", "US4321368", "US20020103189A1",
        "KR1019950700951",
    ]:
        p = find_pdf(pid)
        print(f"{pid:<30} -> {p}")
