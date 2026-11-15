# -*- coding: utf-8 -*-
"""
扫描 benchmark_D2C-RND 与 benchmark_D2C-UNI 的所有 seg/boxes/*.json，
解码出每条 (office, publication_id, pages)，
生成一份 PDF 下载清单 CSV：标明每份 PDF 该放到哪个目录、是否已存在、
可用的 Google Patents / Espacenet 检索链接。
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from decode_filename import decode_stem  # noqa: E402
from find_pdf import find_pdf  # noqa: E402

BENCHMARK_ROOT = Path(r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents")
OUT_CSV = BENCHMARK_ROOT / "pdf_download_list.csv"

# 下载后的 PDF 期望路径：分 office 子目录便于管理
PDF_ROOT = BENCHMARK_ROOT / "pdfs_benchmark"

# 兼容已经存在的目录：BENCHMARK_ROOT/pdfs/<cnipa|epo|jpo|kipo|uspto>/*.pdf
LEGACY_PDF_ROOT = BENCHMARK_ROOT / "pdfs"

# office 码 -> 旧的子目录命名（PatCID main dataset 中用的命名）
OFFICE_TO_LEGACY = {
    "cn": "cnipa",
    "eu": "epo",
    "jp": "jpo",
    "kr": "kipo",
    "us": "uspto",
}


def google_patents_url(pub_id: str) -> str:
    return f"https://patents.google.com/patent/{pub_id}/en"


def espacenet_url(pub_id: str) -> str:
    return f"https://worldwide.espacenet.com/patent/search?q={pub_id}"


def find_existing(pub_id: str, office: str) -> str:
    p = find_pdf(pub_id, office)
    return str(p) if p else ""


def main():
    rows = []
    # 同一 publication 多页只输出一行；同时记录涉及的页码集合
    aggregate = defaultdict(lambda: {"pages": set(), "subsets": set(), "prefixes": set()})

    for subset in ["benchmark_D2C-RND", "benchmark_D2C-UNI"]:
        seg_root = BENCHMARK_ROOT / subset / "data" / "seg" / "boxes"
        if not seg_root.exists():
            print(f"[WARN] missing: {seg_root}")
            continue
        for jp in seg_root.rglob("*.json"):
            stem = jp.stem
            info = decode_stem(stem)
            if info is None:
                print(f"[WARN] decode fail: {stem}")
                continue
            key = (info["office"], info["publication_id"])
            agg = aggregate[key]
            agg["pages"].add(info["page"])
            agg["subsets"].add(subset)
            agg["prefixes"].add(stem.rsplit("_", 1)[0])

    for (office, pub), agg in sorted(aggregate.items()):
        existing = find_existing(pub, office)
        rows.append({
            "office": office,
            "publication_id": pub,
            "subsets": ";".join(sorted(agg["subsets"])),
            "filename_prefix": ";".join(sorted(agg["prefixes"])),
            "pages_needed": ",".join(str(p) for p in sorted(agg["pages"])),
            "n_pages_needed": len(agg["pages"]),
            "target_pdf_path": str(PDF_ROOT / office / f"{pub}.pdf"),
            "existing_pdf_path": existing,
            "exists": bool(existing),
            "google_patents_url": google_patents_url(pub),
            "espacenet_url": espacenet_url(pub),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "office", "publication_id", "subsets", "filename_prefix",
            "pages_needed", "n_pages_needed",
            "target_pdf_path", "existing_pdf_path", "exists",
            "google_patents_url", "espacenet_url",
        ])
        w.writeheader()
        w.writerows(rows)

    # 汇总
    by_office = defaultdict(lambda: {"total": 0, "have": 0, "missing": 0})
    for r in rows:
        s = by_office[r["office"]]
        s["total"] += 1
        if r["exists"]:
            s["have"] += 1
        else:
            s["missing"] += 1

    print(f"\nWrote {len(rows)} unique publications -> {OUT_CSV}\n")
    print(f"{'office':<8}{'have':>8}{'missing':>10}{'total':>8}")
    for o in sorted(by_office):
        s = by_office[o]
        print(f"{o:<8}{s['have']:>8}{s['missing']:>10}{s['total']:>8}")

    # 仅缺失列表（便于复制粘贴去抓 PDF）
    missing_csv = BENCHMARK_ROOT / "pdf_download_list_missing.csv"
    with open(missing_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "office", "publication_id", "target_pdf_path",
            "google_patents_url", "espacenet_url",
        ])
        w.writeheader()
        for r in rows:
            if not r["exists"]:
                w.writerow({k: r[k] for k in w.fieldnames})
    print(f"\nMissing-only list -> {missing_csv}")


if __name__ == "__main__":
    main()
