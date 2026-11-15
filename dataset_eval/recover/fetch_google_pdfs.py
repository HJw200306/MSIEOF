# -*- coding: utf-8 -*-
"""
从 Google Patents 解析 citation_pdf_url 并直接下载到本地。
绕开浏览器 "Download PDF" 按钮抽风的问题。

用法:
    python fetch_google_pdfs.py CN103228653B CN102838654B CN104016971B CN103998442B JP2017031138A
    # 或者:
    python fetch_google_pdfs.py --from-missing  # 读取 pdf_download_list_missing.csv

下载位置: 自动根据 office 放到 pdfs/<legacy_subdir>/ 下
    cn -> pdfs/cnipa/   us -> pdfs/uspto/   eu -> pdfs/epo/
    jp -> pdfs/jpo/     kr -> pdfs/kipo/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlparse

import requests

BENCHMARK_ROOT = Path(r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents")
PDFS_ROOT = BENCHMARK_ROOT / "pdfs"
OFFICE_TO_LEGACY = {"cn": "cnipa", "us": "uspto", "eu": "epo", "jp": "jpo", "kr": "kipo"}
MISSING_CSV = BENCHMARK_ROOT / "pdf_download_list_missing.csv"
SERPAPI_BASE = "https://serpapi.com/search.json"
RESOLVED_MAP_CSV = BENCHMARK_ROOT / "pdf_resolved_mapping.csv"
SERPAPI_KEY_FILE = Path(__file__).resolve().parent.parent / ".serpapi_key"


def load_serpapi_key() -> str | None:
    """SerpAPI key 加载顺序: 环境变量 SERPAPI_KEY > dataset_eval/.serpapi_key 单行文件。"""
    key = os.environ.get("SERPAPI_KEY")
    if key:
        return key.strip() or None
    if SERPAPI_KEY_FILE.exists():
        return SERPAPI_KEY_FILE.read_text(encoding="utf-8").strip() or None
    return None


# 不传命令行参数时默认要下载的专利号（5 个本地占位符损坏的 PDF）
DEFAULT_IDS = [
    "CN103228653B",
    "CN102838654B",
    "CN104016971B",
    "CN103998442B",
    "JP2017031138A",
]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

META_RE = re.compile(
    r'<meta\s+name="citation_pdf_url"\s+content="([^"]+)"', re.IGNORECASE
)


def office_of(pub_id: str) -> str:
    pid = pub_id.upper()
    if pid.startswith("CN"):
        return "cn"
    if pid.startswith("US"):
        return "us"
    if pid.startswith("EP"):
        return "eu"
    if pid.startswith("JP"):
        return "jp"
    if pid.startswith("KR"):
        return "kr"
    return ""


def patent_page_url(pub_id: str) -> str:
    return f"https://patents.google.com/patent/{pub_id}/en"


def find_pdf_url(pub_id: str, session: requests.Session) -> str | None:
    url = patent_page_url(pub_id)
    r = session.get(url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  [HTTP {r.status_code}] {url}")
        return None
    m = META_RE.search(r.text)
    if not m:
        print(f"  [NO citation_pdf_url meta] {url}")
        return None
    return m.group(1)


STORAGE_BASE = "https://patentimages.storage.googleapis.com/"


EP_RE = re.compile(r"^EP(\d{8})([A-Z]\d?)?$")
KR_RE = re.compile(r"^KR(\d{13})$")


def normalize_query(pub_id: str) -> str:
    """
    PatCID 的 publication_id 是 5 个专利局异构格式的拼盘，Google 不一定认。
    把 EP/KR 归一化到 Google Patents/SerpAPI 能识别的查询串。

    规则（基于 SerpAPI 实测）：
      - EP, 8 位数字 + 可选 kind：
          * 以 0 开头（旧式公开号被左填充到 8 位）→ 剥 1 个 0 + 保留 kind
            例 EP00103355B1 → EP0103355B1
          * 不以 0 开头（现代 8 位申请号 + 误带 kind）→ 丢弃 kind，按申请号查
            例 EP04790519B1 → EP04790519
      - KR, 13 位（KR + 1 + YYYY + 7位 = 申请号）→ 原样
      - 其他局（CN/US/JP）→ 原样
    """
    pid = pub_id.upper()
    m = EP_RE.match(pid)
    if m:
        digits, kind = m.group(1), m.group(2) or ""
        if digits.startswith("0"):
            return f"EP{digits[1:]}{kind}"
        return f"EP{digits}"  # 申请号，去掉 kind
    return pid


def resolve_via_serpapi(pub_id: str, session: requests.Session, api_key: str,
                        max_retries: int = 4) -> tuple[str, str] | None:
    """
    用 SerpAPI 的 google_patents 引擎查 → 真实 publication_number + pdf URL。
    通过 SerpAPI 的 IP 访问 Google，绕开本机 IP 被 patents.google.com 风控的问题。
    实际 PDF 字节流仍走本机直连 patentimages.storage.googleapis.com（独立域，未被封）。
    """
    q = normalize_query(pub_id)
    if q != pub_id:
        print(f"  [normalize] {pub_id} -> {q}")
    params = {"engine": "google_patents", "q": q, "api_key": api_key}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = session.get(SERPAPI_BASE, params=params, timeout=15)
            break
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries:
                print(f"  [serpapi retry {attempt+1}/{max_retries}] {e}")
                time.sleep(2.0)
            else:
                print(f"  [serpapi ERR] {e}")
                return None
    if r.status_code != 200:
        print(f"  [serpapi HTTP {r.status_code}] {r.text[:200]}")
        return None
    try:
        data = r.json()
    except ValueError:
        print(f"  [serpapi JSON decode fail]")
        return None
    if data.get("error"):
        print(f"  [serpapi error] {data['error']}")
        return None
    for hit in data.get("organic_results", []) or []:
        pub = hit.get("publication_number")
        pdf = hit.get("pdf")
        if pub and pdf:
            return pub, pdf
    return None


def append_mapping(input_id: str, resolved_pub: str, dst: Path):
    """记录 申请号→公开号 的解析结果，便于后续 recover_pages.py 反查。"""
    new = not RESOLVED_MAP_CSV.exists()
    with open(RESOLVED_MAP_CSV, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["input_publication_id", "resolved_publication_number", "local_pdf_path"])
        w.writerow([input_id, resolved_pub, str(dst)])


def load_resolved_map() -> dict[str, Path]:
    """读出 input_id → 本地 PDF 路径，便于重跑时跳过已解析过的 ID（省 SerpAPI credits）。"""
    out: dict[str, Path] = {}
    if not RESOLVED_MAP_CSV.exists():
        return out
    with open(RESOLVED_MAP_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = Path(row.get("local_pdf_path", ""))
            if p.exists():
                out[row["input_publication_id"].upper()] = p
    return out


def resolve_via_xhr(pub_id: str, session: requests.Session) -> tuple[str, str] | None:
    """
    EU/KR benchmark 文件名带的是申请号，直链 /patent/<id>/en 会 404。
    Google Patents 前端用的 xhr/query JSON 接口支持申请号查询，
    返回真正的 publication_number 以及 patentimages storage 的相对 pdf 路径。
    """
    inner = f"q={pub_id}&oq={pub_id}"
    xhr_url = f"https://patents.google.com/xhr/query?url={quote(inner, safe='')}&exp="
    r = session.get(xhr_url, headers=HEADERS, timeout=30)
    if r.status_code != 200:
        print(f"  [xhr HTTP {r.status_code}]")
        return None
    try:
        data = json.loads(r.text)
    except json.JSONDecodeError:
        print(f"  [xhr JSON decode fail]")
        return None
    clusters = data.get("results", {}).get("cluster", []) or []
    for cluster in clusters:
        for hit in cluster.get("result", []) or []:
            patent = hit.get("patent", {}) or {}
            pub = patent.get("publication_number")
            pdf = patent.get("pdf")
            if pub and pdf:
                return pub, STORAGE_BASE + pdf
    return None


def download(pdf_url: str, dst: Path, session: requests.Session) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        with session.get(pdf_url, headers=HEADERS, stream=True, timeout=60) as r:
            if r.status_code != 200:
                print(f"  [HTTP {r.status_code}] {pdf_url}")
                return False
            ctype = r.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower():
                print(f"  [WARN] Content-Type={ctype}  (still saving) {pdf_url}")
            tmp = dst.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
                        n += len(chunk)
            tmp.rename(dst)
            print(f"  [OK] {n/1024/1024:.2f} MB -> {dst}")
            return True
    except Exception as e:
        print(f"  [ERR] {e}  {pdf_url}")
        return False


def run_batch(ids: list[str], session: requests.Session, force: bool,
              via: str, serpapi_key: str | None, sleep: float,
              resolved_map: dict[str, Path] | None = None) -> tuple[int, int]:
    ok = fail = 0
    for i, pid in enumerate(ids, 1):
        print(f"\n[{i}/{len(ids)}] {pid}")
        if fetch_one(pid, session, force, via, serpapi_key, resolved_map):
            ok += 1
        else:
            fail += 1
        if i < len(ids):
            time.sleep(sleep)
    return ok, fail


def load_missing_by_office(office_filter: set[str]) -> list[str]:
    if not MISSING_CSV.exists():
        return []
    out: list[str] = []
    with open(MISSING_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if office_filter and row.get("office", "").lower() not in office_filter:
                continue
            out.append(row["publication_id"])
    return out


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        print(f"{prompt} [auto-yes]")
        return True
    try:
        ans = input(f"{prompt} (y/N): ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


def run_pipeline(session: requests.Session, sleep: float, force: bool, assume_yes: bool) -> None:
    """3 EU 验证 → 2 KR 验证 → 剩余 EU+KR 全跑完，每阶段后交互确认。"""
    key = load_serpapi_key()
    if not key:
        print(f"[FATAL] 缺少 SerpAPI key。"
              f"请在 {SERPAPI_KEY_FILE} 写入单行 key，或设置环境变量 SERPAPI_KEY。")
        sys.exit(2)

    eu_all = load_missing_by_office({"eu"})
    kr_all = load_missing_by_office({"kr"})
    if not eu_all and not kr_all:
        print(f"[FATAL] {MISSING_CSV} 不存在或 EU/KR 缺失列表为空")
        sys.exit(2)

    print(f"[PLAN] EU 缺失 {len(eu_all)} / KR 缺失 {len(kr_all)}（总 {len(eu_all)+len(kr_all)}）")
    print(f"[PLAN] 阶段 1: 3 EU 验证（3 credits）")
    print(f"[PLAN] 阶段 2: 2 KR 验证（2 credits）")
    print(f"[PLAN] 阶段 3: 剩余 {len(eu_all[3:]) + len(kr_all[2:])} 个（{len(eu_all[3:]) + len(kr_all[2:])} credits）")
    print()

    # 阶段 1
    print("=" * 60)
    print("阶段 1: EU 验证")
    print("=" * 60)
    ok1, f1 = run_batch(eu_all[:3], session, force, "serpapi", key, sleep)
    print(f"\n阶段 1 结果: ok={ok1} fail={f1}")
    if f1 > 0 and not confirm("阶段 1 有失败，继续阶段 2？", assume_yes):
        print("退出。")
        return

    # 阶段 2
    print()
    print("=" * 60)
    print("阶段 2: KR 验证")
    print("=" * 60)
    ok2, f2 = run_batch(kr_all[:2], session, force, "serpapi", key, sleep)
    print(f"\n阶段 2 结果: ok={ok2} fail={f2}")
    if f2 > 0 and not confirm("阶段 2 有失败，继续阶段 3？", assume_yes):
        print("退出。前 5 个已下载。")
        return

    # 阶段 3
    remaining = eu_all[3:] + kr_all[2:]
    print()
    print("=" * 60)
    print(f"阶段 3: 剩余 {len(remaining)} 个 EU+KR")
    print("=" * 60)
    if not confirm(f"消耗 {len(remaining)} credits 跑剩余全部？", assume_yes):
        print("中止阶段 3。前 5 个已下载。")
        return
    ok3, f3 = run_batch(remaining, session, force, "serpapi", key, sleep)

    total = len(eu_all[:3]) + len(kr_all[:2]) + len(remaining)
    print(f"\n=== ALL DONE ===  ok={ok1+ok2+ok3}  fail={f1+f2+f3}  total={total}")


def fetch_one(pub_id: str, session: requests.Session, force: bool,
              via: str, serpapi_key: str | None,
              resolved_map: dict[str, Path] | None = None) -> bool:
    office = office_of(pub_id)
    if not office:
        print(f"[SKIP] cannot infer office: {pub_id}")
        return False
    sub = OFFICE_TO_LEGACY[office]
    dst = PDFS_ROOT / sub / f"{pub_id}.pdf"
    if dst.exists() and not force:
        sz = dst.stat().st_size
        print(f"[KEEP] exists {dst} ({sz} bytes)")
        return True
    # 重跑场景：之前已经把 申请号→公开号 的映射写过，PDF 已落到解析后的公开号路径，
    # 这里先查映射表，命中就跳过，避免重复消耗 SerpAPI credits。
    if not force and resolved_map:
        hit = resolved_map.get(pub_id.upper())
        if hit and hit.exists():
            print(f"[KEEP] {pub_id} 已解析 -> {hit} ({hit.stat().st_size} bytes)")
            return True
    print(f"[GET ] {pub_id} ...")

    if via == "serpapi":
        # 本机 IP 已被 patents.google.com 风控，跳过直链与 xhr，直接走 SerpAPI 代理。
        if not serpapi_key:
            print(f"  [FATAL] --via serpapi 需要环境变量 SERPAPI_KEY")
            return False
        resolved = resolve_via_serpapi(pub_id, session, serpapi_key)
        if not resolved:
            print(f"  [serpapi no result for {pub_id}]")
            return False
        real_pub, storage_pdf = resolved
        if real_pub.upper() != pub_id.upper():
            dst = PDFS_ROOT / sub / f"{real_pub}.pdf"
            print(f"  resolved {pub_id} -> {real_pub}")
            if dst.exists() and not force:
                sz = dst.stat().st_size
                print(f"  [KEEP] exists {dst} ({sz} bytes)")
                append_mapping(pub_id, real_pub, dst)
                return True
        print(f"  pdf_url = {storage_pdf}")
        ok = download(storage_pdf, dst, session)
        if ok:
            append_mapping(pub_id, real_pub, dst)
        return ok

    # via == "direct": 老流程（CN/US/JP 仍可用）
    pdf_url = find_pdf_url(pub_id, session)
    if pdf_url:
        print(f"  pdf_url = {pdf_url}")
        return download(pdf_url, dst, session)

    # 直链 404（EU/KR benchmark 用的是申请号）→ 用 xhr/query 把申请号解析成公开号
    resolved = resolve_via_xhr(pub_id, session)
    if not resolved:
        print(f"  [xhr no result for {pub_id}]")
        return False
    real_pub, storage_pdf = resolved
    if real_pub.upper() != pub_id.upper():
        # 用解析出的公开号命名，方便 find_pdf 的 (office_prefix, digits) 主键命中
        dst = PDFS_ROOT / sub / f"{real_pub}.pdf"
        print(f"  resolved {pub_id} -> {real_pub}")
        if dst.exists() and not force:
            sz = dst.stat().st_size
            print(f"  [KEEP] exists {dst} ({sz} bytes)")
            return True
    print(f"  pdf_url = {storage_pdf}")
    return download(storage_pdf, dst, session)


def main():
    ap = argparse.ArgumentParser(
        description="不传参数 = 默认执行 EU+KR 三阶段 SerpAPI 流水线（3 EU 验证 → 2 KR 验证 → 剩余）。"
                    "高级用法可显式传 --from-missing / pub_ids / --via / --offices / --limit。")
    ap.add_argument("pub_ids", nargs="*", help="显式 publication ids；不传则进入默认流水线")
    ap.add_argument("--from-missing", action="store_true",
                    help="读取 pdf_download_list_missing.csv 里的所有 publication_id")
    ap.add_argument("--force", action="store_true", help="即使本地已有也强制重下")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="每个请求之间的间隔秒数（默认 2.0）")
    ap.add_argument("--offices", nargs="+", default=None,
                    choices=["cn", "us", "eu", "jp", "kr"],
                    help="只下载指定 office 的 publication（可多选，如 --offices eu kr）")
    ap.add_argument("--via", choices=["direct", "serpapi"], default=None,
                    help="direct=本机直连 Google Patents；serpapi=通过 SerpAPI 代理。"
                         "默认流水线模式下自动使用 serpapi。")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个 ID（用于小批量验证；0 = 不限制）")
    ap.add_argument("--min-kr-year", type=int, default=0,
                    help="跳过申请年份小于该值的 KR (e.g. 1995)。0 = 不过滤。"
                         "用于排除 Google Patents 几乎不收录的早期 KR 专利。")
    ap.add_argument("--yes", action="store_true", help="跳过每阶段的交互确认（CI/无人值守用）")
    args = ap.parse_args()

    session = requests.Session()
    # 实测本机访问 serpapi.com 必须走系统代理（直连 30s timeout），但代理本身偶发
    # ConnectionReset/Timeout。在 resolve_via_serpapi 里用 4 次重试 + 15s timeout 兜底，
    # 这里保留 trust_env=True 让 requests 自动读 HTTPS_PROXY / WPAD。

    # 默认模式: 不传任何 ID/--from-missing → 走 EU+KR 三阶段 SerpAPI 流水线
    is_pipeline = not args.pub_ids and not args.from_missing
    if is_pipeline:
        run_pipeline(session, args.sleep, args.force, args.yes)
        return

    # 显式模式: 老逻辑（兼容旧命令行）
    via = args.via or "direct"
    serpapi_key: str | None = None
    if via == "serpapi":
        serpapi_key = load_serpapi_key()
        if not serpapi_key:
            print(f"[FATAL] --via serpapi 需要 SerpAPI key（{SERPAPI_KEY_FILE} 或 env SERPAPI_KEY）")
            sys.exit(2)

    office_filter: set[str] | None = set(args.offices) if args.offices else None

    ids: list[str] = []
    for pid in args.pub_ids:
        if office_filter and office_of(pid) not in office_filter:
            continue
        ids.append(pid)

    if args.from_missing:
        ids.extend(load_missing_by_office(office_filter or set()))
        if not ids:
            print(f"[FATAL] {MISSING_CSV} not found or empty after filter")
            sys.exit(2)

    if not ids:
        ids = list(DEFAULT_IDS)
        print(f"[INFO] 未传专利号，使用脚本内置 DEFAULT_IDS（{len(ids)} 个）")

    # KR 早期申请号大多在 Google Patents 索引外，按需过滤
    if args.min_kr_year > 0:
        before = len(ids)
        kept: list[str] = []
        for pid in ids:
            m = re.match(r"^KR10(\d{4})\d{7}$", pid)
            if m and int(m.group(1)) < args.min_kr_year:
                continue
            kept.append(pid)
        ids = kept
        dropped = before - len(ids)
        print(f"[INFO] --min-kr-year {args.min_kr_year} 跳过 {dropped} 个早期 KR，剩 {len(ids)} 个")

    if args.limit > 0:
        ids = ids[:args.limit]
        print(f"[INFO] --limit {args.limit} 生效，只处理前 {len(ids)} 个")

    resolved_map = load_resolved_map() if via == "serpapi" else None
    if resolved_map:
        print(f"[INFO] 已知映射 {len(resolved_map)} 个；命中即跳过，省 SerpAPI credits")
    ok, fail = run_batch(ids, session, args.force, via, serpapi_key, args.sleep, resolved_map)
    print(f"\n=== done ===  ok={ok}  fail={fail}  total={len(ids)}")


if __name__ == "__main__":
    main()
