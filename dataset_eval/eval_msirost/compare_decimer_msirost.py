"""DECIMER-Segmentation vs MSIEOF 在恢复 PatCID 页面评测集上的对比评估。

对每张页面：
    1) 构建一次 GT（build_gt.build_gt_for_page，灰度口径 color_aware=False）；
    2) 跑 MSIEOF 完整流水线（predict.predict_for_page → build_page_pred_masks）；
    3) 跑 DECIMER-Segmentation（predict_decimer.predict_decimer_for_page）；
    4) 两者用同一 scorer（score.score_page @ RECALL_THRESHOLD）各自打分；
    5) 聚合后输出统一对比表。

判定口径（与 eval_msirost/main.py 完全一致）
-------------------------------------------
一个 GT 分子被判为"完整检出" ⇔ 匈牙利 1对1 匹配后，其匹配到的预测像素对该 GT
的覆盖率 `|P∩G|/|G|` ≥ RECALL_THRESHOLD。两方法共用同一 GT、同一 scorer、
同一阈值，因此指标可直接横向比较。

命令行
------
  python -m dataset_eval.eval_msirost.compare_decimer_msirost --limit 5
  python -m dataset_eval.eval_msirost.compare_decimer_msirost            # 全量 449 页
  python -m dataset_eval.eval_msirost.compare_decimer_msirost --method decimer

（cwd 在 msirost 项目根；--method 可单独重跑某一边，默认 both。）

两环境复用：本机 DECIMER 只在 Anaconda（D:\\ProgramData\\Anaconda3）有
tensorflow/decimer_segmentation，MSIEOF 只在 FastEnv（C:\\FastEnv\\msirost）有
torch。无法同进程同时 import 两者时，分别在各自环境跑单边生成 per_page_*.csv，
再用 --decimer-csv / --msirost-csv 喂回来聚合：

  # 环境 A（Anaconda，跑 DECIMER）
  D:\\ProgramData\\Anaconda3\\python.exe -m dataset_eval.eval_msirost.compare_decimer_msirost --method decimer
  # 环境 B（FastEnv，跑 MSIEOF，复用 A 的 DECIMER csv）
  C:\\FastEnv\\msirost\\python.exe -m dataset_eval.eval_msirost.compare_decimer_msirost --decimer-csv "E:\\其他文章及数据集\\eval_compare\\results\\per_page_decimer.csv"

产物
----
  COMPARE_RESULTS_DIR/per_page_msirost.csv   每页一行（MSIEOF 侧）
  COMPARE_RESULTS_DIR/per_page_decimer.csv   每页一行（DECIMER 侧）
  COMPARE_RESULTS_DIR/comparison.csv         统一对比表（列 = 用户要求的 8 项指标）
  COMPARE_RESULTS_DIR/comparison.md          人读版本
"""

from __future__ import annotations

# PyCharm 右键 Run 直接跑时注入包路径（python -m ... 时是 no-op）。
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "dataset_eval.eval_msirost"

import argparse
import csv
import gc
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import cv2

from . import build_gt, predict, predict_decimer, score
from .config import (
    COMPARE_RESULTS_DIR,
    OFFICES,
    RECALL_THRESHOLD,
    SUBSETS,
)
from .main import _aggregate, _iter_pages

# 对比表里的方法 / 数据集标签（对齐论文表头）。
DATASET_LABEL = "恢复 PatCID 页面评测集"
METHOD_LABELS = {
    "msirost": "MSIEOF (Ours)",
    "decimer": "DECIMER",
}

# 统一对比表列顺序（用户要求的 8 项指标 + 前置标识列）。
COMPARE_COLUMNS = [
    "Dataset", "Method", "Pages", "GT", "Pred", "Complete", "Partial",
    "Missed", "False Positive", "Coverage Rate (%)",
]


def _run_method(page_png: Path, page_stem_full: str, method: str,
                crops_root: Path) -> List[Dict]:
    """对一张页面跑指定方法，返回 score.py 口径的预测实例列表。"""
    if method == "msirost":
        records = predict.predict_for_page(
            page_png, page_stem_full, color_aware=False, crops_root=crops_root,
        )
        page_bgr = cv2.imread(str(page_png), cv2.IMREAD_COLOR)
        H, W = page_bgr.shape[:2]
        return predict.build_page_pred_masks(records, (H, W))
    if method == "decimer":
        return predict_decimer.predict_decimer_for_page(page_png)
    raise ValueError(method)


def _make_row(subset: str, office: str, stem: str, s: Dict,
              elapsed: float) -> Dict:
    return {
        "subset": subset,
        "office": office,
        "stem": stem,
        "n_gt": s["n_gt"],
        "n_pred": s["n_pred"],
        "n_complete": s["n_complete"],
        "n_partial": s["n_partial"],
        "n_missed": s["n_missed"],
        "n_false_positive": s["n_false_positive"],
        "coverage_rate": round(s["coverage_rate"], 6),
        "elapsed_s": round(elapsed, 2),
    }


def _empty_overall() -> Dict:
    """与 main._aggregate 的 overall 结构一致的空聚合，用于某方法 0 页成功时兜底。"""
    return dict(
        n_pages=0,
        total_gt=0,
        total_pred=0,
        total_complete=0,
        total_partial=0,
        total_missed=0,
        total_false_positive=0,
        coverage_rate=None,
    )


def _load_rows_from_csv(path: Path) -> List[Dict]:
    """读取已有的 per_page.csv，复用其聚合（跳过该方法的推理）。

    复用场景：本机 DECIMER 只装在 Anaconda 环境、MSIEOF 只装在 FastEnv 环境，
    两者无法在同一进程里同时 import tensorflow 与 torch。此时可分别在各自环境
    跑出 per_page_{method}.csv，再用 --decimer-csv / --msirost-csv 喂回来聚合。
    """
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "subset": r["subset"],
                "office": r["office"],
                "stem": r["stem"],
                "n_gt": int(r["n_gt"]),
                "n_pred": int(r["n_pred"]),
                "n_complete": int(r["n_complete"]),
                "n_partial": int(r["n_partial"]),
                "n_missed": int(r["n_missed"]),
                "n_false_positive": int(r["n_false_positive"]),
                "coverage_rate": float(r["coverage_rate"]),
            })
    return rows


def _build_comparison_table(agg_msirost: Dict, agg_decimer: Dict) -> List[Dict]:
    """把两方法的 overall 聚合拼成统一对比表的行。"""
    def _coverage(ov: Dict) -> str:
        if ov["coverage_rate"] is None:
            return ""
        return f"{ov['coverage_rate'] * 100:.2f}"

    rows = []
    for method in ("decimer", "msirost"):
        ov = agg_decimer["overall"] if method == "decimer" else agg_msirost["overall"]
        rows.append({
            "Dataset": DATASET_LABEL,
            "Method": METHOD_LABELS[method],
            "Pages": ov["n_pages"],
            "GT": ov["total_gt"],
            "Pred": ov["total_pred"],
            "Complete": ov["total_complete"],
            "Partial": ov["total_partial"],
            "Missed": ov["total_missed"],
            "False Positive": ov["total_false_positive"],
            "Coverage Rate (%)": _coverage(ov),
        })
    return rows


def _write_comparison(rows: List[Dict], out_dir: Path) -> None:
    """写 comparison.csv（utf-8-sig 便于 Excel 中文）+ comparison.md。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COMPARE_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "comparison.md"
    header = " | ".join(COMPARE_COLUMNS)
    sep = " | ".join("---" for _ in COMPARE_COLUMNS)
    lines = [
        "# DECIMER vs MSIEOF 在恢复 PatCID 页面评测集上的完整分子提取对比",
        "",
        f"判定口径：`|P∩G|/|G|` ≥ {RECALL_THRESHOLD}（匈牙利 1对1 匹配，两方法共用同一 GT）。",
        "",
        f"| {header} |",
        f"| {sep} |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r[c]) for c in COMPARE_COLUMNS) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--offices", nargs="+", default=list(OFFICES))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 页，0=不限")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 页")
    ap.add_argument("--method", choices=["both", "msirost", "decimer"],
                    default="both", help="只跑某一边（both=都跑）")
    ap.add_argument("--crops-root", type=str, default=None,
                    help="MSIEOF 的临时 crop 落盘根（默认走 predict 内部默认值）")
    ap.add_argument("--decimer-csv", type=str, default=None,
                    help="复用已有 DECIMER per_page.csv（跳过 DECIMER 推理）")
    ap.add_argument("--msirost-csv", type=str, default=None,
                    help="复用已有 MSIEOF per_page.csv（跳过 MSIEOF 推理）")
    args = ap.parse_args()

    out_dir = COMPARE_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_root = Path(args.crops_root) if args.crops_root else None

    todo = list(_iter_pages(args.subsets, args.offices))
    todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"[compare] {len(todo)} pages, threshold={RECALL_THRESHOLD}, "
        f"method={args.method}"
    )

    rows_msirost: List[Dict] = []
    rows_decimer: List[Dict] = []
    if args.msirost_csv:
        rows_msirost = _load_rows_from_csv(Path(args.msirost_csv))
        print(f"[compare] 复用 MSIEOF 结果 {len(rows_msirost)} 行 ← {args.msirost_csv}")
    if args.decimer_csv:
        rows_decimer = _load_rows_from_csv(Path(args.decimer_csv))
        print(f"[compare] 复用 DECIMER 结果 {len(rows_decimer)} 行 ← {args.decimer_csv}")

    run_msirost = (args.method in ("both", "msirost")) and not args.msirost_csv
    run_decimer = (args.method in ("both", "decimer")) and not args.decimer_csv

    errors: List[Dict] = []

    if run_msirost or run_decimer:
        for idx, (subset, office, page_png, mask_png, json_path) in enumerate(todo, 1):
            stem = page_png.stem
            page_stem_full = f"{subset}__{office}__{stem}"
            s_msirost = None
            s_decimer = None
            try:
                gt_insts, _ = build_gt.build_gt_for_page(
                    page_png, mask_png, color_aware=False, json_path=json_path,
                )

                if run_msirost:
                    t0 = time.time()
                    pred_msirost = _run_method(
                        page_png, page_stem_full, "msirost", crops_root,
                    )
                    s_msirost = score.score_page(gt_insts, pred_msirost)
                    rows_msirost.append(_make_row(
                        subset, office, stem, s_msirost, time.time() - t0,
                    ))
                    predict.free_gpu_memory()

                if run_decimer:
                    t0 = time.time()
                    pred_decimer = _run_method(
                        page_png, page_stem_full, "decimer", crops_root,
                    )
                    s_decimer = score.score_page(gt_insts, pred_decimer)
                    rows_decimer.append(_make_row(
                        subset, office, stem, s_decimer, time.time() - t0,
                    ))
                    # 显式释放大数组防止内存累积
                    del pred_decimer
                    # 清理 TensorFlow 显存
                    predict_decimer.free_gpu_memory()

                cov_m = (f"{s_msirost['coverage_rate']*100:.1f}%"
                         if s_msirost is not None else "-")
                cov_d = (f"{s_decimer['coverage_rate']*100:.1f}%"
                         if s_decimer is not None else "-")
                print(
                    f"[{idx}/{len(todo)}] {page_stem_full}  GT={len(gt_insts)}  "
                    f"MSIEOF={cov_m}  DECIMER={cov_d}"
                )

                # 显式释放当前页的资源
                del gt_insts
                if s_msirost is not None:
                    del s_msirost
                if s_decimer is not None:
                    del s_decimer

                # 每5页强制垃圾回收 + 显存清理（更激进）
                if idx % 5 == 0:
                    gc.collect()
                    if run_msirost:
                        predict.free_gpu_memory()
                    if run_decimer:
                        predict_decimer.free_gpu_memory()

            except Exception as e:
                elapsed = time.time()
                errors.append({
                    "subset": subset, "office": office, "stem": stem,
                    "error": str(e), "trace": traceback.format_exc(limit=5),
                })
                print(f"[{idx}/{len(todo)}] ERR {page_stem_full}: {e}")
            finally:
                predict.free_gpu_memory()

    # 落盘各方法 per-page csv
    for method, rows in (("msirost", rows_msirost), ("decimer", rows_decimer)):
        if not rows:
            continue
        p = out_dir / f"per_page_{method}.csv"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # 聚合 + 统一对比表
    agg_msirost = _aggregate(rows_msirost) if rows_msirost else {"overall": _empty_overall()}
    agg_decimer = _aggregate(rows_decimer) if rows_decimer else {"overall": _empty_overall()}
    table = _build_comparison_table(agg_msirost, agg_decimer)
    _write_comparison(table, out_dir)

    print("\n" + "=" * 90)
    print(" | ".join(COMPARE_COLUMNS))
    print("-" * 90)
    for r in table:
        print(" | ".join(str(r[c]) for c in COMPARE_COLUMNS))
    print("=" * 90)
    print(f"Errors (skipped): {len(errors)}")
    print(f"→ comparison.csv: {out_dir / 'comparison.csv'}")
    print(f"→ comparison.md : {out_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
