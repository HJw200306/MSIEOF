"""主评估脚本：遍历 restored/<subset>/<office>/*.png，对每张页面：

    1) 构建 GT（build_gt.build_gt_for_page）
    2) 跑 msirost 完整流水线（predict.predict_for_page → build_page_pred_masks）
    3) 1 对 1 匹配 + Recall≥阈值 判定（score.score_page）
    4) 保存 4-in-1 可视化 (visualize.save_page_overlay)

产物落盘
--------
  results/per_page.csv      每页一行的明细
  results/per_instance.csv  每 GT 分子一行（供分子级 <0.90 筛选）
  results/summary.json      全局聚合 + 每 subset/office 分组
  results/summary.md        人读版本
  results/viz/<stem>.png    每页可视化
  _crops_tmp/<stem>/*.png   extract 的 crop 中间产物（可以事后删）

命令行
------
  python -m dataset_eval.eval_msirost.main --limit 10
  python -m dataset_eval.eval_msirost.main --subsets benchmark_D2C-RND --offices jp
  python -m dataset_eval.eval_msirost.main            # 全量（449 页）

（cwd 应该在 msirost 项目根，即 C:\\Users\\Admin\\PycharmProjects\\msirost）
"""

from __future__ import annotations

# 允许 PyCharm 右键 Run 直接运行本文件：如果 __package__ 未设置（= 被当脚本跑），
# 就把 msirost 项目根塞进 sys.path 并显式设置 __package__，让 `from .xxx import ...`
# 能解析。python -m dataset_eval.eval_msirost.main 时 __package__ 已经是包名，这段
# 是 no-op。
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    __package__ = "dataset_eval.eval_msirost"

import argparse
import csv
import json
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2

from . import build_gt, predict, score, visualize
from .config import (
    MYVAL_CROPS_TMP_DIR,
    MYVAL_DATA_ROOT,
    MYVAL_RESULTS_DIR,
    MYVAL_VIZ_DIR,
    OFFICES,
    PATCID_RESTORED_ROOT,
    RECALL_THRESHOLD,
    RESULTS_DIR,
    SUBSETS,
    VIZ_DIR,
)


def _iter_pages(
    subsets: Iterable[str],
    offices: Iterable[str],
) -> Iterable[Tuple[str, str, Path, Path, "Path | None"]]:
    """yield (subset, office, page_png, mask_png, json_path)

    json_path=None → build_gt 回退到 sibling <stem>.json（PatCID 布局）。
    """
    for subset in subsets:
        for office in offices:
            d = PATCID_RESTORED_ROOT / subset / office
            if not d.exists():
                continue
            for png in sorted(d.glob("*.png")):
                if png.name.endswith(".mask.png") or png.name.endswith(".vis.png"):
                    continue
                mask = png.with_suffix(".mask.png")
                if not mask.exists():
                    continue
                yield subset, office, png, mask, None


def _myval_office(stem: str) -> str:
    """自己验证集的分组标签：专利号(2 大写字母 + 数字，如 CN102863967B) → 'patent'，
    其余（期刊，如 EJMC_2024_263_...） → 'journal'。仅用于分组汇报。"""
    import re
    return "patent" if re.match(r"^[A-Z]{2}\d", stem) else "journal"


def _iter_myval_pages(
    data_root: Path,
    split: str,
) -> Iterable[Tuple[str, str, Path, Path, "Path | None"]]:
    """yield (subset, office, page_png, mask_png, json_path) —— 自己验证集布局。

    结构：<root>/images/<split>/*.png、<root>/json/<split>/*.json、
          <root>/masks/<split>/*.png（json 缺失时的回退源）。
    subset 固定为 split（如 'val'），office 按 patent/journal 分组。
    """
    img_dir = data_root / "images" / split
    json_dir = data_root / "json" / split
    mask_dir = data_root / "masks" / split
    if not img_dir.exists():
        raise FileNotFoundError(img_dir)
    for png in sorted(img_dir.glob("*.png")):
        stem = png.stem
        jpath = json_dir / f"{stem}.json"
        mpath = mask_dir / f"{stem}.png"
        yield split, _myval_office(stem), png, mpath, (jpath if jpath.exists() else None)


def _fmt_row(subset: str, office: str, stem: str,
             page_hw: Tuple[int, int], s: Dict, elapsed: float) -> Dict:
    return {
        "subset": subset,
        "office": office,
        "stem": stem,
        "page_h": page_hw[0],
        "page_w": page_hw[1],
        "n_gt": s["n_gt"],
        "n_pred": s["n_pred"],
        "n_complete": s["n_complete"],
        "n_partial": s["n_partial"],
        "n_missed": s["n_missed"],
        "n_false_positive": s["n_false_positive"],
        "coverage_rate": round(s["coverage_rate"], 4),
        "elapsed_s": round(elapsed, 2),
    }


def _aggregate(rows: List[Dict]) -> Dict:
    """全局 + 分组聚合。分组指标是"合并统计口径"（sum of n_complete / sum of n_gt），
    不是"页平均"，这样避免 GT=0 的页面拉低均值。"""
    def _agg(rs: List[Dict]) -> Dict:
        tg = sum(r["n_gt"] for r in rs)
        tc = sum(r["n_complete"] for r in rs)
        tp = sum(r["n_partial"] for r in rs)
        tm = sum(r["n_missed"] for r in rs)
        tf = sum(r["n_false_positive"] for r in rs)
        tpr = sum(r["n_pred"] for r in rs)
        return dict(
            n_pages=len(rs),
            total_gt=tg,
            total_pred=tpr,
            total_complete=tc,
            total_partial=tp,
            total_missed=tm,
            total_false_positive=tf,
            coverage_rate=(tc / tg) if tg > 0 else None,
        )

    grouped: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for subset in sorted({r["subset"] for r in rows}):
        for office in sorted({r["office"] for r in rows if r["subset"] == subset}):
            sub = [r for r in rows if r["subset"] == subset and r["office"] == office]
            grouped[subset][office] = _agg(sub)

    return {
        "threshold": RECALL_THRESHOLD,
        "overall": _agg(rows),
        "by_subset_office": grouped,
    }


def _write_summary_md(summary: Dict, out_path: Path, n_errors: int) -> None:
    thr = summary["threshold"]
    ov = summary["overall"]
    lines = [
        f"# msirost 在 PatCID benchmark 上的完整检出率评估",
        "",
        f"- 判定口径：一个 GT 分子被判定为**完整检出** ⇔ "
        f"匈牙利 1对1 匹配后，其匹配到的预测像素对该 GT 的覆盖率 "
        f"`|P∩G|/|G|` **≥ {thr}**。",
        f"- 前景合并策略：文字标签（原子标签 / 编号 / 取代基文字）合并进各自分子。",
        f"- 预测来源：extract.py（YOLOv11-seg + 洪水填充）+ post_segment.py（UNet++ 二分类过滤）。",
        f"- 评估失败的页面（异常跳过）：{n_errors}",
        "",
        "## Overall",
        "",
        f"| pages | GT | Pred | Complete | Partial | Missed | FP | Coverage |",
        f"|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| {ov['n_pages']} | {ov['total_gt']} | {ov['total_pred']} | "
        f"{ov['total_complete']} | {ov['total_partial']} | {ov['total_missed']} | "
        f"{ov['total_false_positive']} | "
        f"{('%.2f%%' % (ov['coverage_rate']*100)) if ov['coverage_rate'] is not None else 'n/a'} |",
        "",
        "## By subset × office",
        "",
        f"| subset | office | pages | GT | Pred | Complete | Partial | Missed | FP | Coverage |",
        f"|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for subset, offices in summary["by_subset_office"].items():
        for office, a in offices.items():
            cov = ('%.2f%%' % (a['coverage_rate']*100)) if a['coverage_rate'] is not None else 'n/a'
            lines.append(
                f"| {subset} | {office} | {a['n_pages']} | {a['total_gt']} | "
                f"{a['total_pred']} | {a['total_complete']} | {a['total_partial']} | "
                f"{a['total_missed']} | {a['total_false_positive']} | {cov} |"
            )
    lines.append("")
    lines.append("**Coverage** = total_complete / total_gt（合并口径，不是页平均）。")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ap.add_argument("--offices", nargs="+", default=list(OFFICES))
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 页，0=不限")
    ap.add_argument("--no-viz", action="store_true", help="跳过每页可视化")
    ap.add_argument("--offset", type=int, default=0, help="跳过前 N 页")
    ap.add_argument(
        "--data-root", type=str, default=str(MYVAL_DATA_ROOT),
        help="自己验证集根目录（含 images/json/masks 各 split 子目录）。"
             "默认 = MYVAL_DATA_ROOT，即直接 Run 就跑 my-val 模式：启用颜色感知前景"
             "提取，输出到 MYVAL_EVAL_ROOT，不覆盖 PatCID 结果。"
             "想跑原 PatCID 流程时显式传 --data-root \"\"（空串）走灰度口径。",
    )
    ap.add_argument(
        "--split", type=str, default="val",
        help="my-val 模式下的 split 子目录名（val/test/train），默认 val。",
    )
    args = ap.parse_args()

    myval_mode = bool(args.data_root)
    color_aware = myval_mode
    if myval_mode:
        data_root = Path(args.data_root) if args.data_root else MYVAL_DATA_ROOT
        results_dir, viz_dir, crops_root = (
            MYVAL_RESULTS_DIR, MYVAL_VIZ_DIR, MYVAL_CROPS_TMP_DIR,
        )
    else:
        results_dir, viz_dir, crops_root = RESULTS_DIR, VIZ_DIR, None

    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    if myval_mode:
        todo = list(_iter_myval_pages(data_root, args.split))
    else:
        todo = list(_iter_pages(args.subsets, args.offices))
    todo = todo[args.offset:]
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"[eval] {len(todo)} pages to process, threshold={RECALL_THRESHOLD}, "
        f"mode={'my-val (color-aware)' if myval_mode else 'PatCID (grayscale)'}"
    )

    per_page_rows: List[Dict] = []
    per_inst_rows: List[Dict] = []
    errors: List[Dict] = []
    for idx, (subset, office, page_png, mask_png, json_path) in enumerate(todo, 1):
        stem = page_png.stem
        page_stem_full = f"{subset}__{office}__{stem}"
        t0 = time.time()
        try:
            gt_insts, _gt_label = build_gt.build_gt_for_page(
                page_png, mask_png, color_aware=color_aware, json_path=json_path,
            )

            pred_records = predict.predict_for_page(
                page_png, page_stem_full,
                color_aware=color_aware, crops_root=crops_root,
            )
            page_bgr = cv2.imread(str(page_png), cv2.IMREAD_COLOR)
            H, W = page_bgr.shape[:2]
            pred_insts = predict.build_page_pred_masks(pred_records, (H, W))

            s = score.score_page(gt_insts, pred_insts)

            for gj, rec_val in enumerate(s["gt_recall_per_inst"]):
                if rec_val is None:
                    status = "missed"
                elif rec_val >= RECALL_THRESHOLD:
                    status = "complete"
                else:
                    status = "partial"
                gx1, gy1, gx2, gy2 = gt_insts[gj]["bbox"]
                per_inst_rows.append({
                    "subset": subset,
                    "office": office,
                    "stem": stem,
                    "gt_idx": gj,
                    "recall": round(rec_val, 4) if rec_val is not None else "",
                    "status": status,
                    "gt_area": gt_insts[gj]["area"],
                    "gt_x1": gx1, "gt_y1": gy1, "gt_x2": gx2, "gt_y2": gy2,
                })

            if not args.no_viz:
                import cv2 as _cv
                page_rgb = _cv.cvtColor(page_bgr, _cv.COLOR_BGR2RGB)
                visualize.save_page_overlay(
                    page_rgb=page_rgb,
                    gt_instances=gt_insts,
                    pred_instances=pred_insts,
                    score=s,
                    out_path=viz_dir / f"{page_stem_full}.overlay.png",
                    page_stem=page_stem_full,
                )

            elapsed = time.time() - t0
            row = _fmt_row(subset, office, stem, (H, W), s, elapsed)
            per_page_rows.append(row)
            print(
                f"[{idx}/{len(todo)}] {page_stem_full}  "
                f"GT={s['n_gt']} Pred={s['n_pred']} "
                f"Complete={s['n_complete']} Missed={s['n_missed']} "
                f"Cov={s['coverage_rate']*100:.1f}%  ({elapsed:.1f}s)"
            )
        except Exception as e:
            elapsed = time.time() - t0
            errors.append({
                "subset": subset, "office": office, "stem": stem,
                "error": str(e),
                "trace": traceback.format_exc(limit=5),
            })
            print(f"[{idx}/{len(todo)}] ERR {page_stem_full}: {e}")
        finally:
            predict.free_gpu_memory()

    csv_path = results_dir / "per_page.csv"
    if per_page_rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_page_rows[0].keys()))
            w.writeheader()
            w.writerows(per_page_rows)

    inst_csv_path = results_dir / "per_instance.csv"
    if per_inst_rows:
        with inst_csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_inst_rows[0].keys()))
            w.writeheader()
            w.writerows(per_inst_rows)

    summary = _aggregate(per_page_rows)
    (results_dir / "summary.json").write_text(
        json.dumps({"summary": summary, "n_errors": len(errors), "errors": errors[:20]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_summary_md(summary, results_dir / "summary.md", n_errors=len(errors))

    print("\n" + "=" * 60)
    ov = summary["overall"]
    cov_str = f"{ov['coverage_rate']*100:.2f}%" if ov['coverage_rate'] is not None else 'n/a'
    print(f"pages processed : {ov['n_pages']}")
    print(f"total GT / Pred : {ov['total_gt']} / {ov['total_pred']}")
    print(f"Complete        : {ov['total_complete']}")
    print(f"Partial         : {ov['total_partial']}")
    print(f"Missed          : {ov['total_missed']}")
    print(f"False positive  : {ov['total_false_positive']}")
    print(f"Coverage rate   : {cov_str}   (thr={RECALL_THRESHOLD})")
    print(f"Errors (skipped): {len(errors)}")
    print(f"→ CSV     : {csv_path}")
    print(f"→ Summary : {results_dir / 'summary.md'}")
    print(f"→ Viz     : {viz_dir}")


if __name__ == "__main__":
    main()
