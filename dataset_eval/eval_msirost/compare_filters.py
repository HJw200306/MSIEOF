r"""对比「UNet++ fg_ratio 门控」与「EfficientNet-B0 二分类器」在候选 crop 上的筛选效果。

口径
----
在同一批 extract（YOLOv11-seg + 洪水填充）产出的候选 crop 上，对每个 crop：
  - GT 标签：该 crop 是否命中了某个 **mol** GT 标注（label == 'mol' 的 polygon/rectangle）。
      用 score.score_page 的匈牙利 1 对 1 匹配，matched 的 crop = 真分子(1)，
      未 matched 的 crop = 假阳性(0)。注意 build_gt 默认会把 non_mol 矩形也当 GT，
      这里先把 json 过滤到只留 mol 再喂 build_gt，避免把 3D/表格标注误当真分子。
  - UNet++ 分数 = fg_ratio（越大越像 2D 分子），阈值 FG_RATIO_THRESHOLD=0.01。
  - EffNet 分数 = sigmoid 概率（越大越像 2D 分子），阈值 = checkpoint 里存的 threshold。

输出
----
  outputs/compare_filters/
    summary.json          每档阈值的混淆矩阵 + ROC/PR-AUC + 召回目标档位
    per_crop.csv          逐 crop 的 (unet_fg_ratio, unet_kept, effnet_prob, effnet_kept, gt_label, crop_path)
    disagreement.md       两筛选器判定不一致的样本清单（供视觉复核）

用法（cwd = msirost 项目根）
------
  C:/FastEnv/msirost/python.exe -m dataset_eval.eval_msirost.compare_filters --split test --limit 20
  ... --split test                     # 全 87 页
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if __package__ in (None, ""):
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "dataset_eval.eval_msirost"

from dataset_eval.eval_msirost import build_gt, predict, score  # noqa: E402
from dataset_eval.eval_msirost.config import (  # noqa: E402
    MYVAL_CROPS_TMP_DIR,
    MYVAL_DATA_ROOT,
    RECALL_THRESHOLD,
)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.models import efficientnet_b0  # noqa: E402
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve  # noqa: E402

# --- 参数 ---
CKPT_PATH = _PROJECT_ROOT / "model" / "classer_best.pt"
OUTPUT_DIR = _PROJECT_ROOT / "outputs" / "compare_filters"
IMAGE_SIZE = 320
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 临时过滤后的 json（只留 mol 标注），避免 build_gt 把 non_mol 矩形当真分子
_MOL_ONLY_JSON_DIR = MYVAL_CROPS_TMP_DIR / "_mol_only_json"


# ---------------------------------------------------------------------------
# EfficientNet-B0 加载 + 预处理（复刻训练侧 msieof_cls.data.build_transform 的测试分支）
# ---------------------------------------------------------------------------
class _ResizePad:
    def __init__(self, size: int, fill=(255, 255, 255)):
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        w, h = image.size
        scale = self.size / max(w, h)
        nw = max(1, min(self.size, round(w * scale)))
        nh = max(1, min(self.size, round(h * scale)))
        resized = image.resize((nw, nh), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - nw) // 2, (self.size - nh) // 2))
        return canvas


def _build_effnet_transform() -> transforms.Compose:
    return transforms.Compose([
        _ResizePad(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _load_effnet(device: torch.device):
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    threshold = float(ckpt["threshold"])
    return model, threshold


@torch.inference_mode()
def _effnet_prob(model: nn.Module, transform, image: Image.Image,
                 device: torch.device) -> float:
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        logit = model(tensor).flatten()
    return float(torch.sigmoid(logit).item())


# ---------------------------------------------------------------------------
# mol-only GT
# ---------------------------------------------------------------------------
def _mol_only_json(src_json: Path) -> Path:
    """读原始 LabelMe json，只保留 label == 'mol' 的 shape，写到临时目录。"""
    data = json.loads(src_json.read_text(encoding="utf-8"))
    data["shapes"] = [s for s in data.get("shapes", []) if s.get("label") == "mol"]
    _MOL_ONLY_JSON_DIR.mkdir(parents=True, exist_ok=True)
    out = _MOL_ONLY_JSON_DIR / src_json.name
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return out


def _iter_myval_pages(split: str) -> List[Tuple[Path, Path, Path]]:
    """yield (page_png, mask_png, mol_only_json)。结构与 main._iter_myval_pages 一致。"""
    root = MYVAL_DATA_ROOT
    img_dir = root / "images" / split
    json_dir = root / "json" / split
    mask_dir = root / "masks" / split
    pages = []
    for png in sorted(img_dir.glob("*.png")):
        stem = png.stem
        jpath = json_dir / f"{stem}.json"
        mpath = mask_dir / f"{stem}.png"
        if not jpath.exists():
            continue
        pages.append((png, mpath, _mol_only_json(jpath)))
    return pages


# ---------------------------------------------------------------------------
# 逐 crop 打分 + GT 标签
# ---------------------------------------------------------------------------
def _select_threshold_for_recall(labels, scores, target_recall: float) -> Dict:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    if np.unique(y_true).size != 2:
        return {"target": target_recall, "threshold": None, "neg_recall": None}
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    eligible = np.flatnonzero((tpr >= target_recall) & np.isfinite(thresholds))
    if eligible.size == 0:
        thr = float(np.nextafter(np.min(y_score), -np.inf))
    else:
        best_spec = np.max(1.0 - fpr[eligible])
        best = eligible[np.isclose(1.0 - fpr[eligible], best_spec)]
        thr = float(np.max(thresholds[best]))
    pred = (y_score >= thr).astype(np.int64)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    pos_recall = tp / (tp + fn) if (tp + fn) else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0.0
    return {"target": target_recall, "threshold": thr,
            "pos_recall": pos_recall, "neg_recall": neg_recall,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _confusion_at(labels, scores, thr: float) -> Dict:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(scores, dtype=np.float64)
    pred = (y_score >= thr).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    pos_recall = tp / (tp + fn) if (tp + fn) else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    return {"thr": thr, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "pos_recall": pos_recall, "neg_recall": neg_recall,
            "precision": precision, "n_pos": tp + fn, "n_neg": tn + fp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0=全部")
    ap.add_argument("--unet-threshold", type=float, default=0.01)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effnet_model, effnet_thr = _load_effnet(device)
    effnet_transform = _build_effnet_transform()
    print(f"[effnet] loaded {CKPT_PATH.name}  threshold={effnet_thr:.6f}  ({device})")

    pages = _iter_myval_pages(args.split)
    if args.limit:
        pages = pages[: args.limit]
    print(f"[compare] {len(pages)} pages (split={args.split}), "
          f"unet_thr={args.unet_threshold}, effnet_thr={effnet_thr:.4f}, "
          f"gt_match_recall_thr={RECALL_THRESHOLD}")

    # 汇总：每 crop 一条
    rows: List[Dict] = []
    n_gt_total = 0
    for idx, (page_png, mask_png, mol_json) in enumerate(pages, 1):
        stem = page_png.stem
        page_stem_full = f"myval__{args.split}__{stem}"

        gt_insts, _ = build_gt.build_gt_for_page(
            page_png, mask_png, color_aware=True, json_path=mol_json)
        n_gt_total += len(gt_insts)

        records = predict.predict_for_page(
            page_png, page_stem_full, color_aware=True,
            crops_root=MYVAL_CROPS_TMP_DIR)
        if not records:
            continue

        # 全部 crop 参与匹配（不按 kept 过滤），得到 matched crop 集合
        pred_insts_all = []
        for rec in records:
            x1, y1, x2, y2 = rec["bbox"]
            pred_insts_all.append({"bbox": (x1, y1, x2, y2),
                                   "mask": rec["mask"], "area": int(rec["mask"].sum())})
        s = score.score_page(gt_insts, pred_insts_all)
        matched_pred = {pi for _, pi, _ in s["matches"]}
        matched_recall = {pi: r for _, pi, r in s["matches"]}

        for i, rec in enumerate(records):
            is_mol = i in matched_pred
            gt_recall = matched_recall.get(i, 0.0)
            # EffNet 推理
            try:
                with Image.open(rec["crop_path"]) as im:
                    im.load()
                    prob = _effnet_prob(effnet_model, effnet_transform, im, device)
            except Exception:
                prob = float("nan")
            rows.append({
                "page": stem,
                "crop_idx": i,
                "gt_label": 1 if is_mol else 0,
                "gt_recall": round(float(gt_recall), 4),
                "unet_fg_ratio": float(rec["fg_ratio"]),
                "unet_kept": int(rec["kept"]),
                "effnet_prob": prob,
                "effnet_kept": int(prob >= effnet_thr) if prob == prob else 0,
                "crop_path": str(rec["crop_path"]),
            })
        predict.free_gpu_memory()
        if idx % 10 == 0 or idx == len(pages):
            print(f"  [{idx}/{len(pages)}] {stem}  GT={len(gt_insts)} crops={len(records)}")

    if not rows:
        print("no crops produced; abort")
        return

    labels = [r["gt_label"] for r in rows]
    unet_scores = [r["unet_fg_ratio"] for r in rows]
    effnet_scores = [r["effnet_prob"] for r in rows]

    # 去掉 effnet nan 的样本
    valid = [i for i in range(len(rows)) if effnet_scores[i] == effnet_scores[i]]
    labels_v = [labels[i] for i in valid]
    unet_v = [unet_scores[i] for i in valid]
    effnet_v = [effnet_scores[i] for i in valid]

    summary = {
        "split": args.split,
        "n_pages": len(pages),
        "n_crops": len(rows),
        "n_valid_crops": len(valid),
        "n_gt_total": n_gt_total,
        "n_pos": sum(labels_v),
        "n_neg": sum(1 for l in labels_v if l == 0),
        "unet_threshold": args.unet_threshold,
        "effnet_threshold": effnet_thr,
        "unet": {
            "confusion": _confusion_at(labels_v, unet_v, args.unet_threshold),
            "roc_auc": roc_auc_score(labels_v, unet_v) if len(set(labels_v)) == 2 else None,
            "average_precision": average_precision_score(labels_v, unet_v) if len(set(labels_v)) == 2 else None,
            "recall_targets": {f"r{int(t*1000)}": _select_threshold_for_recall(labels_v, unet_v, t)
                               for t in (0.98, 0.99, 0.995)},
        },
        "effnet": {
            "confusion": _confusion_at(labels_v, effnet_v, effnet_thr),
            "roc_auc": roc_auc_score(labels_v, effnet_v) if len(set(labels_v)) == 2 else None,
            "average_precision": average_precision_score(labels_v, effnet_v) if len(set(labels_v)) == 2 else None,
            "recall_targets": {f"r{int(t*1000)}": _select_threshold_for_recall(labels_v, effnet_v, t)
                               for t in (0.98, 0.99, 0.995)},
        },
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐 crop CSV
    import csv
    with (OUTPUT_DIR / "per_crop.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 分歧样本（UNet++ 与 EffNet 判定不一致）
    disc = [r for r in rows if r["unet_kept"] != r["effnet_kept"]]
    with (OUTPUT_DIR / "disagreement.md").open("w", encoding="utf-8") as f:
        f.write(f"# 两筛选器分歧样本（共 {len(disc)} / {len(rows)}）\n\n")
        f.write("| page | crop | gt | unet_fg_ratio | unet | effnet_prob | effnet | path |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in sorted(disc, key=lambda x: x["effnet_prob"]):
            gt = "MOL" if r["gt_label"] else "FP"
            f.write(f"| {r['page']} | {r['crop_idx']} | {gt} | {r['unet_fg_ratio']:.4f} "
                    f"| {'KEEP' if r['unet_kept'] else 'DROP'} | {r['effnet_prob']:.4f} "
                    f"| {'KEEP' if r['effnet_kept'] else 'DROP'} | {r['crop_path']} |\n")

    # 控制台摘要
    def _show(name, m):
        c = m["confusion"]
        auc = m["roc_auc"]
        ap_ = m["average_precision"]
        print(f"\n=== {name} ===")
        print(f"  混淆矩阵  TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}")
        print(f"  positive_recall (真分子保留率) = {c['pos_recall']*100:.2f}%")
        print(f"  negative_recall (假阳性剔除率) = {c['neg_recall']*100:.2f}%")
        print(f"  precision                      = {c['precision']*100:.2f}%")
        print(f"  ROC-AUC = {auc if auc is None else round(auc,4)}   "
              f"PR-AUC = {ap_ if ap_ is None else round(ap_,4)}")
        for k, v in m["recall_targets"].items():
            if v["threshold"] is None:
                print(f"  {k} (target pos_recall={v['target']}) -> 无负样本，跳过")
            else:
                print(f"  {k} (target pos_recall={v['target']}) -> thr={v['threshold']:.4f} "
                      f"pos_recall={v['pos_recall']*100:.2f}% neg_recall={v['neg_recall']*100:.2f}%")

    print("\n" + "=" * 60)
    print(f"split={args.split}  pages={len(pages)}  crops={len(rows)}  "
          f"pos={summary['n_pos']}  neg={summary['n_neg']}  gt_total={n_gt_total}")
    _show("UNet++ fg_ratio 门控", summary["unet"])
    _show("EfficientNet-B0 分类器", summary["effnet"])
    print(f"\n分歧样本: {len(disc)}  -> {OUTPUT_DIR / 'disagreement.md'}")
    print(f"详细: {OUTPUT_DIR / 'summary.json'}, {OUTPUT_DIR / 'per_crop.csv'}")


if __name__ == "__main__":
    main()
