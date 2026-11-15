r"""给 compare_filters 的结果补可视化：

1. ROC / PR 曲线：UNet++ fg_ratio vs EfficientNet prob（在 1324 个 extract crop 上）。
2. 46 个被 EffNet 误删真分子的缩略拼图（每格标 page/mol# + effnet_prob + unet_fg）。
3. 2 个被 UNet++ 误保留的 non_mol 大图（对照）。

产物写到 outputs/compare_filters/viz/。

用法（cwd = msirost 项目根）：
  C:/FastEnv/msirost/python.exe -m dataset_eval.eval_msirost.visualize_compare
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE = _PROJECT_ROOT / "outputs" / "compare_filters"
VIZ = BASE / "viz"
VIZ.mkdir(parents=True, exist_ok=True)


def _load_rows():
    with (BASE / "per_crop.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_roc_pr(rows):
    labels = [int(r["gt_label"]) for r in rows]
    unet = np.array([float(r["unet_fg_ratio"]) for r in rows])
    effnet = np.array([float(r["effnet_prob"]) for r in rows])

    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    # ROC
    for name, score, color in [("UNet++ (fg_ratio)", unet, "#1f77b4"),
                               ("EfficientNet-B0 (prob)", effnet, "#d62728")]:
        fpr, tpr, _ = roc_curve(labels, score)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f"{name}  AUC={auc(fpr, tpr):.4f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC  (1324 crops, 1316 mol / 8 neg)")
    axes[0].legend(loc="lower right")

    # PR
    for name, score, color in [("UNet++ (fg_ratio)", unet, "#1f77b4"),
                               ("EfficientNet-B0 (prob)", effnet, "#d62728")]:
        prec, rec, _ = precision_recall_curve(labels, score)
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f"{name}  AP={average_precision_score(labels, score):.4f}")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(loc="upper right")
    axes[1].set_xlim(0, 1.01); axes[1].set_ylim(0.9, 1.005)

    fig.tight_layout()
    out = VIZ / "roc_pr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] ROC/PR -> {out}")


def _montage(fn_rows, out_path: Path, cols: int = 8, thumb: int = 260):
    """把 fn_rows 里的 crop 拼成网格，每格标题写 page/mol# + prob + fg。"""
    n = len(fn_rows)
    rows_n = math.ceil(n / cols)
    pad = 6
    title_h = 34
    cell = thumb + title_h
    canvas = Image.new("RGB", (cols * (thumb + pad) + pad,
                               rows_n * (cell + pad) + pad), (245, 245, 245))

    from PIL import ImageDraw
    draw = ImageDraw.Draw(canvas)
    for i, r in enumerate(fn_rows):
        row, col = divmod(i, cols)
        x = pad + col * (thumb + pad)
        y = pad + row * (cell + pad)
        try:
            im = Image.open(r["crop_path"]).convert("RGB")
            im.thumbnail((thumb, thumb), Image.Resampling.LANCZOS)
        except Exception:
            im = Image.new("RGB", (thumb, thumb), (200, 200, 200))
        ox = x + (thumb - im.width) // 2
        oy = y + (thumb - im.height) // 2
        canvas.paste(im, (ox, oy))
        draw.rectangle([x, y, x + thumb - 1, y + thumb - 1], outline=(120, 120, 120), width=1)
        label = f"{r['page'][-16:]} mol{r['crop_idx']}"
        draw.text((x + 3, y + thumb + 4), label, fill=(20, 20, 20))
        sub = f"p={float(r['effnet_prob']):.3f} fg={float(r['unet_fg_ratio']):.2f}"
        draw.text((x + 3, y + thumb + 18), sub, fill=(180, 40, 40))
    canvas.save(out_path)
    print(f"[viz] montage ({n} imgs) -> {out_path}")


def main():
    rows = _load_rows()
    plot_roc_pr(rows)

    fn = sorted([r for r in rows if r["gt_label"] == "1" and r["effnet_kept"] == "0"],
                key=lambda x: float(x["effnet_prob"]))
    _montage(fn, VIZ / "effnet_false_negatives_46.png")

    # 被 UNet++ 误保留的 2 个 non_mol（从 test_nonmol_rejection 的裁图取）
    nonmol_dir = BASE / "_nonmol_tmp"
    unet_missed = []
    if nonmol_dir.exists():
        for p in sorted(nonmol_dir.glob("*.png")):
            unet_missed.append({"crop_path": str(p), "page": p.stem, "crop_idx": "",
                                "effnet_prob": float("nan"), "unet_fg_ratio": float("nan")})
    # 只挑 fg_ratio 高（被 UNet++ KEEP）的不好筛，这里直接拼全部 non_mol 供参考
    print(f"[viz] non_mol 裁图目录（供人工复核）: {nonmol_dir}")


if __name__ == "__main__":
    main()
