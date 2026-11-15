r"""用原版 PatCID MolClassifier（实为 Mask R-CNN，三分类）在 per_crop.csv 的 1324 个
crop 上打分，推理时把 label 重映射成二值：Clean/Markush → 分子(1)，Trash → 非分子(0)。
不改模型、不重训。与 per_crop.csv 里已存的 EffNet / UNet++ 结果对比召回与剔除。

用法（cwd = msirost 项目根）：
  C:/FastEnv/msirost/python.exe -m dataset_eval.eval_msirost.molclass_eval
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT = _PROJECT_ROOT / "model" / "molclassifier_model.chpt"
OUT = _PROJECT_ROOT / "outputs" / "compare_filters"
PER_CROP = OUT / "per_crop.csv"
BATCH = 16

_MOL_KEYS = ("clean", "markush", "molecular", "mol")
_NONMOL_KEYS = ("trash", "background", "non")


def load_model(device):
    ckpt = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    classes = list(ckpt["classes"])
    sd = ckpt["model_state_dict"]
    num_classes = sd["roi_heads.box_predictor.cls_score.weight"].shape[0]
    model = maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None,
        box_nms_thresh=0.1, box_score_thresh=0.8, num_classes=num_classes)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, classes


def build_mapping(classes):
    """label -> 1(mol)/0(non_mol)。label 0 通常是 background，跳过。"""
    mapping = {}
    for i, name in enumerate(classes):
        n = str(name).lower()
        if i == 0 and ("background" in n or n == ""):
            continue
        if any(k in n for k in _NONMOL_KEYS):
            mapping[i] = 0
        elif any(k in n for k in _MOL_KEYS):
            mapping[i] = 1
        else:
            print(f"[warn] 未知类别 label={i} name={name!r} -> 默认按 non_mol(0)")
            mapping[i] = 0
    return mapping


@torch.inference_mode()
def predict_batch(model, images, device):
    tensors = [F.convert_image_dtype(F.pil_to_tensor(im)).to(device) for im in images]
    return model(tensors)


def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    pos_recall = tp / (tp + fn) if (tp + fn) else 0.0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0.0
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "pos_recall": pos_recall, "neg_recall": neg_recall}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_model(device)
    mapping = build_mapping(classes)
    print(f"[loaded] classes={classes}")
    print(f"[loaded] mapping={mapping}")

    with PER_CROP.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"[data] {len(rows)} crops")

    new_rows = []
    n_batches = (len(rows) + BATCH - 1) // BATCH
    for bi in range(n_batches):
        chunk = rows[bi * BATCH:(bi + 1) * BATCH]
        images, paths = [], []
        for r in chunk:
            im = Image.open(r["crop_path"]).convert("RGB")
            images.append(im)
            paths.append(r["crop_path"])
        outputs = predict_batch(model, images, device)
        for r, out in zip(chunk, outputs):
            labels = out["labels"].cpu().tolist()
            scores = out["scores"].cpu().tolist()
            if not labels:
                r["molclass_label"], r["molclass_score"], r["molclass_kept"] = "", 0.0, 0
            else:
                top = max(range(len(scores)), key=lambda i: scores[i])
                r["molclass_label"] = str(classes[labels[top]])
                r["molclass_score"] = round(float(scores[top]), 4)
                r["molclass_kept"] = int(mapping.get(labels[top], 0))
            new_rows.append(r)
        if (bi + 1) % 20 == 0 or (bi + 1) == n_batches:
            print(f"  [{bi + 1}/{n_batches}]")

    gt = [int(r["gt_label"]) for r in new_rows]
    mol_kept = [r["molclass_kept"] for r in new_rows]
    eff_kept = [r["effnet_kept"] for r in new_rows]
    unet_kept = [r["unet_kept"] for r in new_rows]

    # 写逐 crop 结果
    fields = list(new_rows[0].keys())
    with (OUT / "molclass_per_crop.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(new_rows)

    print("\n" + "=" * 60)
    print(f"1324 crops（1316 mol / 8 neg），真分子保留率 = pos_recall，假阳性剔除率 = neg_recall")
    print("=" * 60)
    for name, pred in [("MolClassifier (Clean/Markush→mol)", mol_kept),
                       ("EfficientNet-B0 (prob>=0.4924)", eff_kept),
                       ("UNet++ (fg_ratio>=0.01)", unet_kept)]:
        c = _confusion(gt, pred)
        print(f"\n{name}")
        print(f"  TP={c['tp']:5d}  TN={c['tn']:3d}  FP={c['fp']:3d}  FN={c['fn']:3d}")
        print(f"  真分子保留率 pos_recall = {c['pos_recall']*100:.2f}%")
        print(f"  假阳性剔除率 neg_recall = {c['neg_recall']*100:.2f}%")

    print(f"\n结果已写 -> {OUT / 'molclass_per_crop.csv'}")


if __name__ == "__main__":
    main()
