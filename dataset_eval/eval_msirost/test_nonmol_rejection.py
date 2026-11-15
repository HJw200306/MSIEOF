r"""直接裁 json 里的 non_mol 矩形（3D模型/表格/装置图等真负样本），测两筛选器的剔除能力。

背景：extract 的 YOLO `classes=[0]` 在干净的 test 集上几乎不产生假阳性 crop，导致
compare_filters 里的负类样本数只有 8 且全是匹配口径噪声。这里绕过 YOLO，直接把 GT 里
标记为 non_mol 的区域从原页裁出来，喂给两个筛选器，看它们各自能否正确判定"不是 2D 分子"。

口径
----
  UNet++ ：post_segment._compute_fg_ratio（无 sidecar → 整图前景占比 |UNet++ fg|/H·W），
           阈值 FG_RATIO_THRESHOLD=0.01，低于阈值 → 判为非分子（剔除）。
  EffNet ：sigmoid 概率，阈值 = checkpoint 存的 threshold，低于阈值 → 剔除。

用法（cwd = msirost 项目根）
------
  C:/FastEnv/msirost/python.exe -m dataset_eval.eval_msirost.test_nonmol_rejection --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if __package__ in (None, ""):
    sys.path.insert(0, str(_PROJECT_ROOT))
    __package__ = "dataset_eval.eval_msirost"

from dataset_eval.eval_msirost.config import MYVAL_DATA_ROOT  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torchvision import transforms  # noqa: E402
from torchvision.models import efficientnet_b0  # noqa: E402

from utils import post_segment as _post_segment  # noqa: E402

CKPT_PATH = _PROJECT_ROOT / "model" / "classer_best.pt"
IMAGE_SIZE = 320
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PAD = 10  # 裁 non_mol 矩形时的外扩像素


class _ResizePad:
    def __init__(self, size: int, fill=(255, 255, 255)):
        self.size = int(size); self.fill = fill

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


def _load_effnet(device):
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, float(ckpt["threshold"])


@torch.inference_mode()
def _effnet_prob(model, transform, image, device) -> float:
    t = transform(image).unsqueeze(0).to(device)
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.float16):
        return float(torch.sigmoid(model(t).flatten()).item())


def _shape_bbox(shape) -> tuple:
    pts = shape.get("points") or []
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    effnet, effnet_thr = _load_effnet(device)
    tf = transforms.Compose([_ResizePad(IMAGE_SIZE), transforms.ToTensor(),
                             transforms.Normalize(mean=MEAN, std=STD)])
    unet_model = _post_segment._load_model(device)
    print(f"[loaded] effnet thr={effnet_thr:.4f}  unet thr={_post_segment.FG_RATIO_THRESHOLD}")

    root = MYVAL_DATA_ROOT
    img_dir = root / "images" / args.split
    json_dir = root / "json" / args.split

    # 收集 (page_stem, x1, y1, x2, y2)
    samples: List[dict] = []
    for jpath in sorted(json_dir.glob("*.json")):
        d = json.loads(jpath.read_text(encoding="utf-8"))
        for s in d.get("shapes", []):
            if s.get("label") != "non_mol":
                continue
            samples.append({"stem": jpath.stem, "bbox": _shape_bbox(s)})

    if not samples:
        print("no non_mol found"); return

    tmp = _PROJECT_ROOT / "outputs" / "compare_filters" / "_nonmol_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, sm in enumerate(samples):
        stem = sm["stem"]
        x1, y1, x2, y2 = sm["bbox"]
        page = Image.open(img_dir / f"{stem}.png").convert("RGB")
        W, H = page.size
        cx1 = max(0, int(x1) - PAD); cy1 = max(0, int(y1) - PAD)
        cx2 = min(W, int(x2) + PAD); cy2 = min(H, int(y2) + PAD)
        crop = page.crop((cx1, cy1, cx2, cy2))
        cpath = tmp / f"{stem}__nm{i:02d}.png"
        crop.save(cpath)

        # UNet++ (whole-image fg_ratio，无 sidecar)
        fg_ratio, _, _, _ = _post_segment._compute_fg_ratio(unet_model, cpath, device)
        unet_keep = fg_ratio >= _post_segment.FG_RATIO_THRESHOLD
        # EffNet
        prob = _effnet_prob(effnet, tf, crop, device)
        effnet_keep = prob >= effnet_thr
        rows.append({
            "stem": stem, "bbox_w": int(x2 - x1), "bbox_h": int(y2 - y1),
            "unet_fg_ratio": round(fg_ratio, 4), "unet_keep": int(unet_keep),
            "effnet_prob": round(prob, 4), "effnet_keep": int(effnet_keep),
        })

    n = len(rows)
    unet_rej = sum(1 for r in rows if not r["unet_keep"])
    eff_rej = sum(1 for r in rows if not r["effnet_keep"])
    unet_fg = np.array([r["unet_fg_ratio"] for r in rows])
    eff_p = np.array([r["effnet_prob"] for r in rows])

    print("\n" + "=" * 62)
    print(f"non_mol 真负样本: {n} 个 (split={args.split})")
    print("=" * 62)
    print(f"\n{'样本':<28} {'bbox':>12} {'unet_fg':>9} {'unet':>6} {'effnet_p':>9} {'effnet':>7}")
    for r in sorted(rows, key=lambda x: x["effnet_prob"], reverse=True):
        print(f"{r['stem']:<28} {('%dx%d' % (r['bbox_w'], r['bbox_h'])):>12} "
              f"{r['unet_fg_ratio']:>9.4f} {'KEEP' if r['unet_keep'] else 'DROP':>6} "
              f"{r['effnet_prob']:>9.4f} {'KEEP' if r['effnet_keep'] else 'DROP':>7}")
    print("\n--- 汇总（正确剔除率 = 判为非分子的比例） ---")
    print(f"UNet++  (thr={_post_segment.FG_RATIO_THRESHOLD}): 剔除 {unet_rej}/{n} = {unet_rej/n*100:.1f}%"
          f"   (fg_ratio median={np.median(unet_fg):.4f})")
    print(f"EffNet  (thr={effnet_thr:.4f}):         剔除 {eff_rej}/{n} = {eff_rej/n*100:.1f}%"
          f"   (prob median={np.median(eff_p):.4f})")
    print(f"\n裁图目录: {tmp}")


if __name__ == "__main__":
    main()
