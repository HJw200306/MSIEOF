"""Step-1 diagnostic: where does the 3-class model put the colored atom labels?

Goal: figure out whether the red O / blue N / etc. atom labels are classified as
"bond" (class 2, green) or "background" (class 0, black) by the solid-arrow model
``model/legacy_arrow_models/misp_weights_no_freeze.pth``.  This decides whether a protect-mask built
from the 3-class model's "bond" channel will actually cover the atoms we must NOT
erase, which is the whole premise of the dashed-arrow eraser.

It only READS the model and WRITES panels to ``val_results/diag/``.  No existing
pipeline is touched.

Usage:
    python diagnose_masks.py                        # the two example images
    python diagnose_masks.py --all                  # every image in val/images
    python diagnose_masks.py --images a.png b.png   # explicit list
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_AE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _AE_ROOT.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from split_tools.unet import Unet  # noqa: E402

MODEL_PATH = _PROJECT_ROOT / "model" / "misp_weights_no_freeze.pth"
IMG_DIR = _AE_ROOT / "dataset" / "val" / "images"
OUT_DIR = _AE_ROOT / "val_results" / "diag"

# class index -> name and BGR color (cv2.imwrite uses BGR).
CLASS_NAMES = {0: "background", 1: "arrow", 2: "bond"}
CLASS_BGR = {0: (0, 0, 0), 1: (0, 0, 255), 2: (0, 255, 0)}

DEFAULT_IMAGES = [
    "journal_0001.png",
    "New_ent_-kaurane_diterpenoid_acids_from_Nouelia_insignis_Franch_and_their_anti-inflammator_005_01.png",
]


def imread_robust(path: Path) -> np.ndarray:
    """Read as RGB ndarray; cv2.imdecode(np.fromfile) survives non-ASCII paths."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def classify(model: Unet, img_rgb: np.ndarray) -> np.ndarray:
    """Run the 3-class model, return argmax class map (H, W) uint8 in {0,1,2}."""
    cls = np.array(model.get_miou_png(Image.fromarray(img_rgb)))
    return cls


def colored_mask(img_rgb: np.ndarray, s_thr: int = 40, v_thr: int = 80) -> np.ndarray:
    """Pixels that are visibly colored (colored atom labels / arrow ink), not
    black ink (low V) and not white bg (low S)."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    return (hsv[..., 1] > s_thr) & (hsv[..., 2] > v_thr)


def class_map_bgr(cls: np.ndarray) -> np.ndarray:
    out = np.zeros((*cls.shape, 3), np.uint8)
    for c, bgr in CLASS_BGR.items():
        out[cls == c] = bgr
    return out


def overlay_bgr(img_rgb: np.ndarray, cls: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    out = img_rgb.copy()
    for c, bgr in CLASS_BGR.items():
        if c == 0:
            continue
        m = cls == c
        out[m] = (out[m] * (1 - alpha) + np.array(bgr, np.uint8) * alpha).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def colored_highlight_bgr(img_rgb: np.ndarray, cmask: np.ndarray) -> np.ndarray:
    """Grayscale original with colored pixels painted cyan (BGR 255,255,0)."""
    gray = cv2.cvtColor(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2BGR)
    out = gray.copy()
    out[cmask] = (255, 255, 0)
    return out


def stats_text(img_rgb: np.ndarray, cls: np.ndarray) -> str:
    cmask = colored_mask(img_rgb)
    n = int(cmask.sum())
    lines = [f"colored px: {n}"]
    if n:
        for c in (0, 1, 2):
            k = int((cmask & (cls == c)).sum())
            lines.append(f"  in {CLASS_NAMES[c]:10s}: {k:6d}  ({100 * k / n:.1f}%)")
    return "\n".join(lines)


def build_composite(img_rgb: np.ndarray, cls: np.ndarray, stem: str) -> np.ndarray:
    cmask = colored_mask(img_rgb)
    p1 = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    p2 = class_map_bgr(cls)
    p3 = overlay_bgr(img_rgb, cls)
    p4 = colored_highlight_bgr(img_rgb, cmask)

    h = p1.shape[0]
    div = np.full((h, 3, 3), 80, np.uint8)
    row = np.hstack([p1, div, p2, div, p3, div, p4])

    labels = ["original", "3-class map  (green=bond, red=arrow)",
              "overlay", "colored atoms (cyan)"]
    seg = row.shape[1] // 4
    title = np.full((24, row.shape[1], 3), 0, np.uint8)
    for i, lab in enumerate(labels):
        cv2.putText(title, lab, (i * seg + 6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)

    txt = stats_text(img_rgb, cls)
    bar = np.full((24 + 20 * txt.count("\n"), row.shape[1], 3), 0, np.uint8)
    for i, line in enumerate(txt.split("\n")):
        cv2.putText(bar, line, (10, 22 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1, cv2.LINE_AA)

    return np.vstack([title, row, bar])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="run on every val image")
    ap.add_argument("--images", nargs="*", default=None, help="explicit image filenames")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.images:
        names = args.images
    elif args.all:
        names = sorted(p.name for p in IMG_DIR.glob("*")
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                       and not p.name.endswith("_struct.png"))
    else:
        names = DEFAULT_IMAGES

    print(f"Loading 3-class model from {MODEL_PATH} ...")
    model = Unet(str(MODEL_PATH), num_classes=3, input_shape=[512, 512])
    print("Model loaded.\n")

    for name in names:
        path = IMG_DIR / name
        if not path.exists():
            print(f"  !! missing: {name}")
            continue
        img_rgb = imread_robust(path)
        cls = classify(model, img_rgb)
        composite = build_composite(img_rgb, cls, Path(name).stem)
        out = OUT_DIR / f"{Path(name).stem}_diag.png"
        cv2.imwrite(str(out), composite)
        print(f"[{name}]")
        print(stats_text(img_rgb, cls))
        print(f"    -> {out}\n")

    print(f"Done. Open {OUT_DIR} to inspect.")


if __name__ == "__main__":
    main()
