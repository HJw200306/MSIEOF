"""probsheet.py idx [idx...] -> 可疑样本接触表 (原图 | raw叠加) 便于视觉判定。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_overlay as R  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "probsheets"
OUT.mkdir(parents=True, exist_ok=True)


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def cell(i, w):
    img = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    W, H = img.size
    ov = R.draw_overlay(img, ann, 1, normalized=True)   # 原始像素尺度叠加
    z = max(1, round(w / max(W, 1)))
    a = img.resize((W * z, H * z), Image.LANCZOS)
    b = ov.resize((W * z, H * z), Image.LANCZOS)
    gap = 6
    cv = Image.new("RGB", (a.width + b.width + gap, a.height + 30), (255, 255, 255))
    cv.paste(a, (0, 30))
    cv.paste(b, (a.width + gap, 30))
    dr = ImageDraw.Draw(cv)
    dr.text((2, 6), f"idx {i}  {W}x{H}", fill=(200, 0, 0), font=font(18))
    return cv


def build(idxs, cols, w):
    cells = [cell(i, w) for i in idxs]
    rowh = max(c.height for c in cells)
    roww = max(c.width for c in cells)
    rows = (len(cells) + cols - 1) // cols
    pad = 10
    cv = Image.new("RGB", (cols * (roww + pad) + pad, rows * (rowh + pad) + pad), (245, 245, 245))
    for t, c in enumerate(cells):
        r, k = divmod(t, cols)
        cv.paste(c, (pad + k * (roww + pad), pad + r * (rowh + pad)))
    out = OUT / f"prob_{'_'.join(str(i) for i in idxs[:3])}_{len(idxs)}.png"
    cv.save(out)
    print(out, cv.size)


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    build(args, 3, 380)
