"""verify_corr.py idx [idx...] -> 原图 | corrected 叠加(像素坐标) 并排图。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_overlay as R  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "verify"
OUT.mkdir(parents=True, exist_ok=True)


def one(i, w=380):
    img = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    W, H = img.size
    corr = json.load(open(D / "corrected_annotations" / f"{i:06d}.json", encoding="utf-8"))
    z = max(1, round(w / max(W, 1)))
    a = img.resize((W * z, H * z), Image.LANCZOS)
    b = R.draw_overlay(img, corr, z, normalized=False)
    gap = 6
    cv = Image.new("RGB", (a.width + b.width + gap, a.height + 34), (255, 255, 255))
    cv.paste(a, (0, 34))
    cv.paste(b, (a.width + gap, 34))
    dr = ImageDraw.Draw(cv)
    dr.text((2, 4), f"idx {i} {W}x{H} na={len(corr['atoms'])} nb={len(corr['bonds'])}", fill=(200, 0, 0), font=R.font(18))
    dr.text((2, 18), corr["esmiles"][:160], fill=(0, 0, 180), font=R.font(12))
    return cv


if __name__ == "__main__":
    idxs = [int(a) for a in sys.argv[1:]]
    cells = [one(i) for i in idxs]
    cols = 2
    rowh = max(c.height for c in cells)
    roww = max(c.width for c in cells)
    rows = (len(cells) + cols - 1) // cols
    pad = 10
    cv = Image.new("RGB", (cols * (roww + pad) + pad, rows * (rowh + pad) + pad), (240, 240, 240))
    for t, c in enumerate(cells):
        r, k = divmod(t, cols)
        cv.paste(c, (pad + k * (roww + pad), pad + r * (rowh + pad)))
    out = OUT / f"v_{'_'.join(str(i) for i in idxs[:4])}_{len(idxs)}.png"
    cv.save(out)
    print(out, cv.size)
