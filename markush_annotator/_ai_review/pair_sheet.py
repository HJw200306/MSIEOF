"""pair_sheet.py lo hi [cols] -> 把 ov_batch2/{idx}.png (原图|MolScribe图+编号) 拼成接触表。
用法: pair_sheet.py 202 220 [4]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
SRC = D / "_ai_review" / "ov_batch2"
OUT = D / "_ai_review" / "pair_sheets"
OUT.mkdir(parents=True, exist_ok=True)


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


lo, hi = int(sys.argv[1]), int(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 3
target_w = int(sys.argv[4]) if len(sys.argv) > 4 else 720   # 每格拉伸到该宽度(两联图)
pad, lab = 8, 24
items = [i for i in range(lo, hi + 1) if (SRC / f"{i:06d}.png").exists()]
scaled = []
for i in items:
    im = Image.open(SRC / f"{i:06d}.png").convert("RGB")
    sc = target_w / im.width
    scaled.append((i, im.resize((target_w, max(1, int(im.height * sc))), Image.LANCZOS)))
cellh = max(im.height for _, im in scaled)
rows = (len(scaled) + cols - 1) // cols
W = cols * (target_w + pad) + pad
H = rows * (cellh + pad + lab) + pad
cv = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(cv)
f = font(24)
for t, (i, im) in enumerate(scaled):
    r, c = divmod(t, cols)
    x = pad + c * (target_w + pad)
    y = pad + r * (cellh + pad + lab)
    dr.text((x, y + 2), f"idx {i}", fill=(200, 0, 0), font=f)
    cv.paste(im, (x, y + lab))
out = OUT / f"pair_{lo}_{hi}.png"
cv.save(out)
print(out, cv.size)
