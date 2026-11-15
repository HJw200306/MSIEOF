"""sheet_ov.py lo hi [cols] [cell] -> 把 overlays_corrected/{idx}.png 拼成接触表核对波浪线。"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\_ai_review\overlays_corrected")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


lo, hi = int(sys.argv[1]), int(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 2
cell = int(sys.argv[4]) if len(sys.argv) > 4 else 900
pad, lab = 10, 22
items = [i for i in range(lo, hi + 1) if (D / f"{i:06d}.png").exists()]
rows = (len(items) + cols - 1) // cols
W = cols * (cell + pad) + pad
H = rows * (int(cell * 0.72) + pad + lab) + pad
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
f = font(20)
rowh = int(cell * 0.72)
for t, i in enumerate(items):
    im = Image.open(D / f"{i:06d}.png").convert("RGB")
    sc = cell / im.width
    im = im.resize((cell, max(1, int(im.height * sc))), Image.LANCZOS)
    r, c = divmod(t, cols)
    x = pad + c * (cell + pad)
    y = pad + r * (rowh + pad + lab)
    canvas.paste(im, (x, y + lab))
    dr.text((x, y + 1), f"idx {i}", fill=(200, 0, 0), font=f)
out = TMP / f"ov_{lo}_{hi}.png"
canvas.save(out)
print(out, canvas.size)
