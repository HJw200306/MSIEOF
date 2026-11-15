"""sheet.py lo hi [cols] [cell] -> 拼接原图接触表, 每格标注 idx, 便于快速扫一遍找波浪线。"""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\images")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


lo, hi = int(sys.argv[1]), int(sys.argv[2])
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 6
cell = int(sys.argv[4]) if len(sys.argv) > 4 else 230
pad, lab = 8, 20
rows = (hi - lo + 1 + cols - 1) // cols
W = cols * (cell + pad) + pad
H = rows * (cell + pad + lab) + pad
canvas = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(canvas)
f = font(18)
for t, i in enumerate(range(lo, hi + 1)):
    p = D / f"{i:06d}.png"
    if not p.exists():
        continue
    im = Image.open(p).convert("RGB")
    sc = min(cell / im.width, cell / im.height)
    im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
    r, c = divmod(t, cols)
    x = pad + c * (cell + pad)
    y = pad + r * (cell + pad + lab)
    canvas.paste(im, (x + (cell - im.width) // 2, y + lab + (cell - im.height) // 2))
    dr.text((x, y + 1), f"idx {i}", fill=(200, 0, 0), font=f)
out = TMP / f"sheet_{lo}_{hi}.png"
canvas.save(out)
print(out, canvas.size)
