"""chk.py lo hi [cols] [cell] -> 把 overlays_corrected 拼成核对表(原图|图)."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OV = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\_ai_review\overlays_corrected")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


a = [x for x in sys.argv[1:] if not x.startswith("-")]
if len(a) == 2 and int(a[0]) < int(a[1]):
    idx = [i for i in range(int(a[0]), int(a[1]) + 1) if (OV / f"{i:06d}.png").exists()]
else:
    idx = [int(x) for x in a]
cols = 2
cell = 1080
pad, lab = 8, 20
rows = (len(idx) + cols - 1) // cols
W = cols * (cell + pad) + pad
H = rows * (int(cell * 0.72) + pad + lab) + pad
cv = Image.new("RGB", (W, H), (255, 255, 255))
dr = ImageDraw.Draw(cv)
f = font(20)
for t, i in enumerate(idx):
    im = Image.open(OV / f"{i:06d}.png").convert("RGB")
    sc = cell / im.width
    im = im.resize((cell, max(1, int(im.height * sc))), Image.LANCZOS)
    r, c = divmod(t, cols)
    x = pad + c * (cell + pad)
    y = pad + r * (int(cell * 0.72) + pad + lab)
    cv.paste(im, (x, y + lab))
    dr.text((x, y + 1), f"idx {i}", fill=(200, 0, 0), font=f)
out = TMP / f"chk_{idx[0]}_{idx[-1]}.png"
cv.save(out)
print(out, cv.size)
