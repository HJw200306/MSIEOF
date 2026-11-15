"""grid_crop.py idx fx0 fy0 fx1 fy1 [zoom] [step]
Crop original image to normalized box, upscale, draw a grid in ORIGINAL pixel units.
Prints the original-pixel origin so read-off is exact."""
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\images")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")

i = int(sys.argv[1])
fx0, fy0, fx1, fy1 = (float(x) for x in sys.argv[2:6])
z = int(sys.argv[6]) if len(sys.argv) > 6 else 12
step = int(sys.argv[7]) if len(sys.argv) > 7 else 10
im = Image.open(D / f"{i:06d}.png").convert("L")
w, h = im.size
px0, py0 = int(fx0 * w), int(fy0 * h)
px1, py1 = int(fx1 * w), int(fy1 * h)
c = im.crop((px0, py0, px1, py1)).resize(((px1 - px0) * z, (py1 - py0) * z), Image.LANCZOS)
c = c.convert("RGB")
dr = ImageDraw.Draw(c)
try:
    f = ImageFont.truetype("arial.ttf", 14)
except Exception:
    f = ImageFont.load_default()
for x in range(px0 - px0 % step, px1 + 1, step):
    X = (x - px0) * z
    dr.line([(X, 0), (X, c.height)], fill=(255, 0, 0), width=1)
    dr.text((X + 2, 2), str(x), fill=(255, 0, 0), font=f)
for y in range(py0 - py0 % step, py1 + 1, step):
    Y = (y - py0) * z
    dr.line([(0, Y), (c.width, Y)], fill=(0, 0, 255), width=1)
    dr.text((2, Y + 2), str(y), fill=(0, 0, 255), font=f)
out = TMP / f"g_{i}.png"
c.save(out)
print(f"orig {w}x{h}  box px=({px0},{py0})-({px1},{py1})  -> {out} {c.size}")
