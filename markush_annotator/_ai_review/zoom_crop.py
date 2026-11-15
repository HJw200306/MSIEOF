"""zoom_crop.py idx fx0 fy0 fx1 fy1 [zoom]
按归一化比例裁剪原图并放大, 用于数清骨架顶点。"""
import sys
from pathlib import Path
from PIL import Image

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\images")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")

i = int(sys.argv[1])
fx0, fy0, fx1, fy1 = (float(x) for x in sys.argv[2:6])
z = int(sys.argv[6]) if len(sys.argv) > 6 else 12
im = Image.open(D / f"{i:06d}.png").convert("L")
w, h = im.size
c = im.crop((int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)))
c = c.resize((c.width * z, c.height * z), Image.LANCZOS)
out = TMP / f"c_{i}.png"
c.save(out)
print(out, c.size)
