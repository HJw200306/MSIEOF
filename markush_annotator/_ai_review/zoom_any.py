import sys
from pathlib import Path
from PIL import Image

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\images")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")

for a in sys.argv[1:]:
    i = int(a)
    im = Image.open(D / f"{i:06d}.png").convert("L")
    w, h = im.size
    z = max(1, int(1100 / max(w, h)))
    c = im.resize((w * z, h * z), Image.LANCZOS)
    out = TMP / f"z_{i}.png"
    c.save(out)
    print(i, im.size, "zoom", z, "->", out)
