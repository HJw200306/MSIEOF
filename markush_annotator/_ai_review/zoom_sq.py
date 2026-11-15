from PIL import Image
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\images")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
for idx, x1 in ((171, 0.45), (183, 0.40), (160, 0.45), (150, 0.45)):
    im = Image.open(D / f"{idx:06d}.png").convert("L")
    w, h = im.size
    c = im.crop((0, 0, int(w * x1), h))
    c = c.resize((c.width * 10, c.height * 10), Image.LANCZOS)
    out = TMP / f"sq_{idx}.png"
    c.save(out)
    print(idx, "orig", im.size, "->", c.size, out.name)
