from PIL import Image
from pathlib import Path

SRC = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\PageToSMILES")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")

# idx -> (source_image, bbox[x0,y0,x1,y1])
JOBS = {
    171: ("CN103118679B_178.png", [860, 414, 975, 488]),
    183: ("CN103118679B_178.png", [708, 612, 853, 711]),
    150: ("CN103118679B_172.png", [848, 1106, 965, 1267]),
}

for idx, (src, bb) in JOBS.items():
    p = SRC / src
    if not p.exists():
        print("MISSING", p)
        continue
    im = Image.open(p).convert("L")
    W, H = im.size
    pad = 60
    x0 = max(0, bb[0] - pad); y0 = max(0, bb[1] - pad)
    x1 = min(W, bb[2] + pad); y1 = min(H, bb[3] + pad)
    c = im.crop((x0, y0, x1, y1))
    z = max(1, int(900 / max(c.width, c.height)))
    c = c.resize((c.width * z, c.height * z), Image.LANCZOS)
    out = TMP / f"src_{idx}.png"
    c.save(out)
    print(idx, src, "page", (W, H), "crop", (x0, y0, x1, y1), "zoom", z, "->", out.name, c.size)
