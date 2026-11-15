from PIL import Image
from pathlib import Path

SRC = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\PageToSMILES")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")

# 只裁波浪键那一小块，超高清放大
JOBS = {
    "171": ("CN103118679B_178.png", (852, 415, 930, 472)),
    "183": ("CN103118679B_178.png", (700, 610, 790, 680)),
    "190": ("CN103118679B_178.png", (548, 800, 640, 872)),
}

for tag, (src, bb) in JOBS.items():
    im = Image.open(SRC / src).convert("L")
    c = im.crop(bb)
    z = 18
    c = c.resize((c.width * z, c.height * z), Image.LANCZOS)
    # 二值化，去掉灰边
    c = c.point(lambda v: 0 if v < 150 else 255, "L")
    out = TMP / f"wv_{tag}.png"
    c.save(out)
    print(tag, bb, "->", out.name, c.size)
