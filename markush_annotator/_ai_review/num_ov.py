"""num_ov.py idx [idx...]  在源图上叠加 RAW 原子序号, 用于辨认波浪线伪原子。"""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "zoom"
OUT.mkdir(parents=True, exist_ok=True)

for a in sys.argv[1:]:
    i = int(a)
    ann = json.loads((D / "annotations" / f"{i:06d}.json").read_text(encoding="utf-8"))
    im = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    W, H = im.size
    S = 8
    im = im.resize((W * S, H * S), Image.LANCZOS)
    dr = ImageDraw.Draw(im)
    try:
        fnt = ImageFont.truetype("arial.ttf", 34)
    except Exception:
        fnt = ImageFont.load_default()
    for k, at in enumerate(ann["atoms"]):
        x, y = at["x"] * W * S, at["y"] * H * S
        r = 6
        dr.ellipse([x - r, y - r, x + r, y + r], fill=(255, 0, 0))
        dr.text((x + 8, y - 40), str(k), fill=(0, 0, 255), font=fnt)
    im.save(OUT / f"_n{i:06d}.png")
    print(i, "ok", W, H, len(ann["atoms"]))
