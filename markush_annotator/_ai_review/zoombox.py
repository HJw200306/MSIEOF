"""zoombox.py idx x0 y0 x1 y1 [--px] [--ov] -> 原图裁剪放大(可叠加拓扑)。
x0..y1 为归一化 [0,1] 比例(默认)或像素(--px)。--ov 叠加 raw 拓扑+编号。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_overlay as R  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = D / "_ai_review" / "zoom"
TMP.mkdir(parents=True, exist_ok=True)


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def run(i, box, px=False, ov=False, target=880):
    img = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    W, H = img.size
    x0, y0, x1, y1 = box
    if not px:
        x0, y0, x1, y1 = x0 * W, y0 * H, x1 * W, y1 * H
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    cw, ch = crop.size
    z = max(1, round(target / max(cw, 1)))
    big = crop.resize((cw * z, ch * z), Image.LANCZOS)
    if ov:
        ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
        dr = ImageDraw.Draw(big)
        for k, a in enumerate(ann["atoms"]):
            ax, ay = a["x"] * W, a["y"] * H
            if not (x0 <= ax <= x1 and y0 <= ay <= y1):
                continue
            px_, py_ = (ax - x0) * z, (ay - y0) * z
            r = 5
            dr.ellipse([px_ - r, py_ - r, px_ + r, py_ + r], outline=(255, 0, 0), width=2)
            dr.text((px_ + 6, py_ - 12), str(k), fill=(220, 0, 0), font=font(20))
    tag = "ov" if ov else "raw"
    out = TMP / f"z{i:06d}_{int(x0)}_{int(y0)}_{int(x1)}_{int(y1)}_{tag}.png"
    big.save(out)
    print(out, big.size)


if __name__ == "__main__":
    i = int(sys.argv[1])
    box = [float(x) for x in sys.argv[2:6]]
    run(i, box, px="--px" in sys.argv, ov="--ov" in sys.argv)
