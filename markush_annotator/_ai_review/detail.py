"""detail.py idx [idx...] [--grid] [--zoom N] -> 高倍「原图 | 拓扑+编号」并排图 (供视觉精判)。
坐标: annotations 为归一化 [0,1]，像素 = 归一化 * PNG 尺寸。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_overlay as R  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = D / "_ai_review" / "detail"
TMP.mkdir(parents=True, exist_ok=True)


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def one(i, zoom=None, grid=False):
    img = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    W, H = img.size
    z = zoom or max(1, min(14, round(1000 / max(W, 1))))
    clean = img.resize((W * z, H * z), Image.NEAREST if grid else Image.LANCZOS)
    ov = R.draw_overlay(img, ann, z, normalized=True)
    if grid:
        dr = ImageDraw.Draw(clean, "RGBA")
        for gx in range(0, W * z, z * 10):
            dr.line([(gx, 0), (gx, H * z)], fill=(0, 0, 255, 90), width=1)
        for gy in range(0, H * z, z * 10):
            dr.line([(0, gy), (W * z, gy)], fill=(0, 0, 255, 90), width=1)

    gap, top = 20, 36
    cv = Image.new("RGB", (W * z * 2 + gap, H * z + top), (255, 255, 255))
    cv.paste(clean, (0, top))
    cv.paste(ov, (W * z + gap, top))
    dr = ImageDraw.Draw(cv)
    dr.text((4, 8), f"idx {i}  {W}x{H}  zoom={z}  atoms={len(ann['atoms'])} bonds={len(ann['bonds'])}",
            fill=(0, 0, 0), font=font(20))
    out = TMP / f"d{i:06d}.png"
    cv.save(out)
    return out


if __name__ == "__main__":
    grid = "--grid" in sys.argv
    zoom = None
    if "--zoom" in sys.argv:
        zoom = int(sys.argv[sys.argv.index("--zoom") + 1])
    idxs = [int(a) for a in sys.argv[1:] if not a.startswith("--")]
    for i in idxs:
        print(one(i, zoom=zoom, grid=grid))
