"""渲染「原图 | 带原子编号的拓扑叠加图」拼图，供视觉模型对号入座。

坐标约定: annotations/*.json 的 atom.x/y 为归一化 [0,1]，像素 = 归一化 * PNG 实际尺寸。
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "overlays"

ELEM_COLOR = {
    "C": (20, 20, 20), "N": (0, 70, 255), "O": (220, 0, 0), "S": (200, 140, 0),
    "F": (0, 165, 0), "Cl": (0, 150, 0), "Br": (150, 60, 0), "I": (130, 0, 130),
    "P": (210, 90, 0), "Si": (110, 110, 110), "H": (130, 130, 130),
    "*": (200, 0, 200), "_AP": (200, 0, 200),
}
BOND_COLOR = {
    "single": (0, 150, 0), "double": (0, 0, 220), "triple": (150, 0, 180),
    "aromatic": (0, 140, 140), "wedge": (220, 30, 90), "dash": (220, 30, 90),
    "wave": (220, 30, 200),
}
FONT_CACHE = {}


def font(size):
    if size not in FONT_CACHE:
        f = None
        for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
            try:
                f = ImageFont.truetype(name, size)
                break
            except Exception:
                pass
        FONT_CACHE[size] = f or ImageFont.load_default()
    return FONT_CACHE[size]


def pixel_atoms(ann, W, H, normalized=True):
    pts = []
    for a in ann["atoms"]:
        sym = a.get("atom_symbol") or a.get("symbol") or "C"
        sx = a["x"] * W if normalized else a["x"]
        sy = a["y"] * H if normalized else a["y"]
        pts.append((sx, sy, sym, a.get("label")))
    return pts


def draw_overlay(img, ann, scale, normalized=True):
    W, H = img.size
    ov = img.resize((W * scale, H * scale), Image.LANCZOS).convert("RGB")
    ov = Image.blend(ov, Image.new("RGB", ov.size, (255, 255, 255)), 0.5)
    dr = ImageDraw.Draw(ov)
    pts = pixel_atoms(ann, W, H, normalized)
    P = [(x * scale, y * scale) for x, y, _, _ in pts]

    for b in ann["bonds"]:
        if isinstance(b, dict):
            i, j = b["endpoint_atoms"]
            bt = b.get("bond_type", "single")
        else:
            i, j, order = int(b[0]), int(b[1]), int(b[2])
            bt = {1: "single", 2: "double", 3: "triple", 4: "aromatic",
                  5: "wedge", 6: "dash", 7: "wave"}.get(order, "single")
        if i >= len(P) or j >= len(P):
            continue
        dr.line([P[i], P[j]], fill=BOND_COLOR.get(bt, (0, 150, 0)), width=max(2, scale))

    for k, (px, py) in enumerate(P):
        _, _, sym, label = pts[k]
        col = ELEM_COLOR.get(sym, (255, 0, 150))
        r = max(3, scale * 2)
        dr.ellipse([px - r, py - r, px + r, py + r], outline=col,
                   width=max(2, scale), fill=(255, 255, 255))
        dr.text((px + r + 1, py - r - 2), str(k), fill=(210, 0, 0),
                font=font(max(12, scale * 7)))
        if label:
            dr.text((px + r + 1, py + r), str(label), fill=(120, 0, 120),
                    font=font(max(11, scale * 6)))
    return ov


def render(idx, out_dir=OUT, src="annotations"):
    out_dir.mkdir(parents=True, exist_ok=True)
    img = Image.open(D / "images" / f"{idx:06d}.png").convert("RGB")
    ann = json.load(open(D / src / f"{idx:06d}.json", encoding="utf-8"))
    W, H = img.size
    scale = max(1, min(6, round(700 / max(W, 1))))
    clean = img.resize((W * scale, H * scale), Image.LANCZOS)
    ov = draw_overlay(img, ann, scale, normalized=(src == "annotations"))
    gap, top = 24, 34
    canvas = Image.new("RGB", (W * scale * 2 + gap, H * scale + top), (255, 255, 255))
    canvas.paste(clean, (0, top))
    canvas.paste(ov, (W * scale + gap, top))
    dr = ImageDraw.Draw(canvas)
    n = len(ann.get("atoms", []))
    nb = len(ann.get("bonds", []))
    dr.text((4, 7), f"idx {idx}  {W}x{H}  n_atoms={n} n_bonds={nb}  LEFT=original RIGHT=graph+index",
            fill=(0, 0, 0), font=font(16))
    path = out_dir / f"{idx:06d}.png"
    canvas.save(path)
    return path


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    src = "corrected_annotations" if "--corrected" in sys.argv else "annotations"
    out = OUT if src == "annotations" else (D / "_ai_review" / "overlays_corrected")
    idxs = [int(a) for a in args] or list(range(53, 100))
    for i in idxs:
        p = render(i, out, src)
        print(p)
