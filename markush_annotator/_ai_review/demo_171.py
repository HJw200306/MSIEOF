"""对 idx171 演示「正确画法」：波浪线 = 一根 type-7 键 + 外端 * 连接点。
左=原图  中=MolScribe 现状  右=应该改成这样
"""
import json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "demo_171_correct.png"
IDX = 171
SCALE = 7
FONT = None
for nm in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
    try:
        FONT = ImageFont.truetype(nm, 22); break
    except Exception:
        pass
if FONT is None:
    FONT = ImageFont.load_default()


def wave(dr, p, q, color=(210, 30, 200), width=3, amp=7.0, cycles=5):
    x1, y1 = p; x2, y2 = q
    L = math.hypot(x2 - x1, y2 - y1)
    if L == 0: return
    dx, dy = (x2 - x1) / L, (y2 - y1) / L
    nx, ny = -dy, dx
    n = cycles * 12
    pts = []
    for s in range(n + 1):
        t = s / n
        off = math.sin(t * math.pi * 2 * cycles) * amp
        pts.append((x1 + dx * L * t + nx * off, y1 + dy * L * t + ny * off))
    dr.line(pts, fill=color, width=width, joint="curve")


def panel(img, atoms, bonds, title, draw_star=False):
    W, H = img.size
    im = img.resize((W * SCALE, H * SCALE), Image.LANCZOS).convert("RGB")
    dr = ImageDraw.Draw(im)
    P = [(a["x"] * W * SCALE, a["y"] * H * SCALE) for a in atoms]
    for b in bonds:
        i, j, t = b
        if t == 7:
            wave(dr, P[i], P[j])
        elif t == 2:
            dr.line([P[i], P[j]], fill=(30, 30, 220), width=3)
            dx, dy = P[j][0] - P[i][0], P[j][1] - P[i][1]
            L = math.hypot(dx, dy) or 1
            nx, ny = -dy / L * 5, dx / L * 5
            dr.line([(P[i][0] + nx, P[i][1] + ny), (P[j][0] + nx, P[j][1] + ny)], fill=(30, 30, 220), width=2)
            dr.line([(P[i][0] - nx, P[i][1] - ny), (P[j][0] - nx, P[j][1] - ny)], fill=(30, 30, 220), width=2)
        else:
            dr.line([P[i], P[j]], fill=(0, 150, 0), width=3)
    for k, a in enumerate(atoms):
        sym = a.get("atom_symbol") or a.get("symbol") or "C"
        px, py = P[k]
        if sym == "*":
            r = 11
            dr.ellipse([px - r, py - r, px + r, py + r], fill=(255, 230, 255), outline=(210, 30, 200), width=3)
            dr.text((px - 6, py - 11), "*", fill=(210, 30, 200), font=FONT)
        else:
            r = 5
            dr.ellipse([px - r, py - r, px + r, py + r], outline=(60, 60, 60), width=2)
    return im


img = Image.open(D / "images" / f"{IDX:06d}.png").convert("RGB")
raw = json.load(open(D / "annotations" / f"{IDX:06d}.json", encoding="utf-8"))

# MolScribe 现状
raw_atoms = [{"x": a["x"], "y": a["y"], "atom_symbol": a["atom_symbol"]} for a in raw["atoms"]]
raw_bonds = [(b["endpoint_atoms"][0], b["endpoint_atoms"][1],
              {"single": 1, "double": 2, "triple": 3}.get(b["bond_type"], 1)) for b in raw["bonds"]]

# 正确版: 去掉 0,1 (假波浪碳), 原子2 改成 *
fix_atoms = [{"x": a["x"], "y": a["y"], "atom_symbol": ("*" if k == 2 else a["atom_symbol"])}
             for k, a in enumerate(raw["atoms"]) if k not in (0, 1)]
# 索引重映射: 原 2,3,4,5,6 -> 0,1,2,3,4
fix_bonds = [(0, 1, 7), (1, 2, 1), (2, 3, 1), (2, 4, 2)]

p1 = panel(img, raw_atoms, raw_bonds, "MolScribe raw")
p2 = panel(img, fix_atoms, fix_bonds, "correct")

W, H = img.size
sw, sh = W * SCALE, H * SCALE
gap, top = 30, 56
cols = [0, sw + gap, 2 * (sw + gap)]
canvas = Image.new("RGB", (3 * sw + 2 * gap, sh + top), (255, 255, 255))
canvas.paste(img.resize((sw, sh), Image.LANCZOS), (cols[0], top))
canvas.paste(p1, (cols[1], top))
canvas.paste(p2, (cols[2], top))
dr = ImageDraw.Draw(canvas)
dr.text((cols[0] + 6, 8), "1. original", fill=(0, 0, 0), font=FONT)
dr.text((cols[1] + 6, 8), "2. MolScribe raw", fill=(180, 0, 0), font=FONT)
dr.text((cols[2] + 6, 8), "3. correct", fill=(0, 130, 0), font=FONT)
dr.text((cols[1] + 6, 30), "squiggle became fake carbons 0,1", fill=(180, 0, 0), font=FONT)
dr.text((cols[2] + 6, 30), "* ==wave bond== CH2 ...", fill=(0, 130, 0), font=FONT)
canvas.save(OUT)
print(OUT)
