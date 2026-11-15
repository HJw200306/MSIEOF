"""Render corrected_annotations over the original image, side by side, for visual QA."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_overlay import draw_overlay, font  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\_ai_review\check")


def render(idx):
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.open(D / "images" / f"{idx:06d}.png").convert("RGB")
    raw = json.load(open(D / "corrected_annotations" / f"{idx:06d}.json", encoding="utf-8"))
    W, H = img.size
    # corrected files store PIXEL coords -> renormalize so draw_overlay's *W is a no-op-safe 1:1
    ann = {
        "atoms": [{"x": a["x"] / W, "y": a["y"] / H,
                   "atom_symbol": a.get("symbol"), "label": a.get("label")}
                  for a in raw.get("atoms", [])],
        "bonds": raw.get("bonds", []),
    }
    scale = max(1, min(6, round(700 / max(W, 1))))
    clean = img.resize((W * scale, H * scale), Image.LANCZOS)
    ov = draw_overlay(img, ann, scale)
    gap, top = 24, 34
    canvas = Image.new("RGB", (W * scale * 2 + gap, H * scale + top), (255, 255, 255))
    canvas.paste(clean, (0, top))
    canvas.paste(ov, (W * scale + gap, top))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 7), f"idx {idx} CORRECTED  n_atoms={len(raw.get('atoms',[]))} "
                    f"n_bonds={len(raw.get('bonds',[]))}", fill=(0, 0, 0), font=font(16))
    p = OUT / f"{idx:06d}.png"
    canvas.save(p)
    return p


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(render(int(a)))
