"""渲染 [lo-hi] 的「原图 | RDKit(corrected esmiles)」并排图, 用于视觉逐张核对。
用法: python __render_range.py 1078 1103
"""
import sys, io, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def rend(smi, W, H):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(W, H)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert("RGB")


def pair(i, W=460):
    orig = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
    H = max(60, int(W * orig.height / orig.width))
    o = orig.resize((W, H))
    esm = json.load(open(D / "corrected_annotations" / f"{i:06d}.json", encoding="utf-8"))["esmiles"]
    r = rend(esm, W, H)
    rh = r.height if r else H
    c = Image.new("RGB", (W + W + 12, max(H, rh) + 24), "white")
    c.paste(o, (0, 24))
    if r:
        c.paste(r, (W + 12, 24))
    dr = ImageDraw.Draw(c)
    tag = f"idx {i}"
    dr.text((4, 3), tag, fill=(200, 0, 0), font=font(18))
    dr.text((W + 16, 3), esm[:70], fill=(0, 90, 0), font=font(12))
    return c


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    for i in range(lo, hi + 1):
        c = pair(i)
        p = TMP / f"rv_{i:06d}.png"
        c.save(p)
    print("done", lo, hi)
