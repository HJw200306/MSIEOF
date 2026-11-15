"""orig_sheet.py lo hi [per] -> 原图接触表(不含拓扑), 用于快速扫波浪线。
边框色: 红=A(孤立C对) 橙=B(悬空末端C) 蓝=异常token 绿=其他。
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = D / "_ai_review" / "sheets"
TMP.mkdir(parents=True, exist_ok=True)

A_SET = {202, 206, 207, 209, 210, 211, 214, 215, 216, 221, 222, 223, 226,
         267, 268, 269, 270, 371}
B_SET = {224, 229, 230, 234, 247, 248, 249, 250, 258, 259, 286, 287, 288, 289,
         290, 291, 297, 298, 300, 301, 302, 303, 304, 305, 306, 307, 308, 310,
         314, 315, 317, 318, 321, 323, 324, 328, 332, 333, 335, 336, 337, 346,
         347, 348, 349, 358, 359, 360, 369, 380, 383, 387, 389, 391, 393, 395,
         396, 399}
FLAG_SET = {255, 286, 388, 389, 392, 398}


def font(sz):
    for n in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(n, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def build(lo, hi, cols=4, cell=430):
    idxs = list(range(lo, hi + 1))
    rows = (len(idxs) + cols - 1) // cols
    rowh = int(cell * 0.78)
    pad, lab = 8, 26
    W = cols * (cell + pad) + pad
    H = rows * (rowh + pad + lab) + pad
    cv = Image.new("RGB", (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(cv)
    f = font(22)
    for t, i in enumerate(idxs):
        im = Image.open(D / "images" / f"{i:06d}.png").convert("RGB")
        sc = min(cell / im.width, rowh / im.height)
        im = im.resize((max(1, int(im.width * sc)), max(1, int(im.height * sc))), Image.LANCZOS)
        r, c = divmod(t, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (rowh + pad + lab)
        col = (220, 0, 0) if i in A_SET else (240, 140, 0) if i in B_SET else (0, 90, 220) if i in FLAG_SET else (150, 150, 150)
        dr.rectangle([x, y, x + cell, y + lab], outline=col, width=3)
        dr.text((x + 4, y + 3), f"idx {i}", fill=col, font=f)
        cv.paste(im, (x, y + lab))
    out = TMP / f"orig_{lo}_{hi}.png"
    cv.save(out)
    print(out, cv.size)


if __name__ == "__main__":
    lo, hi = int(sys.argv[1]), int(sys.argv[2])
    cols = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    cell = int(sys.argv[4]) if len(sys.argv) > 4 else 430
    build(lo, hi, cols, cell)
