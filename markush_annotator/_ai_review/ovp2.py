"""紧凑叠加渲染: 把 corrected 的键/原子编号半透明画在放大后的原图上(单图)。"""
import json, math, sys
from PIL import Image, ImageDraw
B = r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles"


def render(i, S=4, src="images"):
    d = json.load(open(f"{B}\\corrected_annotations\\{i:06d}.json", encoding="utf-8"))
    im = Image.open(f"{B}\\{src}\\{i:06d}.png").convert("RGB")
    im = im.resize((im.size[0] * S, im.size[1] * S), Image.LANCZOS)
    dr = ImageDraw.Draw(im, "RGBA")
    A = d["atoms"]

    def P(k):
        return (A[k]["x"] * S, A[k]["y"] * S)

    for bd in d["bonds"]:
        i1, j1, o = bd
        x1, y1 = P(i1); x2, y2 = P(j1)
        dx, dy = x2 - x1, y2 - y1
        L = math.hypot(dx, dy) or 1
        ux, uy = dx / L, dy / L; px, py = -uy, ux
        if o in (5, 6):
            w = 7
            if o == 5:
                dr.polygon([(x1, y1), (x2 + px * w, y2 + py * w), (x2 - px * w, y2 - py * w)],
                           fill=(255, 0, 0, 90), outline=(255, 0, 0, 255))
            else:
                for t in range(1, 9):
                    f = t / 9.0; cx, cy = x1 + dx * f, y1 + dy * f; ww = w * f
                    dr.line((cx + px * ww, cy + py * ww, cx - px * ww, cy - py * ww),
                            fill=(0, 120, 255, 220), width=2)
        elif o == 2:
            dr.line((x1, y1, x2, y2), fill=(255, 140, 0, 210), width=3)
        elif o == 7:
            dr.line((x1, y1, x2, y2), fill=(255, 0, 255, 255), width=3)
        else:
            dr.line((x1, y1, x2, y2), fill=(0, 200, 0, 170), width=2)
    for k, a in enumerate(A):
        x, y = P(k)
        c = {"C": (0, 150, 0), "N": (0, 0, 255), "O": (230, 0, 0), "H": (90, 90, 90),
             "*": (200, 0, 200)}.get(a["symbol"], (0, 0, 0))
        dr.ellipse((x - 9, y - 9, x + 9, y + 9), outline=c, width=3)
        lab = a.get("label")
        dr.text((x + 10, y - 22), str(k) + (lab or ""), fill=(0, 0, 0, 255))
    out = f"{B}\\_ai_review\\zoom\\_c{i:06d}.png"
    im.save(out)
    return out, d


if __name__ == "__main__":
    i = int(sys.argv[1])
    out, d = render(i)
    print(out)
    print("esmiles:", d["esmiles"])
