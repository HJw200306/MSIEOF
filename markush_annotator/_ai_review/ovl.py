import sys, json, os
import numpy as np
from PIL import Image, ImageDraw

DATA = r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles'
OUT  = r'C:\Users\Admin\AppData\Local\Temp'

def main():
    for a in sys.argv[1:]:
        i = int(a)
        orig = Image.open(os.path.join(DATA, 'images', f'{i:06d}.png')).convert('RGB')
        g = np.array(orig.convert('L'))
        ys, xs = np.where(g < 128)
        bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
        d = json.load(open(os.path.join(DATA, 'corrected_annotations', f'{i:06d}.json'), encoding='utf-8'))
        at = d['atoms']; bo = d['bonds']
        ax = [p['x'] for p in at]; ay = [p['y'] for p in at]
        ax0, ax1, ay0, ay1 = min(ax), max(ax), min(ay), max(ay)
        sx = (bx1-bx0)/(ax1-ax0) if ax1 > ax0 else 1
        sy = (by1-by0)/(ay1-ay0) if ay1 > ay0 else 1
        W, H = orig.size
        sc = 6
        big = orig.resize((W*sc, H*sc), Image.NEAREST)
        dr = ImageDraw.Draw(big)
        def P(x, y): return ((bx0 + (x-ax0)*sx)*sc, (by0 + (y-ay0)*sy)*sc)
        for b in bo:
            p, q, t = b[0], b[1], b[2]
            col = (255,0,0) if t == 7 else (0,110,255)
            dr.line([P(at[p]['x'], at[p]['y']), P(at[q]['x'], at[q]['y'])], fill=col, width=3)
        for at2 in at:
            x, y = P(at2['x'], at2['y'])
            dr.ellipse([x-5, y-5, x+5, y+5], outline=(0,170,0), width=2)
        outp = os.path.join(OUT, f'ovl_{i:06d}.png')
        big.save(outp)
        print(outp)

main()
