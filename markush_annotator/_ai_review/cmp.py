import sys, json, os
from PIL import Image, ImageDraw

DATA = r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles'
OUT  = r'C:\Users\Admin\AppData\Local\Temp'

def draw_cor(idx, W=520):
    p = os.path.join(DATA, 'corrected_annotations', f'{idx:06d}.json')
    d = json.load(open(p, encoding='utf-8'))
    aw, ah = d.get('image_size', [1, 1])
    atoms = d['atoms']; bonds = d['bonds']
    H = int(W * ah / aw)
    sc = W / aw
    img = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(img)
    def xy(a): return (a['x'] * sc, a['y'] * sc)
    for b in bonds:
        i, j, t = b[0], b[1], b[2]
        x1, y1 = xy(atoms[i]); x2, y2 = xy(atoms[j])
        if t == 4:  # aromatic
            dr.line([x1, y1, x2, y2], fill=(120,180,255), width=2)
        elif t == 7:  # wave
            dr.line([x1, y1, x2, y2], fill=(255,0,0), width=3)
        elif t == 2:
            dr.line([x1, y1, x2, y2], fill=(0,0,0), width=1)
            dr.line([x1, y1, x2, y2], fill=(0,0,0), width=1)  # naive
        else:
            dr.line([x1, y1, x2, y2], fill=(0,0,0), width=1)
    for a in atoms:
        x, y = xy(a)
        r = 4
        dr.ellipse([x-r, y-r, x+r, y+r], fill=(0,150,0))
        lbl = a.get('label')
        if lbl: dr.text((x+5, y-14), lbl, fill=(200,0,0))
    return img

def main():
    for idx in sys.argv[1:]:
        i = int(idx)
        orig = Image.open(os.path.join(DATA, 'images', f'{i:06d}.png')).convert('RGB')
        W = 520
        o2 = orig.resize((W, int(W * orig.height / orig.width)))
        cor = draw_cor(i, W)
        H = max(o2.height, cor.height)
        canvas = Image.new('RGB', (W*2+10, H), 'white')
        canvas.paste(o2, (0, 0)); canvas.paste(cor, (W+10, 0))
        outp = os.path.join(OUT, f'cmp_{i:06d}.png')
        canvas.save(outp)
        print(outp, 'image_size', json.load(open(os.path.join(DATA,'corrected_annotations',f'{i:06d}.json'),encoding='utf-8')).get('image_size'))

main()
