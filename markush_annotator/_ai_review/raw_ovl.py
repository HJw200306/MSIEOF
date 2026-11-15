import sys, json, os
from PIL import Image, ImageDraw

DATA = r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles'
OUT  = r'C:\Users\Admin\AppData\Local\Temp'

def main():
    for a in sys.argv[1:]:
        i = int(a)
        raw = json.load(open(os.path.join(DATA, 'annotations', f'{i:06d}.json'), encoding='utf-8'))
        W, H = raw.get('image_size', [0, 0])
        orig = Image.open(os.path.join(DATA, 'images', f'{i:06d}.png')).convert('RGB')
        sc = 5
        big = orig.resize((orig.width * sc, orig.height * sc), Image.NEAREST)
        sx = big.width / W if W else sc
        sy = big.height / H if H else sc
        dr = ImageDraw.Draw(big)
        at = raw['atoms']
        def P(k):
            return (at[k]['x'] * sx, at[k]['y'] * sy)
        for b in raw['bonds']:
            u, v, t = b['endpoint_atoms'][0], b['endpoint_atoms'][1], b['bond_type']
            col = (255, 0, 0) if t == 'wave' else (0, 120, 255)
            dr.line([P(u), P(v)], fill=col, width=2)
        for k in range(len(at)):
            x, y = P(k)
            dr.ellipse([x-4, y-4, x+4, y+4], outline=(0, 170, 0), width=2)
        outp = os.path.join(OUT, f'rawovl_{i:06d}.png')
        big.save(outp)
        print(outp)

main()
