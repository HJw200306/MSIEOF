import sys, json, os
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D

DATA = r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles'
OUT  = r'C:\Users\Admin\AppData\Local\Temp'

def render(smi, W, H):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(W, H)
    d.drawOptions().addStereoAnnotation = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    import io
    return Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')

def main():
    W = 380
    for a in sys.argv[1:]:
        i = int(a)
        orig = Image.open(os.path.join(DATA, 'images', f'{i:06d}.png')).convert('RGB')
        H = int(W * orig.height / orig.width)
        o2 = orig.resize((W, H))
        cor = json.load(open(os.path.join(DATA, 'corrected_annotations', f'{i:06d}.json'), encoding='utf-8'))
        r = render(cor['esmiles'], W, H)
        canvas = Image.new('RGB', (W, H*2+24), 'white')
        canvas.paste(o2, (0, 0))
        if r is not None:
            canvas.paste(r, (0, H+24))
        dr = ImageDraw.Draw(canvas)
        dr.text((3, H+4), str(i), fill=(200, 0, 0))
        outp = os.path.join(OUT, f'side_{i:06d}.png')
        canvas.save(outp)
        print(outp)

main()
