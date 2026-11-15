import sys,io,json
sys.path.insert(0,'.')
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
D=Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP=Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
def rend(smi,W,H):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    d=rdMolDraw2D.MolDraw2DCairo(W,H); rdMolDraw2D.PrepareAndDrawMolecule(d,m); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')
for i in range(833,923):
    orig=Image.open(D/'images'/f'{i:06d}.png').convert('RGB')
    esm=json.load(open(D/'corrected_annotations'/f'{i:06d}.json',encoding='utf-8'))['esmiles']
    W=420; H=int(W*orig.height/orig.width)
    o=orig.resize((W,H)); r=rend(esm,W,H)
    c=Image.new('RGB',(W+430,max(H,r.height if r else H)+26),'white')
    c.paste(o,(0,26))
    if r: c.paste(r,(W+10,26))
    dr=ImageDraw.Draw(c); dr.text((4,4),f'idx {i}',fill=(200,0,0),font=font(20))
    c.save(TMP/f'fin_{i:06d}.png')
print('ok')
