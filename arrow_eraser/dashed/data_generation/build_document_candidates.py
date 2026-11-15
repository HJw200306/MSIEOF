"""Train-only patent/journal style candidates with synchronized SVG masks.

Molecules come only from existing synthetic training SMILES, excluding canonical
structures in held-out synthetic splits. Layouts do not assert real reactions.
"""
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cairosvg
from PIL import Image, ImageDraw
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT=Path(__file__).resolve().parents[1]
NS='{http://www.w3.org/2000/svg}'


def render(svg):
    return np.array(Image.open(io.BytesIO(cairosvg.svg2png(bytestring=svg.encode()))).convert('RGB'))


def canonical(group):
    if 'SMILES: ' not in group: return None
    mol=Chem.MolFromSmiles(group.split('SMILES: ',1)[1])
    return Chem.MolToSmiles(mol) if mol else None


def molecule(smiles,x,y,width,rotation):
    mol=Chem.MolFromSmiles(smiles)
    drawer=rdMolDraw2D.MolDraw2DSVG(300,300)
    opt=drawer.drawOptions(); opt.useBWAtomPalette(); opt.bondLineWidth=width
    opt.rotate=rotation; opt.fixedFontSize=17
    drawer.DrawMolecule(mol); drawer.FinishDrawing()
    svg=ET.fromstring(drawer.GetDrawingText())
    for child in list(svg):
        if child.tag.endswith('rect'): svg.remove(child)
    all_paths=''.join(ET.tostring(c,encoding='unicode') for c in svg)
    bonds=''.join(ET.tostring(c,encoding='unicode') for c in svg
                  if re.match(r'bond-\d+',c.get('class','')))
    return tuple(f'<g transform="translate({x},{y})">{s}</g>' for s in (all_paths,bonds))


def wrap(s):
    return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="480"><rect width="800" height="480" fill="white"/>'+s+'</svg>'


def main():
    source=ROOT/'dataset/DashedArrow-Seg'
    rows=list(csv.DictReader((source/'manifest.csv').open(encoding='utf-8-sig')))
    held={canonical(r['group']) for r in rows if r['split']!='train'}-{None}
    pool=sorted({canonical(r['group']) for r in rows if r['split']=='train'}-held-{None})
    assert len(pool)>2
    out=ROOT/'dataset/DocumentStyle-Candidates-v1'
    out.mkdir(exist_ok=False)
    for name in ['images','masks','previews']: (out/name).mkdir()
    rng=np.random.default_rng(20261028)
    records=[]
    for i in range(120):
        style='journal' if i%2==0 else 'patent'
        negative=i%10==0
        curved=i%3==0
        selected=list(rng.choice(pool,2,replace=False))
        width=float(rng.uniform(.8,2.0))
        parts=[molecule(selected[0],10,110,width,int(rng.integers(-25,26))),
               molecule(selected[1],490,110,width,int(rng.integers(-25,26)))]
        mol=''.join(p[0] for p in parts); bonds=''.join(p[1] for p in parts)
        dash=int(rng.integers(3,12)); gap=int(rng.integers(3,14))
        line=float(rng.uniform(.8,2.4)); color=int(rng.integers(0,95))
        path='M 300 205 Q 400 85 500 205' if curved else 'M 310 240 L 490 240'
        head='M 500 205 L 489 201 M 500 205 L 498 193' if curved else 'M 490 240 L 479 234 M 490 240 L 479 246'
        arrow='' if negative else f'<g fill="none" stroke="rgb({color},{color},{color})" stroke-width="{line:.3f}" stroke-linecap="round"><path d="{path}" stroke-dasharray="{dash} {gap}"/><path d="{head}"/></g>'
        font='serif' if style=='journal' else 'sans-serif'
        heading='Scheme 1' if style=='journal' else f'Example {i+1}'
        text=f'<g font-family="{font}" fill="black"><text x="30" y="45" font-size="20">{heading}</text><text x="135" y="440" font-size="17">({i+1})</text><text x="625" y="440" font-size="17">({i+2})</text><text x="340" y="290" font-size="14">conditions</text></g>'
        rgb=render(wrap(arrow+mol+text))
        a=render(wrap(arrow)).min(2)<250
        b=render(wrap(bonds)).min(2)<250
        # Text/atom glyph overlap is excluded from supervision, not erased.
        m=render(wrap(mol+text)).min(2)<250
        mask=np.zeros(a.shape,np.uint8); mask[a]=1; mask[b]=2
        mask[a&m&~b]=255
        Image.fromarray(rgb).save(out/'images'/f'{i:04d}.png')
        Image.fromarray(mask).save(out/'masks'/f'{i:04d}.png')
        colors=rgb.copy(); colors[mask==1]=[255,0,0]; colors[mask==2]=[0,150,0]; colors[mask==255]=128
        if i<12:
            Image.fromarray(np.concatenate([rgb,colors],axis=1)).save(out/'previews'/f'{i:04d}.png')
        records.append({'id':f'{i:04d}','split':'train_candidate','style':style,
            'smiles':selected,'negative':negative,'curved':curved,'dash':dash,'gap':gap,
            'line_width':line,'molecule_line_width':width,'arrow_gray':color,
            'arrow_pixels':int((mask==1).sum()),'bond_pixels':int((mask==2).sum()),
            'image_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()})
    assert all(r['bond_pixels']>0 and (r['negative'] or r['arrow_pixels']>0) for r in records)
    assert not ({s for r in records for s in r['smiles']}&held)
    (out/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'quality_report.json').write_text(json.dumps({'samples':120,'negative':12,
        'heldout_canonical_smiles_overlap':0,'valid_labels':[0,1,2,255],
        'geometry':'same SVG rendering for image and label','seed':20261028,
        'status':'candidates, not merged into training; visual review and training evaluation required'},indent=2),encoding='utf-8')
    print(out)


if __name__=='__main__': main()
