import sys, io
sys.path.insert(0,'.')
import proto_wave as P
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
TMP=Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
# 846/849 copy from 847
from backend.data_manager import graph_to_smiles
cache={}
def new(i):
    if i in (846,849): i=847
    if i in cache: return cache[i]
    A,Bo,(W,H)=P.wave_connect(i)
    s=graph_to_smiles(A,Bo,[]); cache[i]=s; return s
for i in range(833,923):
    s=new(i)
    orig=Image.open(P.D+r'\images\%06d.png'%i).convert('RGB')
    W=420; H=int(W*orig.height/orig.width)
    o=orig.resize((W,H)); r=P.rend(s,W,H)
    rh = r.height if r else H
    c=Image.new('RGB',(W+420+10,max(H,rh)+24),(255,255,255))
    c.paste(o,(0,24))
    if r: c.paste(r,(W+10,24))
    dr=ImageDraw.Draw(c); dr.text((4,4),f'idx {i}',fill=(200,0,0),font=font(20))
    c.save(TMP/f'new_{i:06d}.png')
print('ok', len(cache))
