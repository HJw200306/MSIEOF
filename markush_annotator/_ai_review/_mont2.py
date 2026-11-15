import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
TMP=Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
lo,hi=map(int,sys.argv[1:3])
cols=2; cell=640; pad,lab=8,26
items=list(range(lo,hi+1))
rows=(len(items)+cols-1)//cols
canvas=Image.new("RGB",(cols*(cell+pad)+pad, rows*(int(cell*0.72)+pad+lab)+pad),(235,235,235))
dr=ImageDraw.Draw(canvas); f=font(22); rowh=int(cell*0.72)
for t,i in enumerate(items):
    im=Image.open(TMP/f"ov2_{i:06d}.png").convert("RGB")
    sc=cell/im.width; im=im.resize((cell,max(1,int(im.height*sc))))
    if im.height>rowh: im=im.crop((0,0,cell,rowh))
    r,c=divmod(t,cols); x=pad+c*(cell+pad); y=pad+r*(rowh+pad+lab)
    canvas.paste(im,(x,y+lab)); dr.text((x,y+2),f"idx {i}",fill=(200,0,0),font=f)
out=TMP/f"m2_{lo}_{hi}.png"; canvas.save(out); print(out,canvas.size)
