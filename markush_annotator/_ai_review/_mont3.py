import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
TMP=Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
lo,hi=map(int,sys.argv[1:3]); cols=2; cell=760; pad,lab=8,0
items=list(range(lo,hi+1)); rows=(len(items)+cols-1)//cols
from PIL import Image as I
sample=I.open(TMP/f"new_{lo:06d}.png"); rowh=sample.height*cell//sample.width
canvas=Image.new("RGB",(cols*(cell+pad)+pad, rows*(rowh+pad)+pad),(235,235,235))
for t,i in enumerate(items):
    im=I.open(TMP/f"new_{i:06d}.png").convert("RGB")
    sc=cell/im.width; im=im.resize((cell,int(im.height*sc)))
    r,c=divmod(t,cols); x=pad+c*(cell+pad); y=pad+r*(rowh+pad)
    canvas.paste(im,(x,y))
out=TMP/f"newsheet_{lo}_{hi}.png"; canvas.save(out); print(out,canvas.size)
